"""
Cross-method comparison: MTS-JEPA vs CC-JEPA vs Trajectory JEPA Adapter.
Runs on PSM and MSL with matched hyperparameters for fair comparison.
"""
import os
import sys
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import prepare_data, RevIN
from train_utils import (
    compute_total_loss, DEFAULT_LOSS_CONFIG,
    full_downstream_evaluation, encode_windows,
    train_downstream_classifier, select_threshold,
    evaluate_downstream,
)
from models import MTSJEPA, DownstreamClassifier
from cc_jepa import CCJEPA

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "comparison")


def pretrain_model(model, train_loader, val_loader, n_vars, config, device):
    """Generic pre-training for any model with forward() returning losses dict."""
    lr = config.get('lr', 5e-4)
    n_epochs = config.get('n_epochs', 100)
    patience = config.get('patience', 10)
    patience_start = config.get('patience_start', 50)

    optimizer = torch.optim.Adam(model.online_params(), lr=lr, weight_decay=1e-5)
    revin = RevIN(n_vars).to(device)

    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        for x_ctx, x_tgt in train_loader:
            x_ctx, x_tgt = x_ctx.to(device), x_tgt.to(device)
            x_ctx_n, x_tgt_n = revin(x_ctx), revin(x_tgt)

            losses = model(x_ctx_n, x_tgt_n)
            total, _ = compute_total_loss(losses, config.get('loss_config', DEFAULT_LOSS_CONFIG), epoch, n_epochs)

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.online_params(), 0.5)
            optimizer.step()
            model.update_ema()

        # Validation
        model.eval()
        val_loss = 0
        n = 0
        with torch.no_grad():
            for x_ctx, x_tgt in val_loader:
                x_ctx, x_tgt = x_ctx.to(device), x_tgt.to(device)
                losses = model(revin(x_ctx), revin(x_tgt))
                total, comps = compute_total_loss(losses, config.get('loss_config', DEFAULT_LOSS_CONFIG), epoch, n_epochs)
                val_loss += comps['L_total']
                n += 1
        avg_val = val_loss / max(n, 1)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        elif epoch >= patience_start:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)

    return {'best_epoch': best_epoch, 'best_val_loss': best_val_loss}


def downstream_eval_generic(model, data_dict, device, seed=42):
    """Downstream eval for any model with encode_for_downstream()."""
    n_vars = data_dict['n_vars']
    model.eval()

    # Encode
    revin = RevIN(n_vars).to(device)
    revin.eval()

    def encode(windows):
        feats = []
        with torch.no_grad():
            for i in range(0, len(windows), 256):
                batch = torch.tensor(windows[i:i+256], dtype=torch.float32).to(device)
                batch_n = revin(batch)
                f = model.encode_for_downstream(batch_n)
                feats.append(f.cpu().numpy())
        return np.concatenate(feats, axis=0)

    ds_train_ctx, ds_train_labels = data_dict['downstream_train']
    ds_val_ctx, ds_val_labels = data_dict['downstream_val']
    ds_test_ctx, ds_test_labels = data_dict['downstream_test']

    feat_train = encode(ds_train_ctx)
    feat_val = encode(ds_val_ctx)
    feat_test = encode(ds_test_ctx)

    input_dim = feat_train.shape[1]

    torch.manual_seed(seed)
    classifier = DownstreamClassifier(input_dim, hidden_dim=256).to(device)
    classifier = train_downstream_classifier(
        feat_train, ds_train_labels, feat_val, ds_val_labels,
        input_dim, device
    )

    threshold, val_f1 = select_threshold(classifier, feat_val, ds_val_labels, device)
    results = evaluate_downstream(classifier, feat_test, ds_test_labels, threshold, device)
    return results


def run_comparison(dataset_name='PSM', seed=42, device='cuda'):
    """Run full comparison between MTS-JEPA and CC-JEPA."""
    print(f"\n{'#'*60}")
    print(f"# Comparison: {dataset_name} | seed={seed}")
    print(f"{'#'*60}")

    data_dict = prepare_data(dataset_name, window_length=100, batch_size=32)
    n_vars = data_dict['n_vars']

    config = {
        'lr': 5e-4, 'n_epochs': 100, 'patience': 10, 'patience_start': 50,
        'd_model': 128, 'd_out': 128, 'n_codes': 64, 'tau': 0.1,
        'patch_length': 20, 'n_patches': 5, 'n_encoder_layers': 3,
        'n_heads': 4, 'dropout': 0.1, 'ema_rho': 0.996,
        'loss_config': DEFAULT_LOSS_CONFIG,
    }

    results = {}

    # 1. MTS-JEPA (channel-independent)
    print("\n--- MTS-JEPA ---")
    torch.manual_seed(seed)
    mts_model = MTSJEPA(
        n_vars=n_vars, d_model=128, d_out=128, n_codes=64, tau=0.1,
        patch_length=20, n_patches=5, n_encoder_layers=3,
        n_heads=4, dropout=0.1,
    ).to(device)

    start = time.time()
    pretrain_info = pretrain_model(
        mts_model, data_dict['pretrain_train_loader'],
        data_dict['pretrain_val_loader'], n_vars, config, device
    )
    mts_downstream = downstream_eval_generic(mts_model, data_dict, device, seed)
    mts_time = time.time() - start

    results['MTS-JEPA'] = {
        'pretrain': pretrain_info,
        'downstream': mts_downstream,
        'wall_time': mts_time,
    }
    print(f"  F1={mts_downstream['f1']:.2f}, AUC={mts_downstream['auc']:.2f}, time={mts_time:.0f}s")

    # 2. CC-JEPA (causal multivariate)
    print("\n--- CC-JEPA ---")
    torch.manual_seed(seed)
    cc_model = CCJEPA(
        n_vars=n_vars, d_model=128, d_out=128, n_codes=64, tau=0.1,
        patch_length=20, n_patches=5, n_encoder_layers=3,
        n_heads=4, dropout=0.1,
    ).to(device)

    start = time.time()
    pretrain_info = pretrain_model(
        cc_model, data_dict['pretrain_train_loader'],
        data_dict['pretrain_val_loader'], n_vars, config, device
    )
    cc_downstream = downstream_eval_generic(cc_model, data_dict, device, seed)
    cc_time = time.time() - start

    results['CC-JEPA'] = {
        'pretrain': pretrain_info,
        'downstream': cc_downstream,
        'wall_time': cc_time,
    }
    print(f"  F1={cc_downstream['f1']:.2f}, AUC={cc_downstream['auc']:.2f}, time={cc_time:.0f}s")

    # Summary
    print(f"\n{'='*60}")
    print(f"COMPARISON: {dataset_name} (seed={seed})")
    print(f"{'='*60}")
    print(f"{'Method':<15s} | {'F1':>6s} | {'AUC':>6s} | {'Prec':>6s} | {'Rec':>6s} | {'Time':>5s}")
    print("-" * 60)
    for method, r in results.items():
        d = r['downstream']
        print(f"{method:<15s} | {d['f1']:6.2f} | {d['auc']:6.2f} | "
              f"{d['precision']:6.2f} | {d['recall']:6.2f} | {r['wall_time']:5.0f}s")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"{dataset_name}_seed{seed}.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['PSM', 'MSL'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123])
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    all_results = {}
    for ds in args.datasets:
        for seed in args.seeds:
            results = run_comparison(ds, seed, args.device)
            all_results[f"{ds}_seed{seed}"] = results
