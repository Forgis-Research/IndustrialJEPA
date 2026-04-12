"""
Benchmark Trajectory JEPA (adapted) on anomaly prediction datasets.
Uses the same pre-train -> freeze -> downstream pipeline as MTS-JEPA.
"""
import os
import sys
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import prepare_data, RevIN
from trajectory_jepa_adapter import (
    TrajectoryJEPAForAnomalyPrediction,
    pretrain_trajectory_jepa,
)
from train_utils import (
    train_downstream_classifier, select_threshold,
    evaluate_downstream,
)
from models import DownstreamClassifier

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "traj_jepa")


def encode_windows_traj(model, windows, n_vars, device, batch_size=256):
    """Encode windows using Trajectory JEPA."""
    model.eval()
    revin = RevIN(n_vars).to(device)
    revin.eval()

    all_features = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = torch.tensor(windows[i:i+batch_size], dtype=torch.float32).to(device)
            batch_n = revin(batch)
            features = model.encode_for_downstream(batch_n)  # (B, d_model)
            all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def run_traj_jepa_benchmark(dataset_name, seed=42, device='cuda',
                             d_model=128, n_layers=2, n_epochs=100):
    """Run Trajectory JEPA pre-train + downstream on anomaly prediction."""
    print(f"\n{'='*60}")
    print(f"Trajectory JEPA Benchmark: {dataset_name} | seed={seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    data_dict = prepare_data(dataset_name, window_length=100, batch_size=32)
    n_vars = data_dict['n_vars']

    # Build model
    model = TrajectoryJEPAForAnomalyPrediction(
        n_vars=n_vars,
        d_model=d_model,
        n_heads=4,
        n_layers=n_layers,
        d_ff=d_model * 2,
        dropout=0.1,
        ema_momentum=0.99,
        predictor_hidden=d_model,
        window_length=100,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Pre-train
    print(f"  Pre-training...")
    start = time.time()
    best_val = pretrain_trajectory_jepa(
        model, data_dict['pretrain_train_loader'],
        data_dict['pretrain_val_loader'],
        n_vars, device=device, n_epochs=n_epochs, lr=3e-4,
    )
    pretrain_time = time.time() - start
    print(f"  Pre-training done: val_loss={best_val:.4f}, time={pretrain_time:.0f}s")

    # Downstream
    print(f"  Downstream evaluation...")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    ds_train_ctx, ds_train_labels = data_dict['downstream_train']
    ds_val_ctx, ds_val_labels = data_dict['downstream_val']
    ds_test_ctx, ds_test_labels = data_dict['downstream_test']

    feat_train = encode_windows_traj(model, ds_train_ctx, n_vars, device)
    feat_val = encode_windows_traj(model, ds_val_ctx, n_vars, device)
    feat_test = encode_windows_traj(model, ds_test_ctx, n_vars, device)

    input_dim = feat_train.shape[1]
    print(f"  Feature dim: {input_dim}")

    torch.manual_seed(seed)
    classifier = train_downstream_classifier(
        feat_train, ds_train_labels,
        feat_val, ds_val_labels,
        input_dim, device,
    )

    threshold, val_f1 = select_threshold(classifier, feat_val, ds_val_labels, device)
    results = evaluate_downstream(classifier, feat_test, ds_test_labels, threshold, device)

    print(f"  F1={results['f1']:.2f}, AUC={results['auc']:.2f}, "
          f"Precision={results['precision']:.2f}, Recall={results['recall']:.2f}")

    # Unfreeze
    for p in model.parameters():
        p.requires_grad = True

    result = {
        'dataset': dataset_name,
        'seed': seed,
        'method': 'TrajectoryJEPA',
        'n_params': n_params,
        'pretrain': {
            'best_val_loss': best_val,
            'wall_time_seconds': pretrain_time,
        },
        'downstream': results,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, f"{dataset_name}_seed{seed}.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    return result


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
            result = run_traj_jepa_benchmark(ds, seed, args.device)
            all_results[f"{ds}_seed{seed}"] = result

    # Print summary
    print(f"\n{'#'*60}")
    print("Trajectory JEPA Benchmark Summary")
    print(f"{'#'*60}")
    for key, r in all_results.items():
        d = r['downstream']
        print(f"  {key}: F1={d['f1']:.2f}, AUC={d['auc']:.2f}")
