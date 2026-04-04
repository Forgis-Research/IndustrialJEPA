"""
Extend few-shot results with 2 additional seeds (seeds 7 and 8) to get 5 seeds total.
Only runs JEPA V2 and Transformer supervised at critical N values (10, all)
to minimize runtime while improving statistical power.

Extends fewshot_curves.json with the new measurements.
"""

import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from src.models import MechanicalJEPAV2
from baselines_comparison import TransformerSupervised, train_supervised_model
from paderborn_transfer import create_paderborn_loaders, CLASSES
from fewshot_transfer_curves import get_paderborn_loaders, subsample_loader

PADERBORN_DIR = Path('/home/sagemaker-user/IndustrialJEPA/datasets/data/paderborn')
JEPA_V2_CKPT = Path(__file__).parent / 'checkpoints/jepa_v2_20260401_003619.pt'
CWRU_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/data/bearings')
RESULTS_FILE = Path(__file__).parent / 'results/fewshot_curves.json'

N_SHOTS_CRITICAL = [10, -1]  # Only the two critical points
EXTRA_SEEDS = [7, 8]  # Two new seeds (not used before)
SUB_SEEDS = [0, 1, 2]  # 3 sub-seeds per main seed

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def run_linear_probe(train_loader, test_loader, embed_dim=512, n_classes=3, device=None, epochs=100, lr=1e-3):
    """Train a linear probe on pre-extracted embeddings."""
    if device is None:
        device = DEVICE
    probe = nn.Linear(embed_dim, n_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Extract all to tensors
    X_tr, y_tr = [], []
    for batch in train_loader:
        X_tr.append(batch[0])
        y_tr.append(batch[1] if isinstance(batch[1], torch.Tensor) else torch.tensor(batch[1]))
    X_tr = torch.cat(X_tr).to(device)
    y_tr = torch.cat(y_tr).to(device)

    for ep in range(epochs):
        probe.train()
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), 32):
            idx = perm[i:i+32]
            loss = loss_fn(probe(X_tr[idx]), y_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    probe.eval()
    X_te, y_te = [], []
    for batch in test_loader:
        X_te.append(batch[0])
        y_te.append(batch[1] if isinstance(batch[1], torch.Tensor) else torch.tensor(batch[1]))
    X_te = torch.cat(X_te).to(device)
    y_te = torch.cat(y_te)

    with torch.no_grad():
        preds = probe(X_te).argmax(dim=1).cpu().numpy()
    return f1_score(y_te.numpy(), preds, average='macro')


def get_jepa_embeddings(loader, device=DEVICE):
    """Extract JEPA V2 embeddings from a data loader."""
    ckpt = torch.load(str(JEPA_V2_CKPT), map_location=device, weights_only=False)
    model = MechanicalJEPAV2(
        n_channels=3, window_size=4096, patch_size=256, embed_dim=512,
        encoder_depth=4, predictor_depth=4, n_heads=4,
        mask_ratio=0.625, ema_decay=0.996, predictor_pos='sinusoidal',
        loss_fn='l1', var_reg_lambda=0.1,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    all_embs, all_y = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            emb = model.get_embeddings(x, pool='mean')
            all_embs.append(emb.cpu())
            y = batch[1]
            all_y.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))

    return torch.cat(all_embs), torch.cat(all_y)


def get_transformer_embeddings(loader, seed, device=DEVICE):
    """Extract Transformer supervised embeddings (trained on CWRU)."""
    from src.data import create_dataloaders
    cwru_train_loader, cwru_test_loader, _ = create_dataloaders(
        data_dir=CWRU_DIR, batch_size=32, window_size=4096, stride=2048,
        test_ratio=0.2, seed=seed, num_workers=0, n_channels=3,
        dataset_filter='cwru',  # Only CWRU bearings, same as original fewshot script
    )
    tr_model = TransformerSupervised(n_classes=4).to(device)
    tr_model = train_supervised_model(tr_model, cwru_train_loader, cwru_test_loader,
                                      epochs=100, lr=1e-4, device=device)
    tr_model.eval()

    all_embs, all_y = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            emb = tr_model.get_embeddings(x)
            all_embs.append(emb.cpu())
            y = batch[1]
            all_y.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))

    return torch.cat(all_embs), torch.cat(all_y)


def main():
    print(f"Device: {DEVICE}")
    print(f"Extra seeds: {EXTRA_SEEDS}, Sub-seeds: {SUB_SEEDS}")
    print(f"Critical N values: {N_SHOTS_CRITICAL}")

    # Load existing results
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    # Track new measurements for the two critical methods
    new_measurements = {
        'jepa_v2': {'10': [], '-1': []},
        'transformer_supervised': {'10': [], '-1': []},
    }

    for seed in EXTRA_SEEDS:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        # Load Paderborn train and test
        pad_train_loader, pad_test_loader = get_paderborn_loaders(seed=seed)
        if pad_train_loader is None:
            print("ERROR: Paderborn not available"); continue

        # Extract JEPA embeddings (fast — no training needed)
        print("Extracting JEPA V2 embeddings...")
        t0 = time.time()
        train_emb, train_y = get_jepa_embeddings(pad_train_loader)
        test_emb, test_y = get_jepa_embeddings(pad_test_loader)
        print(f"  Done in {time.time()-t0:.1f}s. train={train_emb.shape}, test={test_emb.shape}")

        for n in N_SHOTS_CRITICAL:
            for sub_seed in SUB_SEEDS:
                # Subsample training embeddings
                if n > 0:
                    sub_train_loader = subsample_loader(
                        DataLoader(TensorDataset(train_emb, train_y), batch_size=32),
                        n_per_class=n, seed=sub_seed
                    )
                else:
                    sub_train_loader = DataLoader(TensorDataset(train_emb, train_y), batch_size=32, shuffle=True)

                test_loader_emb = DataLoader(TensorDataset(test_emb, test_y), batch_size=256)
                f1 = run_linear_probe(sub_train_loader, test_loader_emb, embed_dim=512, n_classes=3)
                n_key = str(n)
                new_measurements['jepa_v2'][n_key].append(f1)
                print(f"  JEPA@N={n}: F1={f1:.3f} (sub_seed={sub_seed})")

        # Extract Transformer embeddings (requires training CWRU model — ~3-4 min)
        print("Training + extracting Transformer supervised embeddings...")
        t0 = time.time()
        try:
            tr_train_emb, tr_train_y = get_transformer_embeddings(pad_train_loader, seed=seed)
            tr_test_emb, tr_test_y = get_transformer_embeddings(pad_test_loader, seed=seed)
            print(f"  Done in {time.time()-t0:.1f}s.")

            for n in N_SHOTS_CRITICAL:
                for sub_seed in SUB_SEEDS:
                    if n > 0:
                        sub_train_loader = subsample_loader(
                            DataLoader(TensorDataset(tr_train_emb, tr_train_y), batch_size=32),
                            n_per_class=n, seed=sub_seed
                        )
                    else:
                        sub_train_loader = DataLoader(TensorDataset(tr_train_emb, tr_train_y), batch_size=32, shuffle=True)

                    test_loader_emb = DataLoader(TensorDataset(tr_test_emb, tr_test_y), batch_size=256)
                    f1 = run_linear_probe(sub_train_loader, test_loader_emb, embed_dim=512, n_classes=3)
                    n_key = str(n)
                    new_measurements['transformer_supervised'][n_key].append(f1)
                    print(f"  Transformer@N={n}: F1={f1:.3f} (sub_seed={sub_seed})")
        except Exception as e:
            print(f"Transformer error: {e}")
            import traceback; traceback.print_exc()

    # Merge new measurements into results
    print(f"\n{'='*60}")
    print("Merging new measurements...")
    for method in ['jepa_v2', 'transformer_supervised']:
        for n_key in ['10', '-1']:
            new_vals = new_measurements[method][n_key]
            if not new_vals:
                continue

            old_vals = results.get(method, {}).get(n_key, {}).get('all', [])
            combined = old_vals + new_vals
            n_combined = len(combined)

            results[method][n_key]['mean'] = np.mean(combined)
            results[method][n_key]['std'] = np.std(combined)
            results[method][n_key]['all'] = combined
            results[method][n_key]['n'] = n_combined

            print(f"  {method}@N={n_key}: {old_vals} + {new_vals} = "
                  f"{np.mean(combined):.3f} +/- {np.std(combined):.3f} (n={n_combined})")

    # Save updated results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nUpdated results saved to {RESULTS_FILE}")

    # Print final comparison
    jepa_10 = results['jepa_v2']['10']
    tr_all = results['transformer_supervised']['-1']
    print(f"\nFinal comparison (n={jepa_10.get('n', 9)} measurements):")
    print(f"  JEPA V2 @ N=10: {jepa_10['mean']:.3f} +/- {jepa_10['std']:.3f}")
    print(f"  Transformer @ N=all: {tr_all['mean']:.3f} +/- {tr_all['std']:.3f}")

    # New t-test
    from scipy import stats
    n1 = jepa_10.get('n', 9)
    n2 = tr_all.get('n', 9)
    se_diff = np.sqrt(jepa_10['std']**2/n1 + tr_all['std']**2/n2)
    t_stat = (jepa_10['mean'] - tr_all['mean']) / se_diff if se_diff > 0 else 0
    df = n1 + n2 - 2
    p_val = 1 - stats.t.cdf(t_stat, df=df)  # one-sided
    print(f"  t={t_stat:.3f}, p={p_val:.4f} (one-sided), n1={n1}, n2={n2}")


if __name__ == '__main__':
    main()
