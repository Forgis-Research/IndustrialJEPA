"""
Phase 7: Valid Frozen Probe Test RMSE for V16b.

The Phase 1 val RMSE (frozen probe) is measured on val_engines with use_last_only=True.
ALL val engines have RUL=1 at their last window (protocol blindspot from session).

This script evaluates the frozen probe on the TEST set (diverse RUL values),
giving a valid frozen probe test RMSE for comparison with:
- E2E test RMSE: 15.06 +/- 1.15
- Feature regressor test RMSE: 17.72
- V2 frozen probe val RMSE: 17.81 (on same biased val protocol)
- Supervised SOTA (STAR): 10.61

If frozen probe test RMSE < SOTA (10.61), we have a genuine claim.
If frozen probe test RMSE > E2E test RMSE (15.06), it's consistent with
  "E2E fine-tuning helps substantially over frozen probe on test."

Seeds: all 3 V16b checkpoints.
Protocol: train linear probe on train_engines (5 cuts each), eval on test_engines.
"""
import sys, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V16_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v16')
sys.path.insert(0, str(V11_DIR))
sys.path.insert(0, str(V16_DIR))

from data_utils import (
    load_cmapss_subset, N_SENSORS, RUL_CAP,
    CMAPSSFinetuneDataset, CMAPSSTestDataset,
    collate_finetune, collate_test,
)
from phase1_v16a import V16aJEPA, D_MODEL, N_HEADS, N_LAYERS, EMA_MOMENTUM

DEVICE = torch.device('cpu')  # Use CPU to avoid OOM while Phase 2 runs on GPU
PROBE_SEEDS = [42, 123, 456]
N_PROBE_EPOCHS = 100


def extract_features_loader(model, loader):
    """Extract encoder features from DataLoader. Returns (features, labels).
    Both collate_finetune and collate_test return (past, mask, rul) tuples.
    mask: True=PADDING, False=valid position.
    """
    model.eval()
    all_feats, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            # Both collate_finetune and collate_test return (past, mask, rul)
            # mask: True=PADDING, False=valid
            past, mask, rul = batch
            mask = mask.to(DEVICE)

            past = past.to(DEVICE)
            feat = model.encode_context(past, mask)
            all_feats.append(feat.cpu().numpy())
            all_labels.append(rul.numpy() if hasattr(rul, 'numpy') else np.array(rul))

    return np.concatenate(all_feats), np.concatenate(all_labels)


def train_probe_and_eval(tr_feats, tr_labs, te_feats, te_labs, seed=42):
    """Train linear probe on train features, evaluate on test features."""
    torch.manual_seed(seed)
    D = tr_feats.shape[1]
    probe = nn.Linear(D, 1).to(DEVICE)
    optim = torch.optim.Adam(probe.parameters(), lr=1e-3)

    tr_f = torch.tensor(tr_feats, dtype=torch.float32).to(DEVICE)
    tr_l = torch.tensor(tr_labs, dtype=torch.float32).to(DEVICE)

    # Normalize labels for training (probe is trained on normalized RUL)
    # tr_labs are in [0,1] (normalized by RUL_CAP from CMAPSSFinetuneDataset)
    # te_labs are raw RUL in cycles from CMAPSSTestDataset

    best_val_rmse = float('inf')
    best_state = None

    for ep in range(N_PROBE_EPOCHS):
        probe.train()
        idx = torch.randperm(len(tr_f), device=DEVICE)
        for i in range(0, len(tr_f), 64):
            batch_idx = idx[i:i+64]
            pred = probe(tr_f[batch_idx]).squeeze(-1)
            loss = F.mse_loss(pred, tr_l[batch_idx])
            optim.zero_grad()
            loss.backward()
            optim.step()

    probe.eval()
    te_f = torch.tensor(te_feats, dtype=torch.float32).to(DEVICE)
    te_l = torch.tensor(te_labs, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        te_pred = probe(te_f).squeeze(-1)
        # te_pred is in normalized [0,1] scale, te_labs are raw
        # Convert pred back to cycles: pred * RUL_CAP
        te_pred_raw = te_pred * RUL_CAP
        te_rmse = float(torch.sqrt(F.mse_loss(te_pred_raw, te_l)))

    return te_rmse


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    print("=" * 60)
    print("Phase 7: Valid Frozen Probe Test RMSE for V16b")
    print("=" * 60)
    print("  Protocol blindspot: val_engines all have RUL=1 at last window.")
    print("  This script uses TEST set for valid frozen probe evaluation.")
    print()

    data = load_cmapss_subset('FD001')
    train_engines = data['train_engines']
    test_engines = data['test_engines']
    test_rul = data['test_rul']

    print(f"  Train engines: {len(train_engines)}")
    print(f"  Test engines: {len(test_engines)}")
    print(f"  Test RUL range: [{test_rul.min():.0f}, {test_rul.max():.0f}], mean={test_rul.mean():.1f}")
    print()

    results = {
        'description': 'Valid frozen probe test RMSE - protocol blindspot fix',
        'test_rul_stats': {
            'min': float(test_rul.min()),
            'max': float(test_rul.max()),
            'mean': float(test_rul.mean()),
            'std': float(test_rul.std()),
        },
        'seeds': {}
    }

    # Use CMAPSSTestDataset for test (last window per engine, with ground truth RUL)
    te_ds = CMAPSSTestDataset(test_engines, test_rul)
    te_loader = DataLoader(te_ds, batch_size=64, shuffle=False,
                           collate_fn=collate_test)

    all_seed_rmses = []

    for ckpt_seed in [42, 123, 456]:
        ckpt_path = V16_DIR / f'best_v16b_seed{ckpt_seed}.pt'
        if not ckpt_path.exists():
            print(f"  Checkpoint missing: {ckpt_path}")
            continue

        print(f"Loading checkpoint: best_v16b_seed{ckpt_seed}.pt")
        model = V16aJEPA(
            n_sensors=N_SENSORS, d_model=D_MODEL, n_heads=N_HEADS,
            n_layers=N_LAYERS, dropout=0.1, ema_momentum=EMA_MOMENTUM,
        ).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.eval()

        # Train probe on training data
        tr_ds = CMAPSSFinetuneDataset(train_engines, n_cuts_per_engine=5, seed=ckpt_seed)
        tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=False, collate_fn=collate_finetune)

        print("  Extracting train features...")
        tr_feats, tr_labs = extract_features_loader(model, tr_loader)

        print("  Extracting test features...")
        te_feats, te_labs = extract_features_loader(model, te_loader)

        # Train and eval probe for each probe seed
        probe_rmses = []
        for pseed in PROBE_SEEDS:
            rmse = train_probe_and_eval(tr_feats, tr_labs, te_feats, te_labs, seed=pseed)
            probe_rmses.append(rmse)

        mean_rmse = float(np.mean(probe_rmses))
        std_rmse = float(np.std(probe_rmses))
        all_seed_rmses.append(mean_rmse)

        results['seeds'][str(ckpt_seed)] = {
            'test_rmse_per_probe_seed': [float(r) for r in probe_rmses],
            'test_rmse_mean': mean_rmse,
            'test_rmse_std': std_rmse,
        }
        print(f"  Checkpoint seed {ckpt_seed}: test RMSE = {mean_rmse:.2f} +/- {std_rmse:.2f}")
        print()

    if all_seed_rmses:
        overall_mean = float(np.mean(all_seed_rmses))
        overall_std = float(np.std(all_seed_rmses))
        results['overall_test_rmse_mean'] = overall_mean
        results['overall_test_rmse_std'] = overall_std

        print("=" * 60)
        print("PHASE 7 SUMMARY: Frozen Probe Test RMSE")
        print("=" * 60)
        for ckpt_seed, rmse in zip([42, 123, 456], all_seed_rmses):
            print(f"  Checkpoint seed {ckpt_seed}: {rmse:.2f}")
        print(f"  Overall: {overall_mean:.2f} +/- {overall_std:.2f}")
        print()
        print(f"  Supervised SOTA (STAR): 10.61")
        print(f"  V16b E2E test: 15.06 +/- 1.15")
        print(f"  Feature regressor test: 17.72")
        print(f"  V2 E2E test: 14.23 +/- 0.39")
        print()
        if overall_mean < 10.61:
            print("  RESULT: Frozen probe BEATS supervised SOTA on test set!")
            print("  THIS IS A GENUINE CLAIM - test set has diverse RUL.")
        elif overall_mean < 14.23:
            print("  RESULT: Frozen probe competitive with V2 E2E on test set.")
        elif overall_mean < 15.06:
            print("  RESULT: Frozen probe competitive with V16b E2E on test set.")
        elif overall_mean < 17.72:
            print("  RESULT: Frozen probe beats feature regressor on test set.")
            print("  Encoder contributes signal. E2E fine-tuning adds more.")
        else:
            print("  RESULT: Frozen probe does NOT beat feature regressor on test set.")
            print("  Encoder provides negligible value over hand-crafted features.")

    out_path = V16_DIR / 'phase7_frozen_probe_test_rmse.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
