"""
V16a E2E Evaluation (with bug fix).

The original phase1_v16a.py had a bug in eval_e2e:
  CMAPSSTestDataset returns RAW cycles (not normalized).
  But the code multiplied targets by RUL_CAP=125 again -> 125x too large.

This script loads each saved checkpoint and evaluates E2E correctly.

Bug fix:
  OLD: targets = np.concatenate(targets) * RUL_CAP  (wrong)
  NEW: targets = np.concatenate(targets)  (correct - already raw cycles)

Outputs:
  experiments/v16/phase1_v16a_e2e_corrected.json
"""

import sys, json, copy
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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 123, 456]


def eval_e2e_fixed(model, data, seed=42, lr=3e-4, n_epochs=100):
    """
    Fine-tune entire model end-to-end. Returns test RMSE.
    BUG FIX: CMAPSSTestDataset returns raw cycles, don't multiply by RUL_CAP.
    """
    torch.manual_seed(seed)
    model_ft = copy.deepcopy(model)
    probe = nn.Linear(D_MODEL, 1).to(DEVICE)

    all_params = list(model_ft.parameters()) + list(probe.parameters())
    optim = torch.optim.AdamW(all_params, lr=lr, weight_decay=0.01)

    tr_ds = CMAPSSFinetuneDataset(data['train_engines'], n_cuts_per_engine=10, seed=seed)
    va_ds = CMAPSSFinetuneDataset(data['val_engines'], use_last_only=True)
    te_ds = CMAPSSTestDataset(data['test_engines'], data['test_rul'])
    tr = DataLoader(tr_ds, batch_size=32, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(va_ds, batch_size=32, shuffle=False, collate_fn=collate_finetune)
    te = DataLoader(te_ds, batch_size=32, shuffle=False, collate_fn=collate_test)

    best_val = float('inf')
    best_state = copy.deepcopy({'model': model_ft.state_dict(), 'probe': probe.state_dict()})
    patience = 15
    no_impr = 0

    for ep in range(n_epochs):
        model_ft.train()
        probe.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            optim.zero_grad()
            h = model_ft.encode_context(past, mask)
            loss = F.mse_loss(probe(h).squeeze(-1), rul)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optim.step()

        model_ft.eval()
        probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                pv.append(probe(model_ft.encode_context(past, mask)).squeeze(-1).cpu().numpy())
                tv.append(rul.numpy())
        val_rmse = float(np.sqrt(np.mean(
            (np.concatenate(pv) * RUL_CAP - np.concatenate(tv) * RUL_CAP) ** 2)))

        if val_rmse < best_val:
            best_val = val_rmse
            best_state = copy.deepcopy({'model': model_ft.state_dict(), 'probe': probe.state_dict()})
            no_impr = 0
        else:
            no_impr += 1
            if no_impr >= patience:
                break

    # Load best model and evaluate on test
    model_ft.load_state_dict(best_state['model'])
    probe.load_state_dict(best_state['probe'])
    model_ft.eval()
    probe.eval()
    preds, targets = [], []
    with torch.no_grad():
        for past, mask, rul in te:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            preds.append(probe(model_ft.encode_context(past, mask)).squeeze(-1).cpu().numpy())
            targets.append(rul.numpy())

    preds = np.concatenate(preds) * RUL_CAP
    targets = np.concatenate(targets)  # BUG FIX: already raw cycles, don't multiply by RUL_CAP
    test_rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))

    print(f"  Val RMSE: {best_val:.2f}, Test RMSE: {test_rmse:.2f}")
    return test_rmse, best_val


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    print("=" * 60)
    print("V16a E2E Evaluation (Bug Fixed)")
    print("Fix: CMAPSSTestDataset returns raw cycles (not normalized)")
    print("Original phase1_v16a.py multiplied targets by RUL_CAP=125 again")
    print("=" * 60)

    print("\nLoading FD001 data...")
    data = load_cmapss_subset('FD001')

    results = {
        'description': 'V16a E2E evaluation with corrected test RMSE computation',
        'bug_fix': 'targets = np.concatenate(targets) not * RUL_CAP',
        'seeds': SEEDS,
        'e2e_rmse_per_seed': [],
        'val_rmse_per_seed': [],
    }

    for seed in SEEDS:
        ckpt_path = V16_DIR / f'best_v16a_seed{seed}.pt'
        if not ckpt_path.exists():
            print(f"\nSeed {seed}: checkpoint not found at {ckpt_path}")
            print("  Skipping - wait for pretraining to complete")
            results['e2e_rmse_per_seed'].append(None)
            results['val_rmse_per_seed'].append(None)
            continue

        print(f"\nSeed {seed}: Loading checkpoint from {ckpt_path}")
        model = V16aJEPA(
            n_sensors=N_SENSORS, d_model=D_MODEL, n_heads=N_HEADS,
            n_layers=N_LAYERS, dropout=0.1, ema_momentum=EMA_MOMENTUM,
        ).to(DEVICE)
        state_dict = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"  Checkpoint loaded (probe=best at time of save)")

        test_rmse, val_rmse = eval_e2e_fixed(model, data, seed=seed)
        results['e2e_rmse_per_seed'].append(test_rmse)
        results['val_rmse_per_seed'].append(val_rmse)

        # Save intermediate results
        with open(V16_DIR / 'phase1_v16a_e2e_corrected.json', 'w') as f:
            json.dump(results, f, indent=2)

    # Compute summary statistics (only for completed seeds)
    valid_e2e = [x for x in results['e2e_rmse_per_seed'] if x is not None]
    if valid_e2e:
        results['e2e_mean'] = float(np.mean(valid_e2e))
        results['e2e_std'] = float(np.std(valid_e2e))
    else:
        results['e2e_mean'] = None
        results['e2e_std'] = None

    print("\n" + "=" * 60)
    print("V16a E2E Summary (corrected)")
    print("=" * 60)
    for seed, e2e in zip(SEEDS, results['e2e_rmse_per_seed']):
        if e2e is None:
            print(f"  Seed {seed}: SKIPPED (checkpoint not found)")
        else:
            print(f"  Seed {seed}: E2E test RMSE = {e2e:.2f}")

    if valid_e2e:
        print(f"\n  Mean: {results['e2e_mean']:.2f} +/- {results['e2e_std']:.2f}")
        print(f"  V2 E2E baseline: 14.23 +/- 0.39")

    with open(V16_DIR / 'phase1_v16a_e2e_corrected.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {V16_DIR / 'phase1_v16a_e2e_corrected.json'}")
