"""
V16 Phase 4: Cross-Machine Generalization.

Test zero-shot transfer: train on FD001, evaluate frozen probe on FD002/FD003/FD004.

Context:
- FD001: single operating condition, 100 engines, 14 sensors
- FD002: 6 operating conditions, 260 engines, 14 sensors (harder)
- FD003: single condition + HPC degradation, 100 engines
- FD004: 6 conditions + HPC degradation, 249 engines (hardest)

V2 baseline cross-machine: FD002 frozen ~28 (estimated from V12)
V14 cross-sensor FD003 frozen: 17.75 +/- 0.58

For each model checkpoint (V2, V15-SIGReg best, V16a):
1. Load checkpoint (pretrained on FD001 only)
2. For each target domain (FD002/FD003/FD004):
   a. Train frozen linear probe on target domain train data
   b. Evaluate on target domain test data
3. Report RMSE for each target domain

This tests whether bidi context + SIGReg creates more transferable representations.

Outputs:
  experiments/v16/phase4_cross_machine_results.json
"""

import sys, json, time, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V14_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v14')
V15_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v15')
V16_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v16')
sys.path.insert(0, str(V11_DIR))
sys.path.insert(0, str(V15_DIR))

from data_utils import (
    load_cmapss_subset, N_SENSORS, RUL_CAP,
    CMAPSSFinetuneDataset, CMAPSSTestDataset,
    collate_finetune, collate_test,
)
from models import TrajectoryJEPA, RULProbe

try:
    import wandb
    HAS_WANDB = True
except Exception:
    HAS_WANDB = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
V16_DIR.mkdir(parents=True, exist_ok=True)

TARGET_DOMAINS = ['FD002', 'FD003', 'FD004']
D_MODEL = 256


def eval_frozen_probe(encode_fn, train_engines, val_engines, test_engines, test_rul,
                      d_model=D_MODEL, seed=42):
    """
    Train frozen probe on target domain train data.
    Evaluate on target domain test data.
    Returns val_rmse, test_rmse.
    """
    torch.manual_seed(seed)
    probe = nn.Linear(d_model, 1).to(DEVICE)
    optim = torch.optim.Adam(probe.parameters(), lr=1e-3)

    tr_ds = CMAPSSFinetuneDataset(train_engines, n_cuts_per_engine=5, seed=seed)
    va_ds = CMAPSSFinetuneDataset(val_engines, use_last_only=True)
    te_ds = CMAPSSTestDataset(test_engines, test_rul)

    tr = DataLoader(tr_ds, batch_size=32, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(va_ds, batch_size=32, shuffle=False, collate_fn=collate_finetune)
    te = DataLoader(te_ds, batch_size=32, shuffle=False, collate_fn=collate_test)

    best_val = float('inf')
    best_state = copy.deepcopy(probe.state_dict())
    no_impr = 0

    for ep in range(100):
        probe.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            optim.zero_grad()
            with torch.no_grad():
                h = encode_fn(past, mask)
            loss = F.mse_loss(probe(h).squeeze(-1), rul)
            loss.backward()
            optim.step()

        probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                pv.append(probe(encode_fn(past, mask)).squeeze(-1).cpu().numpy())
                tv.append(rul.numpy())
        val_rmse = float(np.sqrt(np.mean(
            (np.concatenate(pv) * RUL_CAP - np.concatenate(tv) * RUL_CAP) ** 2)))

        if val_rmse < best_val:
            best_val = val_rmse
            best_state = copy.deepcopy(probe.state_dict())
            no_impr = 0
        else:
            no_impr += 1
            if no_impr >= 10:
                break

    # Load best probe and evaluate on test
    probe.load_state_dict(best_state)
    probe.eval()
    preds, targets = [], []
    with torch.no_grad():
        for past, mask, rul in te:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            preds.append(probe(encode_fn(past, mask)).squeeze(-1).cpu().numpy())
            targets.append(rul.numpy())
    preds = np.concatenate(preds) * RUL_CAP
    targets = np.concatenate(targets)  # CMAPSSTestDataset returns raw cycles (not normalized)
    test_rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))

    return best_val, test_rmse


def evaluate_model_cross_machine(model_name: str, encode_fn, results: dict):
    """
    Evaluate cross-machine transfer for a model.
    Freezes the encoder and trains a new probe on each target domain.
    """
    print(f"\n{'='*50}")
    print(f"Cross-Machine: {model_name}")
    print(f"{'='*50}")

    domain_results = {}
    for fd in TARGET_DOMAINS:
        print(f"\n  Target domain: {fd}")
        data = load_cmapss_subset(fd)
        print(f"    Train: {len(data['train_engines'])} engines, "
              f"Test: {len(data['test_engines'])} engines")

        # Average over 3 seeds for stable estimate
        val_rmses, test_rmses = [], []
        for seed in [42, 123, 456]:
            val_rmse, test_rmse = eval_frozen_probe(
                encode_fn, data['train_engines'], data['val_engines'],
                data['test_engines'], data['test_rul'], seed=seed)
            val_rmses.append(val_rmse)
            test_rmses.append(test_rmse)

        val_mean = float(np.mean(val_rmses))
        test_mean = float(np.mean(test_rmses))
        test_std = float(np.std(test_rmses))

        domain_results[fd] = {
            'val_rmse': val_mean,
            'test_rmse_mean': test_mean,
            'test_rmse_std': test_std,
            'test_rmse_per_seed': test_rmses,
        }
        print(f"    Val RMSE: {val_mean:.2f}")
        print(f"    Test RMSE: {test_mean:.2f} +/- {test_std:.2f}")

    results[model_name] = domain_results
    return domain_results


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    print("=" * 60)
    print("V16 Phase 4: Cross-Machine Generalization")
    print("Train on FD001, zero-shot frozen probe on FD002/FD003/FD004")
    print("=" * 60)

    results = {
        'description': 'Zero-shot transfer: FD001 pretrained encoder, new probe on target domain',
        'target_domains': TARGET_DOMAINS,
        'baselines': {
            'V2_FD001_frozen': 17.81,
            'V14_FD003_cross_sensor': 17.75,
        }
    }

    # --- V2 Baseline (V11 causal encoder) ---
    print("\nLoading V2 baseline (V11 TrajectoryJEPA)...")
    v2_ckpt = V14_DIR / 'best_pretrain_full_sequence.pt'
    if v2_ckpt.exists():
        v2_model = TrajectoryJEPA(
            n_sensors=N_SENSORS, d_model=D_MODEL, n_heads=4,
            n_layers=2, d_ff=512, ema_momentum=0.99
        ).to(DEVICE)
        v2_model.load_state_dict(torch.load(v2_ckpt, map_location=DEVICE))
        v2_model.eval()

        def v2_encode(past, mask):
            return v2_model.encode_past(past, mask)

        evaluate_model_cross_machine('V2_full_seq', v2_encode, results)
    else:
        print(f"  V2 checkpoint not found at {v2_ckpt}")
        results['V2_full_seq'] = 'SKIPPED - checkpoint not found'

    # --- V15-SIGReg best (use seed 42 checkpoint if available) ---
    # Note: V15 phase1 script doesn't save checkpoints to disk currently.
    # Will add checkpoint saving to phase1_sigreg.py separately.
    # For now, skip V15-SIGReg cross-machine eval.
    print("\n  V15-SIGReg checkpoint: Not saved to disk by phase1_sigreg.py")
    print("  Cross-machine eval requires checkpoint. Add checkpoint saving to phase1.")
    results['V15_SIGReg'] = 'SKIPPED - no checkpoint saved'

    # --- V16a best (will be available after phase1_v16a.py runs) ---
    v16a_ckpt = V16_DIR / 'best_v16a_seed42.pt'
    if v16a_ckpt.exists():
        sys.path.insert(0, str(V16_DIR))
        from phase1_v16a import V16aJEPA

        v16a_model = V16aJEPA(n_sensors=N_SENSORS).to(DEVICE)
        v16a_model.load_state_dict(torch.load(v16a_ckpt, map_location=DEVICE))
        v16a_model.eval()

        def v16a_encode(past, mask):
            return v16a_model.encode_context(past, mask)

        evaluate_model_cross_machine('V16a', v16a_encode, results)
    else:
        print(f"\nV16a checkpoint not found at {v16a_ckpt}")
        print("Run phase1_v16a.py first, then re-run this script.")
        results['V16a'] = 'PENDING - run phase1_v16a.py first'

    # --- V16b best (VICReg + LR warmup, all 3 seeds available) ---
    # V16b uses same V16aJEPA architecture, just different training regime
    v16b_ckpt = V16_DIR / 'best_v16b_seed42.pt'
    if v16b_ckpt.exists():
        from phase1_v16a import V16aJEPA

        v16b_model = V16aJEPA(n_sensors=N_SENSORS).to(DEVICE)
        v16b_model.load_state_dict(torch.load(v16b_ckpt, map_location=DEVICE))
        v16b_model.eval()

        def v16b_encode(past, mask):
            return v16b_model.encode_context(past, mask)

        evaluate_model_cross_machine('V16b', v16b_encode, results)
    else:
        print(f"\nV16b checkpoint not found at {v16b_ckpt}")
        results['V16b'] = 'SKIPPED - no checkpoint'

    # Save results
    out_path = V16_DIR / 'phase4_cross_machine_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Cross-Machine Transfer Summary")
    print("=" * 60)
    for model_name, domain_results in results.items():
        if isinstance(domain_results, dict) and not model_name.startswith('V') or \
           isinstance(domain_results, str):
            continue
        if isinstance(domain_results, str):
            continue
        if model_name in ['description', 'target_domains', 'baselines']:
            continue
        print(f"\n{model_name}:")
        if isinstance(domain_results, str):
            print(f"  {domain_results}")
            continue
        for fd, res in domain_results.items():
            if isinstance(res, dict):
                print(f"  {fd}: test RMSE = {res['test_rmse_mean']:.2f} +/- {res['test_rmse_std']:.2f}")
