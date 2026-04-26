"""
Phase 1b: Weight Decay E2E

Add L2 weight decay (1e-4) to E2E. The prior v13 session tested this at 100%
and found it was worse (15.00 vs 14.48). We also test at 5% labels where
the from-scratch ablation showed high variance (E2E std=5.27 at 5%).

5 seeds, 100% + 5% labels, FD001.

Output: experiments/v13/weight_decay_results.json
"""

import sys
import json
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V13_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v13')
sys.path.insert(0, str(V11_DIR))

from data_utils import (
    load_cmapss_subset, N_SENSORS, RUL_CAP,
    CMAPSSFinetuneDataset, CMAPSSTestDataset, collate_finetune, collate_test
)
from models import TrajectoryJEPA, RULProbe
from train_utils import subsample_engines

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRETRAIN_CKPT = V11_DIR / 'best_pretrain_L1_v2.pt'
SEEDS = [42, 123, 456, 789, 1024]
N_EPOCHS = 100
PATIENCE = 20
BATCH_SIZE = 16

print(f"Phase 1b: Weight Decay E2E")
print(f"Device: {DEVICE}")
t0_global = time.time()

data = load_cmapss_subset('FD001')
all_train_engines = data['train_engines']
val_engines = data['val_engines']
test_engines = data['test_engines']
test_rul = data['test_rul']


def run_e2e(train_sub, seed, weight_decay=0.0):
    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1, d_model=256, n_heads=4, n_layers=2,
        d_ff=512, dropout=0.1, ema_momentum=0.996, predictor_hidden=256
    ).to(DEVICE)
    model.load_state_dict(torch.load(str(PRETRAIN_CKPT), map_location=DEVICE))
    probe = RULProbe(256).to(DEVICE)

    for p in model.context_encoder.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        list(model.context_encoder.parameters()) +
        list(model.predictor.parameters()) +
        list(probe.parameters()),
        lr=1e-4, weight_decay=weight_decay
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = CMAPSSFinetuneDataset(train_sub, n_cuts_per_engine=5, seed=seed)
    val_ds = CMAPSSFinetuneDataset(val_engines, use_last_only=True)
    test_ds = CMAPSSTestDataset(test_engines, test_rul)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_finetune)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_finetune)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_test)

    best_val, best_ps, best_es, ni = float('inf'), None, None, 0
    for ep in range(1, N_EPOCHS+1):
        model.train(); probe.train()
        for past, mask, rul in train_loader:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            optimizer.zero_grad()
            h = model.encode_past(past, mask)
            loss = F.mse_loss(probe(h), rul)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(probe.parameters()), 1.0)
            optimizer.step()
        model.eval(); probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in val_loader:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                pv.append(probe(model.encode_past(past, mask)).cpu().numpy())
                tv.append(rul.numpy())
        vr = float(np.sqrt(np.mean((np.concatenate(pv)*RUL_CAP - np.concatenate(tv)*RUL_CAP)**2)))
        if vr < best_val:
            best_val = vr; best_ps = copy.deepcopy(probe.state_dict()); best_es = copy.deepcopy(model.context_encoder.state_dict()); ni = 0
        else:
            ni += 1
            if ni >= PATIENCE: break
    probe.load_state_dict(best_ps); model.context_encoder.load_state_dict(best_es)
    model.eval(); probe.eval()
    pt, tt = [], []
    with torch.no_grad():
        for past, mask, rg in test_loader:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            pt.append(probe(model.encode_past(past, mask)).cpu().numpy()*RUL_CAP)
            tt.append(rg.numpy())
    return float(np.sqrt(np.mean((np.concatenate(pt)-np.concatenate(tt))**2)))


results = {}

for budget_label, budget_frac in [('100pct', 1.0), ('5pct', 0.05)]:
    print(f"\n{'='*60}")
    print(f"Budget: {budget_label}")
    print(f"{'='*60}")

    wd_rmses = []
    nowd_rmses = []

    for seed in SEEDS:
        print(f"\n--- {budget_label}, seed={seed} ---")

        if budget_frac < 1.0:
            train_sub = subsample_engines(all_train_engines, budget_frac, seed=seed)
        else:
            train_sub = all_train_engines

        # With weight decay
        rmse_wd = run_e2e(train_sub, seed, weight_decay=1e-4)
        # Without weight decay (baseline)
        rmse_nowd = run_e2e(train_sub, seed, weight_decay=0.0)

        print(f"  WD=1e-4: {rmse_wd:.3f}")
        print(f"  WD=0:    {rmse_nowd:.3f}")
        print(f"  Delta:   {rmse_wd - rmse_nowd:+.3f}")

        wd_rmses.append(rmse_wd)
        nowd_rmses.append(rmse_nowd)

    results[budget_label] = {
        'with_wd': {'mean': float(np.mean(wd_rmses)), 'std': float(np.std(wd_rmses)), 'all': wd_rmses},
        'without_wd': {'mean': float(np.mean(nowd_rmses)), 'std': float(np.std(nowd_rmses)), 'all': nowd_rmses},
        'delta': float(np.mean(wd_rmses) - np.mean(nowd_rmses)),
    }

    print(f"\n  {budget_label} Summary:")
    print(f"    WD=1e-4: {results[budget_label]['with_wd']['mean']:.3f} +/- {results[budget_label]['with_wd']['std']:.3f}")
    print(f"    WD=0:    {results[budget_label]['without_wd']['mean']:.3f} +/- {results[budget_label]['without_wd']['std']:.3f}")
    print(f"    Delta:   {results[budget_label]['delta']:+.3f}")

    # Check if WD reduces variance at 5%
    if budget_label == '5pct':
        wd_std = results[budget_label]['with_wd']['std']
        nowd_std = results[budget_label]['without_wd']['std']
        print(f"    Variance reduction: WD std={wd_std:.3f} vs no-WD std={nowd_std:.3f}")

results['wall_time_total_s'] = time.time() - t0_global

out_path = V13_DIR / 'weight_decay_results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
print(f"Total wall time: {time.time()-t0_global:.1f}s")
