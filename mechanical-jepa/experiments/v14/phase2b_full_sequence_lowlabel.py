"""
V14 Phase 2b: full-sequence target encoder at LOW label budgets.

Phase 2 showed frozen probe improves -2.11 RMSE at 100% labels. Test
whether the gain also holds where it matters most: 5% and 10% labels
(the grey-swan regime, where the frozen probe beat STAR in V13 at 5%).

5 seeds each at budgets {5%, 10%, 20%}. Frozen-only (matches the
V13 STAR comparison). Uses the checkpoint from Phase 2.

Output: experiments/v14/full_sequence_lowlabel.json
"""

import sys, json, time, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V14_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v14')
sys.path.insert(0, str(V11_DIR))

from data_utils import (
    load_cmapss_subset, N_SENSORS, RUL_CAP,
    CMAPSSFinetuneDataset, CMAPSSTestDataset, collate_finetune, collate_test,
)
from models import TrajectoryJEPA, RULProbe
from train_utils import subsample_engines

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = V14_DIR / 'best_pretrain_full_sequence.pt'
D_MODEL = 256
BATCH_SIZE = 16
N_EPOCHS = 100
PATIENCE = 20


def run_frozen(train_sub, data, seed):
    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1, d_model=D_MODEL, n_heads=4,
        n_layers=2, d_ff=512, dropout=0.1,
    ).to(DEVICE)
    model.load_state_dict(torch.load(str(CKPT), map_location=DEVICE))
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    probe = RULProbe(D_MODEL).to(DEVICE)
    torch.manual_seed(seed); np.random.seed(seed)
    optim = torch.optim.Adam(probe.parameters(), lr=1e-3)

    tr = DataLoader(CMAPSSFinetuneDataset(train_sub, n_cuts_per_engine=5, seed=seed),
                    batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(CMAPSSFinetuneDataset(data['val_engines'], use_last_only=True),
                    batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_finetune)
    te = DataLoader(CMAPSSTestDataset(data['test_engines'], data['test_rul']),
                    batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_test)

    best_val = float('inf'); best_ps = None; no_impr = 0
    for _ in range(N_EPOCHS):
        probe.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            with torch.no_grad():
                h = model.encode_past(past, mask)
            optim.zero_grad()
            loss = F.mse_loss(probe(h), rul)
            loss.backward(); optim.step()

        probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                h = model.encode_past(past, mask)
                pv.append(probe(h).cpu().numpy()); tv.append(rul.numpy())
        val_rmse = float(np.sqrt(np.mean(
            (np.concatenate(pv)*RUL_CAP - np.concatenate(tv)*RUL_CAP)**2)))
        if val_rmse < best_val:
            best_val = val_rmse; best_ps = copy.deepcopy(probe.state_dict()); no_impr = 0
        else:
            no_impr += 1
            if no_impr >= PATIENCE: break

    probe.load_state_dict(best_ps)
    probe.eval()
    pt, tt = [], []
    with torch.no_grad():
        for past, mask, rul_gt in te:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = model.encode_past(past, mask)
            pt.append(probe(h).cpu().numpy() * RUL_CAP)
            tt.append(rul_gt.numpy())
    return float(np.sqrt(np.mean((np.concatenate(pt) - np.concatenate(tt))**2))), best_val


def main():
    print(f"V14 Phase 2b: full-sequence target, low-label frozen probe")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CKPT}")
    t0 = time.time()

    data = load_cmapss_subset('FD001')
    budgets = [0.20, 0.10, 0.05]
    seeds = [42, 123, 456, 789, 1024]

    # V2 baseline for comparison (from v13 RESULTS.md / paper tab)
    v2_frozen = {0.20: (19.83, 0.3), 0.10: (19.93, 0.9), 0.05: (21.53, 2.0)}
    star = {0.20: (17.74, 3.6), 0.10: (18.72, 2.8), 0.05: (24.55, 6.5)}

    results = {}
    for b in budgets:
        key = f"{int(b*100)}pct"
        print(f"\n=== budget={key} ===")
        rmses, vals = [], []
        for seed in seeds:
            train_sub = subsample_engines(data['train_engines'], b, seed=seed)
            rmse, val = run_frozen(train_sub, data, seed)
            print(f"  seed={seed} | test RMSE={rmse:.3f} | val={val:.3f} | n_engines={len(train_sub)}")
            rmses.append(rmse); vals.append(val)
        mean = float(np.mean(rmses)); std = float(np.std(rmses))
        v2_m, v2_s = v2_frozen[b]
        s_m, s_s = star[b]
        print(f"  -- V14: {mean:.3f} +/- {std:.3f}")
        print(f"  -- V2:  {v2_m:.3f} +/- {v2_s:.3f}  (delta {mean - v2_m:+.3f})")
        print(f"  -- STAR: {s_m:.3f} +/- {s_s:.3f}  (delta {mean - s_m:+.3f})")
        results[key] = {
            'budget': b,
            'v14_frozen_mean': mean, 'v14_frozen_std': std,
            'per_seed': rmses, 'per_seed_val': vals,
            'v2_frozen_mean': v2_m, 'v2_frozen_std': v2_s,
            'star_mean': s_m, 'star_std': s_s,
            'delta_vs_v2': mean - v2_m,
            'delta_vs_star': mean - s_m,
        }

    results['wall_time_s'] = time.time() - t0
    out = V14_DIR / 'full_sequence_lowlabel.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}")
    print(f"Total wall time: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
