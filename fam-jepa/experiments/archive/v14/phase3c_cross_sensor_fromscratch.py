"""
V14 Phase 3c: From-scratch ablation on the cross-sensor architecture.

Following v13 Phase 0c (which showed pretraining gives +8.8 RMSE at 100%
on V2), we run the analogous ablation on the cross-sensor architecture.

Question: is Phase 3's frozen 14.98 an SSL win or an architecture win?
If random-init cross-sensor E2E reaches ~15 RMSE, the win is mostly
architectural. If it lands near V2's from-scratch E2E (22.99) or worse,
pretraining does real work on this architecture too.

Protocol:
- Same cross-sensor architecture (d=128, n_pairs=2, 961K params)
- 3 seeds at 100% labels
- Also 3 seeds at 10% labels (where V2 had +15.6 pretraining delta)
- E2E fine-tune from random init (no pretrain) vs from pretrained
  checkpoint (re-run pretrained for a matched-seed comparison).

Output: experiments/v14/phase3_from_scratch.json
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
sys.path.insert(0, str(V14_DIR))

from data_utils import (
    load_cmapss_subset, N_SENSORS, RUL_CAP,
    CMAPSSFinetuneDataset, CMAPSSTestDataset,
    collate_finetune, collate_test,
)
from models import RULProbe
from train_utils import subsample_engines
from phase3_cross_sensor import CrossSensorJEPA, D_MODEL, N_HEADS, N_PAIRS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = V14_DIR / 'best_pretrain_cross_sensor.pt'


def run_e2e(use_pretrained, data, budget, seed):
    train_eng = data['train_engines']
    if budget < 1.0:
        train_eng = subsample_engines(train_eng, budget, seed=seed)

    model = CrossSensorJEPA(d_model=D_MODEL, n_heads=N_HEADS, n_pairs=N_PAIRS,
                             dropout=0.1).to(DEVICE)
    if use_pretrained:
        model.load_state_dict(torch.load(str(CKPT), map_location=DEVICE))

    for p in model.context_encoder.parameters(): p.requires_grad = True
    probe = RULProbe(D_MODEL).to(DEVICE)

    torch.manual_seed(seed); np.random.seed(seed)

    tr = DataLoader(CMAPSSFinetuneDataset(train_eng, n_cuts_per_engine=5, seed=seed),
                    batch_size=16, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(CMAPSSFinetuneDataset(data['val_engines'], use_last_only=True),
                    batch_size=16, shuffle=False, collate_fn=collate_finetune)
    te = DataLoader(CMAPSSTestDataset(data['test_engines'], data['test_rul']),
                    batch_size=16, shuffle=False, collate_fn=collate_test)

    optim = torch.optim.Adam(
        list(model.context_encoder.parameters()) + list(probe.parameters()), lr=1e-4)

    best_val = float('inf'); best_ps = None; best_es = None; no_impr = 0
    for ep in range(100):
        model.train(); probe.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            optim.zero_grad()
            h = model.encode_past(past, mask)
            loss = F.mse_loss(probe(h), rul)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.context_encoder.parameters()) + list(probe.parameters()), 1.0)
            optim.step()
        model.eval(); probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                h = model.encode_past(past, mask)
                pv.append(probe(h).cpu().numpy()); tv.append(rul.numpy())
        val_rmse = float(np.sqrt(np.mean(
            (np.concatenate(pv)*RUL_CAP - np.concatenate(tv)*RUL_CAP)**2)))
        if val_rmse < best_val:
            best_val = val_rmse
            best_ps = copy.deepcopy(probe.state_dict())
            best_es = copy.deepcopy(model.context_encoder.state_dict())
            no_impr = 0
        else:
            no_impr += 1
            if no_impr >= 20: break

    probe.load_state_dict(best_ps)
    model.context_encoder.load_state_dict(best_es)
    model.eval(); probe.eval()
    pt, tt = [], []
    with torch.no_grad():
        for past, mask, rul_gt in te:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = model.encode_past(past, mask)
            pt.append(probe(h).cpu().numpy() * RUL_CAP)
            tt.append(rul_gt.numpy())
    return float(np.sqrt(np.mean((np.concatenate(pt) - np.concatenate(tt))**2))), best_val


def main():
    print(f"V14 Phase 3c: from-scratch ablation on cross-sensor architecture")
    t0 = time.time()

    data = load_cmapss_subset('FD001')
    budgets = [1.0, 0.10]
    seeds = [42, 123, 456]

    results = {}
    for b in budgets:
        key = f"{int(b*100)}pct"
        print(f"\n=== budget={key} ===")
        results[key] = {'pretrained': [], 'scratch': []}
        for seed in seeds:
            rmse_pre, val_pre = run_e2e(True, data, b, seed)
            rmse_scr, val_scr = run_e2e(False, data, b, seed)
            delta = rmse_scr - rmse_pre
            print(f"  seed={seed} pretrained={rmse_pre:.3f} scratch={rmse_scr:.3f} "
                  f"delta={delta:+.3f}")
            results[key]['pretrained'].append(
                {'seed': seed, 'test_rmse': rmse_pre, 'val_rmse': val_pre})
            results[key]['scratch'].append(
                {'seed': seed, 'test_rmse': rmse_scr, 'val_rmse': val_scr})
        pre_vals = [r['test_rmse'] for r in results[key]['pretrained']]
        scr_vals = [r['test_rmse'] for r in results[key]['scratch']]
        results[key]['pretrained_mean'] = float(np.mean(pre_vals))
        results[key]['pretrained_std'] = float(np.std(pre_vals))
        results[key]['scratch_mean'] = float(np.mean(scr_vals))
        results[key]['scratch_std'] = float(np.std(scr_vals))
        results[key]['delta_mean'] = results[key]['scratch_mean'] - results[key]['pretrained_mean']
        print(f"  -- pretrained E2E: {results[key]['pretrained_mean']:.3f} +/- "
              f"{results[key]['pretrained_std']:.3f}")
        print(f"  -- from-scratch E2E: {results[key]['scratch_mean']:.3f} +/- "
              f"{results[key]['scratch_std']:.3f}")
        print(f"  -- delta (scratch - pretrained): {results[key]['delta_mean']:+.3f}")

    results['wall_time_s'] = time.time() - t0
    # V2 baselines from v13 phase 0c
    results['v2_baselines'] = {
        '100pct': {'pretrained': 14.18, 'scratch': 22.99, 'delta': 8.81},
        '10pct': {'pretrained': 19.97, 'scratch': 35.59, 'delta': 15.62},
    }

    out = V14_DIR / 'phase3_from_scratch.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}")
    print(f"Wall time: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
