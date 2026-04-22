"""
V18 Phase 9: NASA Scoring Function at matched label budgets.

Round-4 reviewer flagged: "Add NASA Scoring Function (asymmetric prognostic
metric) as a second column to tab:main_results."

NASA-S (Saxena 2008): asymmetric penalty favoring early predictions.
  For each engine i with prediction error d_i = pred_i - true_i:
    score_i = exp(-d_i / 13) - 1     if d_i < 0  (early, softer)
    score_i = exp( d_i / 10) - 1     if d_i >= 0 (late, harsher)
  Total = sum over test engines.

We retrain FAM E2E at each of {100%, 20%, 10%, 5%} labels (3 seeds) from the
v17_seed42 backbone, saving per-engine predictions, and compute NASA-S.

Output: experiments/v18/phase9_nasa_score.json
"""

import sys, json, copy, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

V11 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V17 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v17')
V18 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v18')
ROOT = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')
sys.path.insert(0, str(V11)); sys.path.insert(0, str(ROOT))

from models import TrajectoryJEPA, RULProbe
from data_utils import (load_cmapss_subset, N_SENSORS, RUL_CAP,
                        CMAPSSFinetuneDataset, CMAPSSTestDataset,
                        collate_finetune, collate_test)
from train_utils import subsample_engines

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D_MODEL = 256; N_HEADS = 4; N_LAYERS = 2; D_FF = 4 * D_MODEL
PRED_HIDDEN = D_FF; EMA_MOMENTUM = 0.99
CKPT_SEED = 42
BUDGETS = [1.0, 0.2, 0.1, 0.05]
SEEDS = [0, 1, 2]
E2E_LR = 1e-4; E2E_EPOCHS = 50; E2E_PATIENCE = 15


def nasa_score(preds, targs):
    """NASA Prognostic Score (Saxena 2008)."""
    d = preds - targs
    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(score.sum()), float(score.mean())


def load_v17():
    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=PRED_HIDDEN,
    ).to(DEVICE)
    sd = torch.load(V17/'ckpts'/f'v17_seed{CKPT_SEED}_best.pt',
                    map_location=DEVICE, weights_only=False)
    model.load_state_dict(sd)
    return model


def run_e2e(data, budget, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = load_v17(); probe = RULProbe(D_MODEL).to(DEVICE)
    sub = subsample_engines(data['train_engines'], budget, seed=seed)
    tr = DataLoader(CMAPSSFinetuneDataset(sub, n_cuts_per_engine=5, seed=seed),
                    batch_size=16, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(CMAPSSFinetuneDataset(data['val_engines'], n_cuts_per_engine=10,
                                           seed=seed+111),
                    batch_size=16, shuffle=False, collate_fn=collate_finetune)
    te = DataLoader(CMAPSSTestDataset(data['test_engines'], data['test_rul']),
                    batch_size=16, shuffle=False, collate_fn=collate_test)
    for p in model.context_encoder.parameters(): p.requires_grad = True
    for p in model.predictor.parameters(): p.requires_grad = True
    params = (list(model.context_encoder.parameters())
              + list(model.predictor.parameters())
              + list(probe.parameters()))
    opt = torch.optim.Adam(params, lr=E2E_LR)
    best_val = float('inf'); best_pr = best_e = best_p = None; no_impr = 0
    for ep in range(E2E_EPOCHS):
        model.train(); probe.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            opt.zero_grad()
            h = model.encode_past(past, mask); pred = probe(h)
            loss = F.mse_loss(pred, rul)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
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
            best_pr = copy.deepcopy(probe.state_dict())
            best_e = copy.deepcopy(model.context_encoder.state_dict())
            best_p = copy.deepcopy(model.predictor.state_dict())
            no_impr = 0
        else:
            no_impr += 1
            if no_impr >= E2E_PATIENCE: break
    probe.load_state_dict(best_pr)
    model.context_encoder.load_state_dict(best_e)
    model.predictor.load_state_dict(best_p)
    model.eval(); probe.eval()
    pt, tt = [], []
    with torch.no_grad():
        for past, mask, rul_gt in te:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = model.encode_past(past, mask)
            pt.append(probe(h).cpu().numpy()*RUL_CAP); tt.append(rul_gt.numpy())
    preds = np.concatenate(pt); targs = np.concatenate(tt)
    test_rmse = float(np.sqrt(np.mean((preds - targs)**2)))
    nasa_total, nasa_mean = nasa_score(preds, targs)
    return {'budget': budget, 'seed': seed,
            'test_rmse': test_rmse,
            'nasa_score_total': nasa_total,
            'nasa_score_mean': nasa_mean,
            'preds': preds.tolist(), 'targets': targs.tolist()}


def main():
    data = load_cmapss_subset('FD001')
    t0 = time.time()
    results = {}
    for budget in BUDGETS:
        key = f'{int(budget*100)}%'
        print(f"\n=== budget={key} ===", flush=True)
        rs = []
        for s in SEEDS:
            r = run_e2e(data, budget, s)
            rs.append(r)
            print(f"  seed={s}: rmse={r['test_rmse']:.2f} "
                  f"NASA-S total={r['nasa_score_total']:.1f} "
                  f"mean={r['nasa_score_mean']:.3f}", flush=True)
        results[key] = {
            'rmse_mean': float(np.mean([r['test_rmse'] for r in rs])),
            'rmse_std':  float(np.std([r['test_rmse'] for r in rs])),
            'nasa_total_mean': float(np.mean([r['nasa_score_total'] for r in rs])),
            'nasa_total_std':  float(np.std([r['nasa_score_total'] for r in rs])),
            'nasa_mean_mean':  float(np.mean([r['nasa_score_mean'] for r in rs])),
            'per_seed': rs,
        }
        with open(V18/'phase9_nasa_score.json', 'w') as f:
            json.dump({'config': 'v18_phase9_nasa_score',
                       'ckpt_backbone': f'v17_seed{CKPT_SEED}_best.pt',
                       'seeds': SEEDS, 'budgets': BUDGETS,
                       'results': results,
                       'runtime_min': (time.time()-t0)/60,
                       'star_paper_ref_nasa_score': 169,
                       'star_paper_ref_rmse': 10.61}, f, indent=2, default=float)

    print("\n" + "=" * 65)
    print("V18 Phase 9: FAM E2E + NASA Scoring Function on FD001")
    print("=" * 65)
    print(f"{'budget':>8} {'RMSE':>14} {'NASA-S total':>16} {'NASA-S mean':>14}")
    for key, r in results.items():
        print(f"{key:>8} {r['rmse_mean']:>6.2f}+-{r['rmse_std']:<5.2f} "
              f"{r['nasa_total_mean']:>10.1f}+-{r['nasa_total_std']:<5.1f} "
              f"{r['nasa_mean_mean']:>11.3f}")
    print(f"\nSTAR paper reference: RMSE 10.61, NASA-S 169")
    print(f"Runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
