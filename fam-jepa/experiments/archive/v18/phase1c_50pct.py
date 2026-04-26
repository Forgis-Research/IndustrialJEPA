"""
V18 Phase 1c: Fill the 50% label budget cell in tab:main_results.

Phase 1b ran 100/20/10/5%. The 50% column was left blank (\todo). This script
runs the same E2E and frozen protocol at 50% label budget with 5 seeds.

Reuses v17_seed42 backbone. Output: phase1c_50pct_results.json
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
from evaluation.grey_swan_metrics import anomaly_metrics as _anomaly_metrics

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D_MODEL = 256; N_HEADS = 4; N_LAYERS = 2; D_FF = 4 * D_MODEL
PRED_HIDDEN = D_FF; EMA_MOMENTUM = 0.99
CKPT_SEED = 42
BUDGET = 0.5
SEEDS = [0, 1, 2, 3, 4]
K_EVAL_LIST = [10, 20, 30, 50]

E2E_LR = 1e-4; E2E_EPOCHS = 50; E2E_PATIENCE = 15
FROZEN_LR = 1e-3; FROZEN_WD = 1e-2
FROZEN_EPOCHS = 200; FROZEN_PATIENCE = 25


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


def f1_multi_k(preds, targs):
    out = {}
    for ke in K_EVAL_LIST:
        y = (targs <= ke).astype(int); score = -preds
        thr = float(np.percentile(score[y == 0], 95)) if (y == 0).sum() > 0 else 0.0
        m = _anomaly_metrics(score, y, threshold=thr)
        out[ke] = {'f1': float(m['f1_non_pa']), 'auc_pr': float(m['auc_pr'])}
    return out


def run_one(data, seed, mode):
    torch.manual_seed(seed); np.random.seed(seed)
    model = load_v17()
    probe = RULProbe(D_MODEL).to(DEVICE)
    sub = subsample_engines(data['train_engines'], BUDGET, seed=seed)
    tr = DataLoader(CMAPSSFinetuneDataset(sub, n_cuts_per_engine=5, seed=seed),
                    batch_size=16, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(CMAPSSFinetuneDataset(data['val_engines'],
                                          n_cuts_per_engine=10, seed=seed+111),
                    batch_size=16, shuffle=False, collate_fn=collate_finetune)
    te = DataLoader(CMAPSSTestDataset(data['test_engines'], data['test_rul']),
                    batch_size=16, shuffle=False, collate_fn=collate_test)

    if mode == 'frozen':
        for p in model.parameters(): p.requires_grad = False
        opt = torch.optim.AdamW(probe.parameters(), lr=FROZEN_LR, weight_decay=FROZEN_WD)
        n_ep, pat = FROZEN_EPOCHS, FROZEN_PATIENCE
    else:
        for p in model.context_encoder.parameters(): p.requires_grad = True
        for p in model.predictor.parameters(): p.requires_grad = True
        params = (list(model.context_encoder.parameters())
                  + list(model.predictor.parameters())
                  + list(probe.parameters()))
        opt = torch.optim.Adam(params, lr=E2E_LR)
        n_ep, pat = E2E_EPOCHS, E2E_PATIENCE

    best_val = float('inf'); best_pr = best_e = best_p = None; no_impr = 0
    for ep in range(n_ep):
        if mode == 'e2e': model.train()
        else: model.eval()
        probe.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            opt.zero_grad()
            if mode == 'frozen':
                with torch.no_grad(): h = model.encode_past(past, mask)
            else:
                h = model.encode_past(past, mask)
            loss = F.mse_loss(probe(h), rul)
            loss.backward()
            if mode == 'e2e': torch.nn.utils.clip_grad_norm_(params, 1.0)
            else: torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            opt.step()
        model.eval(); probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                h = model.encode_past(past, mask)
                pv.append(probe(h).cpu().numpy()); tv.append(rul.numpy())
        preds = np.concatenate(pv)*RUL_CAP; targs = np.concatenate(tv)*RUL_CAP
        val_rmse = float(np.sqrt(np.mean((preds - targs)**2)))
        if val_rmse < best_val:
            best_val = val_rmse
            best_pr = copy.deepcopy(probe.state_dict())
            if mode == 'e2e':
                best_e = copy.deepcopy(model.context_encoder.state_dict())
                best_p = copy.deepcopy(model.predictor.state_dict())
            no_impr = 0
        else:
            no_impr += 1
            if no_impr >= pat: break
    probe.load_state_dict(best_pr)
    if mode == 'e2e' and best_e is not None:
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
    return {'mode': mode, 'seed': seed, 'val_rmse': best_val,
            'test_rmse': test_rmse, 'f1_by_k': f1_multi_k(preds, targs)}


def main():
    data = load_cmapss_subset('FD001')
    t0 = time.time()
    results = {'frozen': [], 'e2e': []}
    for mode in ['e2e', 'frozen']:
        for s in SEEDS:
            r = run_one(data, s, mode)
            results[mode].append(r)
            print(f"  [{mode:6s} b=50% s={s}] val={r['val_rmse']:.2f} "
                  f"test={r['test_rmse']:.2f} F1@30={r['f1_by_k'][30]['f1']:.3f}",
                  flush=True)

    def agg(rs):
        return {'mean': float(np.mean([r['test_rmse'] for r in rs])),
                'std': float(np.std([r['test_rmse'] for r in rs])),
                'per_seed': [r['test_rmse'] for r in rs]}

    summary = {
        'config': 'v18_phase1c_50pct',
        'budget': BUDGET, 'seeds': SEEDS,
        'frozen': {'test_rmse': agg(results['frozen']),
                   'per_seed_raw': results['frozen']},
        'e2e': {'test_rmse': agg(results['e2e']),
                'per_seed_raw': results['e2e']},
        'runtime_min': (time.time() - t0) / 60,
    }
    with open(V18/'phase1c_50pct_results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"\n50% labels:")
    print(f"  Frozen: {summary['frozen']['test_rmse']['mean']:.2f} "
          f"+/- {summary['frozen']['test_rmse']['std']:.2f}")
    print(f"  E2E:    {summary['e2e']['test_rmse']['mean']:.2f} "
          f"+/- {summary['e2e']['test_rmse']['std']:.2f}")
    print(f"Runtime: {summary['runtime_min']:.1f} min")


if __name__ == '__main__':
    main()
