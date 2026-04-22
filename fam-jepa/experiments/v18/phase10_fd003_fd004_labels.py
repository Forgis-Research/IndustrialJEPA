"""
V18 Phase 10: FD003/FD004 label-efficiency sweep under honest protocol.

Round-4 reviewer M5 + limitations: "Full label-efficiency sweeps on FD003/
FD004... remain future work. Do them."

For each subset in {FD003, FD004}:
  For each budget in {100%, 20%, 10%, 5%}:
    For each seed in {0, 1, 2}:
      Fine-tune v14 full-seq ckpt E2E with honest val (n_cuts=10).
      Report test RMSE + NASA-S + F1@30.

Total: 2 subsets * 4 budgets * 3 seeds = 24 E2E runs.

Output: experiments/v18/phase10_fd003_fd004_labels.json
"""

import sys, json, copy, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

V11 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V14 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v14')
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
D_MODEL = 256; N_HEADS = 4; N_LAYERS = 2; D_FF = 512
PRED_HIDDEN = 256; EMA_MOMENTUM = 0.99
BUDGETS = [1.0, 0.2, 0.1, 0.05]
SEEDS = [0, 1, 2]
E2E_LR = 1e-4; E2E_EPOCHS = 50; E2E_PATIENCE = 15
K_EVAL_LIST = [10, 20, 30, 50]

CKPTS = {
    'FD003': V14 / 'best_pretrain_full_sequence_fd003.pt',
    'FD004': V14 / 'best_pretrain_full_sequence_fd004.pt',
}


def nasa_score(preds, targs):
    d = preds - targs
    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(score.sum()), float(score.mean())


def load_model(ckpt_path):
    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=PRED_HIDDEN,
    ).to(DEVICE)
    sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if isinstance(sd, dict) and 'model_state_dict' in sd: sd = sd['model_state_dict']
    model.load_state_dict(sd)
    return model


def run_e2e(data, ckpt_path, budget, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = load_model(ckpt_path); probe = RULProbe(D_MODEL).to(DEVICE)
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
    f1_by_k = {}
    for ke in K_EVAL_LIST:
        y = (targs <= ke).astype(int); score = -preds
        thr = float(np.percentile(score[y == 0], 95)) if (y == 0).sum() > 0 else 0.0
        m = _anomaly_metrics(score, y, threshold=thr)
        f1_by_k[ke] = {'f1': float(m['f1_non_pa']), 'auc_pr': float(m['auc_pr'])}
    return {'budget': budget, 'seed': seed,
            'val_rmse': best_val, 'test_rmse': test_rmse,
            'nasa_score_total': nasa_total, 'nasa_score_mean': nasa_mean,
            'f1_by_k': f1_by_k}


def main():
    t0 = time.time()
    all_results = {}
    for subset, ckpt in CKPTS.items():
        print(f"\n{'='*60}\n{subset}\n{'='*60}", flush=True)
        data = load_cmapss_subset(subset)
        print(f"  train={len(data['train_engines'])} val={len(data['val_engines'])} "
              f"test={len(data['test_engines'])}", flush=True)
        all_results[subset] = {}
        for budget in BUDGETS:
            key = f'{int(budget*100)}%'
            print(f"\n--- {subset} budget={key} ---", flush=True)
            rs = []
            for s in SEEDS:
                r = run_e2e(data, ckpt, budget, s)
                rs.append(r)
                print(f"  seed={s}: rmse={r['test_rmse']:.2f} "
                      f"NASA-S={r['nasa_score_total']:.1f} "
                      f"F1@30={r['f1_by_k'][30]['f1']:.3f}", flush=True)
            all_results[subset][key] = {
                'rmse_mean': float(np.mean([r['test_rmse'] for r in rs])),
                'rmse_std':  float(np.std([r['test_rmse'] for r in rs])),
                'nasa_total_mean': float(np.mean([r['nasa_score_total'] for r in rs])),
                'nasa_total_std':  float(np.std([r['nasa_score_total'] for r in rs])),
                'f1_at_30_mean': float(np.mean([r['f1_by_k'][30]['f1'] for r in rs])),
                'per_seed': rs,
            }
            with open(V18/'phase10_fd003_fd004_labels.json', 'w') as f:
                json.dump({'config': 'v18_phase10_fd003_fd004_label_eff',
                           'ckpts': {k: str(v) for k,v in CKPTS.items()},
                           'seeds': SEEDS, 'budgets': BUDGETS,
                           'star_paper_ref': {'FD003': 10.71, 'FD004': 15.87},
                           'results': all_results,
                           'runtime_min': (time.time()-t0)/60}, f, indent=2, default=float)

    print("\n" + "=" * 60)
    print("V18 Phase 10: FD003/FD004 E2E label-efficiency SUMMARY")
    print("=" * 60)
    print(f"{'subset':>8} {'100%':>12} {'20%':>12} {'10%':>12} {'5%':>12}")
    for subset in ['FD003', 'FD004']:
        row = f"{subset:>8}"
        for budget in ['100%', '20%', '10%', '5%']:
            r = all_results[subset][budget]
            row += f" {r['rmse_mean']:5.2f}+-{r['rmse_std']:<4.2f}"
        print(row)
    print(f"\nSTAR paper ref: FD003 10.71, FD004 15.87 (100% labels)")
    print(f"Runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
