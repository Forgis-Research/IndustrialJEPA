"""
V18 Phase 1b: E2E fine-tuning from V17 best checkpoints.

For each label budget in {100%, 20%, 10%, 5%}:
  For each seed in {0, 1, 2, 3, 4} (random seed for init/subsample):
    Load v17_seed42_best.pt (shared pretrained backbone).
    Unfreeze entire context_encoder + predictor + add linear head.
    Adam LR=1e-4, 50 epochs, MSE loss, honest val (n_cuts=10).
    Report test RMSE + F1 at k in {10,20,30,50}.

Also reports the "frozen probe baseline" from same ckpt with same label budget,
using an honest val protocol.

Output: experiments/v18/phase1b_e2e_results.json
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

# V17 arch
D_MODEL = 256; N_HEADS = 4; N_LAYERS = 2; D_FF = 4 * D_MODEL
PRED_HIDDEN = D_FF; EMA_MOMENTUM = 0.99

CKPT_SEED = 42  # shared pretrained backbone
BUDGETS = [1.0, 0.2, 0.1, 0.05]
SEEDS = [0, 1, 2, 3, 4]  # 5 seeds for init/subsample
K_EVAL_LIST = [10, 20, 30, 50]

# E2E config (mirrors v11)
E2E_LR = 1e-4
E2E_EPOCHS = 50
E2E_PATIENCE = 15
# Frozen-probe protocol matches honest v18 phase 1a
FROZEN_LR = 1e-3
FROZEN_WD = 1e-2
FROZEN_EPOCHS = 200
FROZEN_PATIENCE = 25


def load_v17_model():
    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=PRED_HIDDEN,
    ).to(DEVICE)
    sd = torch.load(V17 / 'ckpts' / f'v17_seed{CKPT_SEED}_best.pt',
                    map_location=DEVICE, weights_only=False)
    model.load_state_dict(sd)
    return model


def f1_multi_k(preds, targs, k_list=K_EVAL_LIST):
    out = {}
    for ke in k_list:
        y = (targs <= ke).astype(int)
        score = -preds
        thr = float(np.percentile(score[y == 0], 95)) if (y == 0).sum() > 0 else 0.0
        m = _anomaly_metrics(score, y, threshold=thr)
        out[ke] = {
            'f1': float(m['f1_non_pa']),
            'auc_pr': float(m['auc_pr']),
            'precision': float(m['precision_non_pa']),
            'recall': float(m['recall_non_pa']),
        }
    return out


def eval_test(model, probe, te_loader):
    model.eval(); probe.eval()
    pt, tt = [], []
    with torch.no_grad():
        for past, mask, rul_gt in te_loader:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = model.encode_past(past, mask)
            pt.append(probe(h).cpu().numpy() * RUL_CAP); tt.append(rul_gt.numpy())
    preds = np.concatenate(pt); targs = np.concatenate(tt)
    return float(np.sqrt(np.mean((preds - targs) ** 2))), preds, targs


def run_one(data, budget, seed, mode):
    """mode = 'frozen' or 'e2e'"""
    torch.manual_seed(seed); np.random.seed(seed)
    model = load_v17_model()
    probe = RULProbe(D_MODEL).to(DEVICE)

    sub_eng = subsample_engines(data['train_engines'], budget, seed=seed)

    tr_ds = CMAPSSFinetuneDataset(sub_eng, n_cuts_per_engine=5, seed=seed)
    # HONEST val protocol (n_cuts=10)
    va_ds = CMAPSSFinetuneDataset(data['val_engines'], n_cuts_per_engine=10,
                                   seed=seed + 111)
    te_ds = CMAPSSTestDataset(data['test_engines'], data['test_rul'])
    tr = DataLoader(tr_ds, batch_size=16, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(va_ds, batch_size=16, shuffle=False, collate_fn=collate_finetune)
    te = DataLoader(te_ds, batch_size=16, shuffle=False, collate_fn=collate_test)

    if mode == 'frozen':
        for p in model.parameters(): p.requires_grad = False
        opt = torch.optim.AdamW(probe.parameters(), lr=FROZEN_LR, weight_decay=FROZEN_WD)
        n_epochs = FROZEN_EPOCHS; patience = FROZEN_PATIENCE
    else:
        # E2E: unfreeze context_encoder + predictor, train with probe
        for p in model.context_encoder.parameters(): p.requires_grad = True
        for p in model.predictor.parameters(): p.requires_grad = True
        # target_encoder & EMA stay frozen
        params = (list(model.context_encoder.parameters())
                  + list(model.predictor.parameters())
                  + list(probe.parameters()))
        opt = torch.optim.Adam(params, lr=E2E_LR)
        n_epochs = E2E_EPOCHS; patience = E2E_PATIENCE

    best_val = float('inf'); best_p = None; best_e = None; best_pr = None; no_impr = 0

    for ep in range(n_epochs):
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
            pred = probe(h)
            loss = F.mse_loss(pred, rul)
            loss.backward()
            if mode == 'e2e':
                torch.nn.utils.clip_grad_norm_(params, 1.0)
            else:
                torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            opt.step()

        model.eval(); probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                h = model.encode_past(past, mask)
                pv.append(probe(h).cpu().numpy()); tv.append(rul.numpy())
        preds = np.concatenate(pv) * RUL_CAP; targs = np.concatenate(tv) * RUL_CAP
        val_rmse = float(np.sqrt(np.mean((preds - targs) ** 2)))
        if val_rmse < best_val:
            best_val = val_rmse
            best_pr = copy.deepcopy(probe.state_dict())
            if mode == 'e2e':
                best_e = copy.deepcopy(model.context_encoder.state_dict())
                best_p = copy.deepcopy(model.predictor.state_dict())
            no_impr = 0
        else:
            no_impr += 1
            if no_impr >= patience: break

    probe.load_state_dict(best_pr)
    if mode == 'e2e' and best_e is not None:
        model.context_encoder.load_state_dict(best_e)
        model.predictor.load_state_dict(best_p)

    test_rmse, preds_test, targs_test = eval_test(model, probe, te)
    f1_k = f1_multi_k(preds_test, targs_test)
    return dict(mode=mode, budget=budget, seed=seed,
                val_rmse=best_val, test_rmse=test_rmse, f1_by_k=f1_k,
                n_train_engines=len(sub_eng))


def main():
    V18.mkdir(exist_ok=True)
    data = load_cmapss_subset('FD001')
    print(f"FD001 loaded. Backbone: v17_seed{CKPT_SEED}_best.pt", flush=True)

    t0 = time.time()
    all_results = {'frozen': {}, 'e2e': {}}

    # Running e2e first as it's the headline. Frozen is a cheap baseline.
    for mode in ['e2e', 'frozen']:
        for budget in BUDGETS:
            all_results[mode][str(budget)] = []
            for seed in SEEDS:
                r = run_one(data, budget, seed, mode)
                all_results[mode][str(budget)].append(r)
                f1_30 = r['f1_by_k'][30]['f1']
                print(f"  [{mode:6s} b={budget*100:4.0f}% s={seed}] "
                      f"val={r['val_rmse']:.2f} test={r['test_rmse']:.2f} "
                      f"F1@30={f1_30:.3f}", flush=True)
            # incremental save
            out = _summarize(all_results, t0)
            with open(V18 / 'phase1b_e2e_results.json', 'w') as f:
                json.dump(out, f, indent=2, default=float)

    out = _summarize(all_results, t0)
    with open(V18 / 'phase1b_e2e_results.json', 'w') as f:
        json.dump(out, f, indent=2, default=float)

    print("\n" + "=" * 80)
    print("V18 Phase 1b: E2E FINE-TUNE SUMMARY")
    print("=" * 80)
    print(f"{'mode':<8} {'100%':>16} {'20%':>16} {'10%':>16} {'5%':>16}")
    for mode in ['frozen', 'e2e']:
        row = f"{mode:<8}"
        for budget in BUDGETS:
            s = out[mode][str(budget)]['test_rmse']
            row += f" {s['mean']:6.2f}+/-{s['std']:<4.2f}"
        print(row)
    print(f"\nV11 E2E (ref)  : 100%=13.80, 20%=16.54, 10%=18.66, 5%=25.33")
    print(f"V2  frozen (ref): 100%=17.81, 20%=19.83, 10%=19.93, 5%=21.53")
    print(f"Runtime: {(time.time()-t0)/60:.1f} min")


def _summarize(all_results, t0):
    out = {'config': 'v18_phase1b_e2e_sweep',
           'ckpt_backbone': f'v17_seed{CKPT_SEED}_best.pt',
           'budgets': BUDGETS, 'seeds': SEEDS,
           'k_eval_list': K_EVAL_LIST,
           'e2e_config': {'lr': E2E_LR, 'epochs': E2E_EPOCHS,
                          'patience': E2E_PATIENCE,
                          'unfreeze': ['context_encoder', 'predictor']},
           'frozen_config': {'lr': FROZEN_LR, 'wd': FROZEN_WD,
                             'epochs': FROZEN_EPOCHS,
                             'patience': FROZEN_PATIENCE},
           'v11_reference': {'100%': 13.80, '20%': 16.54,
                             '10%': 18.66, '5%': 25.33},
           'v2_frozen_reference': {'100%': 17.81, '20%': 19.83,
                                   '10%': 19.93, '5%': 21.53},
           'runtime_min': (time.time() - t0) / 60,
           'frozen': {}, 'e2e': {}}
    for mode in ['frozen', 'e2e']:
        for budget in BUDGETS:
            bs = str(budget)
            if bs not in all_results[mode]: continue
            rs = all_results[mode][bs]
            if not rs: continue
            agg = {
                'test_rmse': {
                    'mean': float(np.mean([r['test_rmse'] for r in rs])),
                    'std': float(np.std([r['test_rmse'] for r in rs])),
                    'per_seed': [r['test_rmse'] for r in rs],
                },
                'val_rmse': {
                    'mean': float(np.mean([r['val_rmse'] for r in rs])),
                    'std': float(np.std([r['val_rmse'] for r in rs])),
                },
                'f1_by_k': {
                    ke: {
                        'f1_mean': float(np.mean([r['f1_by_k'][ke]['f1'] for r in rs])),
                        'f1_std': float(np.std([r['f1_by_k'][ke]['f1'] for r in rs])),
                        'auc_pr_mean': float(np.mean([r['f1_by_k'][ke]['auc_pr'] for r in rs])),
                    } for ke in K_EVAL_LIST
                },
                'n_seeds': len(rs),
                'per_seed_raw': rs,
            }
            out[mode][bs] = agg
    return out


if __name__ == '__main__':
    main()
