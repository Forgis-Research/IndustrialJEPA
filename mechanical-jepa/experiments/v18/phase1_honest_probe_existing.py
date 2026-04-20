"""
V18 Phase 1a: Honest probe of existing V17 best checkpoints.

Re-probes each v17_seed{S}_best.pt under the honest protocol, reports:
  - Test RMSE
  - F1, precision, recall, AUC-PR at k in {10, 20, 30, 50}

This supersedes v17 Phase 2 (which only reported F1 at k=30) with multi-k
coverage, and confirms the 15.38 headline number persists across 3 seeds.

Output: experiments/v18/phase1a_frozen_multi_k.json
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

from models import TrajectoryJEPA
from data_utils import (load_cmapss_subset, N_SENSORS, RUL_CAP,
                        CMAPSSFinetuneDataset, CMAPSSTestDataset,
                        collate_finetune, collate_test)
from evaluation.grey_swan_metrics import anomaly_metrics as _anomaly_metrics

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# V17 arch
D_MODEL = 256; N_HEADS = 4; N_LAYERS = 2; D_FF_V17 = 4 * D_MODEL
PRED_HIDDEN_V17 = D_FF_V17; EMA_MOMENTUM = 0.99
SEEDS = [42, 123, 456]
K_EVAL_LIST = [10, 20, 30, 50]


def load_v17_model(seed):
    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF_V17,
        dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=PRED_HIDDEN_V17,
    ).to(DEVICE)
    sd = torch.load(V17 / 'ckpts' / f'v17_seed{seed}_best.pt',
                    map_location=DEVICE, weights_only=False)
    model.load_state_dict(sd)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    return model


def f1_metrics_at_k(preds, targs, k_eval):
    y = (targs <= k_eval).astype(int)
    score = -preds
    thr = float(np.percentile(score[y == 0], 95)) if (y == 0).sum() > 0 else 0.0
    m = _anomaly_metrics(score, y, threshold=thr)
    return {
        'f1': float(m['f1_non_pa']),
        'auc_pr': float(m['auc_pr']),
        'precision': float(m['precision_non_pa']),
        'recall': float(m['recall_non_pa']),
    }


def honest_probe(model, data, seed):
    torch.manual_seed(seed)
    probe = nn.Sequential(nn.Linear(D_MODEL, 1), nn.Sigmoid()).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-2)

    tr_ds = CMAPSSFinetuneDataset(data['train_engines'], n_cuts_per_engine=5, seed=seed)
    va_ds = CMAPSSFinetuneDataset(data['val_engines'], n_cuts_per_engine=10, seed=seed + 111)
    te_ds = CMAPSSTestDataset(data['test_engines'], data['test_rul'])
    tr = DataLoader(tr_ds, batch_size=32, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(va_ds, batch_size=32, shuffle=False, collate_fn=collate_finetune)
    te = DataLoader(te_ds, batch_size=32, shuffle=False, collate_fn=collate_test)

    best_val = float('inf'); best_state = None; no_impr = 0
    for ep in range(200):
        probe.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            with torch.no_grad(): h = model.encode_past(past, mask)
            loss = F.mse_loss(probe(h).squeeze(-1), rul)
            opt.zero_grad(); loss.backward(); opt.step()

        probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                h = model.encode_past(past, mask)
                pv.append(probe(h).squeeze(-1).cpu().numpy()); tv.append(rul.numpy())
        preds = np.concatenate(pv) * RUL_CAP; targs = np.concatenate(tv) * RUL_CAP
        val_rmse = float(np.sqrt(np.mean((preds - targs) ** 2)))
        if val_rmse < best_val:
            best_val = val_rmse; best_state = copy.deepcopy(probe.state_dict()); no_impr = 0
        else:
            no_impr += 1
            if no_impr >= 25: break

    probe.load_state_dict(best_state)
    probe.eval()
    p_test, t_test = [], []
    with torch.no_grad():
        for past, mask, rul_gt in te:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = model.encode_past(past, mask)
            p_test.append(probe(h).squeeze(-1).cpu().numpy() * RUL_CAP)
            t_test.append(rul_gt.numpy())
    preds_test = np.concatenate(p_test); targs_test = np.concatenate(t_test)
    test_rmse = float(np.sqrt(np.mean((preds_test - targs_test) ** 2)))

    out = dict(seed=seed, val_rmse=best_val, test_rmse=test_rmse, f1_by_k={})
    for ke in K_EVAL_LIST:
        out['f1_by_k'][ke] = f1_metrics_at_k(preds_test, targs_test, ke)
    return out


def main():
    V18.mkdir(exist_ok=True)
    data = load_cmapss_subset('FD001')

    t0 = time.time()
    rs = []
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===", flush=True)
        model = load_v17_model(seed)
        r = honest_probe(model, data, seed)
        print(f"  val={r['val_rmse']:.2f} test={r['test_rmse']:.2f}", flush=True)
        for ke in K_EVAL_LIST:
            m = r['f1_by_k'][ke]
            print(f"  k={ke:2d}: F1={m['f1']:.3f} P={m['precision']:.3f} "
                  f"R={m['recall']:.3f} AUC-PR={m['auc_pr']:.3f}", flush=True)
        rs.append(r); del model; torch.cuda.empty_cache()

    summary = {
        'config': 'v18_phase1a_honest_probe_existing_v17_ckpts',
        'arch': {'d_model': D_MODEL, 'n_heads': N_HEADS, 'n_layers': N_LAYERS,
                 'd_ff': D_FF_V17, 'predictor_hidden': PRED_HIDDEN_V17},
        'seeds': SEEDS,
        'k_eval_list': K_EVAL_LIST,
        'per_seed': rs,
        'agg': {
            'test_rmse': {
                'mean': float(np.mean([r['test_rmse'] for r in rs])),
                'std': float(np.std([r['test_rmse'] for r in rs])),
                'per_seed': [r['test_rmse'] for r in rs],
            },
            'val_rmse': {
                'mean': float(np.mean([r['val_rmse'] for r in rs])),
                'std': float(np.std([r['val_rmse'] for r in rs])),
                'per_seed': [r['val_rmse'] for r in rs],
            },
            'f1_by_k': {
                ke: {
                    'f1_mean': float(np.mean([r['f1_by_k'][ke]['f1'] for r in rs])),
                    'f1_std': float(np.std([r['f1_by_k'][ke]['f1'] for r in rs])),
                    'auc_pr_mean': float(np.mean([r['f1_by_k'][ke]['auc_pr'] for r in rs])),
                    'precision_mean': float(np.mean([r['f1_by_k'][ke]['precision'] for r in rs])),
                    'recall_mean': float(np.mean([r['f1_by_k'][ke]['recall'] for r in rs])),
                } for ke in K_EVAL_LIST
            },
        },
        'runtime_minutes': (time.time() - t0) / 60,
    }
    out_path = V18 / 'phase1a_frozen_multi_k.json'
    with open(out_path, 'w') as f: json.dump(summary, f, indent=2, default=float)
    print(f"\nSaved: {out_path}", flush=True)

    print("\n" + "=" * 60)
    print("V18 Phase 1a: HONEST FROZEN PROBE (v17 best ckpts)")
    print("=" * 60)
    a = summary['agg']
    print(f"Test RMSE: {a['test_rmse']['mean']:.2f} +/- {a['test_rmse']['std']:.2f} "
          f"(v17 ref 15.38)")
    for ke in K_EVAL_LIST:
        m = a['f1_by_k'][ke]
        print(f"  k={ke:2d}: F1={m['f1_mean']:.3f} AUC-PR={m['auc_pr_mean']:.3f}")


if __name__ == '__main__':
    main()
