"""
V18 Phase 6b: Honest-protocol probe of v14 full-sequence FD001.

Round-4 reviewer MAJOR-4: "Rerun V16b / full-seq under the honest protocol -
the finding likely survives but the paper needs the number."

This probes v14/best_pretrain_full_sequence.pt (d=256, d_ff=512, V2 arch
with full-sequence target) on FD001 under the honest protocol (AdamW WD=1e-2,
val n_cuts=10). 3 probe seeds. Reports test RMSE + F1@30.

v14 reported old-protocol: frozen 15.70, E2E 14.32.

Output: experiments/v18/phase6b_fullseq_fd001.json
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

from models import TrajectoryJEPA
from data_utils import (load_cmapss_subset, N_SENSORS, RUL_CAP,
                        CMAPSSFinetuneDataset, CMAPSSTestDataset,
                        collate_finetune, collate_test)
from evaluation.grey_swan_metrics import anomaly_metrics as _anomaly_metrics

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D_MODEL = 256; N_HEADS = 4; N_LAYERS = 2; D_FF = 512
PRED_HIDDEN = 256; EMA_MOMENTUM = 0.99
CKPT = V14 / 'best_pretrain_full_sequence.pt'
SEEDS = [42, 123, 456]
K_EVAL_LIST = [10, 20, 30, 50]


def load_model():
    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=PRED_HIDDEN,
    ).to(DEVICE)
    sd = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    if isinstance(sd, dict) and 'model_state_dict' in sd: sd = sd['model_state_dict']
    model.load_state_dict(sd)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    return model


def honest_probe(model, data, seed):
    torch.manual_seed(seed)
    probe = nn.Sequential(nn.Linear(D_MODEL, 1), nn.Sigmoid()).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-2)
    tr_ds = CMAPSSFinetuneDataset(data['train_engines'], n_cuts_per_engine=5, seed=seed)
    va_ds = CMAPSSFinetuneDataset(data['val_engines'], n_cuts_per_engine=10, seed=seed+111)
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
        preds = np.concatenate(pv)*RUL_CAP; targs = np.concatenate(tv)*RUL_CAP
        val_rmse = float(np.sqrt(np.mean((preds - targs)**2)))
        if val_rmse < best_val:
            best_val = val_rmse; best_state = copy.deepcopy(probe.state_dict()); no_impr = 0
        else:
            no_impr += 1
            if no_impr >= 25: break
    probe.load_state_dict(best_state); probe.eval()
    pt, tt = [], []
    with torch.no_grad():
        for past, mask, rul_gt in te:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = model.encode_past(past, mask)
            pt.append(probe(h).squeeze(-1).cpu().numpy()*RUL_CAP); tt.append(rul_gt.numpy())
    preds = np.concatenate(pt); targs = np.concatenate(tt)
    test_rmse = float(np.sqrt(np.mean((preds - targs)**2)))
    f1_by_k = {}
    for ke in K_EVAL_LIST:
        y = (targs <= ke).astype(int); score = -preds
        thr = float(np.percentile(score[y == 0], 95)) if (y == 0).sum() > 0 else 0.0
        m = _anomaly_metrics(score, y, threshold=thr)
        f1_by_k[ke] = {'f1': float(m['f1_non_pa']), 'auc_pr': float(m['auc_pr'])}
    return {'seed': seed, 'val_rmse': best_val, 'test_rmse': test_rmse, 'f1_by_k': f1_by_k}


def main():
    data = load_cmapss_subset('FD001')
    print(f"Loading {CKPT}...", flush=True)
    model = load_model()
    t0 = time.time()
    rs = []
    for seed in SEEDS:
        r = honest_probe(model, data, seed)
        rs.append(r)
        print(f"  seed={seed}: val={r['val_rmse']:.2f} test={r['test_rmse']:.2f} "
              f"F1@30={r['f1_by_k'][30]['f1']:.3f}", flush=True)

    summary = {
        'config': 'v18_phase6b_v14_fullseq_fd001_honest_probe',
        'ckpt': str(CKPT),
        'per_seed': rs,
        'test_rmse_mean': float(np.mean([r['test_rmse'] for r in rs])),
        'test_rmse_std': float(np.std([r['test_rmse'] for r in rs])),
        'v14_old_protocol_reference': {'frozen': 15.70, 'e2e': 14.32},
        'v17_honest_reference': 15.53,
        'v2_honest_reference': 15.73,
        'runtime_min': (time.time() - t0) / 60,
    }
    with open(V18 / 'phase6b_fullseq_fd001.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"\nV14 full-seq FD001 honest probe: "
          f"{summary['test_rmse_mean']:.2f} +/- {summary['test_rmse_std']:.2f}")
    print(f"V14 old-protocol reference:     15.70 frozen")
    print(f"V17 honest reference:           15.53")
    print(f"V2  honest reference:           15.73")


if __name__ == '__main__':
    main()
