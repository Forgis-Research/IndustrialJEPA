"""
V18 Phase 6c: Honest-protocol E2E fine-tuning on FD003 and FD004.

Phase 6 delivered FD003/FD004 FROZEN probe numbers. Reviewers asked for E2E
under honest protocol too (round-4 M4: "rerun FD003/FD004 with 5 probe seeds
under honest protocol"). This phase fine-tunes the v14 full-sequence (V2-
arch) checkpoints end-to-end on FD003 and FD004 with the honest val protocol
(n_cuts=10) and 3 probe seeds per subset.

V14 old-protocol references:
  FD003 fullseq E2E: 13.67
  FD004 fullseq E2E: 25.27

Output: experiments/v18/phase6c_fd003_fd004_e2e.json
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
from evaluation.grey_swan_metrics import anomaly_metrics as _anomaly_metrics

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D_MODEL = 256; N_HEADS = 4; N_LAYERS = 2; D_FF = 512
PRED_HIDDEN = 256; EMA_MOMENTUM = 0.99
E2E_LR = 1e-4; E2E_EPOCHS = 50; E2E_PATIENCE = 15
SEEDS = [0, 1, 2]
K_EVAL_LIST = [10, 20, 30, 50]

CKPTS = {
    'FD003': V14 / 'best_pretrain_full_sequence_fd003.pt',
    'FD004': V14 / 'best_pretrain_full_sequence_fd004.pt',
}


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


def run_e2e(data, ckpt_path, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = load_model(ckpt_path); probe = RULProbe(D_MODEL).to(DEVICE)

    tr_ds = CMAPSSFinetuneDataset(data['train_engines'], n_cuts_per_engine=5, seed=seed)
    va_ds = CMAPSSFinetuneDataset(data['val_engines'], n_cuts_per_engine=10, seed=seed+111)
    te_ds = CMAPSSTestDataset(data['test_engines'], data['test_rul'])
    tr = DataLoader(tr_ds, batch_size=16, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(va_ds, batch_size=16, shuffle=False, collate_fn=collate_finetune)
    te = DataLoader(te_ds, batch_size=16, shuffle=False, collate_fn=collate_test)

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
        preds = np.concatenate(pv)*RUL_CAP; targs = np.concatenate(tv)*RUL_CAP
        val_rmse = float(np.sqrt(np.mean((preds - targs)**2)))
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

    f1_by_k = {}
    for ke in K_EVAL_LIST:
        y = (targs <= ke).astype(int); score = -preds
        thr = float(np.percentile(score[y == 0], 95)) if (y == 0).sum() > 0 else 0.0
        m = _anomaly_metrics(score, y, threshold=thr)
        f1_by_k[ke] = {'f1': float(m['f1_non_pa']), 'auc_pr': float(m['auc_pr'])}
    return {'seed': seed, 'val_rmse': best_val, 'test_rmse': test_rmse, 'f1_by_k': f1_by_k}


def main():
    t0 = time.time()
    results = {}
    for subset in ['FD003', 'FD004']:
        print(f"\n=== {subset} E2E (v14 full-seq, honest protocol) ===", flush=True)
        data = load_cmapss_subset(subset)
        rs = []
        for s in SEEDS:
            r = run_e2e(data, CKPTS[subset], s)
            rs.append(r)
            print(f"  seed={s}: val={r['val_rmse']:.2f} test={r['test_rmse']:.2f} "
                  f"F1@30={r['f1_by_k'][30]['f1']:.3f}", flush=True)
        results[subset] = {
            'per_seed': rs,
            'test_rmse_mean': float(np.mean([r['test_rmse'] for r in rs])),
            'test_rmse_std': float(np.std([r['test_rmse'] for r in rs])),
            'val_rmse_mean': float(np.mean([r['val_rmse'] for r in rs])),
        }

    summary = {
        'config': 'v18_phase6c_fd003_fd004_e2e_honest',
        'results': results,
        'v14_old_protocol_e2e_reference': {'FD003': 13.67, 'FD004': 25.27},
        'runtime_min': (time.time() - t0) / 60,
    }
    with open(V18 / 'phase6c_fd003_fd004_e2e.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n" + "=" * 60)
    print("V18 Phase 6c: FD003/FD004 E2E honest protocol")
    print("=" * 60)
    for subset in ['FD003', 'FD004']:
        r = results[subset]
        print(f"  {subset}: {r['test_rmse_mean']:.2f} +/- {r['test_rmse_std']:.2f}  "
              f"(v14 old-protocol ref: {summary['v14_old_protocol_e2e_reference'][subset]})")
    print(f"Runtime: {summary['runtime_min']:.1f} min")


if __name__ == '__main__':
    main()
