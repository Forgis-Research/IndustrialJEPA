"""V23 Phase 4: PhysioNet 2019 Sepsis - pretrain + pred-FT.

Dataset:
  34 clinical variables + SepsisLabel, per-patient .psv files.  Hourly
  resolution, variable-length stays (min ~6, max ~336 hours).

Splits (from data.sepsis.load_sepsis):
  pretrain_patients : non-septic set-A ft_train patients (~14.8k stays).
  ft_train          : set A 80%  (~16.3k stays)
  ft_val            : set A 20%  (~ 4.1k stays)
  ft_test           : set B      (~20.0k stays)

Window = 24 hours.  Horizons = [1, 2, 3, 6, 12, 24, 36, 48] hours.

Protocol per seed (42, 123, 456):
  1. Pretrain TrajectoryJEPA(n_sensors=34, window=24) on pretrain_patients.
  2. Freeze encoder; train EventHead + predictor via pos-weighted BCE on
     ft_train / ft_val.
  3. Evaluate probability surface on ft_test.  Compute AUPRC (primary),
     AUROC (SOTA comparison), F1, per-horizon AUPRC.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

ROOT = Path('/home/sagemaker-user/IndustrialJEPA')
FAM = ROOT / 'fam-jepa'
V11 = FAM / 'experiments' / 'v11'
V21 = FAM / 'experiments' / 'v21'
V22 = FAM / 'experiments' / 'v22'
V23 = FAM / 'experiments' / 'v23'

sys.path.insert(0, str(V11))
sys.path.insert(0, str(V21))
sys.path.insert(0, str(V22))
sys.path.insert(0, str(V23))
sys.path.insert(0, str(FAM))

from models import TrajectoryJEPA, count_parameters  # noqa: E402
from pred_ft_utils import (  # noqa: E402
    AnomalyWindowDataset, collate_anomaly_window,
    EventHead, train_bce, evaluate_surface, estimate_pos_weight,
    save_surface,
)
from evaluation.surface_metrics import (  # noqa: E402
    evaluate_probability_surface, auprc_per_horizon,
    monotonicity_violation_rate,
)
from data.sepsis import load_sepsis, N_CHANNELS  # noqa: E402

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- architecture ---------------------------------------------------------
WINDOW     = 24
W_FUT      = 4             # future-target window (hours)
K_MAX      = 48            # predict up to 48 hours ahead
HORIZONS   = [1, 2, 3, 6, 12, 24, 36, 48]
D_MODEL    = 256
N_HEADS    = 4
N_LAYERS   = 2
D_FF       = 1024

# ---- pretrain hyperparams (matching v22 baseline) -------------------------
BATCH_SIZE = 128           # stays are short, so batch more
LR         = 3e-4
WD         = 0.01
EMA_MOM    = 0.99
LAMBDA_VAR = 0.04
N_SAMPLES  = 8000          # samples per epoch from the patient cohort

# ---- pred-FT -------------------------------------------------------------
TRAIN_CFG = dict(lr=1e-3, wd=1e-2, n_epochs=30, patience=6)
STRIDE_TR = 2              # stride through each patient
STRIDE_EV = 1              # dense eval


# ---------------------------------------------------------------------------
# Pretraining dataset over patient stays
# ---------------------------------------------------------------------------

class SepsisPretrainDataset(Dataset):
    """Samples (past, future, k) across a list of patient stays.

    Each sample picks a patient uniformly at random, then a cut t such that
    past = patient.x[t-WINDOW:t] and future = patient.x[t+k:t+k+W_FUT].
    Patients whose stay is too short for any valid cut are excluded.
    """

    def __init__(self, patients: List[Dict], n_samples: int = N_SAMPLES,
                 window: int = WINDOW, w: int = W_FUT, K_max: int = K_MAX,
                 seed: int = 42):
        self.window = window
        self.w = w
        self.K_max = K_max
        rng = np.random.RandomState(seed)
        # Only keep patients with T >= window + 1 + w
        self.pats = [p for p in patients
                     if len(p['x']) >= window + 1 + w]
        if not self.pats:
            raise ValueError('no patients long enough for WINDOW+W_FUT')
        self.samples = []
        n = len(self.pats)
        for _ in range(n_samples):
            i = int(rng.randint(0, n))
            T = len(self.pats[i]['x'])
            t_lo = window
            t_hi = T - 1 - w
            if t_hi < t_lo:
                continue
            t = int(rng.randint(t_lo, t_hi + 1))
            k_hi = min(K_max, T - t - w)
            if k_hi < 1:
                continue
            u = rng.uniform(0.0, math.log(max(2.0, float(k_hi))))
            k = int(math.exp(u))
            k = max(1, min(k, k_hi))
            self.samples.append((i, t, k))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        i, t, k = self.samples[idx]
        x = self.pats[i]['x']
        past = torch.from_numpy(x[t - self.window: t]).float()
        future = torch.from_numpy(x[t + k: t + k + self.w]).float()
        return past, future, k


def collate_fixed(batch):
    pasts, futures, ks = zip(*batch)
    x_past = torch.stack(list(pasts), dim=0)
    x_fut = torch.stack(list(futures), dim=0)
    past_mask = torch.zeros(x_past.shape[:2], dtype=torch.bool)
    fut_mask = torch.zeros(x_fut.shape[:2], dtype=torch.bool)
    k_t = torch.tensor(ks, dtype=torch.long)
    return x_past, past_mask, x_fut, fut_mask, k_t


def l1_loss_with_var(pred, target, lam_var=LAMBDA_VAR):
    pred_n = F.normalize(pred, dim=-1)
    targ_n = F.normalize(target.detach(), dim=-1)
    l_pred = F.l1_loss(pred_n, targ_n)
    std = pred.std(dim=0)
    l_var = F.relu(1.0 - std).mean()
    return l_pred + lam_var * l_var


def pretrain(patients: List[Dict], seed: int, max_epochs: int, patience: int,
             verbose: bool = True) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    model = TrajectoryJEPA(
        n_sensors=N_CHANNELS, patch_length=1,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        dropout=0.1, ema_momentum=EMA_MOM, predictor_hidden=D_FF,
    ).to(DEVICE)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs)
    best = {'loss': float('inf'), 'sd': None, 'ep': 0}
    no_impr = 0
    losses = []
    t0 = time.time()
    for ep in range(1, max_epochs + 1):
        ds = SepsisPretrainDataset(patients, n_samples=N_SAMPLES,
                                   seed=seed * 1000 + ep)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_fixed, num_workers=0)
        model.train()
        tot = 0.0; n = 0
        for past, past_mask, fut, fut_mask, k in loader:
            past = past.to(DEVICE); past_mask = past_mask.to(DEVICE)
            fut = fut.to(DEVICE); fut_mask = fut_mask.to(DEVICE)
            k = k.to(DEVICE)
            opt.zero_grad()
            pred, targ, _ = model.forward_pretrain(past, past_mask,
                                                   fut, fut_mask, k)
            loss = l1_loss_with_var(pred, targ, LAMBDA_VAR)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            model.update_ema()
            B = past.shape[0]
            tot += loss.item() * B; n += B
        sched.step()
        ep_loss = tot / max(n, 1)
        losses.append(ep_loss)
        if ep_loss < best['loss'] - 1e-4:
            best = {'loss': ep_loss, 'sd': copy.deepcopy(model.state_dict()),
                    'ep': ep}
            no_impr = 0
        else:
            no_impr += 1
        if verbose and (ep % 3 == 0 or ep == 1):
            print(f"    ep {ep:3d}  L={ep_loss:.4f}  best={best['loss']:.4f}  "
                  f"ni={no_impr}  ({(time.time()-t0)/60:.1f}m)", flush=True)
        if no_impr >= patience:
            if verbose:
                print(f"    Early stop at ep {ep} (best={best['loss']:.4f})",
                      flush=True)
            break
    if best['sd'] is not None:
        model.load_state_dict(best['sd'])
    return {'model': model, 'best_loss': best['loss'],
            'best_ep': best['ep'], 'final_ep': ep, 'losses': losses,
            'runtime_min': (time.time() - t0) / 60,
            'params': count_parameters(model)}


# ---------------------------------------------------------------------------
# Pred-FT on patient-level splits
# ---------------------------------------------------------------------------

def _build_ds(patients: List[Dict], stride: int, max_future: int
              ) -> ConcatDataset:
    parts = []
    for p in patients:
        if len(p['x']) < WINDOW + 1:
            continue
        parts.append(AnomalyWindowDataset(
            p['x'], p['labels'],
            window=WINDOW, stride=stride, max_future=max_future))
    return ConcatDataset(parts)


def pred_ft(d, seed: int, model: TrajectoryJEPA) -> dict:
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    max_fut = max(HORIZONS) + 1
    tr_ds = _build_ds(d['ft_train'], STRIDE_TR, max_fut)
    va_ds = _build_ds(d['ft_val'],   STRIDE_EV, max_fut)
    te_ds = _build_ds(d['ft_test'],  STRIDE_EV, max_fut)
    print(f"    ft_train windows={len(tr_ds)} ft_val={len(va_ds)} "
          f"ft_test={len(te_ds)}", flush=True)
    tr = DataLoader(tr_ds, batch_size=256, shuffle=True,
                    collate_fn=collate_anomaly_window, num_workers=0)
    va = DataLoader(va_ds, batch_size=256, shuffle=False,
                    collate_fn=collate_anomaly_window, num_workers=0)
    te = DataLoader(te_ds, batch_size=256, shuffle=False,
                    collate_fn=collate_anomaly_window, num_workers=0)
    pw = estimate_pos_weight(tr, HORIZONS)
    head = EventHead(D_MODEL).to(DEVICE)
    train_out = train_bce(model, head, tr, va, mode='pred_ft',
                          pos_weight=pw, horizons_eval=HORIZONS,
                          device=DEVICE, **TRAIN_CFG)
    surf = evaluate_surface(model, head, te, mode='pred_ft',
                            horizons=HORIZONS, device=DEVICE)
    p, y = surf['p_surface'], surf['y_surface']
    (V23 / 'surfaces').mkdir(exist_ok=True)
    surf_path = V23 / 'surfaces' / f'sepsis_pred_ft_seed{seed}.npz'
    save_surface(surf_path, p, y, HORIZONS, surf['t_index'],
                 metadata={'dataset': 'sepsis', 'seed': seed,
                           'mode': 'pred_ft', 'pos_weight': float(pw)})
    prim = evaluate_probability_surface(p, y)
    per_h = auprc_per_horizon(p, y, horizon_labels=HORIZONS)
    mono = monotonicity_violation_rate(p)
    return {'seed': seed,
            'primary': prim, 'per_horizon': per_h,
            'monotonicity': mono,
            'train': {'best_val': train_out['best_val'],
                      'final_epoch': train_out['final_epoch']},
            'pos_weight': float(pw),
            'n_train_windows': len(tr_ds),
            'n_test_windows': len(te_ds),
            'surface_file': str(surf_path),
            'runtime_s': time.time() - t0}


def _agg(rs):
    out = {'n_seeds': len(rs)}
    for name, fn in [
        ('auprc',          lambda r: r['primary']['auprc']),
        ('auroc',          lambda r: r['primary']['auroc']),
        ('f1_best',        lambda r: r['primary']['f1_best']),
        ('precision_best', lambda r: r['primary']['precision_best']),
        ('recall_best',    lambda r: r['primary']['recall_best']),
        ('mono_violation', lambda r: r['monotonicity']['violation_rate']),
    ]:
        vals = np.array([fn(r) for r in rs], float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            out[f'{name}_mean'] = float('nan')
            out[f'{name}_std'] = float('nan')
        else:
            out[f'{name}_mean'] = float(vals.mean())
            out[f'{name}_std'] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    ap.add_argument('--max-epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=5)
    args = ap.parse_args()

    (V23 / 'ckpts').mkdir(exist_ok=True)
    (V23 / 'surfaces').mkdir(exist_ok=True)
    out_path = V23 / 'phase4_sepsis.json'

    print('Loading sepsis dataset...', flush=True)
    d = load_sepsis(verbose=True)

    all_results = []
    t_all = time.time()
    for seed in args.seeds:
        print(f'\n=== sepsis seed {seed} ===', flush=True)
        pr = pretrain(d['pretrain_patients'], seed,
                      max_epochs=args.max_epochs, patience=args.patience)
        ckpt = V23 / 'ckpts' / f'sepsis_seed{seed}_best.pt'
        torch.save(pr['model'].state_dict(), ckpt)
        print(f"    params={pr['params']:,} "
              f"best_L={pr['best_loss']:.4f}@ep{pr['best_ep']} "
              f"({pr['runtime_min']:.1f}m)", flush=True)
        fr = pred_ft(d, seed, pr['model'])
        fr['pretrain'] = {'best_loss': pr['best_loss'],
                          'best_ep': pr['best_ep'],
                          'final_ep': pr['final_ep'],
                          'runtime_min': pr['runtime_min'],
                          'params': pr['params']}
        all_results.append(fr)
        print(f"    [sepsis s{seed}] AUPRC={fr['primary']['auprc']:.3f} "
              f"AUROC={fr['primary']['auroc']:.3f} "
              f"F1={fr['primary']['f1_best']:.3f} "
              f"(P={fr['primary']['precision_best']:.3f} "
              f"R={fr['primary']['recall_best']:.3f}) "
              f"mono={fr['monotonicity']['violation_rate']:.3f} "
              f"({fr['runtime_s']:.0f}s)", flush=True)
        del pr
        torch.cuda.empty_cache()
        with open(out_path, 'w') as f:
            json.dump({'per_seed': all_results, 'agg': _agg(all_results),
                       'seeds': args.seeds,
                       'horizons': HORIZONS,
                       'window': WINDOW,
                       'runtime_min': (time.time() - t_all) / 60},
                      f, indent=2, default=float)

    print(f'\nDONE in {(time.time()-t_all)/60:.1f}m -> {out_path}')
    a = _agg(all_results)
    print('\n' + '=' * 72)
    print('V23 PHASE 4 SUMMARY (PhysioNet 2019 Sepsis, pred-FT)')
    print('=' * 72)
    print(f"AUPRC = {a['auprc_mean']:.3f}±{a['auprc_std']:.3f}")
    print(f"AUROC = {a['auroc_mean']:.3f}±{a['auroc_std']:.3f}   "
          f"(SOTA ~0.78-0.85)")
    print(f"F1    = {a['f1_best_mean']:.3f}±{a['f1_best_std']:.3f}  "
          f"(P={a['precision_best_mean']:.3f}  "
          f"R={a['recall_best_mean']:.3f})")


if __name__ == '__main__':
    main()
