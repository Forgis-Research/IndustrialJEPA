"""V23 Phase 5+6: Patch tokenization on SMAP (and PSM).

Baseline tokenizes each timestep as one token (patch_length=1 -> 100 tokens
over window=100).  With patch_length=L, we group L timesteps into one token
via Linear(C * L, d), giving window / L tokens.

Protocol matches v22 baseline pretrain exactly, EXCEPT for patch_length:
  Fixed past window = 100 (must be divisible by L)
  Random future window w = 10 at horizon k ~ LogUniform[1, 150]
  L1 loss on L2-normalized EMA-target embeddings + var_reg (lambda=0.04)
  AdamW lr=3e-4, cosine, batch=64, grad clip=1, EMA momentum=0.99
  patience=5, max 50 epochs

After pretraining we freeze the encoder and run pred-FT with entity splits.

Outputs:
  v23/ckpts/patch_L{L}_{dataset}_seed{S}_best.pt
  v23/phase5_patch_pretrain_{dataset}.json
  v23/phase6_patch_predft_{dataset}.json
  v23/surfaces/{dataset}_patch_L{L}_pred_ft_seed{S}.npz
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

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
    save_surface, HORIZONS_STEPS,
)
from evaluation.surface_metrics import (  # noqa: E402
    evaluate_probability_surface, auprc_per_horizon,
    monotonicity_violation_rate,
)
from data import (  # noqa: E402
    load_smap, load_psm, split_smap_entities,
)
from data.smap_msl import WINDOW_SIZE  # noqa: E402

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

WINDOW     = 100
W_FUT      = 10
K_MAX      = 150
N_SAMPLES  = 2000
D_MODEL    = 256
N_HEADS    = 4
N_LAYERS   = 2
D_FF       = 1024
BATCH_SIZE = 64
LR         = 3e-4
WD         = 0.01
EMA_MOM    = 0.99
LAMBDA_VAR = 0.04

TRAIN_CFG = dict(lr=1e-3, wd=1e-2, n_epochs=40, patience=8)


# ---------------------------------------------------------------------------
# Pretraining dataset (fixed window on a concatenated train array)
# ---------------------------------------------------------------------------

class FixedWindowArrayPretrainDataset(Dataset):
    """Samples (past, future, k) from a 2D train array (T, C)."""

    def __init__(self, arr: np.ndarray, n_samples: int,
                 window: int = WINDOW, w: int = W_FUT, K_max: int = K_MAX,
                 seed: int = 42):
        self.arr = arr
        self.window = window
        self.w = w
        T = len(arr)
        rng = np.random.RandomState(seed)
        self.samples = []
        t_lo = window
        t_hi = T - 1 - w
        if t_hi < t_lo:
            raise ValueError(f'train stream too short T={T}')
        for _ in range(n_samples):
            t = int(rng.randint(t_lo, t_hi + 1))
            k_hi = min(K_max, T - t - w)
            if k_hi < 1:
                continue
            u = rng.uniform(0.0, math.log(max(2.0, float(k_hi))))
            k = int(math.exp(u))
            k = max(1, min(k, k_hi))
            self.samples.append((t, k))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t, k = self.samples[idx]
        past = torch.from_numpy(self.arr[t - self.window: t]).float()
        future = torch.from_numpy(self.arr[t + k: t + k + self.w]).float()
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


def build_patch_model(n_sensors: int, patch_length: int) -> TrajectoryJEPA:
    return TrajectoryJEPA(
        n_sensors=n_sensors, patch_length=patch_length,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        dropout=0.1, ema_momentum=EMA_MOM, predictor_hidden=D_FF,
    ).to(DEVICE)


def pretrain(patch_length: int, arr: np.ndarray, n_sensors: int,
             seed: int, max_epochs: int, patience: int,
             verbose: bool = True) -> dict:
    if WINDOW % patch_length != 0:
        raise ValueError(f'WINDOW={WINDOW} not divisible by L={patch_length}')
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_patch_model(n_sensors, patch_length)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs)
    best = {'loss': float('inf'), 'sd': None, 'ep': 0}
    no_impr = 0
    losses = []
    t0 = time.time()
    for ep in range(1, max_epochs + 1):
        ds = FixedWindowArrayPretrainDataset(
            arr, n_samples=N_SAMPLES, seed=seed * 1000 + ep)
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
# Pred-FT on entity splits (SMAP) or stream split (PSM)
# ---------------------------------------------------------------------------

def _build_ds_from_entities(entities, window, stride, max_future):
    return ConcatDataset([
        AnomalyWindowDataset(e['test'], e['labels'],
                             window=window, stride=stride,
                             max_future=max_future) for e in entities])


def _build_ds_from_stream(split_dict, window, stride, max_future):
    return AnomalyWindowDataset(
        split_dict['test'], split_dict['labels'],
        window=window, stride=stride, max_future=max_future)


def _stream_split(data, ratios=(0.6, 0.1, 0.3), gap=WINDOW_SIZE):
    T = len(data['test'])
    t1 = int(ratios[0] * T); t2 = int((ratios[0] + ratios[1]) * T)
    return (
        {'test': data['test'][:t1],           'labels': data['labels'][:t1]},
        {'test': data['test'][t1 + gap: t2],  'labels': data['labels'][t1 + gap: t2]},
        {'test': data['test'][t2 + gap:],     'labels': data['labels'][t2 + gap:]},
    )


def pred_ft(dataset: str, patch_length: int, seed: int,
            model: TrajectoryJEPA) -> dict:
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    max_fut = max(HORIZONS_STEPS) + 1
    if dataset == 'SMAP':
        sp = split_smap_entities()
        tr_ds = _build_ds_from_entities(sp['ft_train'], WINDOW, 4, max_fut)
        va_ds = _build_ds_from_entities(sp['ft_val'],   WINDOW, 4, max_fut)
        te_ds = _build_ds_from_entities(sp['ft_test'],  WINDOW, 4, max_fut)
    elif dataset == 'PSM':
        data = load_psm()
        tr_sp, va_sp, te_sp = _stream_split(data)
        tr_ds = _build_ds_from_stream(tr_sp, WINDOW, 4, max_fut)
        va_ds = _build_ds_from_stream(va_sp, WINDOW, 4, max_fut)
        te_ds = _build_ds_from_stream(te_sp, WINDOW, 4, max_fut)
    else:
        raise ValueError(dataset)
    tr = DataLoader(tr_ds, batch_size=256, shuffle=True,
                    collate_fn=collate_anomaly_window, num_workers=0)
    va = DataLoader(va_ds, batch_size=256, shuffle=False,
                    collate_fn=collate_anomaly_window, num_workers=0)
    te = DataLoader(te_ds, batch_size=256, shuffle=False,
                    collate_fn=collate_anomaly_window, num_workers=0)
    pw = estimate_pos_weight(tr, HORIZONS_STEPS)
    head = EventHead(D_MODEL).to(DEVICE)
    train_out = train_bce(model, head, tr, va, mode='pred_ft',
                          pos_weight=pw, horizons_eval=HORIZONS_STEPS,
                          device=DEVICE, **TRAIN_CFG)
    surf = evaluate_surface(model, head, te, mode='pred_ft',
                            horizons=HORIZONS_STEPS, device=DEVICE)
    p, y = surf['p_surface'], surf['y_surface']
    (V23 / 'surfaces').mkdir(exist_ok=True)
    surf_path = V23 / 'surfaces' / (
        f'{dataset.lower()}_patch_L{patch_length}_pred_ft_seed{seed}.npz')
    save_surface(surf_path, p, y, HORIZONS_STEPS, surf['t_index'],
                 metadata={'dataset': dataset,
                           'patch_length': int(patch_length),
                           'seed': seed, 'mode': 'pred_ft',
                           'pos_weight': float(pw)})
    prim = evaluate_probability_surface(p, y)
    per_h = auprc_per_horizon(p, y, horizon_labels=HORIZONS_STEPS)
    mono = monotonicity_violation_rate(p)
    return {'dataset': dataset, 'patch_length': int(patch_length),
            'seed': seed, 'primary': prim, 'per_horizon': per_h,
            'monotonicity': mono,
            'train': {'best_val': train_out['best_val'],
                      'final_epoch': train_out['final_epoch']},
            'pos_weight': float(pw),
            'surface_file': str(surf_path),
            'runtime_s': time.time() - t0}


def _agg(rs):
    out = {'n_seeds': len(rs)}
    for name, fn in [
        ('auprc',          lambda r: r['primary']['auprc']),
        ('auroc',          lambda r: r['primary']['auroc']),
        ('f1_best',        lambda r: r['primary']['f1_best']),
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
    ap.add_argument('--dataset', default='SMAP', choices=['SMAP', 'PSM'])
    ap.add_argument('--patches', nargs='+', type=int, default=[1, 5, 10, 20])
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    ap.add_argument('--max-epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=5)
    args = ap.parse_args()

    (V23 / 'ckpts').mkdir(exist_ok=True)
    (V23 / 'surfaces').mkdir(exist_ok=True)
    out_pre = V23 / f'phase5_patch_pretrain_{args.dataset.lower()}.json'
    out_ft  = V23 / f'phase6_patch_predft_{args.dataset.lower()}.json'

    if args.dataset == 'SMAP':
        data = load_smap()
    else:
        data = load_psm()
    arr = data['train']
    n_sensors = data['n_channels']
    print(f'{args.dataset}: train shape {arr.shape}, n_sensors={n_sensors}',
          flush=True)

    pretrain_log = {}
    predft_log = {}
    t_all = time.time()
    for L in args.patches:
        print(f'\n=== patch L={L} ===', flush=True)
        pretrain_log[str(L)] = []
        predft_log[str(L)] = {'per_seed': [], 'agg': None}
        for seed in args.seeds:
            print(f'  -- seed {seed} --', flush=True)
            pr = pretrain(L, arr, n_sensors, seed,
                          max_epochs=args.max_epochs,
                          patience=args.patience)
            ckpt = V23 / 'ckpts' / (
                f'patch_L{L}_{args.dataset.lower()}_seed{seed}_best.pt')
            torch.save(pr['model'].state_dict(), ckpt)
            print(f"    params={pr['params']:,} "
                  f"best_L={pr['best_loss']:.4f}@ep{pr['best_ep']} "
                  f"({pr['runtime_min']:.1f}m)", flush=True)
            pretrain_log[str(L)].append({
                'seed': seed, 'ckpt': str(ckpt),
                'best_loss': pr['best_loss'], 'best_ep': pr['best_ep'],
                'final_ep': pr['final_ep'], 'losses': pr['losses'],
                'params': pr['params'],
                'runtime_min': pr['runtime_min'],
            })
            fr = pred_ft(args.dataset, L, seed, pr['model'])
            predft_log[str(L)]['per_seed'].append(fr)
            print(f"    [L={L} s{seed}] AUPRC={fr['primary']['auprc']:.3f} "
                  f"AUROC={fr['primary']['auroc']:.3f} "
                  f"F1={fr['primary']['f1_best']:.3f} "
                  f"mono={fr['monotonicity']['violation_rate']:.3f} "
                  f"FTep={fr['train']['final_epoch']} "
                  f"({fr['runtime_s']:.0f}s)", flush=True)
            del pr, fr
            torch.cuda.empty_cache()
        predft_log[str(L)]['agg'] = _agg(predft_log[str(L)]['per_seed'])
        with open(out_pre, 'w') as f:
            json.dump({'patches': args.patches, 'log': pretrain_log,
                       'runtime_min': (time.time() - t_all) / 60},
                      f, indent=2, default=float)
        with open(out_ft, 'w') as f:
            json.dump({'patches': args.patches, 'log': predft_log,
                       'runtime_min': (time.time() - t_all) / 60},
                      f, indent=2, default=float)

    print(f'\nDONE in {(time.time()-t_all)/60:.1f}m')
    print('\n' + '=' * 72)
    print(f'V23 PHASE 5+6 SUMMARY (patch tokenization on {args.dataset})')
    print('=' * 72)
    for L in args.patches:
        a = predft_log[str(L)]['agg']
        print(f"L={L:3d}  "
              f"AUPRC={a['auprc_mean']:.3f}±{a['auprc_std']:.3f} "
              f"AUROC={a['auroc_mean']:.3f}±{a['auroc_std']:.3f} "
              f"F1={a['f1_best_mean']:.3f}±{a['f1_best_std']:.3f} "
              f"mono={a['mono_violation_mean']:.3f}")


if __name__ == '__main__':
    main()
