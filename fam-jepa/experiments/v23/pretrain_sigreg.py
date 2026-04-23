"""V23 Phase 1: SIGReg pretrain on FD001 (no EMA).

Drop-EMA ablation: target = context_encoder(future).detach() (stop-grad on
the SAME encoder), so there is no EMA to tune. Explicit collapse prevention
comes from a VICReg-style triplet on h_past:

  loss = L1(pred_norm, target_norm) + lam_var * var_loss + lam_cov * cov_loss

  var_loss = mean(relu(1 - std(h_past, dim=0)))                    (VICReg)
  cov_loss = sum_{i!=j} cov(h_past)_{i,j}^2 / d                    (VICReg)

Curriculum on the future horizon k (the predictor is unstable early):

  ep  1-20 :  k ~ LogUniform[1, 10]
  ep 20-40 :  k ~ LogUniform[1, k_hi(ep)]   linearly grows 10 -> 150
  ep 40+   :  k ~ LogUniform[1, 150]

Collapse detection: abort the seed if mean(std(h_past)) < 0.01 at any epoch.

Outputs:
  v23/ckpts/sigreg_fd001_seed{S}_best.pt
  v23/pretrain_sigreg.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path('/home/sagemaker-user/IndustrialJEPA')
FAM = ROOT / 'fam-jepa'
V11 = FAM / 'experiments' / 'v11'
V17 = FAM / 'experiments' / 'v17'
V22 = FAM / 'experiments' / 'v22'
V23 = FAM / 'experiments' / 'v23'

sys.path.insert(0, str(V11))
sys.path.insert(0, str(V17))
sys.path.insert(0, str(V22))
sys.path.insert(0, str(V23))
sys.path.insert(0, str(FAM))

from data_utils import load_cmapss_subset, N_SENSORS  # noqa: E402
from models import TrajectoryJEPA  # noqa: E402

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Shared config (matches v22 baseline for direct comparison)
WINDOW     = 100
W_FUT      = 10
K_MAX      = 150
N_CUTS     = 30
MIN_PAST   = WINDOW
D_MODEL    = 256
N_HEADS    = 4
N_LAYERS   = 2
D_FF       = 1024
BATCH_SIZE = 64
LR         = 3e-4
WD         = 0.01
LAMBDA_VAR = 0.04
LAMBDA_COV = 0.02

# Curriculum on k
K_WARMUP_HI   = 10   # ep 1..20
K_RAMP_END_EP = 40   # linearly grow to K_MAX by this epoch
WARMUP_EP     = 20

COLLAPSE_STD_THRESHOLD = 0.01


def k_hi_for_epoch(ep: int) -> int:
    if ep <= WARMUP_EP:
        return K_WARMUP_HI
    if ep >= K_RAMP_END_EP:
        return K_MAX
    frac = (ep - WARMUP_EP) / (K_RAMP_END_EP - WARMUP_EP)
    return int(round(K_WARMUP_HI + frac * (K_MAX - K_WARMUP_HI)))


class FixedWindowPretrainDataset(Dataset):
    """Fixed past length = WINDOW (=100).  LogUniform k in [1, k_hi]."""

    def __init__(self, engines, n_cuts=N_CUTS, window=WINDOW,
                 K_hi=K_MAX, w=W_FUT, seed=42):
        self.w = w
        self.window = window
        rng = np.random.RandomState(seed)
        self.samples = []
        for eid, arr in engines.items():
            T = len(arr)
            t_lo = window
            t_hi = T - 1 - w
            if t_hi < t_lo:
                continue
            for _ in range(n_cuts):
                t = int(rng.randint(t_lo, t_hi + 1))
                k_hi = min(K_hi, T - t - w)
                if k_hi < 1:
                    continue
                u = rng.uniform(0.0, math.log(max(2.0, float(k_hi))))
                k = int(math.exp(u))
                k = max(1, min(k, k_hi))
                self.samples.append((eid, t, k))
        self.engines = engines

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        eid, t, k = self.samples[idx]
        arr = self.engines[eid]
        past = torch.from_numpy(arr[t - self.window: t]).float()
        future = torch.from_numpy(arr[t + k: t + k + self.w]).float()
        return past, future, k


def collate_fixed(batch):
    pasts, futures, ks = zip(*batch)
    x_past = torch.stack(list(pasts), dim=0)
    x_fut  = torch.stack(list(futures), dim=0)
    past_mask = torch.zeros(x_past.shape[:2], dtype=torch.bool)
    fut_mask  = torch.zeros(x_fut.shape[:2],  dtype=torch.bool)
    k_t = torch.tensor(ks, dtype=torch.long)
    return x_past, past_mask, x_fut, fut_mask, k_t


def vicreg_var_cov(h: torch.Tensor, eps: float = 1e-4):
    """VICReg variance + covariance terms computed on h (B, D)."""
    B, D = h.shape
    # variance: encourage each dim to have std >= 1
    std = h.std(dim=0) + eps
    l_var = F.relu(1.0 - std).mean()
    # covariance: encourage off-diagonal cov to be 0
    h_c = h - h.mean(dim=0, keepdim=True)
    cov = (h_c.T @ h_c) / max(B - 1, 1)             # (D, D)
    off = cov - torch.diag(torch.diag(cov))
    l_cov = (off ** 2).sum() / D
    return l_var, l_cov, std.mean().detach()


def sigreg_loss(pred, target, h_past,
                lam_var=LAMBDA_VAR, lam_cov=LAMBDA_COV):
    pred_n = F.normalize(pred, dim=-1)
    targ_n = F.normalize(target.detach(), dim=-1)
    l_pred = F.l1_loss(pred_n, targ_n)
    l_var, l_cov, mean_std = vicreg_var_cov(h_past)
    return (l_pred + lam_var * l_var + lam_cov * l_cov,
            l_pred, l_var, l_cov, mean_std)


def pretrain_one(seed: int, data: dict, max_epochs: int = 50,
                 patience: int = 5, verbose: bool = True) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)

    # Reuse TrajectoryJEPA infrastructure (keeps downstream loaders working).
    # target_encoder weights are unused during SIGReg pretraining; downstream
    # pred-FT only calls encode_past = context_encoder.
    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        dropout=0.1, ema_momentum=0.99, predictor_hidden=D_FF,
    ).to(DEVICE)
    # Train only context_encoder + predictor.
    for p in model.target_encoder.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs)

    best = {'loss': float('inf'), 'sd': None, 'ep': 0}
    no_impr = 0
    history = {
        'ep': [], 'loss': [], 'l_pred': [], 'l_var': [], 'l_cov': [],
        'mean_std': [], 'k_hi': [],
    }
    collapsed = False
    t0 = time.time()

    for ep in range(1, max_epochs + 1):
        k_hi = k_hi_for_epoch(ep)
        ds = FixedWindowPretrainDataset(
            data['train_engines'], n_cuts=N_CUTS, K_hi=k_hi,
            seed=seed * 1000 + ep)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_fixed, num_workers=0)
        model.train()
        tot = 0.0; n = 0
        lp = lv = lc = ms = 0.0
        for past, past_mask, fut, fut_mask, k in loader:
            past = past.to(DEVICE); past_mask = past_mask.to(DEVICE)
            fut = fut.to(DEVICE); fut_mask = fut_mask.to(DEVICE)
            k = k.to(DEVICE)
            opt.zero_grad()
            # Forward: h_past via context_encoder; target via stop-grad on
            # the SAME encoder applied causally to the future window
            # (matches v17 phase 3 SG variant).
            h_past = model.context_encoder(past, past_mask)
            with torch.no_grad():
                target = model.context_encoder(fut, fut_mask)
            pred = model.predictor(h_past, k)
            loss, l_pred, l_var, l_cov, mean_std = sigreg_loss(
                pred, target, h_past, LAMBDA_VAR, LAMBDA_COV)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            # NO EMA update — that's the point.
            B = past.shape[0]
            tot += loss.item() * B
            lp  += l_pred.item() * B
            lv  += l_var.item() * B
            lc  += l_cov.item() * B
            ms  += mean_std.item() * B
            n += B
        sched.step()
        ep_loss = tot / max(n, 1)
        ep_mean_std = ms / max(n, 1)
        history['ep'].append(ep)
        history['loss'].append(ep_loss)
        history['l_pred'].append(lp / max(n, 1))
        history['l_var'].append(lv / max(n, 1))
        history['l_cov'].append(lc / max(n, 1))
        history['mean_std'].append(ep_mean_std)
        history['k_hi'].append(k_hi)

        # Collapse detection
        if ep_mean_std < COLLAPSE_STD_THRESHOLD:
            if verbose:
                print(f"    !! COLLAPSE at ep {ep} (mean_std={ep_mean_std:.4f})",
                      flush=True)
            collapsed = True
            break

        # Don't track best / patience until the k curriculum has fully
        # ramped.  During warmup (k in [1, 10]) and ramp (k linearly up
        # to 150), loss is incomparable across epochs because the task
        # difficulty is changing.
        tracking = ep >= K_RAMP_END_EP
        if tracking:
            if ep_loss < best['loss'] - 1e-4:
                best = {'loss': ep_loss,
                        'sd': copy.deepcopy(model.state_dict()),
                        'ep': ep}
                no_impr = 0
            else:
                no_impr += 1
        else:
            # Keep latest as best so we save something if max_epochs
            # is hit before the ramp finishes.
            best = {'loss': ep_loss,
                    'sd': copy.deepcopy(model.state_dict()),
                    'ep': ep}
        if verbose and (ep % 2 == 0 or ep == 1):
            flag = 'T' if tracking else 'w'
            print(f"    ep {ep:3d}[{flag}] | k_hi={k_hi} L={ep_loss:.4f} "
                  f"l_pred={lp/n:.4f} l_var={lv/n:.4f} l_cov={lc/n:.4f} "
                  f"std={ep_mean_std:.3f} best={best['loss']:.4f} "
                  f"ni={no_impr} ({(time.time()-t0)/60:.1f}m)",
                  flush=True)
        if tracking and no_impr >= patience:
            if verbose:
                print(f"    Early stop at ep {ep} (best={best['loss']:.4f})",
                      flush=True)
            break

    if best['sd'] is not None:
        model.load_state_dict(best['sd'])
    return {'seed': seed, 'best_loss': best['loss'],
            'best_ep': best['ep'], 'final_ep': ep,
            'collapsed': collapsed, 'history': history,
            'runtime_min': (time.time() - t0) / 60,
            'model': model}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    ap.add_argument('--max-epochs', type=int, default=50)
    ap.add_argument('--patience', type=int, default=5)
    args = ap.parse_args()

    (V23 / 'ckpts').mkdir(exist_ok=True, parents=True)
    out_path = V23 / 'pretrain_sigreg.json'
    data = load_cmapss_subset('FD001')
    print(f'Loaded FD001: {len(data["train_engines"])} engines', flush=True)

    log = []
    t_all = time.time()
    for seed in args.seeds:
        print(f'\n=== SIGReg pretrain seed {seed} ===', flush=True)
        res = pretrain_one(seed, data,
                           max_epochs=args.max_epochs,
                           patience=args.patience)
        ckpt_path = V23 / 'ckpts' / f'sigreg_fd001_seed{seed}_best.pt'
        torch.save(res['model'].state_dict(), ckpt_path)
        print(f"    -> {ckpt_path}  best_L={res['best_loss']:.4f} "
              f"@ep{res['best_ep']}  collapsed={res['collapsed']}  "
              f"({res['runtime_min']:.1f}m)", flush=True)
        log.append({
            'seed': seed, 'ckpt': str(ckpt_path),
            'best_loss': res['best_loss'], 'best_ep': res['best_ep'],
            'final_ep': res['final_ep'], 'collapsed': res['collapsed'],
            'history': res['history'],
            'runtime_min': res['runtime_min'],
        })
        del res
        torch.cuda.empty_cache()
        with open(out_path, 'w') as f:
            json.dump({'log': log, 'config': vars(args),
                       'runtime_min': (time.time() - t_all) / 60},
                      f, indent=2, default=float)

    print(f'\nDONE in {(time.time()-t_all)/60:.1f}m -> {out_path}')


if __name__ == '__main__':
    main()
