"""V22 Phase 4/5: Pretrain encoder variants on FD001.

Three variants pretrained under the SAME protocol for a fair comparison:
  baseline  - v17-style causal transformer (what we've been using).
  variantA  - pure iTransformer encoder (fixed window T=100).
  variantB  - hybrid temporal + cross-channel encoder (variable T).

Protocol (matches v17 for the baseline):
  - Past past_len ∈ [min_past, T_engine].  Variant A requires fixed T=100,
    so we enforce min_past = 100 (only use cuts where past >= 100 cycles;
    take the last 100 cycles as the context).  Same enforcement is applied
    to all three variants for a fair comparison.
  - k ~ LogUniform[1, 150]
  - Target window w = 10 cycles, L1 loss on L2-normalized embeddings,
    variance regularizer lambda_var = 0.04.
  - LR = 3e-4, AdamW, cosine schedule, grad clip = 1.0, batch = 64.
  - Early stopping: patience = 5, max 50 epochs.
  - EMA target encoder momentum 0.99 where applicable (baseline, B; A has
    no temporal transformer to EMA).

Writes:
  v22/ckpts/{variant}_fd001_seed{S}_best.pt
  v22/pretrain_variants.json
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

sys.path.insert(0, str(V11))
sys.path.insert(0, str(V17))
sys.path.insert(0, str(V22))
sys.path.insert(0, str(FAM))

from data_utils import load_cmapss_subset, N_SENSORS  # noqa: E402
from models_variants import build_model, count_params  # noqa: E402

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Config (fixed across variants for fair comparison)
WINDOW     = 100
W_FUT      = 10
K_MAX      = 150
N_CUTS     = 30
MIN_PAST   = WINDOW   # enforce past >= 100 for all variants
D_MODEL    = 256
N_HEADS    = 4
N_LAYERS   = 2
D_FF       = 1024
BATCH_SIZE = 64
LR         = 3e-4
WD         = 0.01
EMA_MOM    = 0.99
LAMBDA_VAR = 0.04


class FixedWindowPretrainDataset(Dataset):
    """Fixed past length = WINDOW (=100).  LogUniform k, target window w.

    Samples: (past_last_W, future_k_to_k+w, k).  All past tensors are
    exactly shape (WINDOW, S).
    """

    def __init__(self, engines, n_cuts=N_CUTS, window=WINDOW,
                 K_max=K_MAX, w=W_FUT, seed=42):
        self.w = w
        self.window = window
        rng = np.random.RandomState(seed)
        self.samples = []
        for eid, arr in engines.items():
            T = len(arr)
            # need t such that t >= window (past is arr[t-window:t]) AND
            # t + k + w <= T  =>  t_max = T - 1 - w (with k=1)
            t_lo = window
            t_hi = T - 1 - w
            if t_hi < t_lo:
                continue
            for _ in range(n_cuts):
                t = int(rng.randint(t_lo, t_hi + 1))
                k_hi = min(K_max, T - t - w)
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
    x_past = torch.stack(list(pasts), dim=0)   # (B, WINDOW, S)
    x_fut  = torch.stack(list(futures), dim=0) # (B, w, S)
    past_mask = torch.zeros(x_past.shape[:2], dtype=torch.bool)
    fut_mask  = torch.zeros(x_fut.shape[:2],  dtype=torch.bool)
    k_t = torch.tensor(ks, dtype=torch.long)
    return x_past, past_mask, x_fut, fut_mask, k_t


def l1_loss_with_var(pred: torch.Tensor, target: torch.Tensor,
                     lambda_var: float = LAMBDA_VAR):
    pred_n = F.normalize(pred, dim=-1)
    targ_n = F.normalize(target.detach(), dim=-1)
    l_pred = F.l1_loss(pred_n, targ_n)
    pred_std = pred.std(dim=0)
    l_var = F.relu(1.0 - pred_std).mean()
    return l_pred + lambda_var * l_var, l_pred, l_var


def pretrain_one(variant: str, data: dict, seed: int, max_epochs: int = 50,
                 patience: int = 5, verbose: bool = True) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)

    model = build_model(variant, n_sensors=N_SENSORS, window=WINDOW,
                        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                        d_ff=D_FF, predictor_hidden=D_FF,
                        ema_momentum=EMA_MOM).to(DEVICE)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs)

    best = {'loss': float('inf'), 'sd': None, 'ep': 0}
    no_impr = 0
    losses = []
    t0 = time.time()

    for ep in range(1, max_epochs + 1):
        ds = FixedWindowPretrainDataset(
            data['train_engines'], n_cuts=N_CUTS, seed=seed * 1000 + ep)
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
            loss, _, _ = l1_loss_with_var(pred, targ, LAMBDA_VAR)
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
        if verbose and (ep % 5 == 0 or ep == 1):
            print(f"    ep {ep:3d} | L={ep_loss:.4f} best={best['loss']:.4f} "
                  f"no_impr={no_impr} ({(time.time()-t0)/60:.1f}m)",
                  flush=True)
        if no_impr >= patience:
            if verbose:
                print(f"    Early stop at ep {ep} (best={best['loss']:.4f})",
                      flush=True)
            break

    if best['sd'] is not None:
        model.load_state_dict(best['sd'])
    return {'variant': variant, 'seed': seed, 'best_loss': best['loss'],
            'best_ep': best['ep'], 'final_ep': ep, 'losses': losses,
            'runtime_min': (time.time() - t0) / 60, 'model': model,
            'params': count_params(model)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='+',
                    default=['baseline', 'variantA', 'variantB'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    ap.add_argument('--max-epochs', type=int, default=50)
    ap.add_argument('--patience', type=int, default=5)
    args = ap.parse_args()

    (V22 / 'ckpts').mkdir(exist_ok=True, parents=True)
    log = {}
    t_all = time.time()
    data = load_cmapss_subset('FD001')
    print(f'Loaded FD001: {len(data["train_engines"])} engines', flush=True)

    for variant in args.variants:
        print(f'\n=== {variant} pretrain ===', flush=True)
        log[variant] = []
        for seed in args.seeds:
            print(f'  -- seed={seed} --', flush=True)
            res = pretrain_one(variant, data, seed,
                               max_epochs=args.max_epochs,
                               patience=args.patience)
            ckpt_path = V22 / 'ckpts' / f'{variant}_fd001_seed{seed}_best.pt'
            torch.save(res['model'].state_dict(), ckpt_path)
            print(f'    -> {ckpt_path}  params={res["params"]:,}  '
                  f'best_L={res["best_loss"]:.4f}@ep{res["best_ep"]}  '
                  f'({res["runtime_min"]:.1f}m)', flush=True)
            log[variant].append({
                'seed': seed, 'ckpt': str(ckpt_path),
                'best_loss': res['best_loss'], 'best_ep': res['best_ep'],
                'final_ep': res['final_ep'], 'losses': res['losses'],
                'params': res['params'], 'runtime_min': res['runtime_min'],
            })
            del res
            torch.cuda.empty_cache()
        # incremental save
        with open(V22 / 'pretrain_variants.json', 'w') as f:
            json.dump({'log': log, 'runtime_min': (time.time() - t_all) / 60,
                       'config': vars(args)}, f, indent=2, default=float)

    print(f'\nDONE in {(time.time()-t_all)/60:.1f}m')


if __name__ == '__main__':
    main()
