"""V22 extra: pretrain variant A on SMAP + pred-FT with entity splits.

Phase 7 of the SESSION_PROMPT is conditional on having a winner on FD001.
We didn't get one, but SMAP's 25 channels (distinct telemetry + command
semantics) are where cross-channel attention is most plausibly useful,
so we still test variant A as a focused follow-up.

Protocol
--------
Pretrain:  same fixed-past-window=100 / LogUniform k ∈ [1, 150] / L1 loss
           as the FD001 variant pretraining, applied to SMAP's train
           stream (135K timesteps, 25 channels).  3 seeds.
Pred-FT:   intra-entity chronological split (same as anomaly_pred_ft_runner),
           frozen encoder, BCE on per-horizon logits, 3 seeds.

Writes ckpts, surfaces, and phase7_variantA_smap.json.
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
from torch.utils.data import Dataset, DataLoader, ConcatDataset

ROOT = Path('/home/sagemaker-user/IndustrialJEPA')
FAM = ROOT / 'fam-jepa'
V11 = FAM / 'experiments' / 'v11'
V22 = FAM / 'experiments' / 'v22'

sys.path.insert(0, str(FAM))
sys.path.insert(0, str(V11))
sys.path.insert(0, str(FAM / 'experiments' / 'v17'))
sys.path.insert(0, str(FAM / 'experiments' / 'v21'))
sys.path.insert(0, str(V22))

from models_variants import build_model, count_params  # noqa: E402
from pred_ft_utils import (  # noqa: E402
    AnomalyWindowDataset, collate_anomaly_window,
    EventHead, train_bce, evaluate_surface, estimate_pos_weight,
    save_surface, HORIZONS_STEPS,
)
from data.smap_msl import load_smap, split_smap_entities, WINDOW_SIZE  # noqa: E402
from evaluation.surface_metrics import (  # noqa: E402
    evaluate_probability_surface, auprc_per_horizon,
    monotonicity_violation_rate,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Pretrain config — matches FD001 variant pretraining
WINDOW     = 100
W_FUT      = 10
K_MAX      = 150
D_MODEL    = 256
N_HEADS    = 4
N_LAYERS   = 2
D_FF       = 1024
BATCH_SIZE = 128
LR         = 3e-4
WD         = 0.01
EMA_MOM    = 0.99
LAMBDA_VAR = 0.04
N_SAMPLES_PER_EPOCH = 20000  # 20K random-cut samples per epoch

# Pred-FT config
TRAIN_CFG = dict(lr=1e-3, wd=1e-2, n_epochs=40, patience=8)


# ---------------------------------------------------------------------------
# Pretrain dataset: fixed past=100 sampled from SMAP train stream
# ---------------------------------------------------------------------------

class FixedWindowArrayPretrainDataset(Dataset):
    """Random (past, future, k) triplets from a single (T, C) numpy array.

    Past is always exactly `window` timesteps.
    """

    def __init__(self, arr: np.ndarray, n_samples: int, window: int = WINDOW,
                 w: int = W_FUT, K_max: int = K_MAX, seed: int = 42):
        T = len(arr)
        t_lo = window
        t_hi = T - 1 - w
        assert t_hi >= t_lo, f"array too short: T={T}"
        rng = np.random.RandomState(seed)
        self.arr = arr
        self.w = w
        self.window = window
        self.samples = []
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


def l1_loss_with_var(pred, target, lambda_var=LAMBDA_VAR):
    pred_n = F.normalize(pred, dim=-1)
    targ_n = F.normalize(target.detach(), dim=-1)
    l_pred = F.l1_loss(pred_n, targ_n)
    pred_std = pred.std(dim=0)
    l_var = F.relu(1.0 - pred_std).mean()
    return l_pred + lambda_var * l_var


def pretrain_variant(variant: str, arr: np.ndarray, n_sensors: int,
                     seed: int, max_epochs: int = 50, patience: int = 5,
                     ) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_model(variant, n_sensors=n_sensors, window=WINDOW,
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
        ds = FixedWindowArrayPretrainDataset(
            arr, n_samples=N_SAMPLES_PER_EPOCH, seed=seed * 1000 + ep)
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
            best = {'loss': ep_loss,
                    'sd': copy.deepcopy(model.state_dict()), 'ep': ep}
            no_impr = 0
        else:
            no_impr += 1
        if ep % 3 == 0 or ep == 1:
            print(f"    ep {ep:3d}  L={ep_loss:.4f}  best={best['loss']:.4f}  "
                  f"no_impr={no_impr}  ({(time.time()-t0)/60:.1f}m)",
                  flush=True)
        if no_impr >= patience:
            print(f"    Early stop at ep {ep} (best={best['loss']:.4f})",
                  flush=True)
            break
    if best['sd'] is not None:
        model.load_state_dict(best['sd'])
    return {'model': model, 'best_loss': best['loss'], 'best_ep': best['ep'],
            'final_ep': ep, 'losses': losses,
            'runtime_min': (time.time() - t0) / 60,
            'params': count_params(model)}


def _build_ds_from_entities(entities, window, stride, max_future):
    return ConcatDataset([
        AnomalyWindowDataset(e['test'], e['labels'],
                             window=window, stride=stride,
                             max_future=max_future) for e in entities])


def pred_ft_one(variant: str, seed: int, model) -> dict:
    """Freeze encoder, pred-FT with BCE on SMAP entity splits."""
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)

    sp = split_smap_entities()
    max_fut = max(HORIZONS_STEPS) + 1
    tr_ds = _build_ds_from_entities(sp['ft_train'], WINDOW, 4, max_fut)
    va_ds = _build_ds_from_entities(sp['ft_val'],   WINDOW, 4, max_fut)
    te_ds = _build_ds_from_entities(sp['ft_test'],  WINDOW, 4, max_fut)
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
    (V22 / 'surfaces').mkdir(exist_ok=True)
    surf_path = V22 / 'surfaces' / f'smap_{variant}_pred_ft_seed{seed}.npz'
    save_surface(surf_path, p, y, HORIZONS_STEPS, surf['t_index'],
                 metadata={'dataset': 'SMAP', 'variant': variant,
                           'seed': seed, 'mode': 'pred_ft'})
    prim = evaluate_probability_surface(p, y)
    per_h = auprc_per_horizon(p, y, horizon_labels=HORIZONS_STEPS)
    mono = monotonicity_violation_rate(p)
    return {'variant': variant, 'seed': seed, 'primary': prim,
            'per_horizon': per_h, 'monotonicity': mono,
            'train': {'best_val': train_out['best_val'],
                      'final_epoch': train_out['final_epoch']},
            'pos_weight': float(pw),
            'n_train_windows': len(tr_ds),
            'n_test_windows': len(te_ds),
            'surface_file': str(surf_path),
            'runtime_s': time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='+', default=['variantA', 'variantB'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    ap.add_argument('--max-epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=5)
    args = ap.parse_args()

    (V22 / 'ckpts').mkdir(exist_ok=True)
    (V22 / 'surfaces').mkdir(exist_ok=True)
    t_all = time.time()
    data = load_smap()
    print(f'SMAP train: {data["train"].shape}', flush=True)

    all_out = {}
    for variant in args.variants:
        print(f'\n=== {variant} ===', flush=True)
        per_seed = []
        for seed in args.seeds:
            # Pretrain
            print(f'  -- pretrain seed={seed} --', flush=True)
            pr = pretrain_variant(variant, data['train'], data['n_channels'],
                                  seed, max_epochs=args.max_epochs,
                                  patience=args.patience)
            ckpt = V22 / 'ckpts' / f'{variant}_smap_seed{seed}_best.pt'
            torch.save(pr['model'].state_dict(), ckpt)
            print(f'    pretrain: params={pr["params"]:,} best_L={pr["best_loss"]:.4f}'
                  f'@ep{pr["best_ep"]} ({pr["runtime_min"]:.1f}m)', flush=True)
            # Pred-FT
            print(f'  -- pred-FT seed={seed} --', flush=True)
            fr = pred_ft_one(variant, seed, pr['model'])
            fr['pretrain'] = {'best_loss': pr['best_loss'],
                              'best_ep': pr['best_ep'],
                              'final_ep': pr['final_ep'],
                              'runtime_min': pr['runtime_min'],
                              'params': pr['params']}
            per_seed.append(fr)
            print(f"    AUPRC={fr['primary']['auprc']:.3f} "
                  f"AUROC={fr['primary']['auroc']:.3f} "
                  f"F1={fr['primary']['f1_best']:.3f} "
                  f"mono={fr['monotonicity']['violation_rate']:.3f} "
                  f"pw={fr['pos_weight']:.1f} "
                  f"ep={fr['train']['final_epoch']} "
                  f"({fr['runtime_s']:.0f}s)", flush=True)
            del pr
            torch.cuda.empty_cache()
        all_out[variant] = {'per_seed': per_seed,
                            'agg': _aggregate(per_seed)}
        with open(V22 / 'phase7_variantA_smap.json', 'w') as f:
            json.dump({'variants': all_out, 'seeds': args.seeds,
                       'runtime_min': (time.time() - t_all) / 60},
                      f, indent=2, default=float)

    print(f'\nDONE in {(time.time()-t_all)/60:.1f}m')
    print('\nSMAP comparison (AUPRC / AUROC / F1-best):')
    print(f"{'Variant':10s} | {'AUPRC':>12s} | {'AUROC':>12s} | {'F1':>12s}")
    print('-' * 60)
    # Prepend baseline from phase1
    try:
        b = json.load(open(V22 / 'phase1_anomaly_pred_ft.json'))
        ba = b['datasets']['SMAP']['agg']
        print(f"{'baseline':10s} | "
              f"{ba['auprc_mean']:.3f}±{ba['auprc_std']:.3f} | "
              f"{ba['auroc_mean']:.3f}±{ba['auroc_std']:.3f} | "
              f"{ba['f1_best_mean']:.3f}±{ba['f1_best_std']:.3f}")
    except Exception:
        pass
    for v, obj in all_out.items():
        a = obj['agg']
        print(f"{v:10s} | "
              f"{a['auprc_mean']:.3f}±{a['auprc_std']:.3f} | "
              f"{a['auroc_mean']:.3f}±{a['auroc_std']:.3f} | "
              f"{a['f1_best_mean']:.3f}±{a['f1_best_std']:.3f}")


def _aggregate(per_seed):
    out = {'n_seeds': len(per_seed), 'per_seed': per_seed}
    metrics = [
        ('auprc', lambda r: r['primary']['auprc']),
        ('auroc', lambda r: r['primary']['auroc']),
        ('f1_best', lambda r: r['primary']['f1_best']),
        ('precision_best', lambda r: r['primary']['precision_best']),
        ('recall_best', lambda r: r['primary']['recall_best']),
        ('mono_violation', lambda r: r['monotonicity']['violation_rate']),
    ]
    for name, fn in metrics:
        vals = np.array([fn(r) for r in per_seed], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            out[f'{name}_mean'] = float('nan')
            out[f'{name}_std'] = float('nan')
            continue
        out[f'{name}_mean'] = float(vals.mean())
        out[f'{name}_std'] = (float(vals.std(ddof=1)) if len(vals) > 1
                              else 0.0)
    return out


if __name__ == '__main__':
    main()
