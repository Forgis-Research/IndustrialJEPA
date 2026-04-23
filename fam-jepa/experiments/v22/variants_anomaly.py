"""V22 extra: generalize phase-7 cross-channel variants across anomaly datasets.

Pretrain variants A/B on the concatenated train stream of each dataset,
then run pred-FT with the matching split strategy:
  entity-split:  SMAP / MSL / SMD
  stream-split:  PSM / MBA  (single-stream, chronological with gap)

Compares to the baseline pred-FT numbers from phase 1.
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
from data import (  # noqa: E402
    load_smap, load_msl, load_psm, load_mba, load_smd,
    split_smap_entities, split_msl_entities, split_smd_entities,
)
from data.smd import load_smd_entities  # noqa: E402
from data.smap_msl import _intra_entity_split, WINDOW_SIZE  # noqa: E402
from evaluation.surface_metrics import (  # noqa: E402
    evaluate_probability_surface, auprc_per_horizon,
    monotonicity_violation_rate,
)
from variantA_smap import (  # noqa: E402
    FixedWindowArrayPretrainDataset, collate_fixed, l1_loss_with_var,
    WINDOW, W_FUT, K_MAX, D_MODEL, N_HEADS, N_LAYERS, D_FF,
    BATCH_SIZE, LR, WD, EMA_MOM, LAMBDA_VAR, N_SAMPLES_PER_EPOCH,
    TRAIN_CFG,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# Registry: (loader for pretrain stream, type, n_sensors used, split fn)
DATASETS = {
    'SMAP': {'type': 'entity', 'n_sensors': 25, 'loader': load_smap,
             'split_fn': split_smap_entities},
    'MSL':  {'type': 'entity', 'n_sensors': 55, 'loader': load_msl,
             'split_fn': split_msl_entities},
    'SMD':  {'type': 'entity', 'n_sensors': 38, 'loader': lambda: load_smd(drop_constant=False),
             'split_fn': lambda: _intra_entity_split(
                 load_smd_entities(drop_constant=False),
                 ratios=(0.6, 0.1, 0.3), window_size=WINDOW_SIZE)},
    'PSM':  {'type': 'stream', 'n_sensors': 25, 'loader': load_psm},
    'MBA':  {'type': 'stream', 'n_sensors': 2,  'loader': load_mba},
}


def pretrain_variant(variant: str, arr: np.ndarray, n_sensors: int,
                     seed: int, max_epochs: int, patience: int,
                     batch_size: int = BATCH_SIZE,
                     n_samples: int = N_SAMPLES_PER_EPOCH,
                     verbose: bool = True) -> dict:
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
            arr, n_samples=n_samples, seed=seed * 1000 + ep)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
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
        if verbose and (ep % 3 == 0 or ep == 1):
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


def _build_ds_from_stream(split_dict, window, stride, max_future):
    return AnomalyWindowDataset(
        split_dict['test'], split_dict['labels'],
        window=window, stride=stride, max_future=max_future)


def _stream_split(data, ratios=(0.6, 0.1, 0.3), gap=WINDOW_SIZE):
    T = len(data['test'])
    t1 = int(ratios[0] * T); t2 = int((ratios[0] + ratios[1]) * T)
    return (
        {'test': data['test'][:t1],             'labels': data['labels'][:t1]},
        {'test': data['test'][t1 + gap: t2],    'labels': data['labels'][t1 + gap: t2]},
        {'test': data['test'][t2 + gap:],       'labels': data['labels'][t2 + gap:]},
    )


def pred_ft_one(dataset: str, variant: str, seed: int, model) -> dict:
    """Freeze encoder, pred-FT with BCE, save surface + metrics."""
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    reg = DATASETS[dataset]
    max_fut = max(HORIZONS_STEPS) + 1
    if reg['type'] == 'entity':
        sp = reg['split_fn']()
        tr_ds = _build_ds_from_entities(sp['ft_train'], WINDOW, 4, max_fut)
        va_ds = _build_ds_from_entities(sp['ft_val'],   WINDOW, 4, max_fut)
        te_ds = _build_ds_from_entities(sp['ft_test'],  WINDOW, 4, max_fut)
    else:
        data = reg['loader']()
        tr_sp, va_sp, te_sp = _stream_split(data)
        tr_ds = _build_ds_from_stream(tr_sp, WINDOW, 4, max_fut)
        va_ds = _build_ds_from_stream(va_sp, WINDOW, 4, max_fut)
        te_ds = _build_ds_from_stream(te_sp, WINDOW, 4, max_fut)
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
    surf_path = V22 / 'surfaces' / f'{dataset.lower()}_{variant}_pred_ft_seed{seed}.npz'
    save_surface(surf_path, p, y, HORIZONS_STEPS, surf['t_index'],
                 metadata={'dataset': dataset, 'variant': variant,
                           'seed': seed, 'mode': 'pred_ft'})
    prim = evaluate_probability_surface(p, y)
    per_h = auprc_per_horizon(p, y, horizon_labels=HORIZONS_STEPS)
    mono = monotonicity_violation_rate(p)
    return {'dataset': dataset, 'variant': variant, 'seed': seed,
            'primary': prim, 'per_horizon': per_h,
            'monotonicity': mono,
            'train': {'best_val': train_out['best_val'],
                      'final_epoch': train_out['final_epoch']},
            'pos_weight': float(pw),
            'n_train_windows': len(tr_ds),
            'n_test_windows': len(te_ds),
            'surface_file': str(surf_path),
            'runtime_s': time.time() - t0}


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
            out[f'{name}_mean'] = float('nan'); out[f'{name}_std'] = float('nan')
            continue
        out[f'{name}_mean'] = float(vals.mean())
        out[f'{name}_std'] = (float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['MSL', 'SMD', 'PSM', 'MBA'])
    ap.add_argument('--variants', nargs='+', default=['variantA', 'variantB'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    ap.add_argument('--max-epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--out', default=str(V22 / 'phase7b_variants_anomaly.json'))
    args = ap.parse_args()

    (V22 / 'ckpts').mkdir(exist_ok=True)
    (V22 / 'surfaces').mkdir(exist_ok=True)
    t_all = time.time()
    all_out = {}

    for ds in args.datasets:
        reg = DATASETS[ds]
        data = reg['loader']()
        ns = data['n_channels']
        print(f'\n=== {ds} (n_channels={ns}, type={reg["type"]}) ===', flush=True)
        all_out[ds] = {}
        for variant in args.variants:
            print(f'\n  --- {variant} on {ds} ---', flush=True)
            per_seed = []
            for seed in args.seeds:
                print(f'    pretrain seed={seed}', flush=True)
                pr = pretrain_variant(variant, data['train'], ns, seed,
                                      max_epochs=args.max_epochs,
                                      patience=args.patience)
                ckpt = V22 / 'ckpts' / f'{variant}_{ds.lower()}_seed{seed}_best.pt'
                torch.save(pr['model'].state_dict(), ckpt)
                print(f'      params={pr["params"]:,} '
                      f'best_L={pr["best_loss"]:.4f}@ep{pr["best_ep"]} '
                      f'({pr["runtime_min"]:.1f}m)', flush=True)
                fr = pred_ft_one(ds, variant, seed, pr['model'])
                fr['pretrain'] = {'best_loss': pr['best_loss'],
                                  'best_ep': pr['best_ep'],
                                  'final_ep': pr['final_ep'],
                                  'runtime_min': pr['runtime_min'],
                                  'params': pr['params']}
                per_seed.append(fr)
                print(f"    [{ds} {variant} s{seed}] "
                      f"AUPRC={fr['primary']['auprc']:.3f} "
                      f"AUROC={fr['primary']['auroc']:.3f} "
                      f"F1={fr['primary']['f1_best']:.3f} "
                      f"mono={fr['monotonicity']['violation_rate']:.3f} "
                      f"pw={fr['pos_weight']:.1f} "
                      f"FTep={fr['train']['final_epoch']} "
                      f"({fr['runtime_s']:.0f}s)", flush=True)
                del pr
                torch.cuda.empty_cache()
            all_out[ds][variant] = {'per_seed': per_seed,
                                    'agg': _aggregate(per_seed)}
            with open(args.out, 'w') as f:
                json.dump({'results': all_out, 'seeds': args.seeds,
                           'runtime_min': (time.time() - t_all) / 60},
                          f, indent=2, default=float)

    print(f'\nDONE in {(time.time()-t_all)/60:.1f}m')
    print('\n' + '=' * 80)
    print('Phase 7b summary: cross-channel variants on anomaly datasets')
    print('=' * 80)
    print(f"{'Dataset':8s} | {'Encoder':10s} | {'AUPRC':>13s} | {'AUROC':>13s} | "
          f"{'F1':>13s}")
    print('-' * 80)
    for ds, vmap in all_out.items():
        for v, obj in vmap.items():
            a = obj['agg']
            print(f"{ds:8s} | {v:10s} | "
                  f"{a['auprc_mean']:.3f}±{a['auprc_std']:.3f} | "
                  f"{a['auroc_mean']:.3f}±{a['auroc_std']:.3f} | "
                  f"{a['f1_best_mean']:.3f}±{a['f1_best_std']:.3f}")


if __name__ == '__main__':
    main()
