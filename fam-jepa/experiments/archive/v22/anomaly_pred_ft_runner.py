"""V22 Anomaly Pred-FT runner.

Finetunes the frozen-encoder predictor + EventHead with positive-weighted
BCE on labeled splits.  Fixes the v21 chronological-split-on-concatenated-
test bug that made AUROC 0.38 on SMAP.

Split strategy per dataset:
  SMAP, MSL, SMD : intra-entity chronological split.  Every entity appears
                   in ft_train / ft_val / ft_test, split 60/10/30 within
                   each entity's test stream, with a gap of WINDOW_SIZE
                   timesteps at each boundary to prevent temporal leakage.
  PSM, MBA       : single stream; chronological split of test stream with
                   the same 60/10/30 layout and gap.

Pipeline per (dataset, seed):
  1. Load pretrained JEPA ckpt.
  2. Build ft_train / ft_val / ft_test torch Datasets (ConcatDataset for
     per-entity streams).
  3. pos_weight = estimate_pos_weight(ft_train loader).
  4. train_bce in 'pred_ft' mode (freeze encoder, train predictor + head).
  5. evaluate_surface on ft_test -> p_surface (N, K), y_surface (N, K),
     t_index (N,).
  6. Save .npz surface.  Compute AUPRC (primary), AUROC, non-PA F1, etc.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path('/home/sagemaker-user/IndustrialJEPA')
FAM = ROOT / 'fam-jepa'
V11 = FAM / 'experiments' / 'v11'
V21 = FAM / 'experiments' / 'v21'
V22 = FAM / 'experiments' / 'v22'
CKPT_OLD = ROOT / 'mechanical-jepa' / 'experiments'

sys.path.insert(0, str(V11))
sys.path.insert(0, str(V21))
sys.path.insert(0, str(V22))
sys.path.insert(0, str(FAM))

from models import TrajectoryJEPA  # noqa: E402
from pred_ft_utils import (  # noqa: E402
    AnomalyWindowDataset, collate_anomaly_window,
    EventHead, estimate_pos_weight, train_bce, evaluate_surface,
    save_surface, HORIZONS_STEPS,
)
from surface_to_legacy import anomaly_legacy_metrics  # noqa: E402

from data.smap_msl import (  # noqa: E402
    split_smap_entities, split_msl_entities, load_smap_entities,
    load_msl_entities, _intra_entity_split, WINDOW_SIZE,
)
from data.smd import load_smd_entities  # noqa: E402
from data.psm import load_psm  # noqa: E402
from data.mba import load_mba  # noqa: E402

from evaluation.surface_metrics import (  # noqa: E402
    evaluate_probability_surface, auprc_per_horizon,
    monotonicity_violation_rate,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BASE_ARCH = dict(patch_length=1, d_model=256, n_heads=4, n_layers=2,
                 d_ff=1024, dropout=0.1, ema_momentum=0.99,
                 predictor_hidden=1024)

# Pred-FT training config
TRAIN_CFG = dict(lr=1e-3, wd=1e-2, n_epochs=40, patience=8)

# Window + stride
WINDOW = WINDOW_SIZE   # 100
STRIDE_TR = 4          # densish train sampling (many windows even for short entities)
STRIDE_EV = 4          # for val / test


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS: Dict[str, dict] = {
    'SMAP': {
        'type': 'entity', 'n_sensors': 25,
        'ckpts': {
            42:  CKPT_OLD / 'v17' / 'ckpts' / 'v17_smap_seed42.pt',
            123: CKPT_OLD / 'v18' / 'ckpts' / 'v18_smap_seed123.pt',
            456: CKPT_OLD / 'v18' / 'ckpts' / 'v18_smap_seed456.pt',
        },
    },
    'MSL': {
        'type': 'entity', 'n_sensors': 55,
        'ckpts': {
            42:  CKPT_OLD / 'v18' / 'ckpts' / 'v18_msl_seed42_ep150.pt',
            123: CKPT_OLD / 'v18' / 'ckpts' / 'v18_msl_seed123_ep150.pt',
            456: CKPT_OLD / 'v18' / 'ckpts' / 'v18_msl_seed456_ep150.pt',
        },
    },
    'SMD': {
        'type': 'entity', 'n_sensors': 38,
        'ckpts': {
            42:  CKPT_OLD / 'v19' / 'ckpts' / 'v19_smd_seed42.pt',
            123: CKPT_OLD / 'v19' / 'ckpts' / 'v19_smd_seed123.pt',
            456: CKPT_OLD / 'v19' / 'ckpts' / 'v19_smd_seed456.pt',
        },
    },
    'PSM': {
        'type': 'stream', 'n_sensors': 25,
        'ckpts': {
            42:  CKPT_OLD / 'v19' / 'ckpts' / 'v19_psm_seed42_ep50.pt',
            123: CKPT_OLD / 'v19' / 'ckpts' / 'v19_psm_seed123_ep50.pt',
            456: CKPT_OLD / 'v19' / 'ckpts' / 'v19_psm_seed456_ep50.pt',
        },
    },
    'MBA': {
        'type': 'stream', 'n_sensors': 2,
        'ckpts': {
            42:  CKPT_OLD / 'v19' / 'ckpts' / 'v19_mba_seed42_ep50.pt',
            123: CKPT_OLD / 'v19' / 'ckpts' / 'v19_mba_seed123_ep50.pt',
            456: CKPT_OLD / 'v19' / 'ckpts' / 'v19_mba_seed456_ep50.pt',
        },
    },
}


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def _load_entity_split(dataset: str) -> Dict[str, List[Dict]]:
    """Return dict with ft_train / ft_val / ft_test entity lists."""
    if dataset == 'SMAP':
        return split_smap_entities()
    if dataset == 'MSL':
        return split_msl_entities()
    if dataset == 'SMD':
        # Use drop_constant=False so channel count matches the v19 ckpt (38).
        ents = load_smd_entities(normalize=True, drop_constant=False)
        return _intra_entity_split(ents, ratios=(0.6, 0.1, 0.3),
                                   window_size=WINDOW_SIZE)
    raise ValueError(dataset)


def _stream_split(data: dict, ratios=(0.6, 0.1, 0.3), gap: int = WINDOW_SIZE
                  ) -> Tuple[Dict, Dict, Dict]:
    """Chronological 60/10/30 split of a single-stream test array with gap."""
    test = data['test']
    labels = data['labels']
    T = len(test)
    t1 = int(ratios[0] * T)
    t2 = int((ratios[0] + ratios[1]) * T)
    tr = {'test': test[:t1],            'labels': labels[:t1]}
    va = {'test': test[t1 + gap: t2],   'labels': labels[t1 + gap: t2]}
    te = {'test': test[t2 + gap:],      'labels': labels[t2 + gap:]}
    return tr, va, te


# ---------------------------------------------------------------------------
# Build ConcatDataset from entity streams
# ---------------------------------------------------------------------------

def _build_ds_from_entities(entities: List[Dict], window: int, stride: int,
                            max_future: int) -> ConcatDataset:
    parts = []
    for e in entities:
        parts.append(AnomalyWindowDataset(
            e['test'], e['labels'],
            window=window, stride=stride, max_future=max_future,
        ))
    return ConcatDataset(parts)


def _build_ds_from_stream(split: Dict, window: int, stride: int,
                          max_future: int) -> AnomalyWindowDataset:
    return AnomalyWindowDataset(
        split['test'], split['labels'],
        window=window, stride=stride, max_future=max_future,
    )


# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------

def _load_model(dataset: str, seed: int) -> TrajectoryJEPA:
    reg = DATASETS[dataset]
    ns = reg['n_sensors']
    arch = dict(BASE_ARCH, n_sensors=ns)
    ckpt = reg['ckpts'][seed]
    if not ckpt.exists():
        raise FileNotFoundError(f'Missing ckpt: {ckpt}')
    m = TrajectoryJEPA(**arch).to(DEVICE)
    sd = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    if isinstance(sd, dict) and 'model' in sd and 'context_encoder.proj.proj.weight' not in sd:
        sd = sd['model']
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f'  load {dataset} s{seed}: missing={len(missing)} '
              f'unexpected={len(unexpected)}', flush=True)
    return m


# ---------------------------------------------------------------------------
# Per-(dataset, seed) run
# ---------------------------------------------------------------------------

def run_single(dataset: str, seed: int, verbose: bool = True,
               pw_cap: float = 1000.0) -> dict:
    """pw_cap: clamp pos_weight to at most this value.  The v22 pos_weight
    ablation (pos_weight_ablation.py) shows that the auto-estimated
    pos_weight ~ N_neg/N_pos (≈40 for SMAP) over-biases the predictor to
    "always anomaly" and drives AUROC below 0.5 on datasets with large
    train→test anomaly-rate shift (SMAP, PSM).  Capping at pw=5 or 10
    recovers AUROC above chance and improves AUPRC.
    """
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)

    reg = DATASETS[dataset]
    max_future = max(HORIZONS_STEPS) + 1

    # Build train/val/test datasets
    if reg['type'] == 'entity':
        sp = _load_entity_split(dataset)
        tr_ds = _build_ds_from_entities(sp['ft_train'], WINDOW, STRIDE_TR, max_future)
        va_ds = _build_ds_from_entities(sp['ft_val'],   WINDOW, STRIDE_EV, max_future)
        te_ds = _build_ds_from_entities(sp['ft_test'],  WINDOW, STRIDE_EV, max_future)
        n_entities = len(sp['ft_train'])
    else:
        # stream: load and chronologically split
        if dataset == 'PSM':
            data = load_psm(drop_constant=True)
        elif dataset == 'MBA':
            data = load_mba()
        else:
            raise ValueError(dataset)
        assert data['n_channels'] == reg['n_sensors'], (
            f'{dataset} n_channels {data["n_channels"]} != reg {reg["n_sensors"]}')
        tr_split, va_split, te_split = _stream_split(data)
        tr_ds = _build_ds_from_stream(tr_split, WINDOW, STRIDE_TR, max_future)
        va_ds = _build_ds_from_stream(va_split, WINDOW, STRIDE_EV, max_future)
        te_ds = _build_ds_from_stream(te_split, WINDOW, STRIDE_EV, max_future)
        n_entities = 1

    tr = DataLoader(tr_ds, batch_size=256, shuffle=True,
                    collate_fn=collate_anomaly_window, num_workers=0)
    va = DataLoader(va_ds, batch_size=256, shuffle=False,
                    collate_fn=collate_anomaly_window, num_workers=0)
    te = DataLoader(te_ds, batch_size=256, shuffle=False,
                    collate_fn=collate_anomaly_window, num_workers=0)

    # pos_weight from train, optionally capped
    pw_raw = estimate_pos_weight(tr, HORIZONS_STEPS)
    pw = min(pw_raw, pw_cap)

    # Model + head
    model = _load_model(dataset, seed)
    head = EventHead(BASE_ARCH['d_model']).to(DEVICE)

    # Train (pred_ft: freeze encoder)
    train_out = train_bce(model, head, tr, va, mode='pred_ft',
                          pos_weight=pw, horizons_eval=HORIZONS_STEPS,
                          device=DEVICE, **TRAIN_CFG)

    # Evaluate on ft_test
    surf = evaluate_surface(model, head, te, mode='pred_ft',
                            horizons=HORIZONS_STEPS, device=DEVICE)
    p, y, tidx = surf['p_surface'], surf['y_surface'], surf['t_index']

    # Save surface
    (V22 / 'surfaces').mkdir(exist_ok=True)
    key = f'{dataset.lower()}_pred_ft_seed{seed}'
    surf_path = V22 / 'surfaces' / f'{key}.npz'
    save_surface(surf_path, p, y, HORIZONS_STEPS, tidx,
                 metadata={'dataset': dataset, 'seed': seed, 'mode': 'pred_ft',
                           'pos_weight': float(pw), 'n_entities': n_entities})

    # Metrics
    prim = evaluate_probability_surface(p, y)
    per_h = auprc_per_horizon(p, y, horizon_labels=HORIZONS_STEPS)
    mono = monotonicity_violation_rate(p)

    # Legacy non-PA F1 from surface (at Δt=100, the MTS-JEPA-equivalent
    # question "any anomaly in next 100 steps"). We report non-PA F1 —
    # PA-F1 is a known-inflated metric and not in the v22 paper table.
    # For stream data we have a global test array; for entity data the
    # surface timestamps are entity-local and do not collapse to a shared
    # timeline, so we report the pooled-surface F1 directly from (p, y).
    legacy_f1 = float(prim['f1_best'])
    legacy_p = float(prim['precision_best'])
    legacy_r = float(prim['recall_best'])

    dt = time.time() - t0
    if verbose:
        print(f'  [{dataset} s{seed}] AUPRC={prim["auprc"]:.3f} '
              f'AUROC={prim["auroc"]:.3f} '
              f'F1={legacy_f1:.3f} (P={legacy_p:.3f} R={legacy_r:.3f}) '
              f'mono={mono["violation_rate"]:.3f} '
              f'pw={pw:.1f} ep={train_out["final_epoch"]} '
              f'({dt:.0f}s) n_tr={len(tr_ds)} n_te={len(te_ds)}',
              flush=True)

    return {
        'dataset': dataset, 'seed': seed, 'mode': 'pred_ft',
        'primary': prim, 'per_horizon': per_h,
        'monotonicity': mono,
        'legacy': {'non_pa_f1': legacy_f1,
                   'non_pa_precision': legacy_p,
                   'non_pa_recall': legacy_r},
        'train': {'best_val': train_out['best_val'],
                  'final_epoch': train_out['final_epoch']},
        'pos_weight': float(pw),
        'pos_weight_raw': float(pw_raw),
        'pos_weight_cap': float(pw_cap),
        'n_train_windows': len(tr_ds),
        'n_val_windows': len(va_ds),
        'n_test_windows': len(te_ds),
        'n_entities': n_entities,
        'surface_file': str(surf_path),
        'runtime_s': dt,
    }


def aggregate(per_seed):
    from scipy.stats import t as t_dist
    out = {'n_seeds': len(per_seed), 'per_seed': per_seed}
    metrics = [
        ('auprc', lambda r: r['primary']['auprc']),
        ('auroc', lambda r: r['primary']['auroc']),
        ('f1_best', lambda r: r['primary']['f1_best']),
        ('precision_best', lambda r: r['primary']['precision_best']),
        ('recall_best', lambda r: r['primary']['recall_best']),
        ('non_pa_f1', lambda r: r['legacy']['non_pa_f1']),
        ('mono_violation', lambda r: r['monotonicity']['violation_rate']),
    ]
    for name, fn in metrics:
        vals = np.array([fn(r) for r in per_seed], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            out[f'{name}_mean'] = float('nan'); out[f'{name}_std'] = float('nan')
            continue
        m = float(vals.mean())
        s = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        if len(vals) > 1:
            tc = float(t_dist.ppf(0.975, df=len(vals) - 1))
            margin = tc * s / np.sqrt(len(vals))
        else:
            margin = float('nan')
        out[f'{name}_mean'] = m
        out[f'{name}_std'] = s
        out[f'{name}_ci95_lo'] = m - margin
        out[f'{name}_ci95_hi'] = m + margin
    return out


def run_dataset_all_seeds(dataset: str, seeds=(42, 123, 456),
                          pw_cap: float = 1000.0):
    per_seed = []
    for s in seeds:
        try:
            r = run_single(dataset, s, verbose=True, pw_cap=pw_cap)
            per_seed.append(r)
        except Exception as e:
            print(f'  ERROR {dataset} s{s}: {e}', flush=True)
            import traceback; traceback.print_exc()
    return {dataset: {'per_seed': per_seed, 'agg': aggregate(per_seed)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=None)
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    ap.add_argument('--out', default=str(V22 / 'phase1_anomaly_pred_ft.json'))
    ap.add_argument('--pw-cap', type=float, default=1000.0,
                    help='clamp pos_weight to at most this value')
    args = ap.parse_args()

    datasets = args.datasets or ['SMAP', 'MSL', 'SMD', 'PSM', 'MBA']
    out_path = Path(args.out)
    V22.mkdir(exist_ok=True)
    (V22 / 'surfaces').mkdir(exist_ok=True)

    t0 = time.time()
    all_out = {}
    for ds in datasets:
        print(f'\n=== {ds} (pw_cap={args.pw_cap}) ===', flush=True)
        all_out.update(run_dataset_all_seeds(
            ds, seeds=tuple(args.seeds), pw_cap=args.pw_cap))
        with open(out_path, 'w') as f:
            json.dump({'datasets': all_out, 'seeds': args.seeds,
                       'pw_cap': args.pw_cap,
                       'runtime_min': (time.time() - t0) / 60},
                      f, indent=2, default=float)

    print(f'\nDONE in {(time.time()-t0)/60:.1f} min. Saved: {out_path}')

    # Summary
    print('\n' + '='*90)
    print('V22 PHASE 1+2 SUMMARY (pred-FT with entity/stream splits)')
    print('='*90)
    print(f"{'Dataset':8s} | {'AUPRC':>17s} | {'AUROC':>17s} | {'F1 (non-PA)':>17s} | {'mono_v':>7s}")
    print('-' * 90)
    for ds, obj in all_out.items():
        a = obj['agg']
        def f(k):
            return f"{a.get(f'{k}_mean', float('nan')):.3f}±{a.get(f'{k}_std', 0):.3f}"
        print(f"{ds:8s} | {f('auprc'):>17s} | {f('auroc'):>17s} | "
              f"{f('f1_best'):>17s} | {a.get('mono_violation_mean', 0):.3f}")


if __name__ == '__main__':
    main()
