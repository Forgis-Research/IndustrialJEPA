"""V32 Phase 3 (FAST): Recompute legacy metrics from stored surfaces.

Same goals as before but with vectorized PA-F1: precompute per-segment max
scores then sweep thresholds in vectorized form. ~100x speedup on SMD-scale
data (200k+ observations).

For each (dataset, label_frac, seed):
  - Load p_surface (N, K), y_surface (N, K).
  - Sweep horizon_for_score in {1, 3, 5, 10, 20, 50, 100, 150}.
  - At each horizon: AUROC + AUPRC on observation level; then sweep
    thresholds for best F1 (PA on the (N,)-timeline; non-PA).
  - Pick the horizon that maximizes pa_f1 / non_pa_f1 / auroc respectively.

For GECCO and BATADAL: produce diagnostics (score quantiles, label rates,
per-horizon AUPRC).

Output: results/legacy_metrics_full.json (incremental save per dataset).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
)

warnings.filterwarnings('ignore')

FAM = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V30_SURF = FAM / 'experiments/v30/surfaces'
V31_SURF = FAM / 'experiments/v31/surfaces'
V32_RES = FAM / 'experiments/v32/results'
V32_RES.mkdir(parents=True, exist_ok=True)

ANOMALY_DATASETS = [
    'SMAP', 'PSM', 'SMD', 'MBA', 'SKAB', 'ETTm1', 'GECCO', 'BATADAL', 'MSL',
]
HORIZON_SWEEP = [1, 3, 5, 10, 20, 50, 100, 150]
SEEDS = [42, 123, 456]
N_THRS = 200


def _surf_path(ds: str, lf: int, seed: int) -> Optional[Path]:
    if lf == 100:
        cands = [
            V30_SURF / f'{ds}_revin_discrete_hazard_td20_p3_s{seed}.npz',
            V30_SURF / f'{ds}_none_discrete_hazard_td20_p3_s{seed}.npz',
            V30_SURF / f'{ds}_revin_discrete_hazard_td20_p2_s{seed}.npz',
        ]
    elif lf == 10:
        cands = [
            V31_SURF / f'{ds}_revin_discrete_hazard_td20_lf10_p1lf10_s{seed}.npz',
            V31_SURF / f'{ds}_revin_discrete_hazard_lf10_p1lf10_s{seed}.npz',
            V30_SURF / f'{ds}_revin_discrete_hazard_td20_lf10_p3_lf10_s{seed}.npz',
            V30_SURF / f'{ds}_none_discrete_hazard_td20_lf10_p3_lf10_s{seed}.npz',
        ]
    else:
        return None
    for c in cands:
        if c.exists():
            return c
    return None


def find_segments(y: np.ndarray):
    """Indices of (start, end_exclusive) for each anomaly segment in y."""
    y = (np.asarray(y, dtype=np.int32) > 0).astype(np.int32)
    if y.sum() == 0:
        return []
    diff = np.diff(np.concatenate(([0], y, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def best_f1_vectorized(scores: np.ndarray, labels: np.ndarray,
                       pa: bool = False, n_thrs: int = N_THRS) -> Dict:
    """Sweep candidate thresholds; return best F1 (PA or non-PA), vectorized.

    PA implementation: compute max-score per anomaly segment. A segment is
    detected at threshold tau iff segment_max_score >= tau, and then counts
    as `len(segment)` true positives. False positives are points outside any
    segment with score >= tau.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32)
    if y.sum() == 0 or y.sum() == len(y):
        return {'f1': float('nan'), 'precision': float('nan'),
                'recall': float('nan'), 'threshold': float('nan')}

    qs = np.linspace(0.0, 1.0, n_thrs + 1)[1:-1]
    thrs = np.unique(np.quantile(s, qs))
    if not pa:
        # Vectorized over thresholds
        best = {'f1': -1.0}
        # Sort scores descending to enable cumulative-FP / cumulative-TP scan
        order = np.argsort(-s)
        ys = y[order]; ss = s[order]
        cum_tp = np.cumsum(ys)
        cum_fp = np.cumsum(1 - ys)
        n_pos = int(y.sum())
        # For each unique threshold tau, find the cutoff index k (smallest k
        # where ss[k] < tau). Then TP = cum_tp[k-1], FP = cum_fp[k-1].
        # Use searchsorted on -ss to find where each tau falls.
        neg_thrs = -np.sort(-thrs)  # descending
        for tau in neg_thrs[::-1]:  # try a range of taus
            k = int(np.searchsorted(-ss, -tau, side='right'))
            if k == 0:
                continue
            tp = int(cum_tp[k - 1]); fp = int(cum_fp[k - 1])
            fn = n_pos - tp
            if tp + fp == 0 or tp == 0:
                continue
            p_ = tp / (tp + fp); r_ = tp / (tp + fn)
            f1 = 2 * p_ * r_ / (p_ + r_)
            if f1 > best['f1']:
                best = {'f1': f1, 'precision': p_, 'recall': r_,
                        'threshold': float(tau), 'tp': tp, 'fp': fp, 'fn': fn}
        if best['f1'] < 0:
            best = {'f1': 0.0, 'precision': 0.0, 'recall': 0.0,
                    'threshold': float('nan')}
        return best

    # PA: precompute per-segment max + lengths
    segs = find_segments(y)
    if not segs:
        return {'f1': float('nan'), 'precision': float('nan'),
                'recall': float('nan'), 'threshold': float('nan')}
    seg_max = np.array([s[a:b].max() for a, b in segs])
    seg_len = np.array([b - a for a, b in segs])
    n_pos_total = int(seg_len.sum())

    # Out-of-segment scores
    in_seg_mask = np.zeros(len(s), dtype=bool)
    for a, b in segs:
        in_seg_mask[a:b] = True
    out_scores = s[~in_seg_mask]

    best = {'f1': -1.0}
    # Sort segments and out-scores once, sweep thresholds vectorized
    # For each tau:
    #   detected_segs = seg_max >= tau (boolean)
    #   tp = sum(seg_len[detected_segs])
    #   fp = sum(out_scores >= tau)
    #   fn = n_pos_total - tp
    # Sort to enable cumulative scan
    s_segmax_sorted = np.sort(seg_max)         # ascending
    seg_len_sorted_by_max_desc = seg_len[np.argsort(-seg_max)]
    cum_seg_len_top = np.cumsum(seg_len_sorted_by_max_desc)  # tp at top-k
    out_scores_sorted_desc = np.sort(-out_scores) * (-1)
    n_out = len(out_scores)
    for tau in thrs:
        # Detected segments = those with seg_max >= tau
        # Use searchsorted on s_segmax_sorted (asc) for "first index >= tau"
        k_seg = int(np.searchsorted(s_segmax_sorted, tau, side='left'))
        n_detected = len(seg_max) - k_seg
        if n_detected == 0:
            continue
        tp = int(cum_seg_len_top[n_detected - 1])
        # Count FPs in out-of-segment scores: scores >= tau
        n_fp = int(n_out - np.searchsorted(out_scores_sorted_desc[::-1], tau, side='left'))
        if tp + n_fp == 0 or tp == 0:
            continue
        fn = n_pos_total - tp
        p_ = tp / (tp + n_fp); r_ = tp / (tp + fn)
        f1 = 2 * p_ * r_ / (p_ + r_)
        if f1 > best['f1']:
            best = {'f1': f1, 'precision': p_, 'recall': r_,
                    'threshold': float(tau), 'tp': tp, 'fp': n_fp, 'fn': fn}
    if best['f1'] < 0:
        best = {'f1': 0.0, 'precision': 0.0, 'recall': 0.0,
                'threshold': float('nan')}
    return best


def metrics_at_horizon(p: np.ndarray, y: np.ndarray, horizons: np.ndarray,
                       h_target: int) -> Dict:
    h_idx = int(np.argmin(np.abs(horizons - h_target)))
    score = p[:, h_idx].astype(np.float64)
    label = (y[:, h_idx] > 0).astype(np.int32)
    if label.sum() == 0 or label.sum() == len(label):
        return {'h': int(horizons[h_idx]), 'pos_rate': float(label.mean()),
                'auroc': float('nan'), 'auprc': float('nan'),
                'pa_f1': float('nan'), 'pa_precision': float('nan'),
                'pa_recall': float('nan'),
                'non_pa_f1': float('nan'), 'non_pa_precision': float('nan'),
                'non_pa_recall': float('nan')}
    auroc = float(roc_auc_score(label, score))
    auprc = float(average_precision_score(label, score))
    pa = best_f1_vectorized(score, label, pa=True)
    npa = best_f1_vectorized(score, label, pa=False)
    return {
        'h': int(horizons[h_idx]),
        'pos_rate': float(label.mean()),
        'auroc': auroc, 'auprc': auprc,
        'pa_f1': pa['f1'], 'pa_precision': pa['precision'],
        'pa_recall': pa['recall'], 'pa_threshold': pa['threshold'],
        'non_pa_f1': npa['f1'], 'non_pa_precision': npa['precision'],
        'non_pa_recall': npa['recall'], 'non_pa_threshold': npa['threshold'],
    }


def horizon_sweep(p: np.ndarray, y: np.ndarray, horizons: np.ndarray) -> Dict:
    out = {}
    for h in HORIZON_SWEEP:
        if h > horizons.max():
            continue
        out[f'h{h}'] = metrics_at_horizon(p, y, horizons, h)
    valid = {k: v for k, v in out.items()
             if not np.isnan(v.get('non_pa_f1', np.nan))}
    if valid:
        best_npa = max(valid.values(), key=lambda v: v['non_pa_f1'])
        best_pa = max(valid.values(), key=lambda v: v['pa_f1'])
        best_auroc = max(valid.values(), key=lambda v: v['auroc'])
    else:
        best_npa = best_pa = best_auroc = {'h': None}
    out['_summary'] = {
        'best_non_pa_f1': best_npa.get('non_pa_f1', float('nan')),
        'best_non_pa_h': best_npa.get('h'),
        'best_pa_f1': best_pa.get('pa_f1', float('nan')),
        'best_pa_h': best_pa.get('h'),
        'best_auroc': best_auroc.get('auroc', float('nan')),
        'best_auroc_h': best_auroc.get('h'),
    }
    return out


def deep_investigation(p: np.ndarray, y: np.ndarray,
                       horizons: np.ndarray) -> Dict:
    inv = {'shape': list(p.shape), 'n_finite_p': int(np.isfinite(p).sum())}
    inv['p_quantiles'] = {f'q{q}': float(np.percentile(p, q))
                          for q in [0.5, 5, 25, 50, 75, 95, 99.5]}
    pos_per_h = {}
    auprc_per_h = {}
    for k, h in enumerate(horizons):
        label = (y[:, k] > 0).astype(np.int32)
        pos_per_h[f'h{int(h)}'] = float(label.mean())
        if 0 < label.sum() < len(label):
            auprc_per_h[f'h{int(h)}'] = float(
                average_precision_score(label, p[:, k]))
    inv['y_pos_rate_per_h'] = pos_per_h
    inv['per_horizon_auprc'] = auprc_per_h
    return inv


def run_one(ds: str, lf: int, seed: int) -> Optional[Dict]:
    p = _surf_path(ds, lf, seed)
    if p is None:
        return None
    d = np.load(p, allow_pickle=True)
    p_surf = d['p_surface']; y_surf = d['y_surface']
    horizons = np.asarray(d['horizons'], dtype=np.int64)
    out = horizon_sweep(p_surf, y_surf, horizons)
    out['_path'] = str(p.relative_to(FAM))
    out['_n_obs'] = int(p_surf.shape[0])
    out['_K'] = int(p_surf.shape[1])
    if ds in ('GECCO', 'BATADAL'):
        out['_diag'] = deep_investigation(p_surf, y_surf, horizons)
    return out


def aggregate(per_seed: List[Dict]) -> Dict:
    runs = [r for r in per_seed if r is not None and '_summary' in r]
    if not runs:
        return {'note': 'no surfaces found'}
    keys = ['best_non_pa_f1', 'best_pa_f1', 'best_auroc']
    out = {}
    for k in keys:
        vals = np.array([r['_summary'][k] for r in runs
                         if r['_summary'].get(k) is not None
                         and not np.isnan(r['_summary'][k])])
        if vals.size:
            out[f'{k}_mean'] = float(vals.mean())
            out[f'{k}_std'] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
            out[f'{k}_per_seed'] = vals.tolist()
    return out


def main():
    full = {}
    out_path = V32_RES / 'legacy_metrics_full.json'
    for ds in ANOMALY_DATASETS:
        for lf in [100, 10]:
            t0 = time.time()
            per_seed = []
            for s in SEEDS:
                r = run_one(ds, lf, s)
                per_seed.append({'seed': s, **(r or {'_missing': True})})
            agg = aggregate(per_seed)
            full[f'{ds}_lf{lf}'] = {'agg': agg, 'per_seed': per_seed}
            elapsed = time.time() - t0
            if 'best_non_pa_f1_mean' in agg:
                print(f'  {ds:7s} lf{lf:3d}: nPA-F1 {agg["best_non_pa_f1_mean"]:.3f}±{agg["best_non_pa_f1_std"]:.3f}, '
                      f'PA-F1 {agg["best_pa_f1_mean"]:.3f}±{agg["best_pa_f1_std"]:.3f}, '
                      f'AUROC {agg["best_auroc_mean"]:.3f}±{agg["best_auroc_std"]:.3f}  '
                      f'({elapsed:.1f}s)', flush=True)
            else:
                print(f'  {ds:7s} lf{lf:3d}: NO SURFACES ({elapsed:.1f}s)',
                      flush=True)
            with open(out_path, 'w') as f:
                json.dump(full, f, indent=2, default=str)
    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
