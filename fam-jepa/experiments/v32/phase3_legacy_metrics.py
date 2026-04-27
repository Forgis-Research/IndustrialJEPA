"""V32 Phase 3: Recompute legacy metrics (PA-F1 / non-PA F1 / AUROC / AUPRC)
from stored probability surfaces for ALL anomaly datasets at 100% and 10%
labels.

For each (dataset, label_frac, seed):
  - Load p_surface (N, K), y_surface (N, K).
  - Sweep horizon_for_score in {1, 3, 5, 10, 20, 50, 100, 150}.
  - At each horizon: compute observation-level AUROC, AUPRC; sweep thresholds
    for best F1 (PA and non-PA on the observation timeline).
  - Pick the horizon that maximizes pa_f1 (kept), and separately the one that
    maximizes non_pa_f1 (kept). Report both.

For GECCO and BATADAL: produce a deep diagnostic with score quantiles, label
prevalence, per-horizon AUPRC breakdown, and best-F1 across all horizons.

Output: results/legacy_metrics_full.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score,
)

warnings.filterwarnings('ignore')

FAM = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V30_SURF = FAM / 'experiments/v30/surfaces'
V31_SURF = FAM / 'experiments/v31/surfaces'
V32_RES = FAM / 'experiments/v32/results'
V32_RES.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(FAM / 'experiments/v21'))
from surface_to_legacy import _adjust_predictions  # noqa: E402

ANOMALY_DATASETS = [
    'SMAP', 'PSM', 'SMD', 'MBA', 'SKAB', 'ETTm1', 'GECCO', 'BATADAL', 'MSL',
]
HORIZON_SWEEP = [1, 3, 5, 10, 20, 50, 100, 150]
SEEDS = [42, 123, 456]


def _surf_path(ds: str, lf: int, seed: int) -> Optional[Path]:
    if lf == 100:
        # 100% labels — v30 dense p3 surfaces
        cands = [
            V30_SURF / f'{ds}_revin_discrete_hazard_td20_p3_s{seed}.npz',
            V30_SURF / f'{ds}_none_discrete_hazard_td20_p3_s{seed}.npz',
            V30_SURF / f'{ds}_revin_discrete_hazard_td20_p2_s{seed}.npz',
        ]
    elif lf == 10:
        # 10% labels — v31 lf10 surfaces
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


def _best_f1_with_pa(scores: np.ndarray, labels: np.ndarray,
                     pa: bool, n_thrs: int = 200) -> Dict:
    """Sweep candidate thresholds (quantiles of scores), return best-F1.

    PA is applied at each candidate threshold before F1 is computed.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return {'f1': float('nan'), 'precision': float('nan'),
                'recall': float('nan'), 'threshold': float('nan')}
    qs = np.linspace(0.0, 1.0, n_thrs + 1)[1:-1]
    thrs = np.unique(np.quantile(scores, qs))
    best = {'f1': -1.0}
    for thr in thrs:
        yp = (scores >= thr).astype(np.int32)
        if pa:
            yp = _adjust_predictions(labels, yp)
        tp = int(((yp == 1) & (labels == 1)).sum())
        fp = int(((yp == 1) & (labels == 0)).sum())
        fn = int(((yp == 0) & (labels == 1)).sum())
        if tp == 0:
            f1 = 0.0; p = 0.0; r = 0.0
        else:
            p = tp / (tp + fp); r = tp / (tp + fn)
            f1 = 2 * p * r / (p + r)
        if f1 > best['f1']:
            best = {'f1': f1, 'precision': p, 'recall': r,
                    'threshold': float(thr), 'tp': tp, 'fp': fp, 'fn': fn}
    return best


def metrics_at_horizon(p: np.ndarray, y: np.ndarray, horizons: np.ndarray,
                       h_target: int) -> Dict:
    """Observation-level metrics at a chosen horizon."""
    h_idx = int(np.argmin(np.abs(horizons - h_target)))
    score = p[:, h_idx].astype(np.float64)
    label = (y[:, h_idx] > 0).astype(np.int32)
    if label.sum() == 0 or label.sum() == len(label):
        return {'h': int(horizons[h_idx]), 'pos_rate': float(label.mean()),
                'auroc': float('nan'), 'auprc': float('nan'),
                'pa_f1': float('nan'), 'pa_precision': float('nan'),
                'pa_recall': float('nan'), 'pa_threshold': float('nan'),
                'non_pa_f1': float('nan'), 'non_pa_precision': float('nan'),
                'non_pa_recall': float('nan'), 'non_pa_threshold': float('nan')}
    auroc = float(roc_auc_score(label, score))
    auprc = float(average_precision_score(label, score))
    pa = _best_f1_with_pa(score, label, pa=True)
    npa = _best_f1_with_pa(score, label, pa=False)
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
    """Compute metrics_at_horizon for each h in HORIZON_SWEEP available."""
    out = {}
    for h in HORIZON_SWEEP:
        if h > horizons.max():
            continue
        out[f'h{h}'] = metrics_at_horizon(p, y, horizons, h)
    # Best across the sweep
    valid = {k: v for k, v in out.items() if not np.isnan(v.get('non_pa_f1', np.nan))}
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


def deep_investigation(ds: str, p: np.ndarray, y: np.ndarray,
                       horizons: np.ndarray) -> Dict:
    """Diagnostic dump for GECCO / BATADAL low-F1 mystery."""
    inv = {}
    inv['shape'] = list(p.shape)
    inv['n_finite_p'] = int(np.isfinite(p).sum())
    inv['p_quantiles'] = {
        f'q{q}': float(np.percentile(p, q))
        for q in [0.5, 5, 25, 50, 75, 95, 99.5]
    }
    inv['y_pos_rate_per_horizon'] = {
        f'h{int(h)}': float(y[:, k].mean())
        for k, h in enumerate(horizons)
    }
    inv['per_horizon_auprc'] = {}
    for k, h in enumerate(horizons):
        label = (y[:, k] > 0).astype(np.int32)
        if label.sum() == 0 or label.sum() == len(label):
            continue
        inv['per_horizon_auprc'][f'h{int(h)}'] = float(
            average_precision_score(label, p[:, k])
        )
    return inv


def run_one(ds: str, lf: int, seed: int) -> Optional[Dict]:
    p = _surf_path(ds, lf, seed)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    p_surf = d['p_surface']; y_surf = d['y_surface']
    horizons = np.asarray(d['horizons'], dtype=np.int64)
    out = horizon_sweep(p_surf, y_surf, horizons)
    out['_path'] = str(p.relative_to(FAM))
    out['_n_obs'] = int(p_surf.shape[0])
    out['_K'] = int(p_surf.shape[1])
    if ds in ('GECCO', 'BATADAL'):
        out['_diag'] = deep_investigation(ds, p_surf, y_surf, horizons)
    return out


def aggregate(per_seed: List[Dict]) -> Dict:
    """Mean ± std over seeds for each summary metric."""
    runs = [r for r in per_seed if r is not None]
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
    for ds in ANOMALY_DATASETS:
        for lf in [100, 10]:
            per_seed = []
            for s in SEEDS:
                r = run_one(ds, lf, s)
                per_seed.append({'seed': s, **(r or {'_missing': True})})
            agg = aggregate(per_seed)
            full[f'{ds}_lf{lf}'] = {'agg': agg, 'per_seed': per_seed}
    out_path = V32_RES / 'legacy_metrics_full.json'
    with open(out_path, 'w') as f:
        json.dump(full, f, indent=2, default=str)
    print(f'wrote {out_path}')
    print('\n=== KEY RESULTS ===')
    for ds in ANOMALY_DATASETS:
        for lf in [100, 10]:
            key = f'{ds}_lf{lf}'
            agg = full[key].get('agg', {})
            if 'best_non_pa_f1_mean' in agg:
                print(f'  {key}: nPA-F1 {agg["best_non_pa_f1_mean"]:.3f}±{agg["best_non_pa_f1_std"]:.3f}, '
                      f'PA-F1 {agg["best_pa_f1_mean"]:.3f}±{agg["best_pa_f1_std"]:.3f}, '
                      f'AUROC {agg["best_auroc_mean"]:.3f}±{agg["best_auroc_std"]:.3f}')
            else:
                print(f'  {key}: NO SURFACES')


if __name__ == '__main__':
    main()
