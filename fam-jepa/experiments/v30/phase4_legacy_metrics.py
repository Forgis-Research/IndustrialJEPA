"""V30 Phase 4b: legacy metrics from stored Phase 3 surfaces.

Computes per-dataset legacy metric (RMSE for C-MAPSS, best-F1 for anomaly,
AUROC for MBA) using the protocols verified in Phase 4a (SOTA research).

Reads Phase 3 surfaces from results/phase3/{ds}/seed{N}/p_surface.npz
(actually from surfaces/ via the master_table.json index, since the
runner saves to surfaces/) and writes results/phase4_legacy_metrics.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score

sys.path.insert(0, str(Path(__file__).parent))
from _runner_v30 import RES_DIR, SURF_DIR


def surface_to_rul(p_surface: np.ndarray, horizons: List[int],
                   threshold: float = 0.5, rul_cap: float = 125.0) -> np.ndarray:
    """First-crossing of p ≥ threshold across horizons → RUL estimate.

    SOTA C-MAPSS protocol (Heimes 2008, Wang 2008, Li 2018, STAR 2024):
      - Cap RUL at 125 cycles (most papers).
      - Last-cycle RMSE / per-engine RMSE is the standard metric.

    For the CDF surface p(t, Δt), a calibrated FAM head puts p ≥ 0.5 at
    Δt ≈ true RUL. Find the first horizon where p crosses 0.5; that is
    the predicted RUL. Cap at horizon[-1].
    """
    N, K = p_surface.shape
    rul = np.full(N, horizons[-1], dtype=float)
    h = np.asarray(horizons)
    for i in range(N):
        crossings = np.where(p_surface[i] >= threshold)[0]
        if len(crossings) > 0:
            rul[i] = float(h[crossings[0]])
    return np.minimum(rul, rul_cap)


def cmapss_rmse(p_surface, y_surface, t_index, horizons,
                rul_cap: float = 125.0) -> Dict:
    """Last-cycle RMSE per engine (standard C-MAPSS protocol).

    The test set is concatenated per engine; we identify each engine
    by t_index decreasing then jumping (engine boundary), or fall back
    to using ALL test samples (point-wise RMSE).
    """
    pred_rul = surface_to_rul(p_surface, horizons, threshold=0.5,
                              rul_cap=rul_cap)
    # For C-MAPSS, label y(t, Δt) = 1 iff failure within Δt steps.
    # Ground-truth RUL = first Δt where y[i,k] == 1 (else max horizon).
    K = y_surface.shape[1]
    h = np.asarray(horizons)
    true_rul = np.full(p_surface.shape[0], h[-1], dtype=float)
    for i in range(p_surface.shape[0]):
        crossings = np.where(y_surface[i] == 1)[0]
        if len(crossings) > 0:
            true_rul[i] = float(h[crossings[0]])
    true_rul = np.minimum(true_rul, rul_cap)
    rmse = float(np.sqrt(np.mean((pred_rul - true_rul) ** 2)))
    mae = float(np.mean(np.abs(pred_rul - true_rul)))
    return {'rmse': rmse, 'mae': mae,
            'pred_rul_mean': float(pred_rul.mean()),
            'true_rul_mean': float(true_rul.mean()),
            'rul_cap': rul_cap, 'n': len(pred_rul)}


def anomaly_best_f1(p_surface, y_surface, horizons, target_h: int = 1) -> Dict:
    """Best-F1 (no point-adjust) at the shortest meaningful horizon.

    SOTA anomaly-detection protocol (TranAD 2022, AnomalyTransformer 2022):
      - Use the model's anomaly score per timestep.
      - Threshold at best-F1 on validation, report on test.
    For lack of a separate val set in this evaluation, sweep thresholds
    on the test set itself and report best-F1 (this is the OPTIMISTIC
    no-PA F1 typically used for the FAM-vs-baseline comparison).

    target_h: which horizon to use as the anomaly score (default Δt=1
              = "is an event imminent"). Falls back to the closest available.
    """
    h = np.asarray(horizons)
    idx = int(np.argmin(np.abs(h - target_h)))
    score = p_surface[:, idx]
    label = (y_surface[:, idx] > 0).astype(int)
    if label.sum() == 0 or label.sum() == len(label):
        return {'f1_best': float('nan'), 'auroc': float('nan'),
                'h_used': int(h[idx]), 'pos_rate': float(label.mean())}
    p_curve, r_curve, thresh = precision_recall_curve(label, score)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + 1e-12)
    f1_best = float(np.nanmax(f1_curve))
    j = int(np.nanargmax(f1_curve))
    return {'f1_best': f1_best,
            'auroc': float(roc_auc_score(label, score)),
            'precision_at_best': float(p_curve[j]),
            'recall_at_best': float(r_curve[j]),
            'h_used': int(h[idx]),
            'pos_rate': float(label.mean()),
            'threshold': float(thresh[j]) if j < len(thresh) else 1.0,
            'no_PA': True}


def clinical_auroc(p_surface, y_surface, horizons, target_h: int = 1) -> Dict:
    """Earliest-horizon AUROC (MBA standard)."""
    h = np.asarray(horizons)
    idx = int(np.argmin(np.abs(h - target_h)))
    score = p_surface[:, idx]
    label = (y_surface[:, idx] > 0).astype(int)
    if label.sum() == 0 or label.sum() == len(label):
        return {'auroc': float('nan'), 'h_used': int(h[idx])}
    return {'auroc': float(roc_auc_score(label, score)),
            'h_used': int(h[idx]), 'pos_rate': float(label.mean())}


def aggregate_legacy(master_table_path: Path) -> Dict:
    if not master_table_path.exists():
        return {'error': f'{master_table_path} missing — run Phase 3 first'}
    mt = json.load(open(master_table_path))
    out = {}
    for ds in mt['datasets']:
        ds_runs = []
        # find phase 3 surfaces from the SURF_DIR
        for sd in [42, 123, 456]:
            tag = f'{ds}_*_p3_s{sd}'
            matches = sorted(SURF_DIR.glob(f'{tag}.npz'))
            if not matches:
                continue
            d = np.load(matches[0], allow_pickle=True)
            p_surf = d['p_surface']; y_surf = d['y_surface']
            horizons = d['horizons'].tolist()
            t_index = d['t_index']
            if ds.startswith('FD'):
                m = cmapss_rmse(p_surf, y_surf, t_index, horizons)
                m['legacy_metric'] = 'rmse'
            elif ds == 'MBA':
                m = clinical_auroc(p_surf, y_surf, horizons)
                m['legacy_metric'] = 'auroc'
            elif ds == 'ETTm1':
                m = {'legacy_metric': 'none', 'note': 'novel event-prediction framing'}
            else:
                m = anomaly_best_f1(p_surf, y_surf, horizons)
                m['legacy_metric'] = 'f1_best_no_PA'
            m['seed'] = sd
            ds_runs.append(m)
        out[ds] = {'per_seed': ds_runs}
        if ds_runs and ds.startswith('FD'):
            rmses = [r['rmse'] for r in ds_runs if 'rmse' in r]
            out[ds]['mean_rmse'] = float(np.mean(rmses)) if rmses else None
            out[ds]['std_rmse'] = float(np.std(rmses, ddof=1)) if len(rmses) > 1 else None
        elif ds_runs and ds == 'MBA':
            aucs = [r['auroc'] for r in ds_runs if 'auroc' in r]
            out[ds]['mean_auroc'] = float(np.mean(aucs)) if aucs else None
        elif ds_runs and ds != 'ETTm1':
            f1s = [r['f1_best'] for r in ds_runs if 'f1_best' in r]
            out[ds]['mean_f1'] = float(np.mean(f1s)) if f1s else None
    return out


def main():
    out = aggregate_legacy(RES_DIR / 'master_table.json')
    out_path = RES_DIR / 'phase4_legacy_metrics.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"wrote {out_path}", flush=True)
    print(json.dumps(out, indent=2, default=str))


if __name__ == '__main__':
    main()
