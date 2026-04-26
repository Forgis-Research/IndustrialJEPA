"""V24: compute all legacy metrics (RMSE, NASA, F1, PA-F1, AUROC) from stored
surfaces so everything is in one place for the RESULTS.md table.

C-MAPSS FDxxx: RMSE + NASA score, predicted via surface_to_rul (threshold
crossing) and surface_to_rul_expected (trapezoid integration). True TTE is
derived from y_surface: TTE(t) = first h_k where y(t, h_k) = 1, else h_max.

Anomaly (SMAP/MSL/PSM/SMD/MBA): per-timestep score from surface (horizon_for
= 100), PA-F1 and non-PA F1 and AUROC on per-timestep labels.

Sepsis: AUROC already in phase6_sepsis.json; also compute AUPRC per horizon.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

V24 = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v24')
SURF = V24 / 'surfaces'
OUT = V24 / 'results' / 'phase10_legacy.json'

sys.path.insert(0, str(V24.parent.parent))
sys.path.insert(0, str(V24.parent / 'v21'))
from surface_to_legacy import (
    surface_to_rul, surface_to_rul_expected, rmse, nasa_score,
)

CMAPSS = ['FD001', 'FD002', 'FD003']
ANOMALY = ['SMAP', 'MSL', 'PSM', 'SMD', 'MBA']
SEEDS = [42, 123, 456]


def true_tte_from_y(y_surface: np.ndarray, horizons: np.ndarray) -> np.ndarray:
    """Derive per-row true TTE from (N, K) label surface.

    y(t, h_k) = 1[TTE <= h_k]. So smallest k where y=1 gives TTE in
    [h_{k-1}, h_k]. We use h_k as the representative TTE value.
    If no y=1 across any horizon, TTE > h_max; use h_max+1 as placeholder
    (or cap at h_max for RMSE).
    """
    y = y_surface.astype(bool)
    K = y.shape[1]
    any_pos = y.any(axis=1)
    first_pos = y.argmax(axis=1)  # 0 if row is all-False
    tte = np.where(any_pos, horizons[first_pos], horizons[-1] + 1)
    return tte.astype(np.float32)


def cmapss_metrics():
    out = {}
    for ds in CMAPSS:
        rows = []
        for seed in SEEDS:
            p = SURF / f'{ds}_s{seed}.npz'
            if not p.exists():
                continue
            d = np.load(p)
            p_surf = d['p_surface']; y_surf = d['y_surface']
            horizons = d['horizons'].astype(np.int64)
            h_max = float(horizons[-1])

            # True TTE (capped at h_max for fair comparison)
            true_tte = true_tte_from_y(y_surf, horizons)
            true_tte_cap = np.minimum(true_tte, h_max)

            # Predicted TTE
            pred_cross = surface_to_rul(p_surf, horizons)  # first h s.t. p>=0.5
            pred_exp = surface_to_rul_expected(p_surf, horizons)

            rmse_cross = rmse(pred_cross, true_tte_cap)
            rmse_exp = rmse(pred_exp, true_tte_cap)
            nasa = nasa_score(pred_exp, true_tte_cap)
            rows.append({
                'seed': seed,
                'rmse_cross': float(rmse_cross),
                'rmse_expected': float(rmse_exp),
                'nasa_score': float(nasa),
                'n': int(len(true_tte_cap)),
            })
        if rows:
            rmse_c = [r['rmse_cross'] for r in rows]
            rmse_e = [r['rmse_expected'] for r in rows]
            nasa_s = [r['nasa_score'] for r in rows]
            out[ds] = {
                'per_seed': rows,
                'rmse_expected_mean': float(np.mean(rmse_e)),
                'rmse_expected_std':  float(np.std(rmse_e)),
                'rmse_cross_mean':    float(np.mean(rmse_c)),
                'rmse_cross_std':     float(np.std(rmse_c)),
                'nasa_mean':          float(np.mean(nasa_s)),
                'nasa_std':           float(np.std(nasa_s)),
                'n_seeds':            len(rows),
            }
            print(f"  {ds}  RMSE (exp)  {np.mean(rmse_e):.2f} +/- {np.std(rmse_e):.2f}   "
                  f"RMSE (cross) {np.mean(rmse_c):.2f} +/- {np.std(rmse_c):.2f}", flush=True)
    return out


def main():
    print("=== C-MAPSS legacy (RMSE, NASA) ===", flush=True)
    cmapss = cmapss_metrics()

    # Anomaly PA-F1 / non-PA F1 (already computed in phase7_pa_f1.json)
    pa_f1_path = V24 / 'results' / 'phase7_pa_f1.json'
    if pa_f1_path.exists():
        anomaly = json.load(open(pa_f1_path))
    else:
        anomaly = {}
    print("\n=== Anomaly PA-F1 (from phase7) ===", flush=True)
    for ds, o in anomaly.items():
        a = o['agg']
        print(f"  {ds}  PA-F1  {a['f1_pa_mean']:.3f} +/- {a['f1_pa_std']:.3f}   "
              f"non-PA-F1 {a['f1_non_pa_mean']:.3f} +/- {a['f1_non_pa_std']:.3f}",
              flush=True)

    # Sepsis: already in phase6_sepsis.json (AUROC + AUPRC)
    sep_path = V24 / 'results' / 'phase6_sepsis.json'
    sepsis = json.load(open(sep_path)) if sep_path.exists() else {}
    if sepsis:
        print(f"\n=== Sepsis ===", flush=True)
        print(f"  AUROC  {sepsis['auroc_mean']:.4f} +/- {sepsis['auroc_std']:.4f}",
              flush=True)

    combined = {
        'cmapss_legacy': cmapss,
        'anomaly_pa_f1': anomaly,
        'sepsis': {k: sepsis[k] for k in
                   ['n_seeds', 'auroc_mean', 'auroc_std', 'auprc_mean', 'auprc_std']
                   if k in sepsis} if sepsis else {},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(combined, indent=2))
    print(f"\nwrote {OUT}", flush=True)


if __name__ == '__main__':
    main()
