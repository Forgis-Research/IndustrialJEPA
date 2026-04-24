"""V27 Phase 1: Baseline diagnostic from existing v26 FD001 surfaces.

The v26 surfaces for FD001 (s42, s123, s456) are already on disk. We just need
to compute the per-horizon AUROC + AUPRC tables and confirm the collapse to
chance at Δt=10. This is the PRIMARY DIAGNOSTIC for the entire session.

If v26 surfaces are missing or inconsistent with the expected collapse,
we retrain; otherwise we reuse.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

FAM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V26_SURF = FAM_DIR / 'experiments/v26/surfaces'
V27_RES = FAM_DIR / 'experiments/v27/results'
V27_RES.mkdir(parents=True, exist_ok=True)


def per_horizon_table(p, y, horizons, label=''):
    """Return list of (dt, auroc, auprc, pos_rate) for each horizon."""
    rows = []
    for i, h in enumerate(horizons):
        y_i = y[:, i].astype(int)
        p_i = p[:, i]
        if y_i.sum() == 0 or y_i.sum() == len(y_i):
            auroc = auprc = float('nan')
        else:
            auroc = roc_auc_score(y_i, p_i)
            auprc = average_precision_score(y_i, p_i)
        pos_rate = float(y_i.mean())
        # Prediction gap: mean(p | y=1) - mean(p | y=0)
        gap = float('nan')
        if y_i.sum() > 0 and (1 - y_i).sum() > 0:
            gap = float(p_i[y_i == 1].mean() - p_i[y_i == 0].mean())
        rows.append({
            'dt': int(h), 'auroc': float(auroc), 'auprc': float(auprc),
            'pos_rate': pos_rate, 'pred_gap': gap,
        })
    return rows


def diagnose(surf_path: Path, label: str) -> dict:
    print(f"\n=== {label} ===")
    print(f"  path: {surf_path}")
    d = np.load(surf_path, allow_pickle=True)
    p = d['p_surface']
    y = d['y_surface']
    horizons = d['horizons'].tolist()
    print(f"  shape: p={p.shape} y={y.shape} horizons={horizons}")
    print(f"  pooled AUPRC = {average_precision_score(y.ravel().astype(int), p.ravel()):.4f}")
    print(f"  pooled AUROC = {roc_auc_score(y.ravel().astype(int), p.ravel()):.4f}")
    rows = per_horizon_table(p, y, horizons, label)
    print(f"  {'dt':>4}  {'AUROC':>7}  {'AUPRC':>7}  {'pos':>6}  {'gap':>7}")
    for r in rows:
        print(f"  {r['dt']:>4}  {r['auroc']:>7.3f}  {r['auprc']:>7.3f}  "
              f"{r['pos_rate']:>6.3f}  {r['pred_gap']:>+7.3f}")
    return {'label': label, 'path': str(surf_path), 'rows': rows,
            'pooled_auprc': float(average_precision_score(y.ravel().astype(int), p.ravel())),
            'pooled_auroc': float(roc_auc_score(y.ravel().astype(int), p.ravel()))}


def main():
    summary = {}
    for subset in ['FD001', 'FD003', 'SMAP', 'MBA']:
        summary[subset] = {}
        for seed in [42, 123, 456]:
            path = V26_SURF / f'{subset}_s{seed}.npz'
            if path.exists():
                summary[subset][f's{seed}'] = diagnose(path, f'{subset}_s{seed}')
            else:
                print(f"MISSING: {path}")

    # Write aggregated per-horizon AUROC table
    out_path = V27_RES / 'phase1_v26_baseline_diagnostic.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out_path}")

    # Headline: FD001 Δt=10 AUROC across seeds
    print("\n=== HEADLINE: FD001 per-horizon AUROC across seeds ===")
    rows_by_seed = []
    for seed_key, diag in summary.get('FD001', {}).items():
        rows_by_seed.append((seed_key, {r['dt']: r['auroc'] for r in diag['rows']}))
    if rows_by_seed:
        horizons = [1, 5, 10, 20, 50, 100, 150]
        print(f"  {'seed':<6}  " + "  ".join(f"{'dt='+str(h):>7}" for h in horizons))
        for seed_key, d in rows_by_seed:
            print(f"  {seed_key:<6}  " + "  ".join(f"{d.get(h, float('nan')):>7.3f}" for h in horizons))
        means = {h: float(np.nanmean([d.get(h, np.nan) for _, d in rows_by_seed])) for h in horizons}
        print(f"  {'mean':<6}  " + "  ".join(f"{means[h]:>7.3f}" for h in horizons))


if __name__ == '__main__':
    main()
