"""V32 Phase 7: verify FAM h-AUROC numbers in paper Table 4 against the
actual stored surfaces. This is a sanity-check pass - if the recomputed
h-AUROC differs from what's currently in the paper, flag it.

Iterates over (dataset, lf, seed) for the v30/v31 dense surfaces and
reports per-horizon AUROC + the mean. Writes results/hauroc_verify.json
and prints a "matches paper?" summary.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import roc_auc_score

FAM = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V30_SURF = FAM / 'experiments/v30/surfaces'
V31_SURF = FAM / 'experiments/v31/surfaces'
V32_RES = FAM / 'experiments/v32/results'

DATASETS = ['FD001', 'FD002', 'FD003', 'SMAP', 'PSM', 'SMD',
            'MBA', 'SKAB', 'ETTm1', 'GECCO', 'BATADAL']
SEEDS = [42, 123, 456]


SPARSE_DATASETS = {'MBA', 'BATADAL', 'SKAB', 'ETTm1'}


def find_surface(ds: str, lf: int, seed: int) -> Optional[Path]:
    if lf == 100:
        if ds in SPARSE_DATASETS:
            cands = [
                V30_SURF / f'{ds}_revin_discrete_hazard_p3b_s{seed}.npz',
                V30_SURF / f'{ds}_revin_discrete_hazard_td20_p3_s{seed}.npz',
            ]
        else:
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


# Currently in paper.tex Table 4 (manually transcribed from lines 305-352)
PAPER_NUMBERS = {
    'FD001_lf100': 0.79, 'FD001_lf10': 0.77,  # C-MAPSS row, treat as FD001
    'SMAP_lf100':  0.60, 'SMAP_lf10':  0.58,
    'PSM_lf100':   0.56, 'PSM_lf10':   0.52,
    'SMD_lf100':   0.65, 'SMD_lf10':   0.53,
    'MBA_lf100':   0.74, 'MBA_lf10':   0.55,
    'SKAB_lf100':  0.71, 'SKAB_lf10':  0.73,
    'ETTm1_lf100': 0.87, 'ETTm1_lf10': 0.77,
    'GECCO_lf100': 0.82, 'GECCO_lf10': 0.35,
    'BATADAL_lf100': 0.61, 'BATADAL_lf10': 0.64,
}


def per_horizon_auroc(p, y, horizons):
    aurocs = []
    per_h = {}
    for k, h in enumerate(horizons):
        lab = (y[:, k] > 0).astype(np.int32)
        if lab.sum() == 0 or lab.sum() == len(lab):
            per_h[f'h{int(h)}'] = float('nan')
            continue
        au = float(roc_auc_score(lab, p[:, k]))
        aurocs.append(au)
        per_h[f'h{int(h)}'] = au
    return float(np.mean(aurocs)) if aurocs else float('nan'), per_h


def main():
    out = {}
    for ds in DATASETS:
        for lf in [100, 10]:
            key = f'{ds}_lf{lf}'
            seeds_aurocs = []
            for s in SEEDS:
                p = find_surface(ds, lf, s)
                if p is None:
                    continue
                d = np.load(p)
                p_surf = d['p_surface']; y_surf = d['y_surface']
                horizons = np.asarray(d['horizons'], dtype=np.int64)
                mean_auroc, per_h = per_horizon_auroc(p_surf, y_surf, horizons)
                seeds_aurocs.append(mean_auroc)
            if seeds_aurocs:
                arr = np.array(seeds_aurocs)
                out[key] = {
                    'mean_h_auroc_recomputed': float(arr.mean()),
                    'std_h_auroc': float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    'n_seeds': int(len(arr)),
                    'paper_value': PAPER_NUMBERS.get(key),
                    'matches': bool(abs(arr.mean() - PAPER_NUMBERS.get(key, 0.0)) < 0.03)
                              if key in PAPER_NUMBERS else None,
                }
            else:
                out[key] = {'note': 'no surfaces found'}

    print('=== h-AUROC verification (recomputed vs paper Table 4) ===')
    print(f'{"key":18s} | {"recomp":>14s} | {"paper":>6s} | match?')
    for k, v in out.items():
        if 'mean_h_auroc_recomputed' in v:
            recomp = f'{v["mean_h_auroc_recomputed"]:.3f}±{v["std_h_auroc"]:.3f}'
            paper = f'{v.get("paper_value", "-"):.2f}' if v.get('paper_value') else '-'
            mtch = 'OK' if v.get('matches') else ('MISMATCH' if v.get('matches') is False else '?')
            print(f'  {k:18s} | {recomp:>14s} | {paper:>6s} | {mtch}')
        else:
            print(f'  {k:18s} | {"--":>14s} | {"--":>6s} | missing')

    out_path = V32_RES / 'hauroc_verify.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
