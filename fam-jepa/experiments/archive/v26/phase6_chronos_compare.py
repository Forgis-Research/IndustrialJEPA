"""V26 Phase 6: FAM (v26 hazard CDF) vs Chronos-2 baseline.

The Chronos-2 baseline is reused from v24 (same test splits, same random
seeds, same frozen pretrained model — splits are bit-identical between
v24 and v26, so the numbers are unchanged by construction). We simply
aggregate v24's Chronos-2 JSONs and place them alongside v26's FAM
numbers for head-to-head comparison.

If --rerun is passed, call baseline_chronos2.py to re-extract and reprobe.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FAM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V24_RES = FAM_DIR / 'experiments/v24/results'
V26_RES = FAM_DIR / 'experiments/v26/results'
V26_RES.mkdir(parents=True, exist_ok=True)


DATASETS = [
    ('FD001', 'phase2', 'cmapss'),
    ('FD002', 'phase2', 'cmapss'),
    ('FD003', 'phase2', 'cmapss'),
    ('SMAP',  'phase3', 'anomaly'),
    ('MSL',   'phase3', 'anomaly'),
    ('PSM',   'phase3', 'anomaly'),
    ('SMD',   'phase3', 'anomaly'),
    ('MBA',   'phase3', 'anomaly'),
]


def load_v26_fam(ds: str, phase: str) -> dict:
    # v26 FAM result JSONs
    path = V26_RES / f'{phase}_{ds}.json'
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    if 'auprc_mean' not in d:
        # might be in-progress with results[] only
        if d.get('results'):
            auprcs = [r['auprc'] for r in d['results']]
            return {'auprc_mean': float(np.mean(auprcs)),
                    'auprc_std':  float(np.std(auprcs)),
                    'n_seeds':    len(auprcs),
                    'source':     'v26_fam_partial'}
        return None
    return {'auprc_mean': d['auprc_mean'],
            'auprc_std':  d['auprc_std'],
            'n_seeds':    d['n_seeds'],
            'source':     'v26_fam'}


def load_v24_chronos(ds: str) -> dict:
    """Aggregate v24 Chronos-2 per-seed JSONs."""
    paths = sorted(V24_RES.glob(f'baseline_chronos2_{ds}_s*.json'))
    # Also the older SMAP single-seed
    if not paths:
        single = V24_RES / f'baseline_chronos2_{ds}.json'
        if single.exists():
            paths = [single]
    if not paths:
        return None
    auprcs = []
    aurocs = []
    for p in paths:
        d = json.loads(p.read_text())
        if d.get('skipped'):
            continue
        prim = d.get('primary')
        if not prim:
            continue
        auprcs.append(float(prim['auprc']))
        aurocs.append(float(prim['auroc']))
    if not auprcs:
        return None
    return {'auprc_mean': float(np.mean(auprcs)),
            'auprc_std':  float(np.std(auprcs)),
            'auroc_mean': float(np.mean(aurocs)),
            'auroc_std':  float(np.std(aurocs)),
            'n_seeds':    len(auprcs),
            'source':     'v24_chronos2 (same splits as v26)'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str,
                    default=str(V26_RES / 'phase6_chronos_compare.json'))
    args = ap.parse_args()

    comparison = {}
    print(f"{'Dataset':<8} {'FAM AUPRC':>12}  {'Chronos-2 AUPRC':>16}  "
          f"{'delta':>8}  {'winner':<8}", flush=True)
    print('-' * 64, flush=True)
    for ds, phase, _ in DATASETS:
        fam = load_v26_fam(ds, phase)
        cr = load_v24_chronos(ds)
        row = {'dataset': ds, 'fam': fam, 'chronos2': cr}
        if fam and cr:
            delta = fam['auprc_mean'] - cr['auprc_mean']
            row['delta'] = float(delta)
            row['winner'] = 'FAM' if delta > 0 else 'Chronos-2'
            print(f"{ds:<8} {fam['auprc_mean']:.4f}+/-{fam['auprc_std']:.4f}  "
                  f"{cr['auprc_mean']:.4f}+/-{cr['auprc_std']:.4f}  "
                  f"{delta:+.4f}  {row['winner']:<8}", flush=True)
        else:
            missing = []
            if fam is None: missing.append('fam')
            if cr is None: missing.append('chronos2')
            print(f"{ds:<8} -- missing: {','.join(missing)}", flush=True)
        comparison[ds] = row

    Path(args.out).write_text(json.dumps(comparison, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
