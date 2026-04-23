"""V24 Phase 7: compute PA-F1 / non-PA F1 from v24 surfaces.

Adapted from experiments/v22/compute_pa_f1_from_surfaces.py for the v24
surface naming convention: {DATASET}_s{seed}.npz in experiments/v24/surfaces/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SURF_DIR = ROOT / 'surfaces'
OUT_PATH = ROOT / 'results' / 'phase7_pa_f1.json'

sys.path.insert(0, str(ROOT.parent.parent))
from evaluation.grey_swan_metrics import anomaly_metrics

DATASETS = ['SMAP', 'MSL', 'PSM', 'SMD', 'MBA']
SEEDS = [42, 123, 456]
PERCENTILES = [90, 92, 94, 95, 96, 98, 99]


def surface_to_score(p_surface):
    return p_surface.max(axis=1)


def surface_to_label(y_surface):
    return (y_surface.max(axis=1) > 0).astype(int)


def best_pa_f1(scores, y_true):
    best = None
    for pct in PERCENTILES:
        m = anomaly_metrics(scores, y_true, threshold_percentile=pct)
        m['threshold_percentile'] = pct
        if best is None or m['f1_pa'] > best['f1_pa']:
            best = m
    return best


def main():
    results = {}
    for ds in DATASETS:
        seed_results = []
        for seed in SEEDS:
            surf = SURF_DIR / f'{ds}_s{seed}.npz'
            if not surf.exists():
                print(f"  SKIP {surf.name} (not found)", flush=True)
                continue
            d = np.load(surf, allow_pickle=True)
            scores = surface_to_score(d['p_surface'])
            y_true = surface_to_label(d['y_surface'])
            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                print(f"  SKIP {surf.name} (degenerate labels)", flush=True)
                continue
            m = best_pa_f1(scores, y_true)
            seed_results.append({
                'seed': seed,
                'f1_pa': m['f1_pa'],
                'f1_non_pa': m['f1_non_pa'],
                'precision_pa': m['precision_pa'],
                'recall_pa': m['recall_pa'],
                'precision_non_pa': m['precision_non_pa'],
                'recall_non_pa': m['recall_non_pa'],
                'auroc': m.get('auroc'),
                'auc_pr': m.get('auc_pr'),
                'threshold_pct': m['threshold_percentile'],
                'threshold_val': m.get('threshold_used'),
                'prevalence': m.get('prevalence'),
            })
            print(f"  {ds} s{seed}: PA-F1={m['f1_pa']:.3f}  "
                  f"non-PA-F1={m['f1_non_pa']:.3f} "
                  f"(pct={m['threshold_percentile']})", flush=True)

        if seed_results:
            pas = [r['f1_pa'] for r in seed_results]
            nonpas = [r['f1_non_pa'] for r in seed_results]
            results[ds] = {
                'per_seed': seed_results,
                'agg': {
                    'f1_pa_mean': float(np.mean(pas)),
                    'f1_pa_std': float(np.std(pas)),
                    'f1_non_pa_mean': float(np.mean(nonpas)),
                    'f1_non_pa_std': float(np.std(nonpas)),
                    'n_seeds': len(seed_results),
                },
            }
            print(f"  {ds} aggregate: PA-F1={np.mean(pas):.3f}+/-{np.std(pas):.3f}  "
                  f"non-PA-F1={np.mean(nonpas):.3f}+/-{np.std(nonpas):.3f}",
                  flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_PATH}", flush=True)


if __name__ == '__main__':
    main()
