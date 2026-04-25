"""V29 Phase 2: transformer-predictor ablation.

Runs FD001, FD003, MBA × {MLP, transformer} predictor × 3 seeds. Same
v29 runner so only the predictor differs (encoder/target/event-head all
identical). Reports mean per-horizon AUROC and stores per-horizon
correlation matrices to detect whether the transformer makes per-horizon
predictions less correlated (more horizon-specific) than the MLP.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

V29_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v29')
sys.path.insert(0, str(V29_DIR))

from _runner_v29 import run_and_persist, RES_DIR


DATASETS = ['FD001', 'FD003', 'MBA']
SEEDS = [42, 123, 456]
PREDICTORS = ['mlp', 'transformer']


def main():
    summary = {}
    for ds in DATASETS:
        summary[ds] = {}
        for pk in PREDICTORS:
            out_json = RES_DIR / f'phase2_{ds}_{pk}.json'
            print(f"\n{'#'*70}\n# Phase 2: {ds}  predictor={pk}  seeds={SEEDS}"
                  f"\n{'#'*70}", flush=True)
            results = run_and_persist(ds, seeds=SEEDS, out_json=out_json,
                                      predictor_kind=pk,
                                      pre_epochs=30, ft_epochs=30)
            aurocs = [r['mean_h_auroc'] for r in results]
            auprcs = [r['pooled_auprc'] for r in results]
            summary[ds][pk] = {
                'mean_h_auroc_mean': float(np.mean(aurocs)) if aurocs else None,
                'mean_h_auroc_std': float(np.std(aurocs, ddof=1))
                if len(aurocs) > 1 else None,
                'pooled_auprc_mean': float(np.mean(auprcs)) if auprcs else None,
                'n_seeds': len(results),
            }

    print("\n" + "=" * 80)
    print("PHASE 2 SUMMARY")
    print("=" * 80)
    print(f"{'Dataset':<10} {'Predictor':<15} {'mean h-AUROC':>20} {'n':>3}")
    print("-" * 60)
    for ds in DATASETS:
        for pk in PREDICTORS:
            s = summary[ds][pk]
            if s['mean_h_auroc_mean'] is None:
                print(f"{ds:<10} {pk:<15} {'(no result)':>20}")
                continue
            std = (f"±{s['mean_h_auroc_std']:.3f}"
                   if s['mean_h_auroc_std'] is not None else "")
            print(f"{ds:<10} {pk:<15} "
                  f"{s['mean_h_auroc_mean']:>14.4f}{std:>6} "
                  f"{s['n_seeds']:>3}")

    out_summary = RES_DIR / 'phase2_summary.json'
    with open(out_summary, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out_summary}", flush=True)


if __name__ == '__main__':
    main()
