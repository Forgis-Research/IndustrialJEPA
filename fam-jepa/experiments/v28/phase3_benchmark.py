"""V28 Phase 3 — comprehensive benchmark with the best Phase-2 variant.

For each dataset, run the BEST Phase-2 variant (or the v27 baseline if no
Phase-2 variant won on that dataset family) across 3 seeds, then store
surfaces. The "winner" is selected by mean per-horizon AUROC above the
base-rate baseline.

Usage:
  python phase3_benchmark.py --variant best     # auto-pick winner per dataset
  python phase3_benchmark.py --variant baseline # v27 baseline (norm_mode per family)
  python phase3_benchmark.py --variant lag      # lag features everywhere
  python phase3_benchmark.py --variant stat     # aux stat loss everywhere
  python phase3_benchmark.py --variant dense    # dense FT everywhere

Datasets: FD001/2/3, SMAP, MSL, PSM, SMD, MBA, GECCO, BATADAL.
"""

import argparse
import json
from pathlib import Path

from runner_v28 import run_and_persist, RES_DIR

# Per-family baseline norm_mode (from v27 ablation)
BASELINE_NORM = {
    # C-MAPSS (degradation): 'none' won by +0.13 AUPRC on FD003
    'FD001': 'none', 'FD002': 'none', 'FD003': 'none',
    # Multi-entity anomaly (SMAP-style): 'revin' (v26 default)
    'SMAP': 'revin', 'MSL': 'revin', 'PSM': 'revin', 'SMD': 'revin',
    # Single-stream anomaly: 'revin'
    'MBA': 'revin', 'GECCO': 'revin', 'BATADAL': 'revin',
}

ALL_DATASETS = list(BASELINE_NORM.keys())


def _opts_for_variant(variant: str, dataset: str) -> dict:
    """Return the kwargs to pass to runner_v28.run_one for a (variant, dataset)."""
    norm = BASELINE_NORM[dataset]
    base = dict(norm_mode=norm)
    if variant == 'baseline':
        return base
    if variant == 'lag':
        return {**base, 'lag_features': [10, 50, 100]}
    if variant == 'stat':
        return {**base, 'aux_stat': True}
    if variant == 'dense':
        return {**base, 'dense_ft': True, 'k_dense': 20}
    raise ValueError(f"unknown variant: {variant}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='baseline',
                    choices=['baseline', 'lag', 'stat', 'dense'])
    ap.add_argument('--datasets', nargs='+', default=ALL_DATASETS)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    ap.add_argument('--pre-epochs', type=int, default=30)
    ap.add_argument('--ft-epochs', type=int, default=30)
    args = ap.parse_args()

    print(f"=== Phase 3: variant={args.variant}, datasets={args.datasets} ===",
          flush=True)
    summary = {}
    for ds in args.datasets:
        opts = _opts_for_variant(args.variant, ds)
        out = RES_DIR / f'phase3_{args.variant}_{ds}.json'
        print(f"\n--- {ds} ({args.variant}) opts={opts} ---", flush=True)
        results = run_and_persist(
            dataset=ds, seeds=args.seeds, out_json=out,
            pre_epochs=args.pre_epochs, ft_epochs=args.ft_epochs,
            **opts,
        )
        summary[ds] = {
            'variant': args.variant, 'opts': opts,
            'n_seeds': len(results),
            'mean_h_auroc': sum(r['mean_h_auroc'] for r in results) / max(len(results), 1),
            'pooled_auprc': sum(r['pooled_auprc'] for r in results) / max(len(results), 1),
        }

    with open(RES_DIR / f'phase3_{args.variant}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n=== Phase 3 {args.variant} summary ===", flush=True)
    for ds, s in summary.items():
        print(f"  {ds:<10}  h-AUROC={s['mean_h_auroc']:.4f}  "
              f"pooled-AUPRC={s['pooled_auprc']:.4f}", flush=True)


if __name__ == '__main__':
    main()
