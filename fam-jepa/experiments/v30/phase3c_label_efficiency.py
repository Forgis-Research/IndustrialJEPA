"""V30 Phase 3c: sub-5% label efficiency on FD001.

The Phase 1 lf10 result (FAM-predft tied FAM-mlp-rand) suggested pretraining
doesn't buy label efficiency at 10%. v20's 5% result (pred-FT 0.261 vs
scratch 0.035) suggested it does dominate at 5%. This isolates the
crossover by running 5% explicitly on FD001.

Two variants × 3 seeds × FD001 only:
  FAM-predft       : frozen FAM encoder + pretrained MLP predictor (warm)
  FAM-mlp-rand     : frozen FAM encoder + RANDOM-init MLP predictor

Sparse horizons (matches Phase 1 ablation protocol).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'v29'))

from phase1_ablation import fam_predft

DS = 'FD001'
SEEDS = [42, 123, 456]
LABEL_FRACTION = 0.05
SPARSE_H = [1, 5, 10, 20, 50, 100, 150]


def main():
    t0 = time.time()
    results = {'fam-predft': {}, 'fam-mlp-rand': {}}
    for sd in SEEDS:
        for variant, rand_init in [('fam-predft', False), ('fam-mlp-rand', True)]:
            print(f"\n>>> {variant} {DS} s{sd} lf{LABEL_FRACTION} <<<\n", flush=True)
            r = fam_predft(DS, sd, SPARSE_H,
                           label_fraction=LABEL_FRACTION,
                           random_init_predictor=rand_init,
                           ft_epochs=30)
            if r is not None:
                results[variant][sd] = r

    # Aggregate
    summary = {'dataset': DS, 'label_fraction': LABEL_FRACTION,
               'horizons': SPARSE_H, 'seeds': SEEDS,
               'time_total_s': time.time() - t0}
    for v, by_seed in results.items():
        if by_seed:
            aurocs = [r['mean_h_auroc'] for r in by_seed.values()]
            summary[v] = {
                'mean': float(np.mean(aurocs)),
                'std': float(np.std(aurocs, ddof=1)) if len(aurocs) > 1 else None,
                'n': len(aurocs),
                'per_seed': {sd: r['mean_h_auroc'] for sd, r in by_seed.items()},
            }

    pf = summary.get('fam-predft', {}).get('mean')
    mr = summary.get('fam-mlp-rand', {}).get('mean')
    if pf and mr:
        summary['delta'] = pf - mr
        summary['verdict'] = ('pred-FT dominates' if (pf - mr) > 0.05
                              else 'tie' if abs(pf - mr) < 0.02
                              else 'pred-FT slight edge')

    out = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v30'
               '/results/phase3c_label_efficiency.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nwrote {out}\n", flush=True)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
