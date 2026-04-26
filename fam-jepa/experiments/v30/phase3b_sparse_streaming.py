"""V30 Phase 3b: sparse-K=8 fallback for streaming-anomaly datasets.

Phase 3 dense K=150 helps lifecycle (FD*) but hurts datasets where the
signal is at SHORT horizons (MBA -0.104, SKAB -0.052, GECCO -0.040,
ETTm1 -0.036, BATADAL -0.030). Re-run those 5 datasets with sparse
horizons {1,5,10,20,50,100,150,200} (the v29 grid) to see if the
regression is recoverable.

Output:
  results/phase3b_sparse_streaming.json
  surfaces/{ds}_revin_discrete_hazard_p3b_s{N}.npz
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _runner_v30 import run_v30, RES_DIR, find_pretrain_ckpt
from _runner_v29 import LOADERS, NORM_POLICY

DATASETS = ['MBA', 'SKAB', 'GECCO', 'ETTm1', 'BATADAL']
SEEDS = [42, 123, 456]
SPARSE_HORIZONS = [1, 5, 10, 20, 50, 100, 150, 200]


def main():
    t0 = time.time()
    results = {ds: {} for ds in DATASETS}
    for ds in DATASETS:
        nm = NORM_POLICY[ds]
        print(f"\n>>> {ds} sparse-h Phase 3b <<<\n", flush=True)
        for sd in SEEDS:
            pre_ckpt = find_pretrain_ckpt(ds, nm, sd)
            try:
                r = run_v30(dataset=ds, seed=sd,
                            eval_horizons=SPARSE_HORIZONS,
                            event_head_kind='discrete_hazard',
                            train_horizons_dense=0,   # use eval horizons every batch
                            tag_suffix='p3b',
                            init_from_ckpt=pre_ckpt,
                            ft_epochs=30, ft_patience=8)
                if r is not None:
                    results[ds][sd] = r
            except Exception as e:
                print(f"  ERROR {ds} s{sd}: {e}", flush=True)
                import traceback; traceback.print_exc()

    # Aggregate
    summary = {'datasets': {}, 'horizons': SPARSE_HORIZONS,
               'time_total_s': time.time() - t0}
    for ds, by_seed in results.items():
        if not by_seed:
            summary['datasets'][ds] = {'mean': None, 'std': None, 'n': 0}
            continue
        aurocs = [r['mean_h_auroc'] for r in by_seed.values()]
        summary['datasets'][ds] = {
            'mean': float(np.mean(aurocs)),
            'std': float(np.std(aurocs, ddof=1)) if len(aurocs) > 1 else None,
            'n': len(aurocs),
            'per_seed': {sd: r['mean_h_auroc'] for sd, r in by_seed.items()},
        }
    out = RES_DIR / 'phase3b_sparse_streaming.json'
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nwrote {out}\n", flush=True)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
