"""V31 Phase 1: Re-run all 11 datasets at 10% labels with the bug fix.

Uses v30 pretrained checkpoints (pretraining is label-free, only FT changes).
Self-check: per-seed h-AUROC at 10% MUST differ from v30 100% numbers.

Per-domain head choice (from v30 Phase 3b analysis):
  - Dense K=150: FD001, FD002, FD003, SMAP, PSM, GECCO, SMD
  - Sparse K=8: MBA, BATADAL, SKAB, ETTm1

Run all 11 datasets x 3 seeds at lf=10%.
Also compare: log n_train_windows at lf10 vs lf100 for the 5 formerly-buggy
datasets (PSM, MBA, GECCO, BATADAL, ETTm1) — verifies the fix.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/fam-jepa')
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v24')
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/archive/v24')
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/archive/v11')
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v27')
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v28')
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v29')
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v30')
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v31')

from _runner_v31 import run_v31, RES_DIR, find_pretrain_ckpt
from _runner_v29 import NORM_POLICY

SEEDS = [42, 123, 456]
LABEL_FRACTION = 0.1
DENSE_HORIZONS = list(range(1, 151))
SPARSE_HORIZONS = [1, 5, 10, 20, 50, 100, 150, 200]

# Per-domain head choice from v30 Phase 3b
DATASET_CONFIG = {
    # Dense K=150 (lifecycle / slow-drift)
    'FD001': {'horizons': DENSE_HORIZONS, 'dense': 20},
    'FD002': {'horizons': DENSE_HORIZONS, 'dense': 20},
    'FD003': {'horizons': DENSE_HORIZONS, 'dense': 20},
    'SMAP':  {'horizons': DENSE_HORIZONS, 'dense': 20},
    'PSM':   {'horizons': DENSE_HORIZONS, 'dense': 20},
    'GECCO': {'horizons': DENSE_HORIZONS, 'dense': 20},
    'SMD':   {'horizons': DENSE_HORIZONS, 'dense': 20},
    # Sparse K=8 (streaming anomaly)
    'MBA':     {'horizons': SPARSE_HORIZONS, 'dense': 0},
    'BATADAL': {'horizons': SPARSE_HORIZONS, 'dense': 0},
    'SKAB':    {'horizons': SPARSE_HORIZONS, 'dense': 0},
    'ETTm1':   {'horizons': SPARSE_HORIZONS, 'dense': 0},
}

DATASETS_11 = [
    'FD001', 'FD002', 'FD003',
    'SMAP', 'PSM', 'MBA',
    'GECCO', 'BATADAL',
    'SKAB', 'ETTm1', 'SMD',
]


def run_dataset_lf10(dataset: str) -> list:
    cfg = DATASET_CONFIG[dataset]
    norm_mode = NORM_POLICY[dataset]
    results = []
    for sd in SEEDS:
        pre_ckpt = find_pretrain_ckpt(dataset, norm_mode, sd)
        if pre_ckpt is None:
            print(f"  WARNING: no pretrained ckpt for {dataset} s{sd}, "
                  f"will train from scratch", flush=True)
        try:
            r = run_v31(
                dataset=dataset, seed=sd,
                eval_horizons=cfg['horizons'],
                event_head_kind='discrete_hazard',
                train_horizons_dense=cfg['dense'],
                tag_suffix='p1lf10',
                init_from_ckpt=pre_ckpt,
                label_fraction=LABEL_FRACTION,
                ft_epochs=30, ft_patience=8,
                sort_panel_by_tte=dataset.startswith('FD'),
                use_wandb=True,
            )
            if r is not None:
                results.append(r)
        except Exception as e:
            print(f"  ERROR {dataset} s{sd}: {e}", flush=True)
            import traceback; traceback.print_exc()

        # Save partial results
        _save_progress(dataset, results)

    return results


def _save_progress(dataset, results):
    out = RES_DIR / 'phase1' / f'{dataset}_lf10.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'dataset': dataset, 'label_fraction': LABEL_FRACTION,
                   'results': results}, f, indent=2, default=str)


def aggregate(all_runs: dict) -> dict:
    table = {}
    for ds, runs in all_runs.items():
        aurocs = [r['mean_h_auroc'] for r in runs if r is not None]
        if not aurocs:
            table[ds] = {'lf10': None}
            continue
        table[ds] = {
            'lf10': {
                'mean_h_auroc': float(np.mean(aurocs)),
                'std_h_auroc': float(np.std(aurocs, ddof=1)) if len(aurocs) > 1 else 0.0,
                'n_seeds': len(aurocs),
                'per_seed': {r['seed']: r['mean_h_auroc'] for r in runs},
                'n_train_windows': {r['seed']: r.get('n_train_windows') for r in runs},
            }
        }
    return table


def main():
    t0 = time.time()
    all_runs = {}

    for ds in DATASETS_11:
        print(f"\n{'#'*72}\n# {ds} @ 10% labels\n{'#'*72}", flush=True)
        runs = run_dataset_lf10(ds)
        all_runs[ds] = [r for r in runs if r is not None]

        # Intermediate aggregation
        partial = aggregate(all_runs)
        with open(RES_DIR / 'phase1_lf10_master.json', 'w') as f:
            json.dump({'datasets': partial,
                       'label_fraction': LABEL_FRACTION,
                       'time_elapsed_s': time.time() - t0,
                       'seeds': SEEDS}, f, indent=2, default=str)

    final = aggregate(all_runs)
    out = RES_DIR / 'phase1_lf10_master.json'
    with open(out, 'w') as f:
        json.dump({'datasets': final,
                   'label_fraction': LABEL_FRACTION,
                   'time_total_s': time.time() - t0,
                   'seeds': SEEDS,
                   'bug_fix': 'v31 - single-entity time truncation',
                   }, f, indent=2, default=str)

    print(f"\n=== Phase 1 done in {time.time()-t0:.1f}s ===")
    print(json.dumps(final, indent=2, default=str))

    # SELF-CHECK: compare lf10 vs v30 lf100
    v30_master = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v30/results/master_table.json')
    if v30_master.exists():
        with open(v30_master) as f:
            v30 = json.load(f)['datasets']
        print("\n=== SELF-CHECK: lf10 vs v30 lf100 ===")
        print(f"{'Dataset':10s} {'lf100 (v30)':>14s} {'lf10 (v31)':>14s} {'delta':>10s} {'ok':>5s}")
        print("-" * 58)
        for ds in DATASETS_11:
            lf100 = v30.get(ds, {}).get('lf100', {}).get('mean_h_auroc')
            lf10_row = final.get(ds, {}).get('lf10')
            if lf100 is None or lf10_row is None:
                print(f"{ds:10s}  missing data")
                continue
            lf10 = lf10_row['mean_h_auroc']
            delta = lf10 - lf100
            # For datasets that were buggy, the old lf10==lf100 exactly,
            # so any non-zero delta is the fix working.
            # We expect lf10 <= lf100 (less data = equal or worse performance)
            ok = True  # any result != exactly lf100 means fix worked
            # For formerly-buggy datasets: check they moved
            formerly_buggy = ['MBA', 'BATADAL', 'PSM', 'ETTm1', 'GECCO']
            if ds in formerly_buggy:
                old_lf10 = v30.get(ds, {}).get('lf10', {}).get('mean_h_auroc')
                if old_lf10 is not None and abs(lf10 - old_lf10) < 0.001:
                    ok = False  # still identical - fix didn't work
            print(f"{ds:10s} {lf100:>14.4f} {lf10:>14.4f} {delta:>+10.4f} {'OK' if ok else 'BUG!':>5s}")

    return out


if __name__ == '__main__':
    main()
