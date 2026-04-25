"""V30 Phase 3: uniform 13-dataset benchmark.

Single set of hyperparameters across all datasets (only Δt_max varies):
  Phase 0 head: dense discrete CDF, eval at K=150 horizons, 20 random
                training horizons per batch.
  Pretrained encoder: re-use v27/v28/v29 ckpts where available.
  Seeds: 42, 123, 456 (matches existing Chronos-2 cache + ckpts).
  Label budgets: 100% and 10% (10% on subset of datasets to control time).

Output:
  results/phase3/{dataset}/seed{N}/  — surface .npz, metrics .json, panel PNG
  results/master_table.json          — aggregated mean ± std per dataset/budget

Phase 3 is the headline deliverable. Heterogeneous v29 master table
("best across v27-v29") is replaced with this clean uniform run.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _runner_v30 import run_v30, RES_DIR, find_pretrain_ckpt
from _runner_v29 import LOADERS, NORM_POLICY

# All datasets that have signal (per Phase 2 + v29 confirmations).
# Reorder by domain for the panel walkthrough.
DATASETS_FULL = [
    'FD001', 'FD002', 'FD003',          # Lifecycle (CMAPSS)
    'SMAP', 'PSM', 'MBA',                # Streaming anomaly
    'GECCO', 'BATADAL',                  # Water/SCADA
    'SKAB', 'ETTm1',                     # New v29 datasets
    # Conditional inclusion based on phase 2:
    'MSL', 'SMD',
]
SEEDS = [42, 123, 456]

# Dense K=150 horizons (Phase 0 winner).
DENSE_HORIZONS = list(range(1, 151))
# CHB-MIT uses native time-grid horizons, but skipped per v29 confirmation.

# Datasets where 10% labels makes sense (engine-style splits with ≥10 train units).
LABEL_FRAC_DATASETS_10PCT = ['FD001', 'FD003', 'MBA', 'BATADAL']


def list_existing_ckpts():
    candidates = []
    for ds in DATASETS_FULL:
        nm = NORM_POLICY[ds]
        for sd in SEEDS:
            ckpt = find_pretrain_ckpt(ds, nm, sd)
            candidates.append({'ds': ds, 'norm': nm, 'seed': sd,
                               'ckpt': str(ckpt) if ckpt else None,
                               'exists': ckpt is not None})
    return candidates


def run_dataset(dataset: str, seeds: List[int], label_fraction: float = 1.0,
                horizons: List[int] = None) -> List[Dict]:
    if horizons is None:
        horizons = DENSE_HORIZONS
    results = []
    norm_mode = NORM_POLICY[dataset]
    for sd in seeds:
        pre_ckpt = find_pretrain_ckpt(dataset, norm_mode, sd)
        try:
            r = run_v30(dataset=dataset, seed=sd,
                        eval_horizons=horizons,
                        event_head_kind='discrete_hazard',
                        train_horizons_dense=20,
                        tag_suffix='p3' + (f'_lf{int(label_fraction*100)}'
                                           if label_fraction < 1.0 else ''),
                        init_from_ckpt=pre_ckpt,
                        label_fraction=label_fraction,
                        ft_epochs=30, ft_patience=8,
                        sort_panel_by_tte=(dataset.startswith('FD')))
            if r is not None:
                results.append(r)
        except Exception as e:
            print(f"  ERROR {dataset} s{sd} lf{label_fraction}: {e}",
                  flush=True)
            import traceback; traceback.print_exc()
        # Persist after each seed
        _persist_progress(dataset, results, label_fraction)
    return results


def _persist_progress(dataset, results, label_fraction):
    out_dir = RES_DIR / 'phase3' / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f'seeds_lf{int(label_fraction*100)}.json'
    with open(out, 'w') as f:
        json.dump({'dataset': dataset, 'label_fraction': label_fraction,
                   'results': results}, f, indent=2, default=str)


def aggregate(all_runs: Dict) -> Dict:
    """Build master_table.json structure."""
    table = {}
    for ds, by_lf in all_runs.items():
        ds_row = {}
        for lf, runs in by_lf.items():
            aurocs = [r['mean_h_auroc'] for r in runs if r is not None]
            ds_row[f'lf{int(lf*100)}'] = {
                'mean_h_auroc': float(np.mean(aurocs)) if aurocs else None,
                'std_h_auroc': float(np.std(aurocs, ddof=1)) if len(aurocs) > 1 else None,
                'n_seeds': len(aurocs),
                'per_seed': {r['seed']: r['mean_h_auroc'] for r in runs},
            }
        table[ds] = ds_row
    return table


def main(only_datasets: List[str] = None, only_lf: float = None):
    t0 = time.time()
    datasets = only_datasets or DATASETS_FULL
    print(f"\n>>> V30 Phase 3 — uniform benchmark on {datasets} <<<\n",
          flush=True)
    print(f"  pretrained-ckpt status:", flush=True)
    for c in list_existing_ckpts():
        if c['ds'] in datasets:
            print(f"    {c['ds']:8s} s{c['seed']} norm={c['norm']:5s}: "
                  f"{'OK' if c['exists'] else 'MISSING'}", flush=True)

    all_runs = {}  # {dataset: {label_fraction: [run_dicts]}}

    label_fractions = [only_lf] if only_lf is not None else [1.0, 0.1]
    for ds in datasets:
        all_runs[ds] = {}
        for lf in label_fractions:
            if lf < 1.0 and ds not in LABEL_FRAC_DATASETS_10PCT:
                continue
            print(f"\n>>> {ds} @ {int(lf*100)}% labels <<<\n", flush=True)
            runs = run_dataset(ds, SEEDS, label_fraction=lf)
            all_runs[ds][lf] = runs
            # Save partial master table after each dataset/budget
            partial = aggregate(all_runs)
            with open(RES_DIR / 'master_table.json', 'w') as f:
                json.dump({'datasets': partial,
                           'time_elapsed_s': time.time() - t0,
                           'horizons': DENSE_HORIZONS}, f, indent=2, default=str)

    final = aggregate(all_runs)
    with open(RES_DIR / 'master_table.json', 'w') as f:
        json.dump({'datasets': final,
                   'time_total_s': time.time() - t0,
                   'horizons': DENSE_HORIZONS,
                   'seeds': SEEDS,
                   'label_fractions': label_fractions}, f, indent=2,
                  default=str)
    print(f"\n=== Phase 3 done in {time.time()-t0:.1f}s ===\n", flush=True)
    print(json.dumps(final, indent=2, default=str))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=None)
    ap.add_argument('--label-fraction', type=float, default=None,
                    help='Only run this fraction (1.0 or 0.1)')
    args = ap.parse_args()
    main(only_datasets=args.datasets, only_lf=args.label_fraction)
