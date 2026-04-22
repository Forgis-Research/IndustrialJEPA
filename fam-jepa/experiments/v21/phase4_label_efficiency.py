"""V21 Phase 4: Label-Efficiency Curve with AUPRC.

Extends v20's label-efficiency study (per-window F1w) to the v21 AUPRC
primary. Compares pred-FT vs E2E on FD001 at label budgets
[100, 50, 20, 10, 5]% with 5 seeds.

Key question: does the pred-FT ≤10% crossover observed with F1w
(v20 Phase 8: p=0.023 at 10%) survive under AUPRC?

5 seeds × 5 budgets × 2 modes = 50 runs. ~5 min on A10G (per Phase 2 speed).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path('/home/sagemaker-user/IndustrialJEPA')
FAM = ROOT / 'fam-jepa'
V21 = FAM / 'experiments' / 'v21'
sys.path.insert(0, str(V21))
sys.path.insert(0, str(FAM / 'experiments' / 'v11'))
sys.path.insert(0, str(FAM))

from cmapss_runner import run_single, aggregate  # noqa: E402
from data_utils import load_cmapss_subset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', default='FD001')
    ap.add_argument('--budgets', nargs='+', type=float,
                    default=[1.0, 0.5, 0.2, 0.1, 0.05])
    ap.add_argument('--seeds', nargs='+', type=int,
                    default=[0, 1, 2, 3, 4])
    ap.add_argument('--modes', nargs='+', default=['pred_ft', 'e2e'])
    ap.add_argument('--out', default=str(V21 / 'phase4_label_efficiency.json'))
    args = ap.parse_args()

    t0 = time.time()
    out_path = Path(args.out)
    data = load_cmapss_subset(args.subset)
    results = {}
    t_start = time.time()
    print(f'=== Phase 4 Label Efficiency on {args.subset} ===', flush=True)
    print(f'{len(args.modes)} modes x {len(args.budgets)} budgets x '
          f'{len(args.seeds)} seeds = {len(args.modes)*len(args.budgets)*len(args.seeds)} runs', flush=True)

    for mode in args.modes:
        for b in args.budgets:
            key = f'{mode}@{b}'
            results[key] = []
            for seed in args.seeds:
                try:
                    r = run_single(data, args.subset, mode, b, seed)
                    results[key].append(r)
                    print(f'  [{mode:8s} b={b*100:5.1f}% s={seed}] '
                          f"AUPRC={r['primary']['auprc']:.3f} "
                          f"AUROC={r['primary']['auroc']:.3f} "
                          f"RMSE={r['legacy']['rmse_expected']:.2f} "
                          f"({r['runtime_s']:.0f}s)", flush=True)
                except Exception as e:
                    print(f'  ERR [{mode} b={b} s={seed}]: {e}', flush=True)

            # incremental save
            agg = {k: aggregate(v) for k, v in results.items()}
            with open(out_path, 'w') as f:
                json.dump({'subset': args.subset, 'budgets': args.budgets,
                           'seeds': args.seeds, 'modes': args.modes,
                           'results': agg,
                           'runtime_min': (time.time() - t0) / 60},
                          f, indent=2, default=float)

    print('\n\nPhase 4 complete. Paired test at each budget:')
    _paired_ttest(results, args.modes, args.budgets)
    print(f'DONE in {(time.time()-t_start)/60:.1f} min. Saved: {out_path}')


def _paired_ttest(results, modes, budgets):
    if set(modes) != {'pred_ft', 'e2e'}:
        return
    from scipy.stats import ttest_rel, wilcoxon
    print(f"{'Budget':>8s} | {'pred_ft AUPRC':>22s} | {'E2E AUPRC':>22s} | "
          f"{'Δ':>7s} | {'t':>7s} | {'p':>7s} | {'wilcoxon p':>11s}")
    print('-' * 100)
    for b in budgets:
        p_runs = results.get(f'pred_ft@{b}', [])
        e_runs = results.get(f'e2e@{b}', [])
        # Align on seeds
        p_map = {r['seed']: r['primary']['auprc'] for r in p_runs}
        e_map = {r['seed']: r['primary']['auprc'] for r in e_runs}
        seeds = sorted(set(p_map) & set(e_map))
        if len(seeds) < 2:
            continue
        pv = np.array([p_map[s] for s in seeds])
        ev = np.array([e_map[s] for s in seeds])
        delta = pv - ev
        try:
            tres = ttest_rel(pv, ev)
            t, p = float(tres.statistic), float(tres.pvalue)
        except Exception:
            t, p = float('nan'), float('nan')
        try:
            wres = wilcoxon(pv, ev)
            wp = float(wres.pvalue)
        except Exception:
            wp = float('nan')
        print(f'{b*100:>7.1f}% | {pv.mean():.3f}±{pv.std(ddof=1):.3f}     ({len(pv)}s) '
              f'| {ev.mean():.3f}±{ev.std(ddof=1):.3f}     ({len(ev)}s) '
              f'| {delta.mean():+.3f} | {t:+.3f} | {p:.3f} | {wp:.3f}')


if __name__ == '__main__':
    main()
