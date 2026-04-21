"""V20 Phase 8: 10-seed FD001 10% labels for pred-FT vs E2E.

Phase 5 (5 seeds) showed the biggest pred-FT vs E2E gap at 10% labels:
+0.177 F1w. This phase doubles seed count to N=10 to power the test.
Combines phase 5 seeds 0-4 with new seeds 5-9.
"""
import sys, json, time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import ttest_rel, wilcoxon

ROOT = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')
V11 = ROOT / 'experiments' / 'v11'
V20 = ROOT / 'experiments' / 'v20'
sys.path.insert(0, str(V11)); sys.path.insert(0, str(V20)); sys.path.insert(0, str(ROOT))

from phase0_pred_ft import run_single
from data_utils import load_cmapss_subset

NEW_SEEDS = [5, 6, 7, 8, 9]
BUDGET = 0.10


def main():
    out_path = V20 / 'phase8_ten_seeds_10pct.json'
    data = load_cmapss_subset('FD001')
    t0 = time.time()
    print("=" * 80)
    print(f"V20 Phase 8: 10-seed FD001 {int(BUDGET*100)}% pred-FT vs E2E")
    print(f"  NEW seeds {NEW_SEEDS}; combines with phase 5 seeds 0-4")
    print("=" * 80, flush=True)

    new_results = {'pred_ft': [], 'e2e': []}
    for mode in ['pred_ft', 'e2e']:
        for seed in NEW_SEEDS:
            t1 = time.time()
            r = run_single(data, mode, BUDGET, seed)
            new_results[mode].append(r)
            dt = time.time() - t1
            print(f"  [{mode:8s} s={seed:2d}] "
                  f"RMSE={r['test_rmse']:.2f} F1w={r['per_window_f1_mean']:.3f} "
                  f"({dt:.0f}s)", flush=True)

    # Combine with phase 5
    with open(V20 / 'phase5_label_efficiency.json') as f:
        phase5 = json.load(f)
    ph5_pred = phase5['results']['pred_ft@0.1']['per_seed']
    ph5_e2e = phase5['results']['e2e@0.1']['per_seed']

    all_pred = ph5_pred + new_results['pred_ft']
    all_e2e = ph5_e2e + new_results['e2e']

    pred_f1 = np.array([r['per_window_f1_mean'] for r in all_pred])
    e2e_f1 = np.array([r['per_window_f1_mean'] for r in all_e2e])
    diff = pred_f1 - e2e_f1

    t_stat, p_two = ttest_rel(pred_f1, e2e_f1)
    cohen_d = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else float('nan')
    try:
        w_stat, w_p = wilcoxon(pred_f1, e2e_f1, alternative='greater')
    except Exception:
        w_stat, w_p = float('nan'), float('nan')

    print("\n" + "=" * 80)
    print(f"PHASE 8 STATISTICAL TEST (10 paired seeds, F1w, FD001 {int(BUDGET*100)}%)")
    print("=" * 80)
    print(f"pred-FT: mean {pred_f1.mean():.4f} ± {pred_f1.std(ddof=1):.4f}")
    print(f"E2E    : mean {e2e_f1.mean():.4f} ± {e2e_f1.std(ddof=1):.4f}")
    print(f"diff   : mean {diff.mean():+.4f} ± {diff.std(ddof=1):.4f}")
    print(f"paired t(9): t={t_stat:.3f}, p two-sided = {p_two:.4f}")
    print(f"  one-sided p (pred > e2e) = {p_two/2:.4f}")
    print(f"Cohen's d = {cohen_d:.3f}")
    print(f"Wilcoxon signed-rank (one-sided, pred>e2e): W={w_stat:.1f}, p={w_p:.4f}")

    n_pred_collapses = int((pred_f1 == 0).sum())
    n_e2e_collapses = int((e2e_f1 == 0).sum())
    print(f"Seed-level collapses (F1w=0): pred-FT {n_pred_collapses}/10, E2E {n_e2e_collapses}/10")

    summary = {
        'config': 'v20_phase8_ten_seeds_10pct',
        'budget': BUDGET,
        'n_seeds_total': len(all_pred),
        'phase5_seeds': [0, 1, 2, 3, 4],
        'phase8_new_seeds': NEW_SEEDS,
        'pred_ft_per_seed': pred_f1.tolist(),
        'e2e_per_seed': e2e_f1.tolist(),
        'diff_per_seed': diff.tolist(),
        'pred_ft_mean': float(pred_f1.mean()),
        'pred_ft_std': float(pred_f1.std(ddof=1)),
        'e2e_mean': float(e2e_f1.mean()),
        'e2e_std': float(e2e_f1.std(ddof=1)),
        'diff_mean': float(diff.mean()),
        'diff_std': float(diff.std(ddof=1)),
        't_stat': float(t_stat),
        'p_two_sided': float(p_two),
        'p_one_sided_pred_gt_e2e': float(p_two/2),
        'cohen_d': cohen_d,
        'wilcoxon_W': float(w_stat) if not np.isnan(w_stat) else None,
        'wilcoxon_p_one_sided': float(w_p) if not np.isnan(w_p) else None,
        'n_pred_collapses': n_pred_collapses,
        'n_e2e_collapses': n_e2e_collapses,
        'runtime_min': (time.time()-t0)/60,
        'new_results': new_results,
    }
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"Runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
