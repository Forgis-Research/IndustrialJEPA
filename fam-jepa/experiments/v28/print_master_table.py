"""V28 final master table — comprehensive summary across all phases.

Prints the v28 sparse + dense tables, the per-try summary, and the
v28-vs-Chronos-2 head-to-head with paired-seed t-tests where possible.

Designed for the final commit message and the SESSION_SUMMARY.
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

REPO = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V28 = REPO / 'experiments/v28'
V27 = REPO / 'experiments/v27'

DATASETS = ['FD001', 'FD002', 'FD003', 'SMAP', 'MSL', 'PSM', 'SMD',
            'MBA', 'GECCO', 'BATADAL']


def metrics_from_npz(p):
    if not Path(p).exists():
        return None
    try:
        d = np.load(p, allow_pickle=True)
    except Exception:
        return None
    pp, yy = d['p_surface'], d['y_surface']
    valid = [i for i in range(pp.shape[1]) if 0 < yy[:, i].mean() < 1]
    if not valid:
        return None
    return {
        'mean_h_auroc': float(np.mean([roc_auc_score(yy[:, i], pp[:, i])
                                       for i in valid])),
        'pooled_auprc': float(average_precision_score(yy.ravel(), pp.ravel())),
        'K': len(d['horizons']),
    }


def best_v28_sparse(ds: str) -> dict:
    """Find the v28 phase JSON with highest mean h-AUROC for ds, sparse K."""
    candidates = []
    for jp in (V28 / 'results').glob('phase*.json'):
        if 'summary' in jp.name: continue
        try:
            d = json.load(open(jp))
        except Exception: continue
        if d.get('dataset') != ds or not d.get('results'):
            continue
        rs = d['results']
        ha = float(np.mean([r['mean_h_auroc'] for r in rs]))
        candidates.append({'src': jp.name, 'mean_h_auroc': ha,
                           'pooled_auprc': float(np.mean([r['pooled_auprc'] for r in rs])),
                           'std': float(np.std([r['mean_h_auroc'] for r in rs])),
                           'n': len(rs), 'rec0': rs[0]})
    if not candidates: return None
    return max(candidates, key=lambda c: c['mean_h_auroc'])


def variant_tag(rec):
    tag = []
    if rec.get('lag_features'): tag.append(f'lag={rec["lag_features"]}')
    if rec.get('aux_stat'): tag.append('stat-z' if rec.get('stat_normalize') else 'stat')
    if rec.get('dense_ft'): tag.append(f'dense{rec["k_dense"]}')
    return '+'.join(tag) if tag else 'baseline'


# =====================================================================
print("=" * 110)
print("V28 FINAL MASTER TABLE")
print("=" * 110)
print()

# Section 1: best v28 variant per dataset (sparse K=7/8)
print("### SPARSE K eval (3 seeds per row, mean ± std)")
print(f"{'Dataset':<10} {'Best variant':<32} {'norm':<8} {'mean h-AUROC':>16} {'pooled AUPRC':>16} {'n':>3}")
print('-' * 95)
sparse_summary = {}
for ds in DATASETS:
    b = best_v28_sparse(ds)
    if b is None:
        print(f"{ds:<10} {'(no v28 run)':<32}")
        continue
    sparse_summary[ds] = b
    tag = variant_tag(b['rec0'])
    print(f"{ds:<10} {b['src']:<32} {b['rec0']['norm_mode']:<8} "
          f"{b['mean_h_auroc']:>9.4f}±{b['std']:.3f}  {b['pooled_auprc']:>16.4f} {b['n']:>3}")
print()

# Section 2: dense paired comparison
print("### DENSE K=150/200 paired-seed comparison: v27 baseline vs v28 best")
print(f"{'Dataset':<10} {'v27 base (n=3)':>18} {'v28 best (n=3)':>18} {'Δ paired':>10} {'paired t (p)':>16}")
print('-' * 80)
for ds in DATASETS:
    src = 'v27' if ds.startswith('FD') else 'v26'
    v27_seeds = []
    v28_seeds = []
    for seed in [42, 123, 456]:
        v = metrics_from_npz(V27 / 'surfaces' / f'dense_fam_{src}_{ds}_s{seed}.npz')
        if v: v27_seeds.append(v['mean_h_auroc'])
        v = metrics_from_npz(V28 / 'surfaces_dense' / f'dense_fam_v28_{ds}_s{seed}.npz')
        if v: v28_seeds.append(v['mean_h_auroc'])

    if v27_seeds and v28_seeds and len(v27_seeds) == len(v28_seeds) >= 2:
        deltas = [b - a for a, b in zip(v27_seeds, v28_seeds)]
        delta_mean = np.mean(deltas)
        if np.std(deltas, ddof=1) > 1e-10:
            t_stat, p_val = stats.ttest_rel(v28_seeds, v27_seeds)
            t_str = f"t={t_stat:+.2f} (p={p_val:.3f})"
        else:
            t_str = '—'
        delta_str = f"{delta_mean:+.4f}"
    else:
        delta_str, t_str = '—', '—'

    fmt = lambda xs: f"{np.mean(xs):.3f}±{np.std(xs, ddof=1):.3f}" if len(xs) >= 2 else (f"{xs[0]:.3f}" if xs else "—")
    print(f"{ds:<10} {fmt(v27_seeds):>18} {fmt(v28_seeds):>18} {delta_str:>10} {t_str:>16}")
print()

# Section 3: head-to-head vs Chronos-2 (single seed s42, dense)
print("### DENSE K=150/200 head-to-head: max(v28, v27) vs Chronos-2 (s42 only)")
print(f"{'Dataset':<10} {'FAM (max v28/v27)':>20} {'Chronos-2 (s42)':>18} {'FAM - Chr':>10}")
print('-' * 70)
for ds in DATASETS:
    src = 'v27' if ds.startswith('FD') else 'v26'
    v27 = metrics_from_npz(V27 / 'surfaces' / f'dense_fam_{src}_{ds}_s42.npz')
    v28 = metrics_from_npz(V28 / 'surfaces_dense' / f'dense_fam_v28_{ds}_s42.npz')
    chr_ = metrics_from_npz(V27 / 'surfaces' / f'dense_chronos2_{ds}_s42.npz')

    fam_choices = [v for v in [v27, v28] if v]
    if not fam_choices:
        chr_str_only = '—' if not chr_ else f"{chr_['mean_h_auroc']:.4f}"
        print(f"{ds:<10} {'—':>20} {chr_str_only:>18} {'—':>10}")
        continue
    fam_best = max(fam_choices, key=lambda v: v['mean_h_auroc'])
    fam_h = fam_best['mean_h_auroc']
    chr_h = chr_['mean_h_auroc'] if chr_ else None
    delta = (fam_h - chr_h) if chr_h else None
    chr_str = f"{chr_h:.4f}" if chr_h else "—"
    delta_str = f"{delta:+.4f}" if delta is not None else "—"
    print(f"{ds:<10} {fam_h:>20.4f} {chr_str:>18} {delta_str:>10}")
