"""
V18 Phase 4e: Statistical significance of the 5% label-efficiency crossover.

Reviewer 2 round-2 suggested: "Run paired Wilcoxon on matched engine subsets."
We don't have matched STAR runs, so instead we compute:
  - Unpaired Welch's t-test on FAM vs STAR 5%-label means
  - Bootstrap 95% CI on the (STAR - FAM) RMSE delta
  - Welch's t-test and bootstrap CI at 10% and 20% as well, for comparison

FAM numbers come from v18/phase1b_e2e_results.json (5 seeds each budget).
STAR numbers come from v11/finetune_results_v2_full.json (public in-repo) or
are taken from the v11/v14 documented runs. We ONLY use STAR numbers that
were run locally in this repo with the same preprocessing pipeline, NOT
paper numbers.

Output: experiments/v18/phase4e_significance.json
"""

import sys, json
from pathlib import Path
import numpy as np
from scipy import stats

V18 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v18')


# FAM E2E and Frozen per-seed test RMSE from Phase 1b
FAM_E2E = {
    '100%': [15.09, 15.05, 14.97, 15.28, 15.03],
    '20%':  [17.48, 18.99, 17.72, 17.95, 17.12],
    '10%':  [20.36, 21.22, 19.68, 19.73, 17.14],
    '5%':   [21.19, 22.97, 20.78, 23.51, 19.31],
}
FAM_FROZEN = {
    '100%': [18.89, 17.27, 15.21, 17.29, 16.36],
    '20%':  [18.42, 19.17, 19.82, 19.82, 20.43],
    '10%':  [20.59, 21.84, 21.40, 20.42, 19.30],
    '5%':   [22.32, 21.09, 21.48, 22.40, 20.04],
}

# STAR per-seed numbers documented in the paper (from our replication).
# Paper body cites STAR 5%: 24.55 +/- 6.4, 10%: 18.72 +/- 2.8, 20%: 17.74 +/- 3.6,
# 100%: 12.19 +/- 0.55. These are used in Table 2 of paper-neurips/paper.tex.
# Without matched-subset re-runs we only have summary stats + implied 5 seeds.
# Approximate per-seed by sampling uniformly from mean +/- std assumption.
# NOTE: this is a *lower bound on significance* because we can't test matched
# pairs. A real paired test requires matched engine subsets.
STAR_SUMMARY = {
    '100%': {'mean': 12.19, 'std': 0.55, 'n': 5},
    '20%':  {'mean': 17.74, 'std': 3.60, 'n': 5},
    '10%':  {'mean': 18.72, 'std': 2.80, 'n': 5},
    '5%':   {'mean': 24.55, 'std': 6.40, 'n': 5},
}


def welch_t(a, b_summary):
    """Welch's unpaired t-test. 'a' is an array of per-seed values; 'b_summary'
    is {mean, std, n} for the comparator. We construct 'b' as an approximation
    by sampling from N(mean, std) - this is crude; the point is to report
    a p-value that reflects the observed spread, not to claim precision."""
    rng = np.random.RandomState(0)
    b = rng.normal(b_summary['mean'], b_summary['std'], size=b_summary['n'])
    t, p = stats.ttest_ind(a, b, equal_var=False)
    # Welch's t-stat from summary stats (independent of the sampled 'b')
    a_mean, a_std = np.mean(a), np.std(a, ddof=1)
    b_mean, b_std = b_summary['mean'], b_summary['std']
    sa = (a_std ** 2) / len(a); sb = (b_std ** 2) / b_summary['n']
    t_true = (a_mean - b_mean) / np.sqrt(sa + sb)
    df_num = (sa + sb) ** 2
    df_den = (sa ** 2) / (len(a) - 1) + (sb ** 2) / (b_summary['n'] - 1)
    df = df_num / max(df_den, 1e-12)
    p_true = 2 * stats.t.sf(abs(t_true), df)
    return {
        't_stat': float(t_true),
        'df': float(df),
        'p_two_sided': float(p_true),
        'a_mean': float(a_mean), 'a_std': float(a_std), 'a_n': len(a),
        'b_mean': b_mean, 'b_std': b_std, 'b_n': b_summary['n'],
        'delta_mean_a_minus_b': float(a_mean - b_mean),
    }


def bootstrap_ci(a, b_summary, n_boot=10000, alpha=0.05, seed=0):
    """Bootstrap 95% CI on the (a_mean - b_mean) difference. 'a' is per-seed
    observations; 'b' is approximated by N(mean, std) since per-seed STAR
    values are not in-repo.
    """
    rng = np.random.RandomState(seed)
    n_a = len(a); n_b = b_summary['n']
    diffs = []
    for _ in range(n_boot):
        a_bs = rng.choice(a, size=n_a, replace=True)
        b_bs = rng.normal(b_summary['mean'], b_summary['std'], size=n_b)
        diffs.append(a_bs.mean() - b_bs.mean())
    diffs = np.array(diffs)
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return {'ci_low': lo, 'ci_high': hi, 'mean_diff': float(diffs.mean()),
            'p_bootstrap_two_sided': float(2 * min(
                (diffs > 0).mean(), (diffs < 0).mean()))}


def main():
    print("V18 Phase 4e: significance of label-efficiency crossover", flush=True)
    print(f"Disclaimer: STAR numbers are from our public replication with "
          f"5 seeds, mean/std only. STAR per-seed values were not saved; we "
          f"approximate the STAR sample by drawing from N(mean, std). A truly "
          f"paired test requires matched engine subsets (not done this session).", flush=True)

    results = {'fam_e2e_vs_star': {}, 'fam_frozen_vs_star': {}}

    print("\n--- FAM E2E vs STAR (Welch's unpaired t, bootstrap CI) ---", flush=True)
    for budget in ['100%', '20%', '10%', '5%']:
        fam = FAM_E2E[budget]
        star = STAR_SUMMARY[budget]
        w = welch_t(np.array(fam), star)
        bs = bootstrap_ci(np.array(fam), star)
        r = {**w, **{'bootstrap_' + k: v for k, v in bs.items()}}
        results['fam_e2e_vs_star'][budget] = r
        sig = 'SIG' if w['p_two_sided'] < 0.05 else 'n.s.'
        print(f"  {budget}: FAM E2E {w['a_mean']:.2f} vs STAR {w['b_mean']:.2f}   "
              f"delta={w['delta_mean_a_minus_b']:+.2f}   t={w['t_stat']:+.2f}   "
              f"p={w['p_two_sided']:.3f}   CI=[{bs['ci_low']:+.2f}, {bs['ci_high']:+.2f}]"
              f"   [{sig}]", flush=True)

    print("\n--- FAM Frozen vs STAR ---", flush=True)
    for budget in ['100%', '20%', '10%', '5%']:
        fam = FAM_FROZEN[budget]
        star = STAR_SUMMARY[budget]
        w = welch_t(np.array(fam), star)
        bs = bootstrap_ci(np.array(fam), star)
        r = {**w, **{'bootstrap_' + k: v for k, v in bs.items()}}
        results['fam_frozen_vs_star'][budget] = r
        sig = 'SIG' if w['p_two_sided'] < 0.05 else 'n.s.'
        print(f"  {budget}: FAM Frozen {w['a_mean']:.2f} vs STAR {w['b_mean']:.2f}   "
              f"delta={w['delta_mean_a_minus_b']:+.2f}   t={w['t_stat']:+.2f}   "
              f"p={w['p_two_sided']:.3f}   CI=[{bs['ci_low']:+.2f}, {bs['ci_high']:+.2f}]"
              f"   [{sig}]", flush=True)

    # Pairwise: FAM E2E vs FAM Frozen at each budget (actual paired data)
    print("\n--- FAM E2E vs FAM Frozen (paired, same seeds, same encoder) ---", flush=True)
    results['fam_e2e_vs_frozen'] = {}
    for budget in ['100%', '20%', '10%', '5%']:
        e2e = np.array(FAM_E2E[budget])
        fr = np.array(FAM_FROZEN[budget])
        t, p = stats.ttest_rel(e2e, fr)
        results['fam_e2e_vs_frozen'][budget] = {
            't_stat': float(t),
            'p_two_sided': float(p),
            'delta_mean_e2e_minus_frozen': float(e2e.mean() - fr.mean()),
        }
        sig = 'SIG' if p < 0.05 else 'n.s.'
        print(f"  {budget}: E2E {e2e.mean():.2f} vs Frozen {fr.mean():.2f}   "
              f"delta={e2e.mean() - fr.mean():+.2f}   paired t={t:+.2f}   "
              f"p={p:.3f}   [{sig}]", flush=True)

    summary = {
        'config': 'v18_phase4e_significance',
        'fam_per_seed_source': 'v18/phase1b_e2e_results.json',
        'star_source': 'v11/finetune_results_v2_full.json (summary mean+-std only)',
        'methodology_note': (
            "STAR per-seed values were not saved; we approximate the STAR "
            "sample by drawing from N(mean, std) with n=5. Tests are Welch's "
            "unpaired (ignores matched-subset structure) + bootstrap 95% CI. "
            "Paired Wilcoxon on matched engine subsets requires re-running "
            "STAR with matched engine seeds - not done this session."),
        'n_boot': 10000,
        'alpha': 0.05,
        'results': results,
    }
    out = V18 / 'phase4e_significance.json'
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nSaved: {out}", flush=True)


if __name__ == '__main__':
    main()
