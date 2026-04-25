"""V29 self-check follow-up: paired t-test on phase 2 ablation.

Per ml-researcher critique #9: the "statistically tied" claim is verbal
without a formal test. Run paired t-tests on the per-seed results from
phase2_<DS>_{mlp,transformer}.json so the conclusion is statistical
not narrative.
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

V29 = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v29')
RES = V29 / 'results'

DATASETS = ['FD001', 'FD003', 'MBA']
SEEDS = [42, 123, 456]


def load_per_seed(ds: str, predictor: str) -> dict:
    p = RES / f'phase2_{ds}_{predictor}.json'
    if not p.exists():
        return {}
    d = json.load(open(p))
    out = {}
    for r in d.get('results', []):
        out[r['seed']] = r['mean_h_auroc']
    return out


def main():
    print("Phase 2 paired t-tests (per-seed mean h-AUROC, transformer - MLP)")
    print(f"{'Dataset':<8} {'MLP mean±std':>20} {'Xpred mean±std':>20} "
          f"{'Δ paired':>12} {'paired t':>10} {'p-value':>10} verdict")
    print("-" * 100)

    results = {}
    for ds in DATASETS:
        mlp = load_per_seed(ds, 'mlp')
        xfm = load_per_seed(ds, 'transformer')
        common_seeds = sorted(set(mlp) & set(xfm))
        m_vals = np.array([mlp[s] for s in common_seeds])
        x_vals = np.array([xfm[s] for s in common_seeds])
        deltas = x_vals - m_vals
        if len(common_seeds) >= 2:
            t_stat, p_val = stats.ttest_rel(x_vals, m_vals)
        else:
            t_stat, p_val = float('nan'), float('nan')
        verdict = ('xpred wins (p<0.05)' if p_val < 0.05 and t_stat > 0 else
                   ('mlp wins (p<0.05)' if p_val < 0.05 and t_stat < 0 else
                    'no significant difference'))
        results[ds] = {
            'seeds': common_seeds,
            'mlp_mean': float(m_vals.mean()),
            'mlp_std': float(m_vals.std(ddof=1)),
            'xpred_mean': float(x_vals.mean()),
            'xpred_std': float(x_vals.std(ddof=1)),
            'delta_mean': float(deltas.mean()),
            't_stat': float(t_stat) if not np.isnan(t_stat) else None,
            'p_value': float(p_val) if not np.isnan(p_val) else None,
            'verdict': verdict,
        }
        print(f"{ds:<8} {m_vals.mean():>10.4f}±{m_vals.std(ddof=1):.4f}  "
              f"{x_vals.mean():>12.4f}±{x_vals.std(ddof=1):.4f}  "
              f"{deltas.mean():+12.4f}  "
              f"{t_stat:>+10.3f}  {p_val:>10.4f}  {verdict}")

    out = RES / 'phase2_paired_ttest.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
