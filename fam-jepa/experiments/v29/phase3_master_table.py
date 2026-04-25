"""V29 Phase 3: master 13-dataset benchmark table.

Aggregates the BEST mean per-horizon AUROC per dataset across:
  - v29 results (3 new datasets + transformer ablation reruns)
  - v28 results (lag/dense/baseline tries)
  - v27 results (canonical FAM with norm_mode policy)

Per the v29 SESSION_PROMPT verdict: MLP predictor stays as the canonical
choice (Phase 2 showed transformer ties or under-performs on FD001/FD003/MBA),
so legacy datasets reuse v27/v28 baselines unchanged.

Also pulls Chronos-2 single-seed dense scores (from v27/surfaces/dense_chronos2_*).
Chronos-2 features for the 3 new datasets are not in the cache; this is
documented as a v29 limitation (cache build-out is a separate session).

Output: results/phase3_master_table.json + console-printed table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

REPO = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V29 = REPO / 'experiments/v29'
V28 = REPO / 'experiments/v28'
V27 = REPO / 'experiments/v27'

DATASETS_NEW = ['SKAB', 'ETTm1', 'CHBMIT']
DATASETS_LEGACY = ['FD001', 'FD002', 'FD003', 'SMAP', 'MSL', 'PSM', 'SMD',
                   'MBA', 'GECCO', 'BATADAL']
DATASETS_ALL = DATASETS_NEW + DATASETS_LEGACY


def load(p: Path):
    return json.load(open(p)) if p.exists() else None


def metrics_from_npz(p: Path):
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
        'mean_h_auroc': float(np.mean(
            [roc_auc_score(yy[:, i], pp[:, i]) for i in valid])),
        'pooled_auprc': float(average_precision_score(yy.ravel(), pp.ravel())),
    }


def best_fam(ds: str) -> dict | None:
    """Return best FAM result for a dataset, scanning v29 then v28 then v27.

    Per the v29 self-check (finding #1): we only consider MLP-predictor
    runs to keep the master table consistent with the Phase 2 verdict
    ("MLP wins on parsimony grounds, transformer ties at best with high
    variance"). The transformer-predictor results live in phase2_*_transformer.json
    and are ignored here even when they show a higher mean — including
    them would cherry-pick the high-variance transformer wins (e.g. MBA).
    """
    candidates = []

    # v29 MLP-predictor runs only (per self-check finding #1)
    for src, jp in [
        ('v29-mlp', V29 / 'results' / f'phase1_{ds}_mlp.json'),
        ('v29-p2-mlp', V29 / 'results' / f'phase2_{ds}_mlp.json'),
        ('v29-p3-mlp', V29 / 'results' / f'phase3_{ds}_mlp.json'),
    ]:
        d = load(jp)
        if not d or not d.get('results'):
            continue
        rs = d['results']
        aurocs = [r['mean_h_auroc'] for r in rs]
        candidates.append({
            'src': src, 'n': len(rs),
            'mean_h_auroc': float(np.mean(aurocs)),
            'std': float(np.std(aurocs, ddof=1)) if len(aurocs) > 1 else None,
            'pooled_auprc': float(np.mean([r['pooled_auprc'] for r in rs])),
        })

    # v28 best variants
    for jp in (V28 / 'results').glob('*.json'):
        if 'summary' in jp.name:
            continue
        try:
            d = json.load(open(jp))
        except Exception:
            continue
        if d.get('dataset') != ds or not d.get('results'):
            continue
        rs = d['results']
        aurocs = [r['mean_h_auroc'] for r in rs]
        candidates.append({
            'src': f'v28:{jp.name}', 'n': len(rs),
            'mean_h_auroc': float(np.mean(aurocs)),
            'std': float(np.std(aurocs, ddof=1)) if len(aurocs) > 1 else None,
            'pooled_auprc': float(np.mean([r['pooled_auprc'] for r in rs])),
        })

    # v27 baselines
    for jp in (V27 / 'results').glob('*.json'):
        try:
            d = json.load(open(jp))
        except Exception:
            continue
        if d.get('dataset') != ds or not d.get('results'):
            continue
        rs = d['results']
        # v27 jsons may not have per-result mean_h_auroc — derive it
        per_runs = []
        for r in rs:
            if 'mean_h_auroc' in r:
                per_runs.append(r['mean_h_auroc'])
            elif 'per_horizon' in r and r['per_horizon']:
                aurocs = [h['auroc'] for h in r['per_horizon']
                          if not np.isnan(h.get('auroc', np.nan))]
                if aurocs:
                    per_runs.append(float(np.mean(aurocs)))
        if not per_runs:
            continue
        candidates.append({
            'src': f'v27:{jp.name}', 'n': len(per_runs),
            'mean_h_auroc': float(np.mean(per_runs)),
            'std': float(np.std(per_runs, ddof=1)) if len(per_runs) > 1 else None,
            'pooled_auprc': float(np.mean([r['pooled_auprc'] for r in rs])),
        })

    if not candidates:
        return None
    return max(candidates, key=lambda c: c['mean_h_auroc'])


def best_chronos(ds: str) -> dict | None:
    """Single-seed Chronos-2 dense surface from v27/surfaces."""
    p = V27 / 'surfaces' / f'dense_chronos2_{ds}_s42.npz'
    return metrics_from_npz(p)


def main():
    print("=" * 100)
    print("V29 MASTER TABLE — best FAM result per dataset (across v27/v28/v29)")
    print("=" * 100)
    print(f"{'Dataset':<10} {'FAM h-AUROC':>15} {'std':>7} {'n':>3} "
          f"{'source':<35} {'Chronos-2':>12} {'Δ':>8}")
    print('-' * 100)

    table = []
    for ds in DATASETS_ALL:
        b = best_fam(ds)
        c = best_chronos(ds)
        c_h = c['mean_h_auroc'] if c else None
        delta = (b['mean_h_auroc'] - c_h) if (b and c_h is not None) else None
        if b is None:
            print(f"{ds:<10} {'(no result)':>15}")
            table.append({'dataset': ds, 'fam': None, 'chronos2': c_h})
            continue
        std_str = f"{b['std']:.3f}" if b['std'] is not None else '—'
        c_str = f"{c_h:.4f}" if c_h is not None else '—'
        d_str = f"{delta:+.4f}" if delta is not None else '—'
        print(f"{ds:<10} {b['mean_h_auroc']:>15.4f} {std_str:>7} {b['n']:>3} "
              f"{b['src']:<35} {c_str:>12} {d_str:>8}")
        table.append({'dataset': ds, 'fam': b, 'chronos2_h_auroc': c_h,
                      'delta_fam_minus_chronos': delta})

    out = V29 / 'results' / 'phase3_master_table.json'
    with open(out, 'w') as f:
        json.dump({'datasets': table}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
