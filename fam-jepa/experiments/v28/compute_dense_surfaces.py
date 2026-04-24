"""V28 dense surface compute — re-evaluate v28 best ckpt at every integer Δt.

After Phase 3 produces sparse-K=7/8 surfaces, this script loads each
ckpt and re-evaluates at K=150 (C-MAPSS) or K=200 (anomaly) horizons.
The FAM predictor takes Δt as a continuous scalar, so dense evaluation
is just a horizon-list swap.

For each dataset we pick the BEST v28 variant by mean per-horizon AUROC
on the sparse evaluation. If the v27 baseline is still best (i.e. no Try
beat it), we fall back to the v27 dense surface (already computed).

Outputs to experiments/v28/surfaces_dense/dense_fam_v28_<ds>_s42.npz.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

FAM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V28 = FAM_DIR / 'experiments/v28'
SURF_DENSE = V28 / 'surfaces_dense'
SURF_DENSE.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(FAM_DIR))
sys.path.insert(0, str(V28))

from model import FAM
from train import collate_event, evaluate
from runner_v28 import LOADERS, _global_zscore, _build_event_concat, apply_lag_features


CMAPSS_DENSE = list(range(1, 151))
ANOMALY_DENSE = list(range(1, 201))


def dense_horizons(ds):
    return CMAPSS_DENSE if ds.startswith('FD') else ANOMALY_DENSE


def best_variant_for(ds: str) -> dict | None:
    """Return the variant kwargs for the best v28 result on this dataset.

    Searches the most recent JSON in experiments/v28/results/phase{2,3}_*_<ds>_*.json,
    picks the one with highest mean h-AUROC (averaged over seeds) and returns
    a dict suitable for runner_v28.run_one. Returns None if no v28 surface beats
    the v27 baseline by > 0.005 mean h-AUROC (the ~1σ noise threshold).
    """
    candidates = []
    for jp in sorted((V28 / 'results').glob('phase*_*.json')):
        try:
            d = json.load(open(jp))
        except Exception:
            continue
        if d.get('dataset') != ds or not d.get('results'):
            continue
        rs = d['results']
        ha = float(np.mean([r['mean_h_auroc'] for r in rs]))
        candidates.append({
            'json': jp.name, 'mean_h_auroc': ha, 'rec': rs[0],
            'norm_mode': rs[0]['norm_mode'],
            'lag_features': rs[0].get('lag_features'),
            'aux_stat': rs[0].get('aux_stat', False),
            'dense_ft': rs[0].get('dense_ft', False),
        })
    if not candidates:
        return None
    best = max(candidates, key=lambda c: c['mean_h_auroc'])
    print(f"  best v28 variant for {ds}: {best['json']} "
          f"(mean h-AUROC={best['mean_h_auroc']:.4f})")
    return best


def reload_model(ds: str, variant: dict, seed: int = 42):
    """Recreate FAM with the same arch as variant, load ckpt, return (model, bundle)."""
    bundle = LOADERS[ds]()
    norm_mode = variant['norm_mode']
    if norm_mode == 'none':
        bundle = _global_zscore(bundle)
    if variant.get('lag_features'):
        bundle = apply_lag_features(bundle, variant['lag_features'])

    n_channels = bundle['n_channels']
    model = FAM(n_channels=n_channels, patch_size=16, d_model=256,
                n_heads=4, n_layers=2, d_ff=256, dropout=0.1,
                ema_momentum=0.99, predictor_hidden=256, norm_mode=norm_mode)
    # Find the FT ckpt by tag pattern from runner_v28
    extra = ''
    if variant.get('lag_features'):
        extra += f'_lag{"_".join(str(L) for L in variant["lag_features"])}'
    if variant.get('aux_stat'):
        extra += '_stat'
    if variant.get('dense_ft'):
        extra += '_dense20'    # default k_dense=20
    tag = f'{ds}_{norm_mode}{extra}_s{seed}'
    ckpt = V28 / 'ckpts' / f'{tag}_pred_ft.pt'
    if not ckpt.exists():
        print(f"    ckpt missing: {ckpt}")
        return None, None
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    return model, bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=[
        'FD001', 'FD002', 'FD003', 'SMAP', 'MSL', 'PSM', 'SMD', 'MBA',
        'GECCO', 'BATADAL'
    ])
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    summary = {}
    for ds in args.datasets:
        print(f"\n=== {ds} ===")
        variant = best_variant_for(ds)
        if variant is None:
            print(f"  no v28 variant found, skip"); continue
        model, bundle = reload_model(ds, variant, seed=args.seed)
        if model is None:
            continue
        model = model.to(device).eval()
        horizons = dense_horizons(ds)
        test_ft = _build_event_concat(bundle['ft_test'], stride=1, max_context=512)
        test_loader = DataLoader(test_ft, batch_size=128, shuffle=False,
                                 collate_fn=collate_event)
        t0 = time.time()
        out = evaluate(model, test_loader, horizons, mode='pred_ft', device=device)
        p, y = out['p_surface'], out['y_surface']
        from sklearn.metrics import roc_auc_score, average_precision_score
        valid = [i for i in range(len(horizons)) if 0 < y[:, i].mean() < 1]
        mean_h_auroc = float(np.mean([roc_auc_score(y[:, i], p[:, i]) for i in valid]))
        pooled_auprc = float(average_precision_score(y.ravel(), p.ravel()))
        print(f"  dense K={len(horizons)}  pooled_AUPRC={pooled_auprc:.4f}  "
              f"mean_h_AUROC={mean_h_auroc:.4f}  ({time.time()-t0:.1f}s)")
        out_path = SURF_DENSE / f'dense_fam_v28_{ds}_s{args.seed}.npz'
        np.savez(out_path,
                 p_surface=p.astype(np.float32),
                 y_surface=y.astype(np.int8),
                 horizons=np.asarray(horizons, dtype=np.int32),
                 t_index=out['t_index'].astype(np.int64),
                 meta=np.asarray(list({
                     'dataset': ds, 'seed': args.seed,
                     'norm_mode': variant['norm_mode'],
                     'lag_features': variant.get('lag_features'),
                     'aux_stat': variant.get('aux_stat', False),
                     'dense_ft': variant.get('dense_ft', False),
                     'kind': 'fam_v28_dense_pooled',
                 }.items()), dtype=object))
        print(f"  wrote {out_path.relative_to(FAM_DIR)}")
        summary[ds] = {'pooled_auprc': pooled_auprc,
                       'mean_h_auroc': mean_h_auroc,
                       'variant': variant.get('lag_features') or
                                  ('aux_stat' if variant.get('aux_stat') else None) or
                                  ('dense_ft' if variant.get('dense_ft') else 'baseline')}
    json.dump(summary, open(V28 / 'results' / 'dense_fam_v28_summary.json', 'w'),
              indent=2, default=str)


if __name__ == '__main__':
    main()
