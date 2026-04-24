"""V28 — compute dense surfaces on additional seeds for variance estimation.

Picked up from compute_dense_surfaces.py: re-evaluate v28 best ckpts at
EVERY integer Δt for seeds 123 and 456 (s42 was done earlier). Then we
can compute paired-seed deltas vs the v27 baseline at matched dense K.

Per ml-researcher self-check: the +0.045 FD003 headline is currently
single-seed; reviewer will demand variance. Same for the GECCO and
BATADAL "FAM vs Chronos-2" comparisons — currently single seed.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

FAM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V28 = FAM_DIR / 'experiments/v28'
SURF = V28 / 'surfaces_dense'
SURF.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(FAM_DIR))
sys.path.insert(0, str(V28))

from model import FAM
from train import collate_event, evaluate
from runner_v28 import LOADERS, _global_zscore, _build_event_concat, apply_lag_features
from compute_dense_surfaces import dense_horizons, best_variant_for, reload_model


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+',
                    default=['FD001', 'FD002', 'FD003', 'MBA', 'GECCO', 'BATADAL'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[123, 456])
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    summary = {}
    for ds in args.datasets:
        print(f"\n=== {ds} ===")
        variant = best_variant_for(ds)
        if variant is None:
            print(f"  no v28 variant found, skip"); continue
        for seed in args.seeds:
            out_path = SURF / f'dense_fam_v28_{ds}_s{seed}.npz'
            if out_path.exists():
                print(f"  s{seed}: surface exists, skip"); continue
            model, bundle = reload_model(ds, variant, seed=seed)
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
            mha = float(np.mean([roc_auc_score(y[:, i], p[:, i]) for i in valid]))
            pa = float(average_precision_score(y.ravel(), p.ravel()))
            print(f"  s{seed} dense K={len(horizons)}: pooled_AUPRC={pa:.4f}  "
                  f"mean_h_AUROC={mha:.4f}  ({time.time()-t0:.1f}s)")
            np.savez(out_path,
                     p_surface=p.astype(np.float32),
                     y_surface=y.astype(np.int8),
                     horizons=np.asarray(horizons, dtype=np.int32),
                     t_index=out['t_index'].astype(np.int64),
                     meta=np.asarray(list({
                         'dataset': ds, 'seed': seed,
                         'norm_mode': variant['norm_mode'],
                         'lag_features': variant.get('lag_features'),
                         'aux_stat': variant.get('aux_stat', False),
                         'dense_ft': variant.get('dense_ft', False),
                     }.items()), dtype=object))
            summary.setdefault(ds, {})[f's{seed}'] = {'mean_h_auroc': mha, 'pooled_auprc': pa}
    json.dump(summary, open(V28 / 'results' / 'dense_fam_v28_more_seeds.json', 'w'),
              indent=2, default=str)


if __name__ == '__main__':
    main()
