"""Compute v27 baseline dense surfaces for seeds 123, 456 (s42 already done in v27).

The v27 phase 7 / dense_pooled_summary used only s42 per dataset. To get a
paired-seed delta vs the v28 best, we need v27 dense at the same seeds.

This script reloads v27 ckpts and re-evaluates at dense K=150 (C-MAPSS) or
K=200 (anomaly) horizons.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

FAM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V26 = FAM_DIR / 'experiments/v26'
V27 = FAM_DIR / 'experiments/v27'
SURF = V27 / 'surfaces'
SURF.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(FAM_DIR))
sys.path.insert(0, str(FAM_DIR / 'experiments/v27'))
sys.path.insert(0, str(FAM_DIR / 'experiments/v28'))

from model import FAM
from train import collate_event, evaluate
from runner_v28 import LOADERS, _global_zscore, _build_event_concat


def dense_horizons(ds):
    if ds.startswith('FD'): return list(range(1, 151))
    return list(range(1, 201))


# (ds, source) -> norm_mode and v27_ckpt_template
V27_RECIPES = {
    # C-MAPSS uses v27 'none' ckpt
    'FD001': ('none', V27 / 'ckpts' / 'FD001_none_s{seed}_pred_ft.pt'),
    'FD002': ('none', V27 / 'ckpts' / 'FD002_none_s{seed}_pred_ft.pt'),
    'FD003': ('none', V27 / 'ckpts' / 'FD003_none_s{seed}_pred_ft.pt'),
    # Anomaly uses v26 'revin' ckpt (the correct baseline per v27 SESSION_SUMMARY)
    'SMAP': ('revin', V26 / 'ckpts' / 'SMAP_s{seed}_pred_ft.pt'),
    'MSL':  ('revin', V26 / 'ckpts' / 'MSL_s{seed}_pred_ft.pt'),
    'PSM':  ('revin', V26 / 'ckpts' / 'PSM_s{seed}_pred_ft.pt'),
    'MBA':  ('revin', V26 / 'ckpts' / 'MBA_s{seed}_pred_ft.pt'),
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=list(V27_RECIPES))
    ap.add_argument('--seeds', type=int, nargs='+', default=[123, 456])
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    summary = {}
    for ds in args.datasets:
        if ds not in V27_RECIPES:
            print(f"  {ds}: no v27 recipe, skip"); continue
        norm_mode, ckpt_tmpl = V27_RECIPES[ds]
        print(f"\n=== {ds} ({norm_mode}) ===")
        bundle = LOADERS[ds]()
        if norm_mode == 'none':
            bundle = _global_zscore(bundle)
        n_channels = bundle['n_channels']
        for seed in args.seeds:
            ckpt = Path(str(ckpt_tmpl).format(seed=seed))
            if not ckpt.exists():
                print(f"  s{seed}: ckpt missing {ckpt}"); continue
            tag = 'v27' if 'v27' in str(ckpt) else 'v26'
            out_path = SURF / f'dense_fam_{tag}_{ds}_s{seed}.npz'
            if out_path.exists():
                print(f"  s{seed}: surface exists, skip"); continue
            model = FAM(n_channels=n_channels, patch_size=16, d_model=256,
                        n_heads=4, n_layers=2, d_ff=256, dropout=0.1,
                        ema_momentum=0.99, predictor_hidden=256,
                        norm_mode=norm_mode)
            model.load_state_dict(torch.load(ckpt, map_location='cpu'))
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
                         'dataset': ds, 'seed': seed, 'norm_mode': norm_mode,
                         'kind': f'fam_{tag}_dense_pooled',
                     }.items()), dtype=object))
            summary.setdefault(ds, {})[f's{seed}'] = {'mean_h_auroc': mha, 'pooled_auprc': pa}
    json.dump(summary, open(V27 / 'results' / 'dense_fam_v27_v26_more_seeds.json', 'w'),
              indent=2, default=str)


if __name__ == '__main__':
    main()
