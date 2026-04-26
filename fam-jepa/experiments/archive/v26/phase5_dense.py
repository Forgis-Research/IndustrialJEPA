"""V26 Phase 5: dense horizon evaluation on best checkpoints.

Re-evaluate FD001, SMAP, MBA seed-42 pred-FT checkpoints at every integer
Δt from 1 to max_horizon. Produces smooth surfaces for heatmap figures
and more accurate pooled AUPRC.

Dense evaluation only — training used sparse horizons. The predictor
already learned a smooth mapping from Δt → representation during pretraining,
so evaluating at novel Δt is valid.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

FAM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V26_DIR = FAM_DIR / 'experiments/v26'
CKPT_DIR = V26_DIR / 'ckpts'
SURF_DIR = V26_DIR / 'surfaces'
RES_DIR = V26_DIR / 'results'

sys.path.insert(0, str(FAM_DIR))
sys.path.insert(0, str(FAM_DIR / 'experiments/v11'))
sys.path.insert(0, str(FAM_DIR / 'experiments/v24'))

from model import FAM
from train import (
    EventDataset, collate_event, evaluate, save_surface,
)

# Plumb in the anomaly loaders + FD001 loader reused from phases 2/3
sys.path.insert(0, str(V26_DIR))
from phase3_anomaly import load_smap, load_mba, build_anomaly_event_concat
from phase2_cmapss import build_event_concat
from _cmapss_raw import load_cmapss_raw


CONFIG = {
    'FD001': {
        'max_horizon': 150, 'n_channels': 14, 'patch_size': 16,
        'max_context': 512, 'min_context': 128,
    },
    'SMAP': {
        'max_horizon': 200, 'n_channels': 25, 'patch_size': 16,
        'max_context': 512, 'min_context': 128,
    },
    'MBA': {
        'max_horizon': 200, 'n_channels': 2, 'patch_size': 16,
        'max_context': 512, 'min_context': 128,
    },
}


def build_fd001_test_loader(max_context=512):
    data = load_cmapss_raw('FD001')
    test_ft = build_event_concat(data['test_engines'], stride=1,
                                 max_context=max_context)
    return DataLoader(test_ft, batch_size=128, shuffle=False,
                      collate_fn=collate_event, num_workers=0)


def build_smap_test_loader(max_context=512):
    _, ft, _ = load_smap()
    test_ft = build_anomaly_event_concat(ft['ft_test'], stride=1,
                                         max_context=max_context)
    return DataLoader(test_ft, batch_size=128, shuffle=False,
                      collate_fn=collate_event, num_workers=0)


def build_mba_test_loader(max_context=512):
    _, ft, _ = load_mba()
    test_ft = build_anomaly_event_concat(ft['ft_test'], stride=1,
                                         max_context=max_context)
    return DataLoader(test_ft, batch_size=128, shuffle=False,
                      collate_fn=collate_event, num_workers=0)


BUILDERS = {
    'FD001': build_fd001_test_loader,
    'SMAP': build_smap_test_loader,
    'MBA': build_mba_test_loader,
}


def run_dense(dataset: str, seed: int = 42, device: str = 'cuda') -> dict:
    cfg = CONFIG[dataset]
    tag = f'{dataset}_s{seed}'
    ft_ckpt = CKPT_DIR / f'{tag}_pred_ft.pt'
    if not ft_ckpt.exists():
        print(f"  SKIP {tag}: no pred_ft checkpoint at {ft_ckpt}", flush=True)
        return None

    print(f"\n=== dense {tag} ===", flush=True)
    print(f"  checkpoint: {ft_ckpt}", flush=True)

    model = FAM(n_channels=cfg['n_channels'], patch_size=cfg['patch_size'],
                d_model=256, n_heads=4, n_layers=2, d_ff=256, dropout=0.1,
                ema_momentum=0.99, predictor_hidden=256)
    model.load_state_dict(torch.load(ft_ckpt, map_location='cpu'))
    model.to(device)
    model.eval()

    # Dense horizons: 1..max
    dense_horizons = list(range(1, cfg['max_horizon'] + 1))
    print(f"  dense horizons: 1..{cfg['max_horizon']} (K={len(dense_horizons)})",
          flush=True)

    loader = BUILDERS[dataset](max_context=cfg['max_context'])
    print(f"  test samples: {len(loader.dataset)}", flush=True)

    t0 = time.time()
    eval_out = evaluate(model, loader, dense_horizons, mode='pred_ft',
                        device=device)
    eval_time = time.time() - t0
    primary = eval_out['primary']
    mono = eval_out['monotonicity']

    print(f"  dense eval took {eval_time:.1f}s", flush=True)
    print(f"  AUPRC (dense pooled): {primary['auprc']:.4f}", flush=True)
    print(f"  AUROC (dense pooled): {primary['auroc']:.4f}", flush=True)
    print(f"  monotonicity violations: {mono['violation_rate']:.6f}",
          flush=True)

    surf_path = SURF_DIR / f'{tag}_dense.npz'
    save_surface(surf_path, eval_out['p_surface'], eval_out['y_surface'],
                 dense_horizons, eval_out['t_index'],
                 metadata={'dataset': dataset, 'seed': seed,
                           'phase': 'v26_p5_dense'})

    return {
        'dataset': dataset, 'seed': seed,
        'max_horizon': cfg['max_horizon'],
        'n_horizons': len(dense_horizons),
        'eval_time_s': eval_time,
        'auprc_dense': float(primary['auprc']),
        'auroc_dense': float(primary['auroc']),
        'f1_best_dense': float(primary['f1_best']),
        'prevalence': float(primary['prevalence']),
        'monotonicity_violation_rate': float(mono['violation_rate']),
        'surface_path': str(surf_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+',
                    default=['FD001', 'SMAP', 'MBA'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=str,
                    default=str(RES_DIR / 'phase5_dense.json'))
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}", flush=True)

    all_results = []
    for ds in args.datasets:
        res = run_dense(ds, seed=args.seed, device=device)
        if res is not None:
            all_results.append(res)
        with open(args.out, 'w') as f:
            json.dump({'datasets': args.datasets, 'seed': args.seed,
                       'results': all_results}, f, indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
