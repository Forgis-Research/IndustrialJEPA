"""V24 Phase 6: PhysioNet 2019 Sepsis Challenge with P=1 (exception to P=16).

Hourly clinical data. Stays are 6-336 hours, most < 128 hours -> P=16 floor
would drop most of the dataset. Use P=1 (1 hour per token) and min_context=8.

Horizons in hours: {1, 2, 3, 6, 12, 24, 48}.

Pretrain on non-septic set A patients (no sepsis exposure).
Pred-FT on all set A (80/20 split), evaluate on set B.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

FAM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V24_DIR = FAM_DIR / 'experiments/v24'
CKPT_DIR = V24_DIR / 'ckpts'
SURF_DIR = V24_DIR / 'surfaces'
LOG_DIR = V24_DIR / 'logs'
RES_DIR = V24_DIR / 'results'
for d in [CKPT_DIR, SURF_DIR, LOG_DIR, RES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(FAM_DIR))

from model import FAM
from train import (
    PretrainDataset, EventDataset, collate_pretrain, collate_event,
    pretrain, finetune, evaluate, save_surface,
)

SEPSIS_HORIZONS = [1, 2, 3, 6, 12, 24, 48]  # hours
SEPSIS_P = 1
SEPSIS_MIN_CTX = 8   # 8 hours = 8 tokens at P=1
SEPSIS_MAX_CTX = 336


def build_sepsis_event_concat(patients, stride, max_context=SEPSIS_MAX_CTX,
                              max_future=200, min_context=SEPSIS_MIN_CTX):
    datasets = []
    for p in patients:
        x = p['x']
        y = p['labels']
        if len(x) <= min_context + 1:
            continue
        ds = EventDataset(x, y, max_context=max_context, stride=stride,
                          max_future=max_future, min_context=min_context)
        if len(ds) > 0:
            datasets.append(ds)
    return ConcatDataset(datasets) if datasets else ConcatDataset([])


def run_seed(seed: int, cached_data=None,
             pre_epochs: int = 20, pre_patience: int = 3,
             ft_epochs: int = 20, ft_patience: int = 4,
             n_cuts: int = 4, device: str = 'cuda') -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    tag = f"sepsis_s{seed}"
    print(f"\n=== {tag} ===", flush=True)

    if cached_data is None:
        from data.sepsis import load_sepsis
        data = load_sepsis(seed=seed, verbose=True,
                           pretrain_from='nonseptic_setA')
    else:
        data = cached_data

    n_channels = int(data['n_channels'])
    print(f"  n_channels={n_channels}", flush=True)

    # Build pretrain dict
    pretrain_seqs = {}
    for i, p in enumerate(data['pretrain_patients']):
        if len(p['x']) >= SEPSIS_MIN_CTX + 2:
            pretrain_seqs[i] = p['x'].astype(np.float32)
    print(f"  n_pretrain_seqs: {len(pretrain_seqs)}", flush=True)

    # Model
    model = FAM(n_channels=n_channels, patch_size=SEPSIS_P, d_model=256,
                n_heads=4, n_layers=2, d_ff=256, dropout=0.1,
                ema_momentum=0.99, predictor_hidden=256)

    ckpt_path = CKPT_DIR / f'{tag}_pretrain.pt'
    if ckpt_path.exists():
        print(f"  pretrain ckpt exists, loading", flush=True)
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        pre_time = 0.0
        pre_best = float('nan')
    else:
        train_pre = PretrainDataset(pretrain_seqs, n_cuts=n_cuts,
                                    max_context=SEPSIS_MAX_CTX,
                                    delta_t_max=48, delta_t_min=1, seed=seed)
        # Build val from a small random subset
        val_keys = list(pretrain_seqs.keys())[-500:]
        val_seqs = {k: pretrain_seqs[k] for k in val_keys}
        val_pre = PretrainDataset(val_seqs, n_cuts=max(2, n_cuts // 2),
                                  max_context=SEPSIS_MAX_CTX,
                                  delta_t_max=48, delta_t_min=1,
                                  seed=seed + 10000)
        print(f"  pretrain samples: train={len(train_pre)}, val={len(val_pre)}",
              flush=True)
        tlo = DataLoader(train_pre, batch_size=128, shuffle=True,
                         collate_fn=collate_pretrain, num_workers=2)
        vlo = DataLoader(val_pre, batch_size=128, shuffle=False,
                         collate_fn=collate_pretrain, num_workers=2)

        t0 = time.time()
        pre_out = pretrain(model, tlo, vlo, lr=3e-4, n_epochs=pre_epochs,
                           patience=pre_patience, device=device)
        pre_time = time.time() - t0
        pre_best = float(pre_out['best_loss'])
        print(f"  pretrain done in {pre_time:.1f}s (best_val={pre_best:.4f})",
              flush=True)
        torch.save(model.state_dict(), ckpt_path)

    # Pred-FT
    train_ft = build_sepsis_event_concat(data['ft_train'], stride=4,
                                         max_context=SEPSIS_MAX_CTX)
    val_ft = build_sepsis_event_concat(data['ft_val'], stride=4,
                                       max_context=SEPSIS_MAX_CTX)
    test_ft = build_sepsis_event_concat(data['ft_test'], stride=1,
                                        max_context=SEPSIS_MAX_CTX)
    print(f"  ft samples: train={len(train_ft)}, val={len(val_ft)}, "
          f"test={len(test_ft)}", flush=True)

    if len(train_ft) == 0 or len(test_ft) == 0:
        print(f"  SKIP {tag}: empty FT datasets", flush=True)
        return None

    tloader = DataLoader(train_ft, batch_size=256, shuffle=True,
                         collate_fn=collate_event, num_workers=2)
    vloader = DataLoader(val_ft, batch_size=256, shuffle=False,
                         collate_fn=collate_event, num_workers=2)
    test_loader = DataLoader(test_ft, batch_size=256, shuffle=False,
                             collate_fn=collate_event, num_workers=2)

    t0 = time.time()
    ft_out = finetune(model, tloader, vloader, SEPSIS_HORIZONS, mode='pred_ft',
                      lr=1e-3, n_epochs=ft_epochs, patience=ft_patience,
                      device=device)
    ft_time = time.time() - t0
    print(f"  finetune done in {ft_time:.1f}s "
          f"(best_val={ft_out['best_val']:.4f})", flush=True)
    ft_ckpt = CKPT_DIR / f'{tag}_pred_ft.pt'
    torch.save(model.state_dict(), ft_ckpt)

    # Evaluate
    t0 = time.time()
    eval_out = evaluate(model, test_loader, SEPSIS_HORIZONS, mode='pred_ft',
                        device=device)
    primary = eval_out['primary']
    per_h = eval_out['per_horizon']
    mono = eval_out['monotonicity']
    eval_time = time.time() - t0

    print(f"  eval done in {eval_time:.1f}s", flush=True)
    print(f"  AUPRC: {primary['auprc']:.4f}  AUROC: {primary['auroc']:.4f}  "
          f"F1: {primary['f1_best']:.4f}", flush=True)
    for h, a in zip(SEPSIS_HORIZONS, per_h['auprc_per_k']):
        print(f"    dt={h:3d}h: AUPRC={a:.4f}", flush=True)

    surf_path = SURF_DIR / f'{tag}.npz'
    save_surface(surf_path, eval_out['p_surface'], eval_out['y_surface'],
                 SEPSIS_HORIZONS, eval_out['t_index'],
                 metadata={'dataset': 'sepsis', 'seed': seed,
                           'phase': 'v24_p6'})

    return {
        'dataset': 'sepsis',
        'seed': seed,
        'n_channels': n_channels,
        'patch_size': SEPSIS_P,
        'pretrain_best_loss': pre_best,
        'pretrain_time_s': pre_time,
        'ft_best_val': float(ft_out['best_val']),
        'ft_time_s': ft_time,
        'eval_time_s': eval_time,
        'auprc': float(primary['auprc']),
        'auroc': float(primary['auroc']),
        'f1_best': float(primary['f1_best']),
        'precision_best': float(primary['precision_best']),
        'recall_best': float(primary['recall_best']),
        'prevalence': float(primary['prevalence']),
        'monotonicity_violation_rate': float(mono['violation_rate']),
        'per_horizon_auprc': {str(h): float(a) for h, a in
                              zip(SEPSIS_HORIZONS, per_h['auprc_per_k'])},
        'per_horizon_auroc': {str(h): float(a) for h, a in
                              zip(SEPSIS_HORIZONS, per_h['auroc_per_k'])},
        'surface_path': str(surf_path),
        'ckpt_path': str(ckpt_path),
        'ft_ckpt_path': str(ft_ckpt),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--pre-epochs', type=int, default=20)
    parser.add_argument('--ft-epochs', type=int, default=20)
    parser.add_argument('--n-cuts', type=int, default=4)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}", flush=True)

    # Load data once (shared across seeds — split depends on seed but data files
    # don't). Reload per seed so the ft_train/val split respects the seed.
    results = []
    for seed in args.seeds:
        res = run_seed(seed, pre_epochs=args.pre_epochs,
                       ft_epochs=args.ft_epochs, n_cuts=args.n_cuts,
                       device=device)
        if res is not None:
            results.append(res)
        out = args.out or (RES_DIR / 'phase6_sepsis.json')
        with open(out, 'w') as f:
            json.dump({'dataset': 'sepsis', 'results': results}, f, indent=2)

    if not results:
        print("NO RESULTS", flush=True)
        return

    auprcs = [r['auprc'] for r in results]
    aurocs = [r['auroc'] for r in results]
    print(f"\n=== SUMMARY sepsis (n={len(auprcs)}) ===", flush=True)
    print(f"  AUPRC: {np.mean(auprcs):.4f} +/- {np.std(auprcs):.4f}", flush=True)
    print(f"  AUROC: {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}", flush=True)

    out = args.out or (RES_DIR / 'phase6_sepsis.json')
    with open(out, 'w') as f:
        json.dump({
            'dataset': 'sepsis',
            'n_seeds': len(results),
            'auprc_mean': float(np.mean(auprcs)),
            'auprc_std': float(np.std(auprcs)),
            'auroc_mean': float(np.mean(aurocs)),
            'auroc_std': float(np.std(aurocs)),
            'results': results,
        }, f, indent=2)
    print(f"wrote {out}", flush=True)


if __name__ == '__main__':
    main()
