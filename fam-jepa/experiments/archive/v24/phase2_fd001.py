"""V24 Phase 2: Full FD001 pretrain + pred-FT, 3 seeds.

Protocol:
- Pretrain: batch=64, lr=3e-4, max 50 epochs, patience=5
- Pred-FT:  batch=256, lr=1e-3, max 40 epochs, patience=8
- Horizons: {1, 5, 10, 20, 50, 100, 150}
- Data:     raw (no pre-normalization); RevIN handles it

Target: AUPRC 0.945 +/- 0.016 (v21 baseline). Stop if <0.90 (regression).
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
V24_DIR = FAM_DIR / 'experiments/v24'
CKPT_DIR = V24_DIR / 'ckpts'
SURF_DIR = V24_DIR / 'surfaces'
LOG_DIR = V24_DIR / 'logs'
RES_DIR = V24_DIR / 'results'
for d in [CKPT_DIR, SURF_DIR, LOG_DIR, RES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(FAM_DIR))
sys.path.insert(0, str(FAM_DIR / 'experiments/v11'))
sys.path.insert(0, str(V24_DIR))

from model import FAM
from train import (
    PretrainDataset, EventDataset, collate_pretrain, collate_event,
    pretrain, finetune, evaluate, get_horizons, save_surface,
)
from _cmapss_raw import load_cmapss_raw


def build_event_concat(engines, stride, max_context=512, max_future=200,
                       min_context=128):
    datasets = []
    for eid, seq in engines.items():
        T = len(seq)
        if T <= min_context:
            continue
        labels = np.zeros(T, dtype=np.int32)
        labels[T - 1] = 1
        datasets.append(EventDataset(seq, labels, max_context=max_context,
                                     stride=stride, max_future=max_future,
                                     min_context=min_context))
    # Filter out zero-length datasets (very short engines)
    datasets = [d for d in datasets if len(d) > 0]
    return ConcatDataset(datasets) if datasets else ConcatDataset([])


def run_seed(subset: str, seed: int, max_context: int = 512,
             pre_epochs: int = 50, pre_patience: int = 5,
             ft_epochs: int = 40, ft_patience: int = 8,
             n_cuts_train: int = 60, n_cuts_val: int = 20,
             device: str = 'cuda') -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    tag = f"{subset}_s{seed}"
    print(f"\n=== {tag} ===", flush=True)

    # ---- Load data ----
    t0 = time.time()
    data = load_cmapss_raw(subset)
    print(f"  raw data loaded in {time.time()-t0:.1f}s "
          f"(train={len(data['train_engines'])}, val={len(data['val_engines'])}, "
          f"test={len(data['test_engines'])})", flush=True)

    # ---- Pretrain ----
    train_pre = PretrainDataset(data['train_engines'], n_cuts=n_cuts_train,
                                max_context=max_context, delta_t_max=150,
                                delta_t_min=1, seed=seed)
    val_pre = PretrainDataset(data['val_engines'], n_cuts=n_cuts_val,
                              max_context=max_context, delta_t_max=150,
                              delta_t_min=1, seed=seed + 10000)
    print(f"  pretrain samples: train={len(train_pre)}, val={len(val_pre)}",
          flush=True)

    train_pre_loader = DataLoader(train_pre, batch_size=64, shuffle=True,
                                  collate_fn=collate_pretrain, num_workers=0)
    val_pre_loader = DataLoader(val_pre, batch_size=64, shuffle=False,
                                collate_fn=collate_pretrain, num_workers=0)

    model = FAM(n_channels=14, patch_size=16, d_model=256, n_heads=4,
                n_layers=2, d_ff=256, dropout=0.1, ema_momentum=0.99,
                predictor_hidden=256)

    ckpt_path = CKPT_DIR / f'{tag}_pretrain.pt'
    if ckpt_path.exists():
        print(f"  pretrain ckpt exists, loading: {ckpt_path}", flush=True)
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        pre_out = {'best_loss': float('nan'), 'history': []}
        pre_time = 0.0
    else:
        t0 = time.time()
        pre_out = pretrain(model, train_pre_loader, val_pre_loader,
                           lr=3e-4, n_epochs=pre_epochs,
                           patience=pre_patience, device=device)
        pre_time = time.time() - t0
        print(f"  pretrain done in {pre_time:.1f}s "
              f"(best_val={pre_out['best_loss']:.4f})", flush=True)
        torch.save(model.state_dict(), ckpt_path)

    # ---- Finetune ----
    horizons = get_horizons(subset)
    train_ft = build_event_concat(data['train_engines'], stride=4,
                                  max_context=max_context)
    val_ft = build_event_concat(data['val_engines'], stride=4,
                                max_context=max_context)
    test_ft = build_event_concat(data['test_engines'], stride=1,
                                 max_context=max_context)
    print(f"  ft samples: train={len(train_ft)}, val={len(val_ft)}, "
          f"test={len(test_ft)}", flush=True)

    train_ft_loader = DataLoader(train_ft, batch_size=256, shuffle=True,
                                 collate_fn=collate_event, num_workers=0)
    val_ft_loader = DataLoader(val_ft, batch_size=256, shuffle=False,
                               collate_fn=collate_event, num_workers=0)
    test_ft_loader = DataLoader(test_ft, batch_size=256, shuffle=False,
                                collate_fn=collate_event, num_workers=0)

    t0 = time.time()
    ft_out = finetune(model, train_ft_loader, val_ft_loader, horizons,
                      mode='pred_ft', lr=1e-3, n_epochs=ft_epochs,
                      patience=ft_patience, device=device)
    ft_time = time.time() - t0
    print(f"  finetune done in {ft_time:.1f}s "
          f"(best_val={ft_out['best_val']:.4f})", flush=True)

    ft_ckpt = CKPT_DIR / f'{tag}_pred_ft.pt'
    torch.save(model.state_dict(), ft_ckpt)

    # ---- Evaluate ----
    t0 = time.time()
    eval_out = evaluate(model, test_ft_loader, horizons, mode='pred_ft',
                        device=device)
    eval_time = time.time() - t0
    primary = eval_out['primary']
    per_h = eval_out['per_horizon']
    mono = eval_out['monotonicity']

    print(f"  eval done in {eval_time:.1f}s", flush=True)
    print(f"  AUPRC: {primary['auprc']:.4f}  AUROC: {primary['auroc']:.4f}  "
          f"F1: {primary['f1_best']:.4f}", flush=True)
    print(f"  per-horizon AUPRC:", flush=True)
    for h, a in zip(horizons, per_h['auprc_per_k']):
        print(f"    dt={h:4d}: {a:.4f}", flush=True)

    # Save surface
    surf_path = SURF_DIR / f'{tag}.npz'
    save_surface(surf_path, eval_out['p_surface'], eval_out['y_surface'],
                 horizons, eval_out['t_index'],
                 metadata={'subset': subset, 'seed': seed, 'phase': 'v24_p2'})

    return {
        'subset': subset,
        'seed': seed,
        'n_params': sum(p.numel() for p in model.parameters()),
        'pretrain_best_loss': float(pre_out['best_loss']),
        'pretrain_time_s': pre_time,
        'pretrain_epochs_run': len(pre_out['history']),
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
                              zip(horizons, per_h['auprc_per_k'])},
        'per_horizon_auroc': {str(h): float(a) for h, a in
                              zip(horizons, per_h['auroc_per_k'])},
        'surface_path': str(surf_path),
        'ckpt_path': str(ckpt_path),
        'ft_ckpt_path': str(ft_ckpt),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', default='FD001')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--pre-epochs', type=int, default=50)
    parser.add_argument('--ft-epochs', type=int, default=40)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}", flush=True)

    results = []
    for seed in args.seeds:
        res = run_seed(args.subset, seed, pre_epochs=args.pre_epochs,
                       ft_epochs=args.ft_epochs, device=device)
        results.append(res)
        # Per-seed early dump (in case of crash)
        out = args.out or (RES_DIR / f'phase2_{args.subset}.json')
        with open(out, 'w') as f:
            json.dump({'subset': args.subset, 'results': results}, f, indent=2)

    # Summary
    auprcs = [r['auprc'] for r in results]
    aurocs = [r['auroc'] for r in results]
    print(f"\n=== SUMMARY {args.subset} (n={len(auprcs)}) ===", flush=True)
    print(f"  AUPRC: {np.mean(auprcs):.4f} +/- {np.std(auprcs):.4f}", flush=True)
    print(f"  AUROC: {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}", flush=True)
    for r in results:
        print(f"  seed {r['seed']:4d}: AUPRC={r['auprc']:.4f}  "
              f"AUROC={r['auroc']:.4f}", flush=True)

    out = args.out or (RES_DIR / f'phase2_{args.subset}.json')
    with open(out, 'w') as f:
        json.dump({
            'subset': args.subset,
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
