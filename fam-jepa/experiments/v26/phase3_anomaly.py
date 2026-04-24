"""V26 Phase 3: anomaly datasets with hazard CDF parameterization.

Datasets: SMAP, MSL, PSM, SMD, MBA. Horizons {1,5,10,20,50,100,150,200}.

Expected improvement vs v24:
  v24 monotonicity violation rates (ordered by worst):
    SMAP: ~11%
    MBA:  ~20-25%
    PSM:  ~7%
    MSL:  ~?%
    SMD:  ~?%
  The hazard CDF parameterization eliminates all violations by construction.
  AUPRC should meet or beat v24 because capacity isn't spent enforcing
  monotonicity.
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
LOG_DIR = V26_DIR / 'logs'
RES_DIR = V26_DIR / 'results'
for d in [CKPT_DIR, SURF_DIR, LOG_DIR, RES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(FAM_DIR))

from model import FAM
from train import (
    PretrainDataset, EventDataset, collate_pretrain, collate_event,
    pretrain, finetune, evaluate, save_surface,
)

ANOMALY_HORIZONS = [1, 5, 10, 20, 50, 100, 150, 200]


def load_smap():
    from data.smap_msl import load_smap_entities, split_smap_entities
    ents = load_smap_entities(normalize=False)
    pretrain_sequences = {i: e['train'].astype(np.float32)
                          for i, e in enumerate(ents) if len(e['train']) >= 128}
    ft = split_smap_entities(normalize=False)
    return pretrain_sequences, ft, 25


def load_msl():
    from data.smap_msl import load_msl_entities, split_msl_entities
    ents = load_msl_entities(normalize=False)
    pretrain_sequences = {i: e['train'].astype(np.float32)
                          for i, e in enumerate(ents) if len(e['train']) >= 128}
    ft = split_msl_entities(normalize=False)
    return pretrain_sequences, ft, 55


def load_psm():
    from data.psm import load_psm
    data = load_psm(normalize=False)
    train = data['train'].astype(np.float32)
    test = data['test'].astype(np.float32)
    labels = data['labels'].astype(np.int32)
    T = len(test)
    gap = 200
    t1 = int(0.6 * T)
    t2 = int(0.7 * T)
    ft = {
        'ft_train': [{'entity_id': 'PSM', 'test': test[:t1],
                      'labels': labels[:t1]}],
        'ft_val':   [{'entity_id': 'PSM', 'test': test[t1 + gap:t2],
                      'labels': labels[t1 + gap:t2]}],
        'ft_test':  [{'entity_id': 'PSM', 'test': test[t2 + gap:],
                      'labels': labels[t2 + gap:]}],
    }
    return {0: train}, ft, train.shape[1]


def load_smd():
    from data.smd import split_smd_entities, load_smd_entities
    ents = load_smd_entities(normalize=False)
    pretrain_sequences = {i: e['train'].astype(np.float32)
                          for i, e in enumerate(ents) if len(e['train']) >= 128}
    ft = split_smd_entities(normalize=False)
    return pretrain_sequences, ft, ents[0]['train'].shape[1]


def load_mba():
    from data.mba import load_mba
    data = load_mba(normalize=False)
    if data is None:
        raise FileNotFoundError("MBA data unavailable")
    train = data['train'].astype(np.float32)
    test = data['test'].astype(np.float32)
    labels = data['labels'].astype(np.int32)
    T = len(test)
    gap = 200
    t1 = int(0.6 * T)
    t2 = int(0.7 * T)
    ft = {
        'ft_train': [{'entity_id': 'MBA', 'test': test[:t1],
                      'labels': labels[:t1]}],
        'ft_val':   [{'entity_id': 'MBA', 'test': test[t1 + gap:t2],
                      'labels': labels[t1 + gap:t2]}],
        'ft_test':  [{'entity_id': 'MBA', 'test': test[t2 + gap:],
                      'labels': labels[t2 + gap:]}],
    }
    return {0: train}, ft, train.shape[1]


LOADERS = {
    'SMAP': load_smap,
    'MSL':  load_msl,
    'PSM':  load_psm,
    'SMD':  load_smd,
    'MBA':  load_mba,
}


def build_anomaly_event_concat(entity_list, stride, max_context=512,
                               max_future=200, min_context=128):
    datasets = []
    for e in entity_list:
        x = e['test']
        y = e['labels']
        if len(x) <= min_context + 1:
            continue
        ds = EventDataset(x, y, max_context=max_context, stride=stride,
                          max_future=max_future, min_context=min_context)
        if len(ds) > 0:
            datasets.append(ds)
    return ConcatDataset(datasets) if datasets else ConcatDataset([])


def run_seed(dataset: str, seed: int, max_context: int = 512,
             pre_epochs: int = 30, pre_patience: int = 5,
             ft_epochs: int = 30, ft_patience: int = 8,
             n_cuts: int = 40, device: str = 'cuda') -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    tag = f"{dataset}_s{seed}"
    print(f"\n=== {tag} ===", flush=True)

    t0 = time.time()
    pretrain_seqs, ft, n_channels = LOADERS[dataset]()
    print(f"  loaded {dataset} in {time.time()-t0:.1f}s "
          f"(n_pretrain_seqs={len(pretrain_seqs)}, n_channels={n_channels})",
          flush=True)
    print(f"    ft_train: {len(ft['ft_train'])} entities", flush=True)
    print(f"    ft_val:   {len(ft['ft_val'])} entities", flush=True)
    print(f"    ft_test:  {len(ft['ft_test'])} entities", flush=True)

    model = FAM(n_channels=n_channels, patch_size=16, d_model=256,
                n_heads=4, n_layers=2, d_ff=256, dropout=0.1,
                ema_momentum=0.99, predictor_hidden=256)

    ckpt_path = CKPT_DIR / f'{tag}_pretrain.pt'
    if ckpt_path.exists():
        print(f"  pretrain ckpt exists, loading: {ckpt_path}", flush=True)
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        pre_time = 0.0
        pre_best = float('nan')
    else:
        train_pre = PretrainDataset(pretrain_seqs, n_cuts=n_cuts,
                                    max_context=max_context, delta_t_max=150,
                                    delta_t_min=1, seed=seed)
        val_seqs = {}
        for k, seq in pretrain_seqs.items():
            L = len(seq)
            cut = int(0.9 * L)
            if L - cut >= 128:
                val_seqs[k] = seq[cut:]
        if not val_seqs:
            val_seqs = pretrain_seqs
        val_pre = PretrainDataset(val_seqs, n_cuts=max(10, n_cuts // 4),
                                  max_context=max_context, delta_t_max=150,
                                  delta_t_min=1, seed=seed + 10000)
        print(f"  pretrain samples: train={len(train_pre)}, val={len(val_pre)}",
              flush=True)

        tlo = DataLoader(train_pre, batch_size=64, shuffle=True,
                         collate_fn=collate_pretrain, num_workers=0)
        vlo = DataLoader(val_pre, batch_size=64, shuffle=False,
                         collate_fn=collate_pretrain, num_workers=0)

        t0 = time.time()
        pre_out = pretrain(model, tlo, vlo, lr=3e-4, n_epochs=pre_epochs,
                           patience=pre_patience, device=device)
        pre_time = time.time() - t0
        pre_best = float(pre_out['best_loss'])
        print(f"  pretrain done in {pre_time:.1f}s (best_val={pre_best:.4f})",
              flush=True)
        torch.save(model.state_dict(), ckpt_path)

    horizons = ANOMALY_HORIZONS
    train_ft = build_anomaly_event_concat(ft['ft_train'], stride=4,
                                          max_context=max_context)
    val_ft = build_anomaly_event_concat(ft['ft_val'], stride=4,
                                        max_context=max_context)
    test_ft = build_anomaly_event_concat(ft['ft_test'], stride=1,
                                         max_context=max_context)
    print(f"  ft samples: train={len(train_ft)}, val={len(val_ft)}, "
          f"test={len(test_ft)}", flush=True)

    if len(train_ft) == 0 or len(test_ft) == 0:
        print(f"  SKIP {tag}: empty FT datasets", flush=True)
        return None

    tloader = DataLoader(train_ft, batch_size=128, shuffle=True,
                         collate_fn=collate_event, num_workers=0)
    vloader = DataLoader(val_ft, batch_size=128, shuffle=False,
                         collate_fn=collate_event, num_workers=0)
    test_loader = DataLoader(test_ft, batch_size=128, shuffle=False,
                             collate_fn=collate_event, num_workers=0)

    t0 = time.time()
    ft_out = finetune(model, tloader, vloader, horizons, mode='pred_ft',
                      lr=1e-3, n_epochs=ft_epochs, patience=ft_patience,
                      device=device)
    ft_time = time.time() - t0
    print(f"  finetune done in {ft_time:.1f}s "
          f"(best_val={ft_out['best_val']:.4f})", flush=True)
    ft_ckpt = CKPT_DIR / f'{tag}_pred_ft.pt'
    torch.save(model.state_dict(), ft_ckpt)

    t0 = time.time()
    eval_out = evaluate(model, test_loader, horizons, mode='pred_ft',
                        device=device)
    primary = eval_out['primary']
    per_h = eval_out['per_horizon']
    mono = eval_out['monotonicity']
    eval_time = time.time() - t0

    print(f"  eval done in {eval_time:.1f}s", flush=True)
    print(f"  AUPRC: {primary['auprc']:.4f}  AUROC: {primary['auroc']:.4f}  "
          f"F1: {primary['f1_best']:.4f}", flush=True)
    print(f"  monotonicity violation rate: "
          f"{mono['violation_rate']:.6f}", flush=True)
    print(f"  per-horizon AUPRC:", flush=True)
    for h, a in zip(horizons, per_h['auprc_per_k']):
        print(f"    dt={h:4d}: {a:.4f}", flush=True)

    surf_path = SURF_DIR / f'{tag}.npz'
    save_surface(surf_path, eval_out['p_surface'], eval_out['y_surface'],
                 horizons, eval_out['t_index'],
                 metadata={'dataset': dataset, 'seed': seed, 'phase': 'v26_p3'})

    return {
        'dataset': dataset,
        'seed': seed,
        'n_params': sum(p.numel() for p in model.parameters()),
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
                              zip(horizons, per_h['auprc_per_k'])},
        'per_horizon_auroc': {str(h): float(a) for h, a in
                              zip(horizons, per_h['auroc_per_k'])},
        'surface_path': str(surf_path),
        'ckpt_path': str(ckpt_path),
        'ft_ckpt_path': str(ft_ckpt),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=list(LOADERS.keys()))
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--pre-epochs', type=int, default=30)
    parser.add_argument('--ft-epochs', type=int, default=30)
    parser.add_argument('--n-cuts', type=int, default=40)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}", flush=True)

    results = []
    for seed in args.seeds:
        res = run_seed(args.dataset, seed, pre_epochs=args.pre_epochs,
                       ft_epochs=args.ft_epochs, n_cuts=args.n_cuts,
                       device=device)
        if res is not None:
            results.append(res)
        out = args.out or (RES_DIR / f'phase3_{args.dataset}.json')
        with open(out, 'w') as f:
            json.dump({'dataset': args.dataset, 'results': results}, f,
                      indent=2)

    if not results:
        print("NO RESULTS", flush=True)
        return

    auprcs = [r['auprc'] for r in results]
    aurocs = [r['auroc'] for r in results]
    f1s = [r['f1_best'] for r in results]
    monos = [r['monotonicity_violation_rate'] for r in results]
    print(f"\n=== SUMMARY {args.dataset} (n={len(auprcs)}) ===", flush=True)
    print(f"  AUPRC: {np.mean(auprcs):.4f} +/- {np.std(auprcs):.4f}",
          flush=True)
    print(f"  AUROC: {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}",
          flush=True)
    print(f"  F1:    {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}", flush=True)
    print(f"  monotonicity violation rate: "
          f"max={max(monos):.6f}, mean={np.mean(monos):.6f}", flush=True)

    out = args.out or (RES_DIR / f'phase3_{args.dataset}.json')
    with open(out, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'n_seeds': len(results),
            'auprc_mean': float(np.mean(auprcs)),
            'auprc_std': float(np.std(auprcs)),
            'auroc_mean': float(np.mean(aurocs)),
            'auroc_std': float(np.std(aurocs)),
            'f1_mean': float(np.mean(f1s)),
            'f1_std': float(np.std(f1s)),
            'monotonicity_max': float(max(monos)),
            'monotonicity_mean': float(np.mean(monos)),
            'results': results,
        }, f, indent=2)
    print(f"wrote {out}", flush=True)


if __name__ == '__main__':
    main()
