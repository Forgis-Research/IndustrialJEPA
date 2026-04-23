"""V24 Phase 11: FAM on 3 new-domain datasets (GECCO, BATADAL, PhysioNet 2012).

Fresh domains added to the paper:
  - GECCO 2018: drinking-water quality (environmental/IoT). P=16.
  - BATADAL:   water-distribution SCADA cyber-attacks (ICS). P=16.
  - PhysioNet 2012: ICU mortality (healthcare, different event from sepsis). P=1.

Same FAM recipe as phase 4-6: per-dataset pretrain, pred-FT with frozen
encoder, 3 seeds. Surfaces stored for downstream legacy-metric computation.
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

from model import FAM
from train import (
    PretrainDataset, EventDataset, collate_pretrain, collate_event,
    pretrain, finetune, evaluate, save_surface,
)

# Horizons and patch per dataset
CONFIG = {
    'GECCO': {
        'horizons': [1, 5, 10, 20, 50, 100, 150, 200],  # minutes
        'patch_size': 16,
        'max_context': 512,
        'min_context': 128,
        'delta_t_max': 200,
        'n_cuts': 40,
        'pre_epochs': 25,
        'ft_epochs': 25,
    },
    'BATADAL': {
        'horizons': [1, 3, 6, 12, 24, 48, 72],  # hours
        'patch_size': 16,
        'max_context': 512,
        'min_context': 128,
        'delta_t_max': 72,
        'n_cuts': 40,
        'pre_epochs': 40,  # small dataset (8761 train)
        'ft_epochs': 40,
    },
    'physionet2012': {
        'horizons': [1, 3, 6, 12, 24, 36, 48],  # hours (of ICU stay prefix)
        'patch_size': 1,
        'max_context': 48,
        'min_context': 8,
        'delta_t_max': 24,
        'n_cuts': 4,
        'pre_epochs': 25,
        'ft_epochs': 20,
    },
}


def _anomaly_event_concat(entity_list, stride, max_context, min_context,
                          max_future=200):
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


def load_dataset(name: str):
    """Return (pretrain_seqs dict, ft_train_ds, ft_val_ds, ft_test_ds,
    n_channels, cfg)."""
    cfg = CONFIG[name]
    if name == 'GECCO':
        from data.gecco import load_gecco
        d = load_gecco(normalize=False)
        pre_seqs = {k: v.astype(np.float32)
                    for k, v in d['pretrain_stream'].items()
                    if len(v) >= cfg['min_context']}
        tr = _anomaly_event_concat(d['ft_train'], 4, cfg['max_context'],
                                   cfg['min_context'])
        va = _anomaly_event_concat(d['ft_val'], 4, cfg['max_context'],
                                   cfg['min_context'])
        te = _anomaly_event_concat(d['ft_test'], 1, cfg['max_context'],
                                   cfg['min_context'])
        return pre_seqs, tr, va, te, d['n_channels'], cfg

    if name == 'BATADAL':
        from data.batadal import load_batadal
        d = load_batadal(normalize=False)
        pre_seqs = {k: v.astype(np.float32)
                    for k, v in d['pretrain_stream'].items()
                    if len(v) >= cfg['min_context']}
        tr = _anomaly_event_concat(d['ft_train'], 4, cfg['max_context'],
                                   cfg['min_context'])
        va = _anomaly_event_concat(d['ft_val'], 4, cfg['max_context'],
                                   cfg['min_context'])
        te = _anomaly_event_concat(d['ft_test'], 1, cfg['max_context'],
                                   cfg['min_context'])
        return pre_seqs, tr, va, te, d['n_channels'], cfg

    if name == 'physionet2012':
        from data.physionet2012 import load_physionet2012
        d = load_physionet2012()
        pre_seqs = {}
        for i, p in enumerate(d['pretrain_patients']):
            if len(p['x']) >= cfg['min_context'] + 2:
                pre_seqs[i] = p['x'].astype(np.float32)

        # EventDataset expects (x, labels) per entity with TTE derived from labels
        # PhysioNet 2012 has label[t] constant (mortality flag). Convert to
        # a sequence-level event at t=last so TTE is decreasing.
        def to_entities(patients):
            out = []
            for p in patients:
                x = p['x']; T = len(x)
                y_const = int(p['death'])
                if y_const == 1:
                    lbl = np.zeros(T, dtype=np.int32)
                    lbl[-1] = 1  # event at end of stay for deceased
                else:
                    lbl = np.zeros(T, dtype=np.int32)
                out.append({'test': x, 'labels': lbl})
            return out

        tr = _anomaly_event_concat(to_entities(d['ft_train']), 4,
                                   cfg['max_context'], cfg['min_context'])
        va = _anomaly_event_concat(to_entities(d['ft_val']), 4,
                                   cfg['max_context'], cfg['min_context'])
        te = _anomaly_event_concat(to_entities(d['ft_test']), 1,
                                   cfg['max_context'], cfg['min_context'])
        return pre_seqs, tr, va, te, d['n_channels'], cfg

    raise ValueError(name)


def run_seed(name: str, seed: int, device: str = 'cuda') -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    tag = f'{name}_s{seed}'
    print(f"\n=== {tag} ===", flush=True)

    t0 = time.time()
    pre_seqs, tr_ds, va_ds, te_ds, n_channels, cfg = load_dataset(name)
    print(f"  loaded in {time.time()-t0:.1f}s  pre_seqs={len(pre_seqs)}  "
          f"n_ch={n_channels}  P={cfg['patch_size']}", flush=True)
    print(f"  ft samples: tr={len(tr_ds)}  va={len(va_ds)}  te={len(te_ds)}",
          flush=True)

    model = FAM(n_channels=n_channels, patch_size=cfg['patch_size'],
                d_model=256, n_heads=4, n_layers=2, d_ff=256,
                dropout=0.1, ema_momentum=0.99, predictor_hidden=256)

    ckpt = CKPT_DIR / f'{tag}_pretrain.pt'
    if ckpt.exists():
        print(f"  loading pretrain ckpt", flush=True)
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        pre_time = 0.0
        pre_best = float('nan')
    else:
        # Build pretrain loader
        train_pre = PretrainDataset(pre_seqs, n_cuts=cfg['n_cuts'],
                                    max_context=cfg['max_context'],
                                    delta_t_max=cfg['delta_t_max'],
                                    delta_t_min=1, seed=seed)
        # val: take last 10% of each sequence if long enough, else reuse
        val_seqs = {}
        for k, seq in pre_seqs.items():
            cut = int(0.9 * len(seq))
            if len(seq) - cut >= cfg['min_context']:
                val_seqs[k] = seq[cut:]
        if not val_seqs:
            val_seqs = pre_seqs
        val_pre = PretrainDataset(val_seqs, n_cuts=max(5, cfg['n_cuts'] // 4),
                                  max_context=cfg['max_context'],
                                  delta_t_max=cfg['delta_t_max'],
                                  delta_t_min=1, seed=seed + 10000)
        print(f"  pretrain samples: train={len(train_pre)} val={len(val_pre)}",
              flush=True)
        tlo = DataLoader(train_pre, batch_size=64, shuffle=True,
                         collate_fn=collate_pretrain, num_workers=2)
        vlo = DataLoader(val_pre, batch_size=64, shuffle=False,
                         collate_fn=collate_pretrain, num_workers=2)
        t0 = time.time()
        pre_out = pretrain(model, tlo, vlo, lr=3e-4,
                           n_epochs=cfg['pre_epochs'], patience=5,
                           device=device)
        pre_time = time.time() - t0
        pre_best = float(pre_out['best_loss'])
        print(f"  pretrain done in {pre_time:.1f}s (best_val={pre_best:.4f})",
              flush=True)
        torch.save(model.state_dict(), ckpt)

    if len(tr_ds) == 0 or len(te_ds) == 0:
        print(f"  SKIP {tag}: empty FT datasets", flush=True)
        return None

    tloader = DataLoader(tr_ds, batch_size=128, shuffle=True,
                         collate_fn=collate_event, num_workers=2)
    vloader = DataLoader(va_ds, batch_size=128, shuffle=False,
                         collate_fn=collate_event, num_workers=2)
    te_loader = DataLoader(te_ds, batch_size=128, shuffle=False,
                           collate_fn=collate_event, num_workers=2)

    t0 = time.time()
    ft_out = finetune(model, tloader, vloader, cfg['horizons'],
                      mode='pred_ft', lr=1e-3,
                      n_epochs=cfg['ft_epochs'], patience=6, device=device)
    ft_time = time.time() - t0
    print(f"  finetune done in {ft_time:.1f}s (best_val={ft_out['best_val']:.4f})",
          flush=True)
    ft_ckpt = CKPT_DIR / f'{tag}_pred_ft.pt'
    torch.save(model.state_dict(), ft_ckpt)

    t0 = time.time()
    eval_out = evaluate(model, te_loader, cfg['horizons'], mode='pred_ft',
                        device=device)
    primary = eval_out['primary']
    per_h = eval_out['per_horizon']
    mono = eval_out['monotonicity']
    eval_time = time.time() - t0

    print(f"  eval done in {eval_time:.1f}s  AUPRC={primary['auprc']:.4f}  "
          f"AUROC={primary['auroc']:.4f}  F1={primary['f1_best']:.4f}",
          flush=True)
    for h, a in zip(cfg['horizons'], per_h['auprc_per_k']):
        print(f"    dt={h:4d}: AUPRC={a:.4f}", flush=True)

    surf = SURF_DIR / f'{tag}.npz'
    save_surface(surf, eval_out['p_surface'], eval_out['y_surface'],
                 cfg['horizons'], eval_out['t_index'],
                 metadata={'dataset': name, 'seed': seed, 'phase': 'v24_p11'})

    return {
        'dataset': name, 'seed': seed,
        'n_channels': n_channels, 'patch_size': cfg['patch_size'],
        'pretrain_best_loss': pre_best, 'pretrain_time_s': pre_time,
        'ft_best_val': float(ft_out['best_val']),
        'ft_time_s': ft_time, 'eval_time_s': eval_time,
        'auprc': float(primary['auprc']), 'auroc': float(primary['auroc']),
        'f1_best': float(primary['f1_best']),
        'precision_best': float(primary['precision_best']),
        'recall_best': float(primary['recall_best']),
        'prevalence': float(primary['prevalence']),
        'monotonicity_violation_rate': float(mono['violation_rate']),
        'per_horizon_auprc': {str(h): float(a) for h, a in
                              zip(cfg['horizons'], per_h['auprc_per_k'])},
        'per_horizon_auroc': {str(h): float(a) for h, a in
                              zip(cfg['horizons'], per_h['auroc_per_k'])},
        'surface_path': str(surf), 'ckpt_path': str(ckpt),
        'ft_ckpt_path': str(ft_ckpt),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(CONFIG.keys()))
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}", flush=True)

    results = []
    for s in args.seeds:
        res = run_seed(args.dataset, s, device=device)
        if res is not None:
            results.append(res)
        out = args.out or (RES_DIR / f'phase11_{args.dataset}.json')
        with open(out, 'w') as f:
            json.dump({'dataset': args.dataset, 'results': results}, f, indent=2)

    if not results:
        return
    auprcs = [r['auprc'] for r in results]
    aurocs = [r['auroc'] for r in results]
    f1s = [r['f1_best'] for r in results]
    print(f"\n=== SUMMARY {args.dataset} (n={len(auprcs)}) ===", flush=True)
    print(f"  AUPRC: {np.mean(auprcs):.4f} +/- {np.std(auprcs):.4f}", flush=True)
    print(f"  AUROC: {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}", flush=True)
    print(f"  F1:    {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}", flush=True)

    out = args.out or (RES_DIR / f'phase11_{args.dataset}.json')
    with open(out, 'w') as f:
        json.dump({'dataset': args.dataset, 'n_seeds': len(results),
                   'auprc_mean': float(np.mean(auprcs)),
                   'auprc_std':  float(np.std(auprcs)),
                   'auroc_mean': float(np.mean(aurocs)),
                   'auroc_std':  float(np.std(aurocs)),
                   'f1_mean': float(np.mean(f1s)),
                   'f1_std':  float(np.std(f1s)),
                   'results': results}, f, indent=2)
    print(f"wrote {out}", flush=True)


if __name__ == '__main__':
    main()
