"""V26 Phase 4: PhysioNet 2012 with hazard CDF parameterization.

Note on patch size: PhysioNet 2012 is 48h of hourly data per patient. With
P=16 this would be only 3 tokens (below the 8-token ARCHITECTURE.md floor).
Following v24 Phase 11 (which reached AUROC 0.858 here), we use P=1 so the
transformer sees 48 tokens. This is consistent with the
'hourly data needs P=1' exception flagged in ARCHITECTURE.md sec 5. The
v26 'P=16 everywhere' rule is meant for datasets with sufficient temporal
resolution — PhysioNet 2012 genuinely doesn't qualify.

Target: compare to v24 AUROC 0.858.
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

CFG = {
    'horizons': [1, 2, 3, 6, 12, 24, 48],  # hours — matches session prompt
    'patch_size': 1,
    'max_context': 48,
    'min_context': 8,
    'delta_t_max': 24,
    'n_cuts': 4,
    'pre_epochs': 25,
    'ft_epochs': 20,
}


def _event_concat(entity_list, stride, max_context, min_context,
                  max_future=48):
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


def load_physionet():
    from data.physionet2012 import load_physionet2012
    d = load_physionet2012()
    pre_seqs = {}
    for i, p in enumerate(d['pretrain_patients']):
        if len(p['x']) >= CFG['min_context'] + 2:
            pre_seqs[i] = p['x'].astype(np.float32)

    def to_entities(patients):
        out = []
        for p in patients:
            x = p['x']
            T = len(x)
            y_const = int(p['death'])
            if y_const == 1:
                lbl = np.zeros(T, dtype=np.int32)
                lbl[-1] = 1
            else:
                lbl = np.zeros(T, dtype=np.int32)
            out.append({'test': x, 'labels': lbl})
        return out

    tr = _event_concat(to_entities(d['ft_train']), 4,
                       CFG['max_context'], CFG['min_context'])
    va = _event_concat(to_entities(d['ft_val']), 4,
                       CFG['max_context'], CFG['min_context'])
    te = _event_concat(to_entities(d['ft_test']), 1,
                       CFG['max_context'], CFG['min_context'])
    return pre_seqs, tr, va, te, d['n_channels']


def run_seed(seed: int, device: str = 'cuda') -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    tag = f'physionet2012_s{seed}'
    print(f"\n=== {tag} ===", flush=True)

    t0 = time.time()
    pre_seqs, tr_ds, va_ds, te_ds, n_channels = load_physionet()
    print(f"  loaded in {time.time()-t0:.1f}s  pre_seqs={len(pre_seqs)}  "
          f"n_ch={n_channels}  P={CFG['patch_size']}", flush=True)
    print(f"  ft samples: tr={len(tr_ds)}  va={len(va_ds)}  te={len(te_ds)}",
          flush=True)

    model = FAM(n_channels=n_channels, patch_size=CFG['patch_size'],
                d_model=256, n_heads=4, n_layers=2, d_ff=256,
                dropout=0.1, ema_momentum=0.99, predictor_hidden=256)

    ckpt = CKPT_DIR / f'{tag}_pretrain.pt'
    if ckpt.exists():
        print(f"  loading pretrain ckpt", flush=True)
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        pre_time = 0.0
        pre_best = float('nan')
    else:
        train_pre = PretrainDataset(pre_seqs, n_cuts=CFG['n_cuts'],
                                    max_context=CFG['max_context'],
                                    delta_t_max=CFG['delta_t_max'],
                                    delta_t_min=1, seed=seed)
        val_seqs = {}
        for k, seq in pre_seqs.items():
            cut = int(0.9 * len(seq))
            if len(seq) - cut >= CFG['min_context']:
                val_seqs[k] = seq[cut:]
        if not val_seqs:
            val_seqs = pre_seqs
        val_pre = PretrainDataset(val_seqs,
                                  n_cuts=max(5, CFG['n_cuts'] // 4),
                                  max_context=CFG['max_context'],
                                  delta_t_max=CFG['delta_t_max'],
                                  delta_t_min=1, seed=seed + 10000)
        print(f"  pretrain samples: train={len(train_pre)} val={len(val_pre)}",
              flush=True)
        tlo = DataLoader(train_pre, batch_size=64, shuffle=True,
                         collate_fn=collate_pretrain, num_workers=2)
        vlo = DataLoader(val_pre, batch_size=64, shuffle=False,
                         collate_fn=collate_pretrain, num_workers=2)
        t0 = time.time()
        pre_out = pretrain(model, tlo, vlo, lr=3e-4,
                           n_epochs=CFG['pre_epochs'], patience=5,
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
    ft_out = finetune(model, tloader, vloader, CFG['horizons'],
                      mode='pred_ft', lr=1e-3,
                      n_epochs=CFG['ft_epochs'], patience=6, device=device)
    ft_time = time.time() - t0
    print(f"  finetune done in {ft_time:.1f}s "
          f"(best_val={ft_out['best_val']:.4f})", flush=True)
    ft_ckpt = CKPT_DIR / f'{tag}_pred_ft.pt'
    torch.save(model.state_dict(), ft_ckpt)

    t0 = time.time()
    eval_out = evaluate(model, te_loader, CFG['horizons'], mode='pred_ft',
                        device=device)
    primary = eval_out['primary']
    per_h = eval_out['per_horizon']
    mono = eval_out['monotonicity']
    eval_time = time.time() - t0

    print(f"  eval done in {eval_time:.1f}s  AUPRC={primary['auprc']:.4f}  "
          f"AUROC={primary['auroc']:.4f}  F1={primary['f1_best']:.4f}",
          flush=True)
    print(f"  monotonicity violation rate: "
          f"{mono['violation_rate']:.6f}", flush=True)
    for h, a in zip(CFG['horizons'], per_h['auprc_per_k']):
        print(f"    dt={h:4d}: AUPRC={a:.4f}", flush=True)

    surf = SURF_DIR / f'{tag}.npz'
    save_surface(surf, eval_out['p_surface'], eval_out['y_surface'],
                 CFG['horizons'], eval_out['t_index'],
                 metadata={'dataset': 'physionet2012', 'seed': seed,
                           'phase': 'v26_p4'})

    return {
        'dataset': 'physionet2012', 'seed': seed,
        'n_channels': n_channels, 'patch_size': CFG['patch_size'],
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
                              zip(CFG['horizons'], per_h['auprc_per_k'])},
        'per_horizon_auroc': {str(h): float(a) for h, a in
                              zip(CFG['horizons'], per_h['auroc_per_k'])},
        'surface_path': str(surf), 'ckpt_path': str(ckpt),
        'ft_ckpt_path': str(ft_ckpt),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}", flush=True)

    results = []
    for s in args.seeds:
        res = run_seed(s, device=device)
        if res is not None:
            results.append(res)
        out = args.out or (RES_DIR / 'phase4_physionet2012.json')
        with open(out, 'w') as f:
            json.dump({'dataset': 'physionet2012', 'results': results}, f,
                      indent=2)

    if not results:
        return
    auprcs = [r['auprc'] for r in results]
    aurocs = [r['auroc'] for r in results]
    monos = [r['monotonicity_violation_rate'] for r in results]
    print(f"\n=== SUMMARY physionet2012 (n={len(auprcs)}) ===", flush=True)
    print(f"  AUPRC: {np.mean(auprcs):.4f} +/- {np.std(auprcs):.4f}",
          flush=True)
    print(f"  AUROC: {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}",
          flush=True)
    print(f"  mono max={max(monos):.6f}", flush=True)

    out = args.out or (RES_DIR / 'phase4_physionet2012.json')
    with open(out, 'w') as f:
        json.dump({'dataset': 'physionet2012', 'n_seeds': len(results),
                   'auprc_mean': float(np.mean(auprcs)),
                   'auprc_std':  float(np.std(auprcs)),
                   'auroc_mean': float(np.mean(aurocs)),
                   'auroc_std':  float(np.std(aurocs)),
                   'monotonicity_max': float(max(monos)),
                   'results': results}, f, indent=2)
    print(f"wrote {out}", flush=True)


if __name__ == '__main__':
    main()
