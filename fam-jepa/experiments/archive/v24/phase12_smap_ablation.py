"""V24 phase 12 (SMAP): predictor pretrain vs random init on SMAP.

Mirror of phase12_predictor_ablation.py but on SMAP where the gap between
FAM and zero-shot Chronos-2 is largest (+0.110 AUPRC). If the pretrained
predictor matters anywhere, it should show up here.
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
V24 = FAM_DIR / 'experiments/v24'
sys.path.insert(0, str(FAM_DIR))

from model import FAM, Predictor
from train import (EventDataset, collate_event, finetune, evaluate)
from data.smap_msl import split_smap_entities

HORIZONS = [1, 5, 10, 20, 50, 100, 150, 200]


def build_anomaly_event(entity_list, stride, max_context=512, min_context=128):
    ds = []
    for e in entity_list:
        x = e['test']; y = e['labels']
        if len(x) <= min_context + 1:
            continue
        d = EventDataset(x, y, max_context=max_context, stride=stride,
                         max_future=200, min_context=min_context)
        if len(d) > 0:
            ds.append(d)
    return ConcatDataset(ds) if ds else ConcatDataset([])


def run_one(ft, seed, reset_predictor, device='cuda'):
    torch.manual_seed(seed); np.random.seed(seed)
    tag = f"SMAP_s{seed}_{'reset' if reset_predictor else 'pretrained'}"
    print(f"\n=== {tag} ===", flush=True)

    tr_ds = build_anomaly_event(ft['ft_train'], stride=4)
    va_ds = build_anomaly_event(ft['ft_val'],   stride=4)
    te_ds = build_anomaly_event(ft['ft_test'],  stride=1)

    model = FAM(n_channels=25, patch_size=16, d_model=256, n_heads=4,
                n_layers=2, d_ff=256, dropout=0.1, ema_momentum=0.99,
                predictor_hidden=256)
    model.load_state_dict(torch.load(
        V24 / f'ckpts/SMAP_s{seed}_pretrain.pt', map_location='cpu'))

    if reset_predictor:
        fresh = Predictor(d_model=256, hidden=256)
        model.predictor.load_state_dict(fresh.state_dict())
        print("  predictor RE-INITIALIZED", flush=True)

    tloader = DataLoader(tr_ds, batch_size=128, shuffle=True, collate_fn=collate_event)
    vloader = DataLoader(va_ds, batch_size=128, shuffle=False, collate_fn=collate_event)
    te_loader = DataLoader(te_ds, batch_size=128, shuffle=False, collate_fn=collate_event)

    t0 = time.time()
    ft_out = finetune(model, tloader, vloader, HORIZONS, mode='pred_ft',
                      lr=1e-3, n_epochs=30, patience=6, device=device)
    ft_time = time.time() - t0
    eval_out = evaluate(model, te_loader, HORIZONS, mode='pred_ft', device=device)
    primary = eval_out['primary']; mono = eval_out['monotonicity']
    print(f"  finetune {ft_time:.1f}s  AUPRC={primary['auprc']:.4f}  "
          f"AUROC={primary['auroc']:.4f}  F1={primary['f1_best']:.4f}",
          flush=True)
    return {
        'seed': seed, 'reset_predictor': reset_predictor,
        'ft_best_val': float(ft_out['best_val']),
        'ft_time_s': ft_time,
        'auprc': float(primary['auprc']),
        'auroc': float(primary['auroc']),
        'f1_best': float(primary['f1_best']),
        'precision_best': float(primary['precision_best']),
        'recall_best': float(primary['recall_best']),
        'prevalence': float(primary['prevalence']),
        'monotonicity_violation_rate': float(mono['violation_rate']),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    ap.add_argument('--out', default=str(V24 / 'results/phase12_smap_ablation.json'))
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ft = split_smap_entities(normalize=False)

    results = []
    for seed in args.seeds:
        for reset in (False, True):
            r = run_one(ft, seed, reset, device=device)
            results.append(r)
            with open(args.out, 'w') as f:
                json.dump({'results': results}, f, indent=2)

    import scipy.stats as stats
    pt = [r['auprc'] for r in results if not r['reset_predictor']]
    rs = [r['auprc'] for r in results if r['reset_predictor']]
    if len(pt) == len(rs) and len(pt) >= 2:
        t, p = stats.ttest_rel(pt, rs)
        print(f"\nSMAP PAIRED: pretrained {np.mean(pt):.4f} +/- {np.std(pt):.4f}  "
              f"vs reset {np.mean(rs):.4f} +/- {np.std(rs):.4f}  "
              f"delta {np.mean(pt) - np.mean(rs):+.4f}  "
              f"t({len(pt)-1})={t:.2f}, p={p:.4f}", flush=True)


if __name__ == '__main__':
    main()
