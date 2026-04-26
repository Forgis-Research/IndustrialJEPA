"""V24 phase 12 (MBA): predictor pretrain vs random init on MBA.

Third dataset for the pretrained-vs-random predictor ablation after FD001
and SMAP. MBA is where FAM wins clearest (AUPRC 0.947 vs Chronos 0.918),
so if the pretrained predictor helps anywhere, it should show up here.
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
from data.mba import load_mba

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
    tag = f"MBA_s{seed}_{'reset' if reset_predictor else 'pretrained'}"
    print(f"\n=== {tag} ===", flush=True)

    tr_ds = build_anomaly_event(ft['ft_train'], stride=4)
    va_ds = build_anomaly_event(ft['ft_val'],   stride=4)
    te_ds = build_anomaly_event(ft['ft_test'],  stride=1)

    model = FAM(n_channels=2, patch_size=16, d_model=256, n_heads=4,
                n_layers=2, d_ff=256, dropout=0.1, ema_momentum=0.99,
                predictor_hidden=256)
    model.load_state_dict(torch.load(
        V24 / f'ckpts/MBA_s{seed}_pretrain.pt', map_location='cpu'))

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
        'monotonicity_violation_rate': float(mono['violation_rate']),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    ap.add_argument('--out', default=str(V24 / 'results/phase12_mba_ablation.json'))
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # MBA is single-stream; replicate the phase4 split logic
    d = load_mba(normalize=False)
    X = d['test'].astype(np.float32); y = d['labels'].astype(np.int32)
    T = len(X); t1 = int(0.6 * T); t2 = int(0.7 * T); gap = 200
    ft = {
        'ft_train': [{'test': X[:t1], 'labels': y[:t1]}],
        'ft_val':   [{'test': X[t1 + gap:t2], 'labels': y[t1 + gap:t2]}],
        'ft_test':  [{'test': X[t2 + gap:], 'labels': y[t2 + gap:]}],
    }

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
        print(f"\nMBA PAIRED: pretrained {np.mean(pt):.4f} +/- {np.std(pt):.4f}  "
              f"vs reset {np.mean(rs):.4f} +/- {np.std(rs):.4f}  "
              f"delta {np.mean(pt) - np.mean(rs):+.4f}  "
              f"t({len(pt)-1})={t:.2f}, p={p:.4f}", flush=True)


if __name__ == '__main__':
    main()
