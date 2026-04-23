"""V24 phase 12: isolate the pretrained-predictor contribution.

The reviewer asks: when we pred-FT, is it the *pretrained* predictor that
matters, or would a freshly-initialized predictor of the same size do just
as well on top of the pretrained encoder?

Protocol on FD001 (3 seeds):
  A. Pred-FT (standard)       : load encoder+predictor from pretrain, freeze
                                 encoder, finetune predictor + head.
  B. Pred-FT (reset predictor): load encoder from pretrain, RE-INIT predictor
                                 to random, freeze encoder, finetune.

Same labels, horizons, splits. Same number of finetuning epochs. Everything
else identical.
"""

import argparse
import copy
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
sys.path.insert(0, str(FAM_DIR / 'experiments/v11'))
sys.path.insert(0, str(V24))

from model import FAM, Predictor
from train import (
    EventDataset, collate_event, finetune, evaluate, get_horizons,
)
from _cmapss_raw import load_cmapss_raw


def build_event_concat(engines, stride, max_context=512, min_context=128):
    ds = []
    for eid, seq in engines.items():
        T = len(seq)
        if T <= min_context:
            continue
        labels = np.zeros(T, dtype=np.int32)
        labels[T - 1] = 1
        d = EventDataset(seq, labels, max_context=max_context, stride=stride,
                         max_future=200, min_context=min_context)
        if len(d) > 0:
            ds.append(d)
    return ConcatDataset(ds) if ds else ConcatDataset([])


def run_one(data, seed: int, reset_predictor: bool, device: str = 'cuda'):
    torch.manual_seed(seed)
    np.random.seed(seed)

    tag = f"FD001_s{seed}_{'reset' if reset_predictor else 'pretrained'}"
    print(f"\n=== {tag} ===", flush=True)

    train_ft = build_event_concat(data['train_engines'], stride=4)
    val_ft   = build_event_concat(data['val_engines'],   stride=4)
    test_ft  = build_event_concat(data['test_engines'],  stride=1)

    model = FAM(n_channels=14, patch_size=16, d_model=256, n_heads=4,
                n_layers=2, d_ff=256, dropout=0.1, ema_momentum=0.99,
                predictor_hidden=256)
    ckpt = V24 / f'ckpts/FD001_s{seed}_pretrain.pt'
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))

    if reset_predictor:
        # Re-init the Predictor module to match the untrained distribution
        fresh = Predictor(d_model=256, hidden=256)
        model.predictor.load_state_dict(fresh.state_dict())
        print("  predictor RE-INITIALIZED", flush=True)

    horizons = get_horizons('FD001')
    tloader = DataLoader(train_ft, batch_size=256, shuffle=True,
                         collate_fn=collate_event)
    vloader = DataLoader(val_ft, batch_size=256, shuffle=False,
                         collate_fn=collate_event)
    te_loader = DataLoader(test_ft, batch_size=256, shuffle=False,
                           collate_fn=collate_event)

    t0 = time.time()
    ft_out = finetune(model, tloader, vloader, horizons, mode='pred_ft',
                      lr=1e-3, n_epochs=40, patience=8, device=device)
    ft_time = time.time() - t0
    print(f"  finetune done in {ft_time:.1f}s best_val={ft_out['best_val']:.4f}",
          flush=True)

    eval_out = evaluate(model, te_loader, horizons, mode='pred_ft', device=device)
    primary = eval_out['primary']
    mono = eval_out['monotonicity']
    print(f"  AUPRC={primary['auprc']:.4f} AUROC={primary['auroc']:.4f} "
          f"F1={primary['f1_best']:.4f}", flush=True)
    return {
        'seed': seed,
        'reset_predictor': reset_predictor,
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
    ap.add_argument('--out', default=str(V24 / 'results/phase12_predictor_ablation.json'))
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = load_cmapss_raw('FD001')

    results = []
    for seed in args.seeds:
        for reset in (False, True):
            r = run_one(data, seed, reset, device=device)
            results.append(r)
            with open(args.out, 'w') as f:
                json.dump({'results': results}, f, indent=2)

    # Paired summary
    import scipy.stats as stats
    by_cond = {'pretrained': [], 'reset': []}
    for r in results:
        by_cond['reset' if r['reset_predictor'] else 'pretrained'].append(r['auprc'])
    pt = np.array(by_cond['pretrained']); rs = np.array(by_cond['reset'])
    if len(pt) == len(rs) and len(pt) >= 2:
        t, p = stats.ttest_rel(pt, rs)
        print(f"\nPAIRED: pretrained {pt.mean():.4f} +/- {pt.std():.4f}  vs  "
              f"reset {rs.mean():.4f} +/- {rs.std():.4f}   "
              f"delta {pt.mean() - rs.mean():+.4f}   "
              f"t({len(pt)-1})={t:.2f}, p={p:.4f}", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
