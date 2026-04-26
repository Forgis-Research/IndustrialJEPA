"""V24 Phase 9: FD001 label efficiency — pred-FT vs E2E at 5/10/50/100% labels.

Engine-level subsampling (v21 protocol). 5 seeds each.
Tests the paper's core claim: a tiny finetunable predictor on top of a
frozen encoder approaches end-to-end finetuning at small label budgets.

Reuses phase2 FD001 pretrain ckpts.
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
RES_DIR = V24_DIR / 'results'
SURF_DIR = V24_DIR / 'surfaces'

sys.path.insert(0, str(FAM_DIR))
sys.path.insert(0, str(FAM_DIR / 'experiments/v11'))
sys.path.insert(0, str(V24_DIR))

from model import FAM
from train import (
    EventDataset, collate_event, finetune, evaluate, get_horizons, save_surface,
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


def subsample_engines(engines: dict, frac: float, seed: int) -> dict:
    """Engine-level random subsample at deterministic seed."""
    ids = sorted(engines.keys())
    n = max(1, int(round(frac * len(ids))))
    rng = np.random.default_rng(seed)
    chosen = list(rng.choice(ids, size=n, replace=False))
    return {i: engines[i] for i in chosen}


def run_one(budget: float, mode: str, seed: int, data: dict,
            pretrain_ckpt: Path, device: str = 'cuda',
            ft_epochs: int = 40, ft_patience: int = 8) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_eng_sub = subsample_engines(data['train_engines'], budget, seed)

    train_ft = build_event_concat(train_eng_sub, stride=4)
    val_ft = build_event_concat(data['val_engines'], stride=4)
    test_ft = build_event_concat(data['test_engines'], stride=1)

    print(f"  [{mode:7s} b={budget*100:5.1f}% s={seed}] "
          f"n_eng={len(train_eng_sub)} train={len(train_ft)} "
          f"val={len(val_ft)} test={len(test_ft)}", flush=True)

    if len(train_ft) == 0:
        return None

    tloader = DataLoader(train_ft, batch_size=256, shuffle=True,
                         collate_fn=collate_event)
    vloader = DataLoader(val_ft, batch_size=256, shuffle=False,
                         collate_fn=collate_event)
    test_loader = DataLoader(test_ft, batch_size=256, shuffle=False,
                             collate_fn=collate_event)

    model = FAM(n_channels=14, patch_size=16, d_model=256, n_heads=4,
                n_layers=2, d_ff=256, dropout=0.1, ema_momentum=0.99,
                predictor_hidden=256)
    model.load_state_dict(torch.load(pretrain_ckpt, map_location='cpu'))

    horizons = get_horizons('FD001')
    t0 = time.time()
    ft_out = finetune(model, tloader, vloader, horizons, mode=mode,
                      lr=1e-3, n_epochs=ft_epochs, patience=ft_patience,
                      device=device)
    ft_time = time.time() - t0

    eval_out = evaluate(model, test_loader, horizons, mode=mode, device=device)
    primary = eval_out['primary']
    mono = eval_out['monotonicity']

    print(f"    -> AUPRC={primary['auprc']:.4f}  AUROC={primary['auroc']:.4f}  "
          f"({ft_time:.0f}s)", flush=True)

    return {
        'budget': budget,
        'mode': mode,
        'seed': seed,
        'n_train_engines': len(train_eng_sub),
        'ft_time_s': ft_time,
        'auprc': float(primary['auprc']),
        'auroc': float(primary['auroc']),
        'f1_best': float(primary['f1_best']),
        'monotonicity_violation_rate': float(mono['violation_rate']),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budgets', nargs='+', type=float,
                    default=[0.05, 0.10, 0.50, 1.0])
    ap.add_argument('--seeds', nargs='+', type=int,
                    default=[42, 123, 456, 789, 1024])
    ap.add_argument('--modes', nargs='+', default=['pred_ft', 'e2e'])
    ap.add_argument('--pretrain-seed', type=int, default=42,
                    help='Which pretrain ckpt to use for all FT runs.')
    ap.add_argument('--out', default=str(RES_DIR / 'phase9_label_eff.json'))
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pretrain_ckpt = CKPT_DIR / f'FD001_s{args.pretrain_seed}_pretrain.pt'
    if not pretrain_ckpt.exists():
        raise FileNotFoundError(pretrain_ckpt)

    print(f"device={device}  pretrain={pretrain_ckpt.name}", flush=True)
    print(f"budgets={args.budgets}  seeds={args.seeds}  modes={args.modes}",
          flush=True)

    data = load_cmapss_raw('FD001')

    results = []
    for mode in args.modes:
        for b in args.budgets:
            for s in args.seeds:
                res = run_one(b, mode, s, data, pretrain_ckpt, device=device)
                if res is not None:
                    results.append(res)
                with open(args.out, 'w') as f:
                    json.dump(results, f, indent=2)

    print('\n=== SUMMARY ===', flush=True)
    for mode in args.modes:
        for b in args.budgets:
            vals = [r['auprc'] for r in results
                    if r['mode'] == mode and r['budget'] == b]
            if vals:
                print(f"  {mode:8s} b={b*100:5.1f}%  AUPRC={np.mean(vals):.4f} "
                      f"+/- {np.std(vals):.4f}  (n={len(vals)})", flush=True)

    # Paired test per budget
    try:
        from scipy.stats import ttest_rel, wilcoxon
        print('\nPaired tests (pred_ft vs e2e):', flush=True)
        for b in args.budgets:
            p = {r['seed']: r['auprc'] for r in results
                 if r['mode'] == 'pred_ft' and r['budget'] == b}
            e = {r['seed']: r['auprc'] for r in results
                 if r['mode'] == 'e2e' and r['budget'] == b}
            seeds = sorted(set(p) & set(e))
            if len(seeds) < 3:
                continue
            pv = np.array([p[s] for s in seeds])
            ev = np.array([e[s] for s in seeds])
            t, pt = ttest_rel(pv, ev)
            try:
                w, pw = wilcoxon(pv, ev, alternative='less')
            except ValueError:
                w, pw = float('nan'), float('nan')
            print(f"  b={b*100:5.1f}%  pred_ft-e2e={np.mean(pv - ev):+.4f}  "
                  f"t={t:.2f}  p_t={pt:.3f}  wilcoxon_p_less={pw:.3f}",
                  flush=True)
    except ImportError:
        pass

    print(f"\nwrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
