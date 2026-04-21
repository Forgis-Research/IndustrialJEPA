"""V20 Phase 3b: EMA vs SIGReg pretraining comparison via pred-FT.

v17 phase 3 trained two SIGReg variants:
  - SIGReg-enc : SIGReg regularization on encoder output
  - SIGReg-pred: SIGReg regularization on predictor output

Both saved as full TrajectoryJEPA state dicts. Baseline (EMA) is
v17_seed{42,123,456}_best.pt used in Phase 0b.

We run pred-FT on FD001 (100% + 5%, 5 seeds) for each SIGReg variant and
compare to the Phase 0b EMA pred-FT numbers.
"""
import sys, json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')
V11 = ROOT / 'experiments' / 'v11'
V17 = ROOT / 'experiments' / 'v17'
V20 = ROOT / 'experiments' / 'v20'
sys.path.insert(0, str(V11)); sys.path.insert(0, str(V20)); sys.path.insert(0, str(ROOT))

from models import TrajectoryJEPA
from data_utils import load_cmapss_subset, N_SENSORS, RUL_CAP
from train_utils import subsample_engines
from pred_ft_utils import (MultiHorizonHead, CMAPSSWindowedDataset,
                           CMAPSSTestWindowedDataset, collate_windowed,
                           train_one, evaluate)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ARCH = dict(n_sensors=N_SENSORS, patch_length=1,
            d_model=256, n_heads=4, n_layers=2, d_ff=1024,
            dropout=0.1, ema_momentum=0.99, predictor_hidden=1024)
N_WINDOWS = 16
BUDGETS = [1.0, 0.05]
SEEDS = [0, 1, 2, 3, 4]
VARIANTS = [
    ('sigreg_enc',  V17 / 'ckpts' / 'v17_phase3_enc_seed42_best.pt'),
    ('sigreg_pred', V17 / 'ckpts' / 'v17_phase3_pred_seed42_best.pt'),
]
MODE_CFG = dict(lr=1e-3, wd=1e-2, n_epochs=100, patience=20)


def load_full_ckpt(path):
    model = TrajectoryJEPA(**ARCH).to(DEVICE)
    sd = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(sd)
    return model


def run_one(data, ckpt_path, budget, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = load_full_ckpt(ckpt_path)
    head = MultiHorizonHead(ARCH['d_model'], n_windows=N_WINDOWS).to(DEVICE)

    sub = subsample_engines(data['train_engines'], budget, seed=seed)
    tr_ds = CMAPSSWindowedDataset(sub, n_cuts_per_engine=5, rul_cap=RUL_CAP, seed=seed)
    va_ds = CMAPSSWindowedDataset(data['val_engines'], n_cuts_per_engine=10,
                                  rul_cap=RUL_CAP, seed=seed + 111)
    te_ds = CMAPSSTestWindowedDataset(data['test_engines'], data['test_rul'])
    tr = DataLoader(tr_ds, batch_size=16, shuffle=True, collate_fn=collate_windowed)
    va = DataLoader(va_ds, batch_size=16, shuffle=False, collate_fn=collate_windowed)
    te = DataLoader(te_ds, batch_size=16, shuffle=False, collate_fn=collate_windowed)

    t = train_one(model, head, tr, va, mode='pred_ft',
                  n_windows=N_WINDOWS, device=DEVICE, **MODE_CFG)
    ev = evaluate(model, head, te, mode='pred_ft',
                  n_windows=N_WINDOWS, rul_cap=RUL_CAP, device=DEVICE)
    return {
        'budget': budget, 'seed': seed,
        'val_mse': t['best_val'], 'final_epoch': t['final_epoch'],
        'test_rmse': ev['legacy']['rmse'], 'test_nasa': ev['legacy']['nasa_score'],
        'per_window_f1_mean': ev['per_window']['f1_mean'],
        'per_window_auroc_mean': ev['per_window']['auroc_mean'],
        'per_window_precision_mean': ev['per_window']['precision_mean'],
        'per_window_recall_mean': ev['per_window']['recall_mean'],
    }


def aggregate(rs):
    from scipy.stats import t as t_dist
    keys = ['test_rmse', 'per_window_f1_mean', 'per_window_auroc_mean']
    out = {'n_seeds': len(rs), 'per_seed': rs}
    for k in keys:
        vals = np.array([r[k] for r in rs if np.isfinite(r[k])])
        if not len(vals):
            out[f'{k}_mean'] = float('nan'); out[f'{k}_std'] = float('nan'); continue
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        if len(vals) > 1:
            tc = float(t_dist.ppf(0.975, df=len(vals) - 1))
            margin = tc * std / np.sqrt(len(vals))
        else:
            margin = float('nan')
        out[f'{k}_mean'] = mean; out[f'{k}_std'] = std
        out[f'{k}_ci95_lo'] = mean - margin
        out[f'{k}_ci95_hi'] = mean + margin
    return out


def main():
    data = load_cmapss_subset('FD001')
    t0 = time.time()
    all_results = {}
    for variant, ckpt in VARIANTS:
        print(f"\n=== {variant} ({ckpt.name}) ===", flush=True)
        for b in BUDGETS:
            key = f"{variant}@{b}"
            all_results[key] = []
            for seed in SEEDS:
                t1 = time.time()
                r = run_one(data, ckpt, b, seed)
                dt = time.time() - t1
                all_results[key].append(r)
                print(f"  [{variant:12s} b={b*100:4.0f}% s={seed}] "
                      f"RMSE={r['test_rmse']:.2f} F1w={r['per_window_f1_mean']:.3f} "
                      f"AUROCw={r['per_window_auroc_mean']:.3f} ({dt:.0f}s)",
                      flush=True)
                with open(V20 / 'phase3_sigreg.json', 'w') as f:
                    json.dump({'config': 'v20_phase3_sigreg',
                               'seeds': SEEDS, 'budgets': BUDGETS,
                               'variants': [v[0] for v in VARIANTS],
                               'runtime_min': (time.time()-t0)/60,
                               'results': {k: aggregate(v) for k, v in all_results.items()}},
                              f, indent=2, default=float)

    print(f"\nRuntime: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
