"""V20 Phase 0b: Pred-FT vs Frozen vs E2E vs Scratch on C-MAPSS FD001.

THE headline experiment. Tests paper contribution #1: predictor finetuning
matches or beats E2E at low label budgets.

5 modes x 2 label fractions (1.0, 0.05) x 5 seeds = 50 runs.

Evaluation: per-window binary F1 over W=16 windows (primary), RMSE/NASA-S (legacy).
"""
import sys, json, time, copy
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
from pred_ft_utils import (
    ProbeH, MultiHorizonHead,
    CMAPSSWindowedDataset, CMAPSSTestWindowedDataset, collate_windowed,
    train_one, evaluate,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_SEED = 42
CKPT_PATH = V17 / 'ckpts' / f'v17_seed{CKPT_SEED}_best.pt'

# V17 architecture
ARCH = dict(n_sensors=N_SENSORS, patch_length=1,
            d_model=256, n_heads=4, n_layers=2, d_ff=1024,
            dropout=0.1, ema_momentum=0.99, predictor_hidden=1024)

N_WINDOWS = 16
BUDGETS = [1.0, 0.05]
SEEDS = [0, 1, 2, 3, 4]
MODES = ['probe_h', 'frozen_multi', 'pred_ft', 'e2e', 'scratch']

# Training config (per mode)
CFG = {
    'probe_h':      dict(lr=1e-3, wd=1e-2, n_epochs=200, patience=25),
    'frozen_multi': dict(lr=1e-3, wd=1e-2, n_epochs=200, patience=25),
    'pred_ft':      dict(lr=1e-3, wd=1e-2, n_epochs=100, patience=20),
    'e2e':          dict(lr=1e-4, wd=1e-4, n_epochs=60,  patience=15),
    'scratch':      dict(lr=1e-4, wd=1e-4, n_epochs=100, patience=25),
}


def load_pretrained_model():
    model = TrajectoryJEPA(**ARCH).to(DEVICE)
    sd = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(sd)
    return model


def fresh_scratch_model(seed: int):
    """Random-init model (no pretraining). Seed affects init."""
    torch.manual_seed(seed); np.random.seed(seed)
    return TrajectoryJEPA(**ARCH).to(DEVICE)


def make_head(mode: str, d_model: int = 256):
    if mode == 'probe_h':
        return ProbeH(d_model).to(DEVICE)
    else:
        return MultiHorizonHead(d_model, n_windows=N_WINDOWS).to(DEVICE)


def run_single(data, mode: str, budget: float, seed: int):
    torch.manual_seed(seed); np.random.seed(seed)
    # Model (fresh copy per run to avoid leakage across modes)
    if mode == 'scratch':
        model = fresh_scratch_model(seed)
    else:
        model = load_pretrained_model()

    head = make_head(mode, ARCH['d_model'])

    sub_eng = subsample_engines(data['train_engines'], budget, seed=seed)

    tr_ds = CMAPSSWindowedDataset(sub_eng, n_cuts_per_engine=5,
                                  rul_cap=RUL_CAP, seed=seed)
    va_ds = CMAPSSWindowedDataset(data['val_engines'], n_cuts_per_engine=10,
                                  rul_cap=RUL_CAP, seed=seed + 111)
    te_ds = CMAPSSTestWindowedDataset(data['test_engines'], data['test_rul'],
                                      rul_cap=RUL_CAP)

    tr = DataLoader(tr_ds, batch_size=16, shuffle=True, collate_fn=collate_windowed)
    va = DataLoader(va_ds, batch_size=16, shuffle=False, collate_fn=collate_windowed)
    te = DataLoader(te_ds, batch_size=16, shuffle=False, collate_fn=collate_windowed)

    cfg = CFG[mode]
    train_out = train_one(model, head, tr, va, mode=mode,
                          n_windows=N_WINDOWS, device=DEVICE, **cfg)
    eval_out = evaluate(model, head, te, mode=mode,
                        n_windows=N_WINDOWS, rul_cap=RUL_CAP, device=DEVICE)

    return {
        'mode': mode, 'budget': budget, 'seed': seed,
        'val_mse': train_out['best_val'],
        'final_epoch': train_out['final_epoch'],
        'test_rmse': eval_out['legacy']['rmse'],
        'test_nasa': eval_out['legacy']['nasa_score'],
        'test_mae': eval_out['legacy']['mae'],
        'per_window_f1_mean': eval_out['per_window']['f1_mean'],
        'per_window_precision_mean': eval_out['per_window']['precision_mean'],
        'per_window_recall_mean': eval_out['per_window']['recall_mean'],
        'per_window_auroc_mean': eval_out['per_window']['auroc_mean'],
        'per_window_detail': eval_out['per_window']['per_window'],
        'n_train_engines': len(sub_eng),
        'n_train_items': len(tr_ds),
    }


def aggregate(per_seed):
    from scipy.stats import t as t_dist
    keys = ['test_rmse', 'test_nasa', 'test_mae',
            'per_window_f1_mean', 'per_window_precision_mean',
            'per_window_recall_mean', 'per_window_auroc_mean',
            'val_mse']
    out = {'n_seeds': len(per_seed), 'per_seed': per_seed}
    for k in keys:
        vals = np.array([r[k] for r in per_seed if np.isfinite(r[k])])
        if len(vals) == 0:
            out[f'{k}_mean'] = float('nan')
            out[f'{k}_std'] = float('nan')
            continue
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        if len(vals) > 1:
            t_crit = float(t_dist.ppf(0.975, df=len(vals) - 1))
            margin = t_crit * std / np.sqrt(len(vals))
        else:
            margin = float('nan')
        out[f'{k}_mean'] = mean
        out[f'{k}_std'] = std
        out[f'{k}_ci95_lo'] = mean - margin
        out[f'{k}_ci95_hi'] = mean + margin
    return out


def fmt(agg, metric, lo_is_better=False):
    m = agg.get(f'{metric}_mean', float('nan'))
    s = agg.get(f'{metric}_std', float('nan'))
    lo = agg.get(f'{metric}_ci95_lo', float('nan'))
    hi = agg.get(f'{metric}_ci95_hi', float('nan'))
    n = agg.get('n_seeds', 0)
    if np.isnan(m):
        return 'N/A'
    if n <= 1 or np.isnan(s):
        return f'{m:.4f} (1s)'
    return f'{m:.4f} ± {s:.4f} ({n}s, 95% CI [{lo:.4f}, {hi:.4f}])'


def main():
    V20.mkdir(exist_ok=True)
    out_path = V20 / 'phase0_pred_ft.json'
    print("=" * 80)
    print("V20 Phase 0b: Pred-FT headline experiment on C-MAPSS FD001")
    print(f"  {len(MODES)} modes x {len(BUDGETS)} budgets x {len(SEEDS)} seeds "
          f"= {len(MODES) * len(BUDGETS) * len(SEEDS)} runs")
    print(f"  Arch: V17 (d_model=256, 2L, 4H), ckpt=v17_seed{CKPT_SEED}_best.pt")
    print(f"  Eval: per-window F1 (W={N_WINDOWS}) + legacy RMSE/NASA-S")
    print("=" * 80, flush=True)

    t0 = time.time()
    data = load_cmapss_subset('FD001')
    print(f"FD001 loaded: {len(data['train_engines'])} train, "
          f"{len(data['val_engines'])} val, {len(data['test_engines'])} test",
          flush=True)

    results = {}
    for mode in MODES:
        for budget in BUDGETS:
            key = f"{mode}@{budget}"
            results[key] = []
            for seed in SEEDS:
                t1 = time.time()
                r = run_single(data, mode, budget, seed)
                dt = time.time() - t1
                results[key].append(r)
                print(f"  [{mode:14s} b={budget*100:4.0f}% s={seed}] "
                      f"val={r['val_mse']:.4f} RMSE={r['test_rmse']:.2f} "
                      f"F1w={r['per_window_f1_mean']:.3f} "
                      f"AUROCw={r['per_window_auroc_mean']:.3f} "
                      f"ep={r['final_epoch']} ({dt:.0f}s)", flush=True)

                # Incremental save every run (overwrite)
                save_all(results, t0, out_path)

    save_all(results, t0, out_path)

    print("\n" + "=" * 80)
    print("V20 Phase 0b SUMMARY (per-window F1 | RMSE)")
    print("=" * 80)
    print(f"{'Mode':16s} | {'100%':>45s} | {'5%':>45s}")
    print("-" * 110)
    for mode in MODES:
        r100 = aggregate(results[f'{mode}@1.0'])
        r5 = aggregate(results[f'{mode}@0.05'])
        f100 = fmt(r100, 'per_window_f1_mean')
        f5 = fmt(r5, 'per_window_f1_mean')
        print(f"{mode:16s} | {f100:>45s} | {f5:>45s}")
    print()
    print("RMSE table:")
    print(f"{'Mode':16s} | {'100%':>45s} | {'5%':>45s}")
    print("-" * 110)
    for mode in MODES:
        r100 = aggregate(results[f'{mode}@1.0'])
        r5 = aggregate(results[f'{mode}@0.05'])
        print(f"{mode:16s} | {fmt(r100, 'test_rmse'):>45s} | {fmt(r5, 'test_rmse'):>45s}")
    print(f"\nRuntime: {(time.time() - t0)/60:.1f} min")


def save_all(results, t0, out_path):
    summary = {
        'config': 'v20_phase0_pred_ft',
        'ckpt': str(CKPT_PATH),
        'arch': ARCH,
        'n_windows': N_WINDOWS,
        'budgets': BUDGETS,
        'seeds': SEEDS,
        'modes': MODES,
        'cfg': CFG,
        'runtime_min': (time.time() - t0) / 60,
        'results': {k: aggregate(v) for k, v in results.items()},
    }
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=float)


if __name__ == '__main__':
    main()
