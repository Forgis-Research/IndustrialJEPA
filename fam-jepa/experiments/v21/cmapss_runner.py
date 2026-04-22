"""V21 Phase 2: C-MAPSS Breakthrough Table with AUPRC.

Uses the pretrained v17 JEPA encoder, replaces v20's scalar-RUL + MSE head
with the v21 per-horizon EventHead + pos-weighted BCE, evaluates on the
canonical horizon grid, and stores the probability surface.

Seeds: 3 default (42, 123, 456).

Modes per subset:
  FD001: probe_h, pred_ft, e2e, scratch at 100% and 5% labels
  FD002: pred_ft at 100% and 5%
  FD003: pred_ft at 100% and 5% (uses 200-epoch ckpt)

Primary metric: AUPRC pooled over p(t, Δt).
Legacy metric: RMSE via surface -> threshold-crossing RUL.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path('/home/sagemaker-user/IndustrialJEPA')
FAM = ROOT / 'fam-jepa'
V11 = FAM / 'experiments' / 'v11'
V21 = FAM / 'experiments' / 'v21'
CKPT_OLD = ROOT / 'mechanical-jepa' / 'experiments'

sys.path.insert(0, str(V11))
sys.path.insert(0, str(V21))
sys.path.insert(0, str(FAM))

from models import TrajectoryJEPA  # noqa: E402
from data_utils import load_cmapss_subset, N_SENSORS  # noqa: E402
from train_utils import subsample_engines  # noqa: E402
from pred_ft_utils import (  # noqa: E402
    CMAPSSSurfaceDataset, CMAPSSSurfaceTestDataset, collate_cmapss_surface,
    EventHead, ProbeHEvent,
    train_bce, evaluate_surface, estimate_pos_weight,
    save_surface, HORIZONS_STEPS,
)
from surface_to_legacy import (  # noqa: E402
    surface_to_rul, surface_to_rul_expected, rmse, nasa_score,
)
from evaluation.surface_metrics import (  # noqa: E402
    evaluate_probability_surface, auprc_per_horizon,
    monotonicity_violation_rate,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# V17 encoder architecture for C-MAPSS
ARCH = dict(n_sensors=N_SENSORS, patch_length=1,
            d_model=256, n_heads=4, n_layers=2, d_ff=1024,
            dropout=0.1, ema_momentum=0.99, predictor_hidden=1024)

# Training config per mode
CFG = {
    'probe_h': dict(lr=1e-3, wd=1e-2, n_epochs=80, patience=15),
    'pred_ft': dict(lr=1e-3, wd=1e-2, n_epochs=60,  patience=12),
    'e2e':     dict(lr=1e-4, wd=1e-4, n_epochs=40,  patience=10),
    'scratch': dict(lr=1e-4, wd=1e-4, n_epochs=60,  patience=15),
}


def _load_v17_ckpt(subset: str, seed: int) -> TrajectoryJEPA:
    """Load the pretrained JEPA.

    FD001: v17 has three seeds (42, 123, 456).
    FD002/FD003: only seed 42 is pretrained on this VM. We reuse that
    checkpoint for all FT seeds; variance comes from head init + FT noise
    (the encoder is frozen in pred_ft mode anyway).
    """
    if subset == 'FD001':
        ckpt = CKPT_OLD / 'v17' / 'ckpts' / f'v17_seed{seed}_best.pt'
        if not ckpt.exists():
            # Fallback to seed 42 (standard practice in v20 label-eff study):
            # FT variance is what we measure; pretrain is shared.
            ckpt = CKPT_OLD / 'v17' / 'ckpts' / 'v17_seed42_best.pt'
    elif subset == 'FD002':
        # Only seed 42 pretrained — reused for all FT seeds.
        candidates = [
            CKPT_OLD / 'v20' / 'ckpts_fd002' / 'v20_fd002_seed42_ep150.pt',
            CKPT_OLD / 'v11' / 'best_pretrain_fd002.pt',
        ]
        ckpt = next((p for p in candidates if p.exists()), candidates[-1])
    elif subset == 'FD003':
        candidates = [
            CKPT_OLD / 'v20' / 'ckpts_fd003' / 'v20_fd003_seed42_ep200.pt',
            CKPT_OLD / 'v20' / 'ckpts_fd003' / 'v20_fd003_seed42_ep100.pt',
            CKPT_OLD / 'v11' / 'best_pretrain_fd003_v2.pt',
        ]
        ckpt = next((p for p in candidates if p.exists()), candidates[-1])
    else:
        raise ValueError(subset)

    assert ckpt.exists(), f'no ckpt: {ckpt}'
    m = TrajectoryJEPA(**ARCH).to(DEVICE)
    sd = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    if isinstance(sd, dict) and 'model' in sd and 'context_encoder.proj.proj.weight' not in sd:
        sd = sd['model']
    m.load_state_dict(sd, strict=False)
    return m


def _make_model(subset: str, seed: int, mode: str) -> TrajectoryJEPA:
    if mode == 'scratch':
        torch.manual_seed(seed); np.random.seed(seed)
        return TrajectoryJEPA(**ARCH).to(DEVICE)
    return _load_v17_ckpt(subset, seed)


def _make_head(mode: str):
    if mode == 'probe_h':
        return ProbeHEvent(ARCH['d_model'], len(HORIZONS_STEPS)).to(DEVICE)
    return EventHead(ARCH['d_model']).to(DEVICE)


def run_single(data: dict, subset: str, mode: str,
               budget: float, seed: int) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time()

    model = _make_model(subset, seed, mode)
    head = _make_head(mode)

    sub_train = subsample_engines(data['train_engines'], budget, seed=seed)
    tr_ds = CMAPSSSurfaceDataset(sub_train, n_cuts_per_engine=5, seed=seed)
    va_ds = CMAPSSSurfaceDataset(data['val_engines'], n_cuts_per_engine=10,
                                 seed=seed + 111)
    te_ds = CMAPSSSurfaceTestDataset(data['test_engines'], data['test_rul'])

    tr = DataLoader(tr_ds, batch_size=16, shuffle=True,
                    collate_fn=collate_cmapss_surface)
    va = DataLoader(va_ds, batch_size=16, shuffle=False,
                    collate_fn=collate_cmapss_surface)
    te = DataLoader(te_ds, batch_size=16, shuffle=False,
                    collate_fn=collate_cmapss_surface)

    pw = estimate_pos_weight(tr, HORIZONS_STEPS)
    cfg = CFG[mode]
    train_out = train_bce(model, head, tr, va, mode=mode,
                          pos_weight=pw, horizons_eval=HORIZONS_STEPS,
                          device=DEVICE, **cfg)

    surf = evaluate_surface(model, head, te, mode=mode,
                            horizons=HORIZONS_STEPS, device=DEVICE)
    p, y = surf['p_surface'], surf['y_surface']

    # Save surface
    (V21 / 'surfaces').mkdir(exist_ok=True)
    key = f'{subset.lower()}_{mode}_b{int(budget*100)}_seed{seed}'
    surf_path = V21 / 'surfaces' / f'{key}.npz'
    save_surface(surf_path, p, y, HORIZONS_STEPS, surf['t_index'],
                 metadata={'dataset': subset, 'seed': seed, 'mode': mode,
                           'budget': budget, 'pos_weight': float(pw)})

    prim = evaluate_probability_surface(p, y)
    per_h = auprc_per_horizon(p, y, horizon_labels=HORIZONS_STEPS)
    mono = monotonicity_violation_rate(p)

    # Legacy RMSE — use surface-threshold crossing
    # True RUL for each test engine:
    true_rul = np.array([float(r) for _, r in te_ds.items], dtype=np.float32)
    pred_rul_cross = surface_to_rul(p, np.asarray(HORIZONS_STEPS))
    pred_rul_exp = surface_to_rul_expected(p, np.asarray(HORIZONS_STEPS))

    # Cap true_rul to horizon range for fair comparison (predictions are
    # clamped at h_max=100 by construction).
    h_max = float(HORIZONS_STEPS[-1])
    true_rul_capped = np.minimum(true_rul, h_max)
    rmse_cross = rmse(pred_rul_cross, true_rul_capped)
    rmse_exp = rmse(pred_rul_exp, true_rul_capped)
    rmse_uncapped = rmse(pred_rul_exp, true_rul)
    nasa_s = nasa_score(pred_rul_exp, true_rul_capped)

    dt = time.time() - t0
    return {
        'subset': subset, 'mode': mode, 'budget': budget, 'seed': seed,
        'primary': prim, 'per_horizon': per_h, 'monotonicity': mono,
        'legacy': {'rmse_cross': rmse_cross, 'rmse_expected': rmse_exp,
                   'rmse_uncapped': rmse_uncapped, 'nasa_score': nasa_s,
                   'horizon_cap': h_max},
        'train': {'best_val': train_out['best_val'],
                  'final_epoch': train_out['final_epoch']},
        'pos_weight': float(pw), 'n_train_items': len(tr_ds),
        'surface_file': str(surf_path), 'runtime_s': dt,
    }


def aggregate(per_seed):
    import numpy as np
    from scipy.stats import t as t_dist
    out = {'n_seeds': len(per_seed), 'per_seed': per_seed}
    metrics = [
        ('auprc', lambda r: r['primary']['auprc']),
        ('auroc', lambda r: r['primary']['auroc']),
        ('f1_best', lambda r: r['primary']['f1_best']),
        ('precision_best', lambda r: r['primary']['precision_best']),
        ('recall_best', lambda r: r['primary']['recall_best']),
        ('rmse_expected', lambda r: r['legacy']['rmse_expected']),
        ('rmse_cross', lambda r: r['legacy']['rmse_cross']),
        ('nasa', lambda r: r['legacy']['nasa_score']),
        ('mono_violation', lambda r: r['monotonicity']['violation_rate']),
    ]
    for name, fn in metrics:
        vals = np.array([fn(r) for r in per_seed], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            out[f'{name}_mean'] = float('nan'); out[f'{name}_std'] = float('nan'); continue
        m = float(vals.mean()); s = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        if len(vals) > 1:
            tc = float(t_dist.ppf(0.975, df=len(vals) - 1))
            margin = tc * s / np.sqrt(len(vals))
        else:
            margin = float('nan')
        out[f'{name}_mean'] = m; out[f'{name}_std'] = s
        out[f'{name}_ci95_lo'] = m - margin; out[f'{name}_ci95_hi'] = m + margin
    return out


def run_matrix(subset: str, modes, budgets, seeds, data=None):
    if data is None:
        data = load_cmapss_subset(subset)
    results = {}
    for mode in modes:
        for b in budgets:
            key = f'{mode}@{b}'
            results[key] = []
            for seed in seeds:
                try:
                    r = run_single(data, subset, mode, b, seed)
                    results[key].append(r)
                    print(f'  [{subset} {mode:8s} b={b*100:4.0f}% s={seed}] '
                          f"AUPRC={r['primary']['auprc']:.3f} "
                          f"AUROC={r['primary']['auroc']:.3f} "
                          f"RMSE={r['legacy']['rmse_expected']:.2f} "
                          f"mono={r['monotonicity']['violation_rate']:.3f} "
                          f"({r['runtime_s']:.0f}s)",
                          flush=True)
                except Exception as e:
                    print(f'  ERR [{subset} {mode} b={b} s={seed}]: {e}',
                          flush=True)
                    import traceback; traceback.print_exc()
    return {k: aggregate(v) for k, v in results.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subsets', nargs='+',
                    default=['FD001', 'FD002', 'FD003'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    ap.add_argument('--out', default=str(V21 / 'phase2_cmapss.json'))
    args = ap.parse_args()

    V21.mkdir(exist_ok=True)
    (V21 / 'surfaces').mkdir(exist_ok=True)
    out_path = Path(args.out)
    t0 = time.time()

    all_out = {}
    for ss in args.subsets:
        print(f'\n=== {ss} ===', flush=True)
        if ss == 'FD001':
            modes = ['probe_h', 'pred_ft', 'e2e', 'scratch']
            budgets = [1.0, 0.05]
        else:
            modes = ['pred_ft']
            budgets = [1.0, 0.05]
        all_out[ss] = run_matrix(ss, modes, budgets, args.seeds)
        with open(out_path, 'w') as f:
            json.dump({'results': all_out, 'seeds': args.seeds,
                       'runtime_min': (time.time() - t0) / 60},
                      f, indent=2, default=float)

    print(f'\n\nDONE in {(time.time()-t0)/60:.1f} min. Saved: {out_path}')


if __name__ == '__main__':
    main()
