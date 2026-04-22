"""V22 Phase 6: Compare encoder variants (baseline / A / B) on FD001 pred-FT.

Each variant is pretrained from scratch via pretrain_variants.py under an
identical protocol (fixed past window = 100, LogUniform horizon k, L1 loss).

This script freezes each pretrained encoder and runs pred-FT with the
v21 EventHead + pos-weighted BCE pipeline.  Same downstream head, same
data, same training budget -> measures representation quality.

Seeds: 3 (42, 123, 456).  Budget: 100% labels.
Writes surfaces, per-seed results, and a summary table.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path('/home/sagemaker-user/IndustrialJEPA')
FAM = ROOT / 'fam-jepa'
V11 = FAM / 'experiments' / 'v11'
V17 = FAM / 'experiments' / 'v17'
V21 = FAM / 'experiments' / 'v21'
V22 = FAM / 'experiments' / 'v22'

sys.path.insert(0, str(V11))
sys.path.insert(0, str(V17))
sys.path.insert(0, str(V21))
sys.path.insert(0, str(V22))
sys.path.insert(0, str(FAM))

from data_utils import load_cmapss_subset, N_SENSORS  # noqa: E402
from train_utils import subsample_engines  # noqa: E402
from models_variants import build_model, count_params  # noqa: E402
from pred_ft_utils import (  # noqa: E402
    EventHead, train_bce, evaluate_surface, estimate_pos_weight,
    save_surface, HORIZONS_STEPS,
)
from surface_to_legacy import (  # noqa: E402
    surface_to_rul, surface_to_rul_expected, rmse,
)
from evaluation.surface_metrics import (  # noqa: E402
    evaluate_probability_surface, auprc_per_horizon,
    monotonicity_violation_rate,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WINDOW = 100
D_MODEL = 256

TRAIN_CFG = dict(lr=1e-3, wd=1e-2, n_epochs=60, patience=12)


# ---------------------------------------------------------------------------
# Fixed-window CMAPSS datasets (past length == WINDOW)
# ---------------------------------------------------------------------------

class FixedWindowSurfaceDataset(Dataset):
    """Sample random cuts from each engine; past = last WINDOW cycles.

    Returns (past, tte) where tte is raw RUL in cycles at cut time t.
    """

    def __init__(self, engines, n_cuts_per_engine=5, window=WINDOW, seed=42):
        rng = np.random.default_rng(seed)
        self.items = []
        for eid, seq in engines.items():
            T = len(seq)
            if T < window + 1:
                continue
            t_lo, t_hi = window, T
            n_cuts = min(n_cuts_per_engine, t_hi - t_lo)
            if n_cuts <= 0:
                continue
            cuts = rng.integers(t_lo, t_hi + 1, size=n_cuts).tolist()
            for t in cuts:
                tte = float(T - t)
                past = torch.from_numpy(seq[t - window: t]).float()
                self.items.append((past, tte))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        past, tte = self.items[i]
        return past, torch.tensor(tte, dtype=torch.float32)


class FixedWindowSurfaceTestDataset(Dataset):
    """Test: one item per engine at its last observed cycle.

    Past = last WINDOW cycles (left-padded with zeros if engine is shorter).
    """

    def __init__(self, test_engines, test_rul, window=WINDOW):
        self.items = []
        for idx, eid in enumerate(sorted(test_engines.keys())):
            seq = test_engines[eid]
            if len(seq) < window:
                pad = np.zeros((window - len(seq), seq.shape[1]),
                               dtype=seq.dtype)
                seq = np.concatenate([pad, seq], axis=0)
            past = torch.from_numpy(seq[-window:]).float()
            tte = float(test_rul[idx])
            self.items.append((past, tte))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        past, tte = self.items[i]
        return past, torch.tensor(tte, dtype=torch.float32)


def collate_fixed(batch):
    pasts, ttes = zip(*batch)
    x = torch.stack(list(pasts), dim=0)  # (B, W, S)
    mask = torch.zeros(x.shape[:2], dtype=torch.bool)
    tte = torch.stack(ttes, dim=0)
    t = torch.zeros(x.shape[0], dtype=torch.long)
    return x, mask, tte, t


# ---------------------------------------------------------------------------
# Load pretrained variant from ckpt
# ---------------------------------------------------------------------------

def _load_variant(variant: str, seed: int):
    model = build_model(variant, n_sensors=N_SENSORS, window=WINDOW,
                        d_model=D_MODEL, n_heads=4, n_layers=2, d_ff=1024,
                        predictor_hidden=1024).to(DEVICE)
    ckpt = V22 / 'ckpts' / f'{variant}_fd001_seed{seed}_best.pt'
    assert ckpt.exists(), f'missing pretrained ckpt: {ckpt}'
    sd = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(sd, strict=False)
    return model


def run_one(variant: str, seed: int, data: dict, budget: float = 1.0
            ) -> dict:
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)

    model = _load_variant(variant, seed)
    head = EventHead(D_MODEL).to(DEVICE)

    sub_tr = subsample_engines(data['train_engines'], budget, seed=seed)
    tr_ds = FixedWindowSurfaceDataset(sub_tr, n_cuts_per_engine=5, seed=seed)
    va_ds = FixedWindowSurfaceDataset(data['val_engines'],
                                      n_cuts_per_engine=10, seed=seed + 111)
    te_ds = FixedWindowSurfaceTestDataset(data['test_engines'],
                                          data['test_rul'])

    tr = DataLoader(tr_ds, batch_size=32, shuffle=True, collate_fn=collate_fixed)
    va = DataLoader(va_ds, batch_size=32, shuffle=False, collate_fn=collate_fixed)
    te = DataLoader(te_ds, batch_size=32, shuffle=False, collate_fn=collate_fixed)

    pw = estimate_pos_weight(tr, HORIZONS_STEPS)
    train_out = train_bce(model, head, tr, va, mode='pred_ft',
                          pos_weight=pw, horizons_eval=HORIZONS_STEPS,
                          device=DEVICE, **TRAIN_CFG)

    surf = evaluate_surface(model, head, te, mode='pred_ft',
                            horizons=HORIZONS_STEPS, device=DEVICE)
    p, y = surf['p_surface'], surf['y_surface']

    (V22 / 'surfaces').mkdir(exist_ok=True)
    key = f'fd001_{variant}_pred_ft_seed{seed}'
    surf_path = V22 / 'surfaces' / f'{key}.npz'
    save_surface(surf_path, p, y, HORIZONS_STEPS, surf['t_index'],
                 metadata={'dataset': 'FD001', 'variant': variant,
                           'seed': seed, 'mode': 'pred_ft',
                           'pos_weight': float(pw)})

    prim = evaluate_probability_surface(p, y)
    per_h = auprc_per_horizon(p, y, horizon_labels=HORIZONS_STEPS)
    mono = monotonicity_violation_rate(p)

    # Legacy RMSE
    true_rul = np.array([float(r) for _, r in te_ds.items], dtype=np.float32)
    pred_rul_cross = surface_to_rul(p, np.asarray(HORIZONS_STEPS))
    pred_rul_exp = surface_to_rul_expected(p, np.asarray(HORIZONS_STEPS))
    h_max = float(HORIZONS_STEPS[-1])
    true_rul_capped = np.minimum(true_rul, h_max)
    rmse_cross = rmse(pred_rul_cross, true_rul_capped)
    rmse_exp = rmse(pred_rul_exp, true_rul_capped)

    dt = time.time() - t0
    return {
        'variant': variant, 'seed': seed,
        'primary': prim, 'per_horizon': per_h,
        'monotonicity': mono,
        'legacy': {'rmse_cross': rmse_cross, 'rmse_expected': rmse_exp,
                   'horizon_cap': h_max},
        'train': {'best_val': train_out['best_val'],
                  'final_epoch': train_out['final_epoch']},
        'pos_weight': float(pw),
        'n_train_items': len(tr_ds), 'n_test_items': len(te_ds),
        'params': count_params(model),
        'surface_file': str(surf_path),
        'runtime_s': dt,
    }


def aggregate(per_seed):
    import numpy as np
    from scipy.stats import t as t_dist
    out = {'n_seeds': len(per_seed), 'per_seed': per_seed}
    metrics = [
        ('auprc', lambda r: r['primary']['auprc']),
        ('auroc', lambda r: r['primary']['auroc']),
        ('f1_best', lambda r: r['primary']['f1_best']),
        ('rmse_expected', lambda r: r['legacy']['rmse_expected']),
        ('rmse_cross', lambda r: r['legacy']['rmse_cross']),
        ('mono_violation', lambda r: r['monotonicity']['violation_rate']),
    ]
    for name, fn in metrics:
        vals = np.array([fn(r) for r in per_seed], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            out[f'{name}_mean'] = float('nan')
            out[f'{name}_std'] = float('nan')
            continue
        m = float(vals.mean())
        s = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out[f'{name}_mean'] = m
        out[f'{name}_std'] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='+',
                    default=['baseline', 'variantA', 'variantB'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    args = ap.parse_args()

    V22.mkdir(exist_ok=True)
    (V22 / 'surfaces').mkdir(exist_ok=True)
    out_path = V22 / 'phase6_variant_cmp.json'
    t0 = time.time()

    data = load_cmapss_subset('FD001')
    print(f'FD001 loaded ({len(data["train_engines"])} train engines)',
          flush=True)

    all_out = {}
    for v in args.variants:
        print(f'\n=== variant {v} ===', flush=True)
        per_seed = []
        for s in args.seeds:
            try:
                r = run_one(v, s, data, budget=1.0)
                per_seed.append(r)
                print(f'  [{v} s{s}] AUPRC={r["primary"]["auprc"]:.3f} '
                      f'AUROC={r["primary"]["auroc"]:.3f} '
                      f'RMSE_exp={r["legacy"]["rmse_expected"]:.2f} '
                      f'mono={r["monotonicity"]["violation_rate"]:.3f} '
                      f'({r["runtime_s"]:.0f}s)', flush=True)
            except Exception as e:
                print(f'  ERROR {v} s{s}: {e}', flush=True)
                import traceback; traceback.print_exc()
        all_out[v] = {'per_seed': per_seed, 'agg': aggregate(per_seed)}
        with open(out_path, 'w') as f:
            json.dump({'variants': all_out, 'seeds': args.seeds,
                       'runtime_min': (time.time() - t0) / 60},
                      f, indent=2, default=float)

    print(f'\nDONE in {(time.time()-t0)/60:.1f}m. Saved: {out_path}')

    print('\n' + '=' * 88)
    print('V22 PHASE 6 SUMMARY (encoder variants, FD001 pred-FT, 100% labels)')
    print('=' * 88)
    print(f"{'Variant':10s} | {'AUPRC':>13s} | {'AUROC':>13s} | "
          f"{'RMSE_exp':>13s} | {'mono_v':>7s}")
    print('-' * 88)
    for v, obj in all_out.items():
        a = obj['agg']
        def f(k):
            return f"{a.get(f'{k}_mean', float('nan')):.3f}±{a.get(f'{k}_std', 0):.3f}"
        print(f"{v:10s} | {f('auprc'):>13s} | {f('auroc'):>13s} | "
              f"{f('rmse_expected'):>13s} | {a.get('mono_violation_mean', 0):.3f}")


if __name__ == '__main__':
    main()
