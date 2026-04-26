"""V23 Phase 2: SIGReg vs EMA pred-FT comparison on FD001.

Loads each of:
  - sigreg: v23/ckpts/sigreg_fd001_seed{S}_best.pt (no EMA, VICReg triplet)
  - baseline (EMA): v22/ckpts/baseline_fd001_seed{S}_best.pt

Freezes encoder, trains EventHead + predictor via pos-weighted BCE on FD001,
stores the probability surface, and reports AUPRC, AUROC, RMSE.

Re-runs baseline too (not just re-using v22 phase6) so both are evaluated
with identical seeds/data/dataloaders for a clean paired comparison.
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
V21 = FAM / 'experiments' / 'v21'
V22 = FAM / 'experiments' / 'v22'
V23 = FAM / 'experiments' / 'v23'

sys.path.insert(0, str(V11))
sys.path.insert(0, str(V21))
sys.path.insert(0, str(V22))
sys.path.insert(0, str(V23))
sys.path.insert(0, str(FAM))

from data_utils import load_cmapss_subset, N_SENSORS  # noqa: E402
from train_utils import subsample_engines  # noqa: E402
from models import TrajectoryJEPA  # noqa: E402
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
from cmapss_cmp_runner import (  # noqa: E402
    FixedWindowSurfaceDataset, FixedWindowSurfaceTestDataset,
    collate_fixed, WINDOW, D_MODEL, TRAIN_CFG,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CKPT_DIRS = {
    'sigreg':   V23 / 'ckpts',
    'baseline': V22 / 'ckpts',
}
CKPT_PATTERN = {
    'sigreg':   'sigreg_fd001_seed{seed}_best.pt',
    'baseline': 'baseline_fd001_seed{seed}_best.pt',
}


def _load_variant(variant: str, seed: int) -> TrajectoryJEPA:
    ck = CKPT_DIRS[variant] / CKPT_PATTERN[variant].format(seed=seed)
    assert ck.exists(), f'missing ckpt: {ck}'
    m = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1,
        d_model=D_MODEL, n_heads=4, n_layers=2, d_ff=1024,
        dropout=0.1, ema_momentum=0.99, predictor_hidden=1024,
    ).to(DEVICE)
    sd = torch.load(ck, map_location=DEVICE, weights_only=False)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f'    load {variant} s{seed}: '
              f'missing={len(missing)} unexpected={len(unexpected)}',
              flush=True)
    return m


def run_one(variant: str, seed: int, data: dict) -> dict:
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)

    model = _load_variant(variant, seed)
    head = EventHead(D_MODEL).to(DEVICE)

    sub_tr = subsample_engines(data['train_engines'], 1.0, seed=seed)
    tr_ds = FixedWindowSurfaceDataset(sub_tr, n_cuts_per_engine=5, seed=seed)
    va_ds = FixedWindowSurfaceDataset(data['val_engines'],
                                      n_cuts_per_engine=10, seed=seed + 111)
    te_ds = FixedWindowSurfaceTestDataset(data['test_engines'],
                                          data['test_rul'])

    tr = DataLoader(tr_ds, batch_size=32, shuffle=True,
                    collate_fn=collate_fixed)
    va = DataLoader(va_ds, batch_size=32, shuffle=False,
                    collate_fn=collate_fixed)
    te = DataLoader(te_ds, batch_size=32, shuffle=False,
                    collate_fn=collate_fixed)

    pw = estimate_pos_weight(tr, HORIZONS_STEPS)
    train_out = train_bce(model, head, tr, va, mode='pred_ft',
                          pos_weight=pw, horizons_eval=HORIZONS_STEPS,
                          device=DEVICE, **TRAIN_CFG)

    surf = evaluate_surface(model, head, te, mode='pred_ft',
                            horizons=HORIZONS_STEPS, device=DEVICE)
    p, y = surf['p_surface'], surf['y_surface']

    (V23 / 'surfaces').mkdir(exist_ok=True)
    key = f'fd001_{variant}_pred_ft_seed{seed}'
    surf_path = V23 / 'surfaces' / f'{key}.npz'
    save_surface(surf_path, p, y, HORIZONS_STEPS, surf['t_index'],
                 metadata={'dataset': 'FD001', 'variant': variant,
                           'seed': seed, 'mode': 'pred_ft',
                           'pos_weight': float(pw)})

    prim = evaluate_probability_surface(p, y)
    per_h = auprc_per_horizon(p, y, horizon_labels=HORIZONS_STEPS)
    mono = monotonicity_violation_rate(p)

    true_rul = np.array([float(r) for _, r in te_ds.items], dtype=np.float32)
    pred_rul_cross = surface_to_rul(p, np.asarray(HORIZONS_STEPS))
    pred_rul_exp = surface_to_rul_expected(p, np.asarray(HORIZONS_STEPS))
    h_max = float(HORIZONS_STEPS[-1])
    true_rul_capped = np.minimum(true_rul, h_max)
    rmse_cross = rmse(pred_rul_cross, true_rul_capped)
    rmse_exp = rmse(pred_rul_exp, true_rul_capped)

    return {
        'variant': variant, 'seed': seed,
        'primary': prim, 'per_horizon': per_h, 'monotonicity': mono,
        'legacy': {'rmse_cross': rmse_cross, 'rmse_expected': rmse_exp,
                   'horizon_cap': h_max},
        'train': {'best_val': train_out['best_val'],
                  'final_epoch': train_out['final_epoch']},
        'pos_weight': float(pw),
        'surface_file': str(surf_path),
        'runtime_s': time.time() - t0,
    }


def agg(rs):
    import numpy as np
    out = {'n_seeds': len(rs)}
    for name, fn in [
        ('auprc',          lambda r: r['primary']['auprc']),
        ('auroc',          lambda r: r['primary']['auroc']),
        ('f1_best',        lambda r: r['primary']['f1_best']),
        ('rmse_expected',  lambda r: r['legacy']['rmse_expected']),
        ('mono_violation', lambda r: r['monotonicity']['violation_rate']),
    ]:
        vals = np.array([fn(r) for r in rs], float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            out[f'{name}_mean'] = float('nan')
            out[f'{name}_std'] = float('nan')
        else:
            out[f'{name}_mean'] = float(vals.mean())
            out[f'{name}_std'] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return out


def paired_stats(rs_a, rs_b, metric_fn):
    """Paired t-test and Wilcoxon on metric(a) - metric(b) across seeds."""
    from scipy.stats import ttest_rel, wilcoxon
    a = np.array([metric_fn(r) for r in rs_a], float)
    b = np.array([metric_fn(r) for r in rs_b], float)
    if len(a) != len(b):
        return None
    d = a - b
    out = {
        'n': len(a), 'a_mean': float(a.mean()), 'b_mean': float(b.mean()),
        'delta_mean': float(d.mean()), 'delta_std': float(d.std(ddof=1)) if len(d) > 1 else 0.0,
    }
    try:
        t = ttest_rel(a, b)
        out['t_stat'] = float(t.statistic); out['t_p'] = float(t.pvalue)
    except Exception:
        pass
    try:
        w = wilcoxon(a, b) if len(a) > 1 else None
        if w is not None:
            out['w_stat'] = float(w.statistic); out['w_p'] = float(w.pvalue)
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='+',
                    default=['sigreg', 'baseline'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    args = ap.parse_args()

    (V23 / 'surfaces').mkdir(exist_ok=True)
    out_path = V23 / 'phase2_sigreg_vs_ema.json'
    data = load_cmapss_subset('FD001')
    print(f'FD001: {len(data["train_engines"])} train engines', flush=True)

    all_out = {}
    t0 = time.time()
    for v in args.variants:
        print(f'\n=== variant {v} ===', flush=True)
        per_seed = []
        for s in args.seeds:
            try:
                r = run_one(v, s, data)
                per_seed.append(r)
                print(f'  [{v} s{s}] AUPRC={r["primary"]["auprc"]:.3f} '
                      f'AUROC={r["primary"]["auroc"]:.3f} '
                      f'RMSE_exp={r["legacy"]["rmse_expected"]:.2f} '
                      f'({r["runtime_s"]:.0f}s)', flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f'  ERROR {v} s{s}: {e}', flush=True)
        all_out[v] = {'per_seed': per_seed, 'agg': agg(per_seed)}
        with open(out_path, 'w') as f:
            json.dump({'variants': all_out, 'seeds': args.seeds,
                       'runtime_min': (time.time() - t0) / 60},
                      f, indent=2, default=float)

    # Paired tests (sigreg vs baseline)
    if 'sigreg' in all_out and 'baseline' in all_out:
        rs_s = all_out['sigreg']['per_seed']
        rs_b = all_out['baseline']['per_seed']
        all_out['paired_sigreg_vs_baseline'] = {
            'auprc': paired_stats(rs_s, rs_b,
                                   lambda r: r['primary']['auprc']),
            'auroc': paired_stats(rs_s, rs_b,
                                   lambda r: r['primary']['auroc']),
            'rmse_expected': paired_stats(rs_s, rs_b,
                                           lambda r: r['legacy']['rmse_expected']),
        }
        with open(out_path, 'w') as f:
            json.dump({'variants': all_out, 'seeds': args.seeds,
                       'runtime_min': (time.time() - t0) / 60},
                      f, indent=2, default=float)

    print(f'\nDONE in {(time.time()-t0)/60:.1f}m -> {out_path}')
    print('\n' + '=' * 72)
    print('V23 PHASE 2 SUMMARY (SIGReg vs EMA, FD001 pred-FT)')
    print('=' * 72)
    for v, obj in all_out.items():
        if v.startswith('paired_'):
            continue
        a = obj['agg']
        print(f"{v:12s} | AUPRC={a['auprc_mean']:.3f}±{a['auprc_std']:.3f} "
              f"AUROC={a['auroc_mean']:.3f}±{a['auroc_std']:.3f} "
              f"RMSE={a['rmse_expected_mean']:.2f}±{a['rmse_expected_std']:.2f}")
    if 'paired_sigreg_vs_baseline' in all_out:
        p = all_out['paired_sigreg_vs_baseline']
        for metric in ['auprc', 'auroc', 'rmse_expected']:
            s = p[metric]
            if s is None:
                continue
            print(f"paired {metric:13s}: delta={s['delta_mean']:+.3f} "
                  f"t={s.get('t_stat', 0):+.2f} p={s.get('t_p', 1):.3f}")


if __name__ == '__main__':
    main()
