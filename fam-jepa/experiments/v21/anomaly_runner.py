"""V21 Anomaly Runner — Mahalanobis Surface + Pred-FT Attempt.

Unified runner for the five anomaly datasets (SMAP, MSL, PSM, SMD, MBA).

Primary pipeline (reliable, matches v18/v19):
  1. Encode train windows (normal-only) -> H_train.
  2. Fit PCA -> Mahalanobis distance d(h).
  3. Encode test windows -> raw scores s(t) = d(h_t).
  4. Calibrate per-horizon sigmoid on a labeled holdout (stratified random
     split of labeled test into fit-cal / eval-surface).
     For horizon Δt_k:  p(t, Δt_k) = sigmoid(a_k * s(t) + b_k)
     (a_k, b_k) fit by logistic regression predicting y(t, Δt_k) from s(t).
  5. Store the surface, compute AUPRC, AUROC, PA-F1 (legacy).

Secondary pipeline (stretch, only if time permits):
  Pred-FT with per-horizon logits + BCE.  Uses IID random split of labeled
  test (chronological split fails due to distribution shift — verified on
  SMAP in Phase 0).

Seeds: 3 (42, 123, 456).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Paths
ROOT = Path('/home/sagemaker-user/IndustrialJEPA')
FAM = ROOT / 'fam-jepa'
V11 = FAM / 'experiments' / 'v11'
V21 = FAM / 'experiments' / 'v21'
CKPT_OLD = ROOT / 'mechanical-jepa' / 'experiments'

sys.path.insert(0, str(V11))
sys.path.insert(0, str(V21))
sys.path.insert(0, str(FAM))

from models import TrajectoryJEPA  # noqa: E402
from pred_ft_utils import (  # noqa: E402
    AnomalyWindowDataset, collate_anomaly_window,
    HORIZONS_STEPS, save_surface,
)
from surface_to_legacy import anomaly_legacy_metrics, binary_prf  # noqa: E402
from data.smap_msl import load_smap, load_msl  # noqa: E402
from data.psm import load_psm  # noqa: E402
from data.smd import load_smd as _load_smd_default  # noqa: E402
from data.mba import load_mba  # noqa: E402

# SMD data is shipped as per-machine .txt under the tranad repo. The canonical
# loader expects combined .npy or `machine-*` dirs. Provide a fallback loader
# that aggregates across all machines (same protocol v19 phase4_smd used).

def load_smd(normalize: bool = True):
    from pathlib import Path
    import glob, os
    import numpy as np
    try:
        d = _load_smd_default(normalize=normalize)
        if d is not None:
            return d
    except Exception:
        pass
    SMD_PATH = Path('/home/sagemaker-user/IndustrialJEPA/paper-replications/'
                    'mts-jepa/data/tranad_repo/data/SMD')
    tr_files = sorted(glob.glob(str(SMD_PATH / 'train' / '*.txt')))
    trains, tests, labels = [], [], []
    for tp in tr_files:
        name = os.path.basename(tp)
        tr = np.loadtxt(tp, delimiter=',').astype(np.float32)
        te = np.loadtxt(str(SMD_PATH / 'test' / name),
                        delimiter=',').astype(np.float32)
        lb = np.loadtxt(str(SMD_PATH / 'labels' / name),
                        delimiter=',').astype(np.int32)
        trains.append(tr); tests.append(te); labels.append(lb)
    train = np.concatenate(trains, axis=0)
    test = np.concatenate(tests, axis=0)
    labels_cat = np.concatenate(labels, axis=0)
    if normalize:
        mu = train.mean(axis=0, keepdims=True)
        std = train.std(axis=0, keepdims=True) + 1e-6
        train = (train - mu) / std
        test = (test - mu) / std
    return {'train': train, 'test': test, 'labels': labels_cat,
            'n_channels': train.shape[1], 'name': 'SMD',
            'anomaly_rate': float(labels_cat.mean())}
from evaluation.surface_metrics import (  # noqa: E402
    evaluate_probability_surface, auprc_per_horizon,
    monotonicity_violation_rate,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Base architecture — n_sensors overridden per dataset
BASE_ARCH = dict(patch_length=1, d_model=256, n_heads=4, n_layers=2,
                 d_ff=1024, dropout=0.1, ema_momentum=0.99,
                 predictor_hidden=1024)

WINDOW = 100  # encoder context length
STRIDE_TRAIN = 10  # how to sample training-set windows for PCA fit
STRIDE_EVAL = 10   # how to stride over test set windows
BATCH = 256
MAX_FUT = max(HORIZONS_STEPS) + 1

# Dataset registry: each has a loader, n_sensors, ckpt paths per seed
DATASETS = {
    'SMAP': {
        'loader': load_smap, 'n_sensors': 25,
        'ckpts': {
            42:  CKPT_OLD / 'v17' / 'ckpts' / 'v17_smap_seed42.pt',
            123: CKPT_OLD / 'v18' / 'ckpts' / 'v18_smap_seed123.pt',
            456: CKPT_OLD / 'v18' / 'ckpts' / 'v18_smap_seed456.pt',
        },
    },
    'MSL': {
        'loader': load_msl, 'n_sensors': 55,
        'ckpts': {
            42:  CKPT_OLD / 'v18' / 'ckpts' / 'v18_msl_seed42_ep150.pt',
            123: CKPT_OLD / 'v18' / 'ckpts' / 'v18_msl_seed123_ep150.pt',
            456: CKPT_OLD / 'v18' / 'ckpts' / 'v18_msl_seed456_ep150.pt',
        },
    },
    'PSM': {
        'loader': load_psm, 'n_sensors': 25,
        'ckpts': {
            42:  CKPT_OLD / 'v19' / 'ckpts' / 'v19_psm_seed42_ep50.pt',
            123: CKPT_OLD / 'v19' / 'ckpts' / 'v19_psm_seed123_ep50.pt',
            456: CKPT_OLD / 'v19' / 'ckpts' / 'v19_psm_seed456_ep50.pt',
        },
    },
    'SMD': {
        'loader': load_smd, 'n_sensors': 38,
        'ckpts': {
            42:  CKPT_OLD / 'v19' / 'ckpts' / 'v19_smd_seed42.pt',
            123: CKPT_OLD / 'v19' / 'ckpts' / 'v19_smd_seed123.pt',
            456: CKPT_OLD / 'v19' / 'ckpts' / 'v19_smd_seed456.pt',
        },
    },
    'MBA': {
        'loader': load_mba, 'n_sensors': 2,
        'ckpts': {
            42:  CKPT_OLD / 'v19' / 'ckpts' / 'v19_mba_seed42_ep50.pt',
            123: CKPT_OLD / 'v19' / 'ckpts' / 'v19_mba_seed123_ep50.pt',
            456: CKPT_OLD / 'v19' / 'ckpts' / 'v19_mba_seed456_ep50.pt',
        },
    },
}


# ---------------------------------------------------------------------------
# Encode windows across an array
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_array(model: TrajectoryJEPA, arr: np.ndarray,
                 stride: int, window: int = WINDOW,
                 labels: np.ndarray | None = None,
                 t_start: int | None = None, t_end: int | None = None,
                 ) -> dict:
    """Encode windows ending at each stride step; return H, starts, ts, tte (if labels)."""
    T = len(arr)
    # Use the same AnomalyWindowDataset for convenience even if labels is None
    if labels is None:
        labels = np.zeros(T, dtype=np.int32)
    ds = AnomalyWindowDataset(arr, labels, window=window, stride=stride,
                              max_future=MAX_FUT, t_start=t_start, t_end=t_end)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False,
                    collate_fn=collate_anomaly_window, num_workers=0)
    H_list, tte_list, t_list = [], [], []
    for x, mask, tte, t in dl:
        x = x.to(DEVICE, non_blocking=True)
        mask = mask.to(DEVICE, non_blocking=True)
        h = model.encode_past(x, mask)
        H_list.append(h.cpu().numpy())
        tte_list.append(tte.numpy())
        t_list.append(t.numpy())
    return {
        'H': np.concatenate(H_list, axis=0).astype(np.float32),
        'tte': np.concatenate(tte_list, axis=0),
        't': np.concatenate(t_list, axis=0),
    }


# ---------------------------------------------------------------------------
# Mahalanobis surface (unsupervised scores + per-horizon calibration)
# ---------------------------------------------------------------------------

def fit_mahal(H_normal: np.ndarray, n_comp: int = 50,
              variance_retention: float = 0.99) -> dict:
    """Fit PCA on normal features; return objects needed to score new features.

    Auto-selects n_components to retain `variance_retention` fraction.
    """
    from sklearn.decomposition import PCA
    mu = H_normal.mean(axis=0)
    Xc = H_normal - mu
    n_max = min(H_normal.shape[0] - 1, H_normal.shape[1])
    pca = PCA(n_components=n_max)
    pca.fit(Xc)
    # Select k for variance_retention
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.argmax(cum >= variance_retention)) + 1 if (cum >= variance_retention).any() else n_max
    k = max(min(k, n_comp, n_max), 1)
    pca2 = PCA(n_components=k)
    pca2.fit(Xc)
    return {'mu': mu, 'pca': pca2, 'k': k,
            'var': pca2.explained_variance_.astype(np.float32)}


def mahal_score(fit: dict, H: np.ndarray) -> np.ndarray:
    Xc = H - fit['mu']
    Z = fit['pca'].transform(Xc)
    d2 = (Z ** 2 / (fit['var'] + 1e-8)).sum(axis=1)
    return d2.astype(np.float32)


def calibrate_surface(scores_cal: np.ndarray, tte_cal: np.ndarray,
                      horizons=HORIZONS_STEPS) -> dict:
    """Per-horizon logistic calibration: fit P(y_k | s) for each Δt_k."""
    from sklearn.linear_model import LogisticRegression
    horizons = np.asarray(horizons, dtype=np.int32)
    K = len(horizons)
    cals = []
    # Build (N, K) label surface
    y = np.zeros((len(tte_cal), K), dtype=np.int32)
    for k, dt in enumerate(horizons):
        y[:, k] = ((tte_cal <= float(dt)) & np.isfinite(tte_cal)).astype(np.int32)

    # Scale scores for numerical stability
    s_mean, s_std = float(np.mean(scores_cal)), float(np.std(scores_cal) + 1e-8)
    s_norm = (scores_cal - s_mean) / s_std
    for k in range(K):
        yk = y[:, k]
        if yk.sum() == 0 or yk.sum() == len(yk):
            # Degenerate: no positives or all positives. Use prevalence.
            cals.append({'a': 0.0, 'b': float(np.log(max(yk.mean(), 1e-6) /
                                                     max(1 - yk.mean(), 1e-6)))})
            continue
        clf = LogisticRegression(max_iter=500, class_weight='balanced',
                                 C=1.0)
        clf.fit(s_norm.reshape(-1, 1), yk)
        cals.append({'a': float(clf.coef_[0, 0]), 'b': float(clf.intercept_[0])})
    return {'cal': cals, 'mean': s_mean, 'std': s_std,
            'horizons': horizons.tolist()}


def apply_surface(cal: dict, scores: np.ndarray) -> np.ndarray:
    """Apply calibration to get (N, K) probability surface."""
    s_norm = (scores - cal['mean']) / cal['std']
    K = len(cal['cal'])
    p = np.zeros((len(scores), K), dtype=np.float32)
    for k, c in enumerate(cal['cal']):
        z = c['a'] * s_norm + c['b']
        p[:, k] = 1.0 / (1.0 + np.exp(-z))
    # Enforce monotonicity along Δt (non-decreasing)
    # The logistic calibration may not guarantee this at the tails
    p = np.maximum.accumulate(p, axis=1)
    return p


# ---------------------------------------------------------------------------
# Single dataset run (Mahalanobis surface, 3 seeds)
# ---------------------------------------------------------------------------

def build_surface_from_mahal(dataset: str, seed: int, verbose: bool = True) -> dict:
    """One dataset + seed → stored surface + metrics."""
    t0 = time.time()
    reg = DATASETS[dataset]
    data = reg['loader']()
    if data is None:
        raise RuntimeError(f'Data missing for {dataset}')
    ns = data.get('n_channels', reg['n_sensors'])
    arch = dict(BASE_ARCH, n_sensors=ns)

    # Load checkpoint
    ckpt_path = reg['ckpts'][seed]
    if not ckpt_path.exists():
        raise FileNotFoundError(f'{ckpt_path}')
    model = TrajectoryJEPA(**arch).to(DEVICE)
    sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if isinstance(sd, dict) and 'model' in sd and 'context_encoder.proj.proj.weight' not in sd:
        sd = sd['model']
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if verbose and (missing or unexpected):
        print(f'  load: missing={len(missing)} unexpected={len(unexpected)}', flush=True)
    model.eval()

    train = data['train']; test = data['test']; labels = data['labels']
    T = len(test)

    # Encode NORMAL (train stream: all normal) for Mahalanobis fit
    enc_tr = encode_array(model, train, stride=STRIDE_TRAIN, window=WINDOW)
    H_train = enc_tr['H']
    fit = fit_mahal(H_train, n_comp=50, variance_retention=0.99)
    if verbose:
        print(f'  mahal fit: k={fit["k"]} using {len(H_train)} normal windows',
              flush=True)

    # Encode full test stream
    enc_te = encode_array(model, test, stride=STRIDE_EVAL, window=WINDOW,
                          labels=labels)
    H_test = enc_te['H']
    scores_te = mahal_score(fit, H_test)
    t_ends = enc_te['t'].astype(np.int64)
    tte_te = enc_te['tte']

    # Split labeled test windows IID into cal / eval (50/50)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(H_test))
    n_cal = len(H_test) // 2
    cal_idx = perm[:n_cal]
    eval_idx = perm[n_cal:]

    cal = calibrate_surface(scores_te[cal_idx], tte_te[cal_idx],
                            horizons=HORIZONS_STEPS)

    # Evaluate on eval_idx: build p_surface + y_surface
    p_eval = apply_surface(cal, scores_te[eval_idx])
    t_eval = t_ends[eval_idx]
    tte_eval = tte_te[eval_idx]
    y_eval = np.zeros((len(eval_idx), len(HORIZONS_STEPS)), dtype=np.int8)
    for k, dt in enumerate(HORIZONS_STEPS):
        y_eval[:, k] = ((tte_eval <= float(dt)) & np.isfinite(tte_eval)
                        ).astype(np.int8)

    # Also build the "full-test" surface (all windows) for legacy PA-F1
    p_full = apply_surface(cal, scores_te)
    y_full = np.zeros((len(H_test), len(HORIZONS_STEPS)), dtype=np.int8)
    for k, dt in enumerate(HORIZONS_STEPS):
        y_full[:, k] = ((tte_te <= float(dt)) & np.isfinite(tte_te)
                        ).astype(np.int8)

    # Save surface (full)
    V21.mkdir(exist_ok=True)
    (V21 / 'surfaces').mkdir(exist_ok=True)
    surf_path = V21 / 'surfaces' / f'{dataset.lower()}_seed{seed}_mahal.npz'
    save_surface(surf_path,
                 p_full, y_full, HORIZONS_STEPS, t_ends,
                 metadata={'dataset': dataset, 'seed': seed,
                           'mode': 'mahal',
                           'cal_mean': cal['mean'], 'cal_std': cal['std'],
                           'pca_k': fit['k']})

    # Metrics on eval split (unbiased AUPRC)
    prim = evaluate_probability_surface(p_eval, y_eval)
    per_h = auprc_per_horizon(p_eval, y_eval, horizon_labels=HORIZONS_STEPS)
    mono = monotonicity_violation_rate(p_eval)

    # Legacy PA-F1: use FULL test surface -> per-timestep score at Δt=100
    legacy = anomaly_legacy_metrics(p_full, t_ends, labels,
                                    np.asarray(HORIZONS_STEPS), 100)

    dt = time.time() - t0
    if verbose:
        print(f'  [{dataset} s{seed}] AUPRC={prim["auprc"]:.3f} '
              f'AUROC={prim["auroc"]:.3f} '
              f'PA-F1={legacy["pa_f1"]:.3f} '
              f'mono={mono["violation_rate"]:.3f} '
              f'({dt:.0f}s)', flush=True)

    return {
        'dataset': dataset, 'seed': seed, 'mode': 'mahal',
        'primary': prim, 'per_horizon': per_h,
        'monotonicity': mono, 'legacy': legacy,
        'cal_info': {'mean': cal['mean'], 'std': cal['std'],
                     'a_per_k': [c['a'] for c in cal['cal']],
                     'b_per_k': [c['b'] for c in cal['cal']]},
        'pca_k': fit['k'],
        'n_train_windows': len(H_train),
        'n_test_windows': len(H_test),
        'n_eval_windows': len(eval_idx),
        'surface_file': str(surf_path),
        'runtime_s': dt,
    }


def aggregate(per_seed):
    """mean±std across seeds for key metrics."""
    import numpy as np
    from scipy.stats import t as t_dist
    out = {'n_seeds': len(per_seed), 'per_seed': per_seed}
    metrics = [
        ('auprc', lambda r: r['primary']['auprc']),
        ('auroc', lambda r: r['primary']['auroc']),
        ('f1_best', lambda r: r['primary']['f1_best']),
        ('precision_best', lambda r: r['primary']['precision_best']),
        ('recall_best', lambda r: r['primary']['recall_best']),
        ('pa_f1', lambda r: r['legacy']['pa_f1']),
        ('pa_precision', lambda r: r['legacy']['pa_precision']),
        ('pa_recall', lambda r: r['legacy']['pa_recall']),
        ('non_pa_f1', lambda r: r['legacy']['non_pa_f1']),
        ('mono_violation', lambda r: r['monotonicity']['violation_rate']),
    ]
    for name, fn in metrics:
        vals = np.array([fn(r) for r in per_seed], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            out[f'{name}_mean'] = float('nan'); out[f'{name}_std'] = float('nan')
            continue
        m = float(vals.mean())
        s = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        if len(vals) > 1:
            tc = float(t_dist.ppf(0.975, df=len(vals) - 1))
            margin = tc * s / np.sqrt(len(vals))
        else:
            margin = float('nan')
        out[f'{name}_mean'] = m; out[f'{name}_std'] = s
        out[f'{name}_ci95_lo'] = m - margin; out[f'{name}_ci95_hi'] = m + margin
    return out


def run_dataset_all_seeds(dataset: str, seeds=(42, 123, 456)):
    per_seed = []
    for s in seeds:
        try:
            r = build_surface_from_mahal(dataset, s, verbose=True)
            per_seed.append(r)
        except Exception as e:
            print(f'  ERROR {dataset} s{s}: {e}', flush=True)
            import traceback; traceback.print_exc()
    return {dataset: {'per_seed': per_seed, 'agg': aggregate(per_seed)}}


def main(datasets=None, seeds=(42, 123, 456)):
    t0 = time.time()
    if datasets is None:
        datasets = ['SMAP', 'MSL', 'PSM', 'SMD', 'MBA']
    out_path = V21 / 'phase1_anomaly.json'
    all_out = {}
    for ds in datasets:
        print(f'\n=== {ds} ===', flush=True)
        all_out.update(run_dataset_all_seeds(ds, seeds=seeds))
        # Save incrementally
        with open(out_path, 'w') as f:
            json.dump({'datasets': all_out,
                       'seeds': list(seeds),
                       'runtime_min': (time.time() - t0) / 60},
                      f, indent=2, default=float)
    print(f'\n\nDONE in {(time.time()-t0)/60:.1f} min. Saved: {out_path}')

    # Summary print
    print('\n' + '='*80)
    print('PHASE 1 SUMMARY')
    print('='*80)
    print(f"{'Dataset':8s} | {'AUPRC':>17s} | {'AUROC':>17s} | {'PA-F1':>17s} | {'mono_v':>8s}")
    print('-' * 80)
    for ds, obj in all_out.items():
        a = obj['agg']
        def f(k):
            return f"{a.get(f'{k}_mean', float('nan')):.3f}±{a.get(f'{k}_std', 0):.3f}"
        print(f"{ds:8s} | {f('auprc'):>17s} | {f('auroc'):>17s} | {f('pa_f1'):>17s} | "
              f"{a.get('mono_violation_mean', 0):.3f}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=None)
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    args = ap.parse_args()
    main(args.datasets, tuple(args.seeds))
