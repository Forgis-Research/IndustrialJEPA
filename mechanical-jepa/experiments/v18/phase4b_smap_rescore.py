"""
V18 Phase 4b: SMAP re-scoring with alternative scoring functions.

Reviewer 4 pointed out that the prediction-error-as-anomaly-score is sign-
unstable for recurrent anomalies. V17 Phase 5 confirmed this: the L1 prediction
error actually anti-correlates with labels on SMAP (anomalies are more
predictable than normals).

This phase reuses v17_smap_seed42.pt (no re-training) and evaluates three new
scoring functions on SMAP:

  1. Representation shift: ||h_past(t) - h_past(t-W)||_2
     Measures how fast the encoder's internal state changes. Sharp changes
     around anomaly onset should produce high scores.

  2. Trajectory divergence: ||gamma(k_short) - gamma(k_long)||_2
     Measures disagreement between short- and long-horizon predictions. High
     when the trajectory is uncertain.

  3. Mahalanobis distance: (h_past(t) - mu_train)^T Sigma_train^{-1} (h_past(t) - mu_train)
     Distance of current representation from training distribution. Uses PCA
     for regularization.

For each scoring function, report:
  - non-PA F1, PA-F1, AUC-PR at threshold = 95th percentile of first 10% of test
  - Sign of score_gap (anomaly_mean - normal_mean)

Output: experiments/v18/phase4b_smap_rescore_results.json
"""

import sys, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

V11 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V17 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v17')
V18 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v18')
ROOT = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')
sys.path.insert(0, str(V11)); sys.path.insert(0, str(ROOT))

from models import TrajectoryJEPA
from data.smap_msl import load_smap
from evaluation.grey_swan_metrics import anomaly_metrics as _anomaly_metrics

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D_MODEL = 256; N_HEADS = 4; N_LAYERS = 2; D_FF = 4 * D_MODEL
EMA_MOMENTUM = 0.99
CKPT_PATH = V17 / 'ckpts' / 'v17_smap_seed42.pt'

WINDOW = 100         # context window size
STRIDE = 10
W_TARGET = 10
K_SHORT = 5
K_LONG = 100
THRESH_PCTL = 95
BATCH = 256


def load_smap_model(C):
    model = TrajectoryJEPA(
        n_sensors=C, patch_length=1,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=D_FF,
    ).to(DEVICE)
    sd = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(sd, dict) and 'model_state_dict' in sd: sd = sd['model_state_dict']
    model.load_state_dict(sd)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    return model


@torch.no_grad()
def encode_all_windows(model, arr, window=WINDOW, stride=STRIDE):
    """Return h_past for sliding windows of size `window`. Map to per-timestep via
    replicated window assignment (for scoring alignment).

    Returns: (starts, H) where starts[i] is t = start+window, H[i] = h_past
    """
    T, C = arr.shape
    starts = list(range(0, T - window, stride))
    H_list = []
    for b in range(0, len(starts), BATCH):
        batch = starts[b:b+BATCH]
        B = len(batch)
        x = np.stack([arr[s: s + window] for s in batch])  # (B, W, C)
        x_t = torch.from_numpy(x).float().to(DEVICE)
        pad_mask = torch.zeros(B, window, dtype=torch.bool, device=DEVICE)
        h = model.encode_past(x_t, pad_mask)  # (B, D)
        H_list.append(h.cpu().numpy())
    H = np.concatenate(H_list, axis=0) if H_list else np.zeros((0, D_MODEL))
    return np.array(starts), H


def rep_shift_score(arr, model, window=WINDOW, stride=STRIDE):
    """Score(t) = ||h_past(t) - h_past(t - stride*steps)||."""
    T = len(arr)
    starts, H = encode_all_windows(model, arr, window=window, stride=stride)
    # H[i] corresponds to h at window ending at start + window
    # rep shift = ||H[i] - H[i-1]||
    shifts = np.zeros(len(H))
    shifts[1:] = np.linalg.norm(H[1:] - H[:-1], axis=1)
    shifts[0] = shifts[1] if len(shifts) > 1 else 0.0
    # Map to per-timestep: spread shift[i] across timesteps [start+window, start+window+stride)
    scores = np.zeros(T, dtype=np.float32)
    counts = np.zeros(T, dtype=np.float32)
    for i, s in enumerate(starts):
        end = s + window
        scores[end: min(end + stride, T)] += shifts[i]
        counts[end: min(end + stride, T)] += 1
    valid = counts > 0
    scores[valid] /= counts[valid]
    if (~valid).any() and valid.any():
        scores[~valid] = float(np.median(scores[valid]))
    return scores


@torch.no_grad()
def traj_divergence_score(arr, model, window=WINDOW, stride=STRIDE,
                           k_short=K_SHORT, k_long=K_LONG):
    """Score(t) = ||gamma(k_short) - gamma(k_long)||."""
    T, C = arr.shape
    starts = list(range(0, T - window - k_long, stride))
    if not starts:
        return np.zeros(T, dtype=np.float32)
    divs = np.zeros(len(starts))
    for b in range(0, len(starts), BATCH):
        batch = starts[b:b+BATCH]
        B = len(batch)
        x = np.stack([arr[s: s + window] for s in batch])
        x_t = torch.from_numpy(x).float().to(DEVICE)
        pad_mask = torch.zeros(B, window, dtype=torch.bool, device=DEVICE)
        h = model.encode_past(x_t, pad_mask)
        gs = model.predictor(h, torch.full((B,), k_short, dtype=torch.long, device=DEVICE))
        gl = model.predictor(h, torch.full((B,), k_long, dtype=torch.long, device=DEVICE))
        divs[b:b+B] = (gs - gl).norm(dim=-1).cpu().numpy()

    scores = np.zeros(T, dtype=np.float32); counts = np.zeros(T, dtype=np.float32)
    for i, s in enumerate(starts):
        end = s + window
        scores[end: min(end + stride, T)] += divs[i]
        counts[end: min(end + stride, T)] += 1
    valid = counts > 0
    scores[valid] /= counts[valid]
    if (~valid).any() and valid.any():
        scores[~valid] = float(np.median(scores[valid]))
    return scores


def mahalanobis_score(arr, model, train_arr, window=WINDOW, stride=STRIDE,
                      n_components=10):
    """Mahalanobis distance of h_past(t) from training h_past distribution.

    Using PCA decomposition with n_components regularization.
    """
    from sklearn.decomposition import PCA
    # Compute training h_past stats
    _, H_train = encode_all_windows(model, train_arr, window=window, stride=stride)
    if len(H_train) < 100:
        return None
    # Center
    mu = H_train.mean(axis=0)
    H_train_c = H_train - mu
    # Regularized covariance via PCA
    k = min(n_components, H_train_c.shape[1], H_train_c.shape[0] - 1)
    pca = PCA(n_components=k)
    Z_train = pca.fit_transform(H_train_c)
    # Variance per component (for inverse)
    var = pca.explained_variance_  # (k,)
    # Mahalanobis = sum_i z_i^2 / var_i

    T = len(arr)
    starts, H_test = encode_all_windows(model, arr, window=window, stride=stride)
    H_test_c = H_test - mu
    Z_test = pca.transform(H_test_c)
    m2 = (Z_test ** 2 / (var + 1e-8)).sum(axis=1)  # (N,)

    scores = np.zeros(T, dtype=np.float32); counts = np.zeros(T, dtype=np.float32)
    for i, s in enumerate(starts):
        end = s + window
        scores[end: min(end + stride, T)] += m2[i]
        counts[end: min(end + stride, T)] += 1
    valid = counts > 0
    scores[valid] /= counts[valid]
    if (~valid).any() and valid.any():
        scores[~valid] = float(np.median(scores[valid]))
    return scores


def evaluate_score(scores, labels, name):
    anom_m = float(scores[labels == 1].mean()) if (labels == 1).any() else 0.0
    norm_m = float(scores[labels == 0].mean()) if (labels == 0).any() else 0.0
    gap = anom_m - norm_m
    n_normal = int(0.1 * len(scores))
    thr = float(np.percentile(scores[:n_normal], THRESH_PCTL))
    m = _anomaly_metrics(scores, labels, threshold=thr)
    result = {
        'name': name,
        'score_gap': gap,  # +ve means anomalies score higher than normals
        'anom_mean': anom_m, 'norm_mean': norm_m,
        'threshold': thr,
        'f1_non_pa': float(m['f1_non_pa']),
        'f1_pa': float(m['f1_pa']),
        'auc_pr': float(m['auc_pr']),
        'precision_non_pa': float(m['precision_non_pa']),
        'recall_non_pa': float(m['recall_non_pa']),
    }
    print(f"  [{name:20s}] gap={gap:+.4f} non-PA F1={m['f1_non_pa']:.3f} "
          f"PA-F1={m['f1_pa']:.3f} AUC-PR={m['auc_pr']:.3f}", flush=True)
    return result


def main():
    V18.mkdir(exist_ok=True)
    print("Loading SMAP...", flush=True)
    data = load_smap()
    C = data['n_channels']
    print(f"  SMAP: train={data['train'].shape} test={data['test'].shape} "
          f"anomaly_rate={data['anomaly_rate']:.3f}", flush=True)

    print(f"Loading model from {CKPT_PATH}", flush=True)
    model = load_smap_model(C)

    t0 = time.time()
    results = []

    # Baseline: L1 prediction error (v17 Phase 5 style, multi-k averaged)
    print("\n1. Representation shift (||h_t - h_{t-1}||)", flush=True)
    scores_rs = rep_shift_score(data['test'], model)
    results.append(evaluate_score(scores_rs, data['labels'], 'rep_shift'))

    print("\n2. Trajectory divergence (||gamma(5) - gamma(100)||)", flush=True)
    scores_td = traj_divergence_score(data['test'], model)
    results.append(evaluate_score(scores_td, data['labels'], 'traj_divergence'))

    print("\n3. Mahalanobis distance (PCA-10 on h_past)", flush=True)
    scores_mh = mahalanobis_score(data['test'], model, data['train'])
    if scores_mh is not None:
        results.append(evaluate_score(scores_mh, data['labels'], 'mahalanobis_pca10'))

    # Also negated rep shift / traj div in case sign is inverted
    print("\n4. Negated representation shift (-||h_t - h_{t-1}||)", flush=True)
    results.append(evaluate_score(-scores_rs, data['labels'], 'neg_rep_shift'))
    print("\n5. Negated trajectory divergence", flush=True)
    results.append(evaluate_score(-scores_td, data['labels'], 'neg_traj_div'))

    summary = {
        'config': 'v18_phase4b_smap_rescore',
        'ckpt': str(CKPT_PATH),
        'data': {'test_shape': list(data['test'].shape),
                 'anomaly_rate': float(data['anomaly_rate'])},
        'settings': {'window': WINDOW, 'stride': STRIDE,
                     'k_short': K_SHORT, 'k_long': K_LONG,
                     'thresh_percentile': THRESH_PCTL},
        'v17_ref_l1_score': {'non_pa_f1': 0.038, 'pa_f1': 0.219, 'gap': -0.61},
        'mts_jepa_ref': {'smap_pa_f1': 0.336},
        'results': results,
        'runtime_min': (time.time() - t0) / 60,
    }
    out = V18 / 'phase4b_smap_rescore_results.json'
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nSaved: {out}", flush=True)

    print("\n" + "=" * 60)
    print("V18 Phase 4b: SMAP Alternative Scoring SUMMARY")
    print("=" * 60)
    print(f"{'method':<22} {'non-PA F1':>10} {'PA-F1':>8} {'AUC-PR':>8} {'gap':>10}")
    print(f"V17 L1 score (ref):    {0.038:>10.3f} {0.219:>8.3f} {'-':>8} {-0.61:>+10.3f}")
    for r in results:
        print(f"{r['name']:<22} {r['f1_non_pa']:>10.3f} "
              f"{r['f1_pa']:>8.3f} {r['auc_pr']:>8.3f} {r['score_gap']:>+10.4f}")
    print(f"MTS-JEPA PA-F1 ref:    {'-':>10} {0.336:>8.3f}")
    print(f"Runtime: {summary['runtime_min']:.1f} min")


if __name__ == '__main__':
    main()
