"""
V18 Phase 4c: Robustness of Mahalanobis SMAP scoring.

Phase 4b showed Mahalanobis(PCA-10) on h_past gives SMAP PA-F1 0.733 - a big
jump from L1 prediction error's 0.219. Reviewers will ask whether this is
robust. This phase varies:

  1. PCA component count: {5, 10, 20, 50, 100} (at most d_model=256)
  2. Training-set subsample for PCA fit (bootstrap): 5 seeds at each PCA size
  3. Lead-time decomposition: are detections onset (TRUE_PREDICTION) or
     continuation (already-active anomaly)?

Reuses `v17_smap_seed42.pt`. No re-training.

Output: experiments/v18/phase4c_mahal_robustness.json
"""

import sys, json, time
from pathlib import Path
import numpy as np
import torch

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

WINDOW = 100; STRIDE = 10; BATCH = 256
THRESH_PCTL = 95
PCA_COMPONENTS = [5, 10, 20, 50, 100]
BOOTSTRAP_SEEDS = [0, 1, 2, 3, 4]


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
def encode_windows(model, arr, window=WINDOW, stride=STRIDE):
    T, C = arr.shape
    starts = list(range(0, T - window, stride))
    H_list = []
    for b in range(0, len(starts), BATCH):
        batch = starts[b:b+BATCH]
        B = len(batch)
        x = np.stack([arr[s:s+window] for s in batch])
        x_t = torch.from_numpy(x).float().to(DEVICE)
        pad_mask = torch.zeros(B, window, dtype=torch.bool, device=DEVICE)
        h = model.encode_past(x_t, pad_mask)
        H_list.append(h.cpu().numpy())
    return np.array(starts), np.concatenate(H_list, axis=0) if H_list else np.zeros((0, D_MODEL))


def mahal_score(H_test, starts_test, H_train_subset, n_components, T_len):
    """Compute per-timestep Mahalanobis score."""
    from sklearn.decomposition import PCA
    mu = H_train_subset.mean(axis=0)
    Htc = H_train_subset - mu
    k = min(n_components, Htc.shape[1], Htc.shape[0] - 1)
    pca = PCA(n_components=k); pca.fit(Htc)
    var = pca.explained_variance_
    Hte_c = H_test - mu
    Zte = pca.transform(Hte_c)
    m2 = (Zte ** 2 / (var + 1e-8)).sum(axis=1)

    scores = np.zeros(T_len, dtype=np.float32); counts = np.zeros(T_len, dtype=np.float32)
    for i, s in enumerate(starts_test):
        end = s + WINDOW
        scores[end: min(end + STRIDE, T_len)] += m2[i]
        counts[end: min(end + STRIDE, T_len)] += 1
    valid = counts > 0
    scores[valid] /= counts[valid]
    if (~valid).any() and valid.any():
        scores[~valid] = float(np.median(scores[valid]))
    return scores


def eval_score(scores, labels):
    n_normal = int(0.1 * len(scores))
    thr = float(np.percentile(scores[:n_normal], THRESH_PCTL))
    m = _anomaly_metrics(scores, labels, threshold=thr)
    return {
        'f1_non_pa': float(m['f1_non_pa']),
        'f1_pa': float(m['f1_pa']),
        'auc_pr': float(m['auc_pr']),
        'threshold': thr,
    }


def lead_time_decomposition(scores, labels, threshold, window=WINDOW):
    """For each detected anomaly point, classify as:
      - CONTINUATION: context window (last W before detection) contains any anomaly
      - TRUE_PREDICTION: context window is entirely normal
    """
    detections = np.where(scores > threshold)[0]
    if len(detections) == 0:
        return {'n_detections': 0, 'n_true_pos': 0, 'continuation_frac': 0.0}

    n_detect = len(detections)
    n_true_pos = 0; n_cont = 0; n_pred = 0
    for d in detections:
        if labels[d] == 1:
            n_true_pos += 1
            ctx_start = max(0, d - window)
            if labels[ctx_start:d].max() == 1:
                n_cont += 1
            else:
                n_pred += 1
    return {
        'n_detections': int(n_detect),
        'n_true_positives': int(n_true_pos),
        'n_continuation': int(n_cont),
        'n_true_prediction': int(n_pred),
        'continuation_frac': float(n_cont / max(1, n_true_pos)),
        'true_prediction_frac': float(n_pred / max(1, n_true_pos)),
    }


def main():
    V18.mkdir(exist_ok=True)
    print("Loading SMAP...", flush=True)
    data = load_smap()
    C = data['n_channels']
    print(f"  SMAP: anomaly_rate={data['anomaly_rate']:.3f}", flush=True)

    print(f"Loading model from {CKPT_PATH}", flush=True)
    model = load_smap_model(C)

    t0 = time.time()
    print("Encoding train and test windows...", flush=True)
    _, H_train_full = encode_windows(model, data['train'])
    starts_test, H_test = encode_windows(model, data['test'])
    T_len = len(data['test'])
    print(f"  train H: {H_train_full.shape}, test H: {H_test.shape}", flush=True)

    results = {'pca_sweep': {}, 'bootstrap': {}, 'lead_time': {}}

    # PCA component sweep (full train fit)
    print("\nPCA component sweep (full train fit):", flush=True)
    for k in PCA_COMPONENTS:
        scores = mahal_score(H_test, starts_test, H_train_full, k, T_len)
        m = eval_score(scores, data['labels'])
        results['pca_sweep'][str(k)] = m
        print(f"  PCA-{k:>3d}: non-PA F1={m['f1_non_pa']:.3f} "
              f"PA-F1={m['f1_pa']:.3f} AUC-PR={m['auc_pr']:.3f}", flush=True)

    # Bootstrap: 5 random subsets of training H, PCA-10
    print("\nBootstrap stability (PCA-10, 50% train subsample x 5 seeds):", flush=True)
    bs_metrics = []
    rng = np.random.RandomState(42)
    for bs_seed in BOOTSTRAP_SEEDS:
        idx = np.random.RandomState(bs_seed).choice(
            len(H_train_full), size=len(H_train_full) // 2, replace=False
        )
        H_sub = H_train_full[idx]
        scores = mahal_score(H_test, starts_test, H_sub, 10, T_len)
        m = eval_score(scores, data['labels'])
        bs_metrics.append(m)
        print(f"  seed={bs_seed}: non-PA F1={m['f1_non_pa']:.3f} "
              f"PA-F1={m['f1_pa']:.3f} AUC-PR={m['auc_pr']:.3f}", flush=True)

    results['bootstrap']['pca10_50pct_subsample'] = {
        'per_seed': bs_metrics,
        'f1_non_pa_mean': float(np.mean([m['f1_non_pa'] for m in bs_metrics])),
        'f1_non_pa_std': float(np.std([m['f1_non_pa'] for m in bs_metrics])),
        'f1_pa_mean': float(np.mean([m['f1_pa'] for m in bs_metrics])),
        'f1_pa_std': float(np.std([m['f1_pa'] for m in bs_metrics])),
        'auc_pr_mean': float(np.mean([m['auc_pr'] for m in bs_metrics])),
    }

    # Lead-time decomposition using full-train PCA-10
    print("\nLead-time decomposition (PCA-10 full train):", flush=True)
    scores = mahal_score(H_test, starts_test, H_train_full, 10, T_len)
    m = eval_score(scores, data['labels'])
    lt = lead_time_decomposition(scores, data['labels'], m['threshold'])
    results['lead_time']['pca10'] = lt
    print(f"  n_detections={lt['n_detections']}, "
          f"true_positives={lt['n_true_positives']}", flush=True)
    print(f"  continuation={lt['continuation_frac']:.3f}, "
          f"true_prediction={lt['true_prediction_frac']:.3f}", flush=True)

    summary = {
        'config': 'v18_phase4c_mahal_robustness',
        'ckpt': str(CKPT_PATH),
        'pca_components_tested': PCA_COMPONENTS,
        'bootstrap_seeds': BOOTSTRAP_SEEDS,
        'window': WINDOW, 'stride': STRIDE, 'thresh_pctl': THRESH_PCTL,
        'mts_jepa_ref': {'smap_pa_f1': 0.336},
        'v17_l1_baseline': {'pa_f1': 0.219, 'non_pa_f1': 0.038},
        'results': results,
        'runtime_min': (time.time() - t0) / 60,
    }
    with open(V18 / 'phase4c_mahal_robustness.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n" + "=" * 60)
    print("V18 Phase 4c: Mahalanobis Robustness SUMMARY")
    print("=" * 60)
    print(f"PCA-k sweep: {[f'k={k}:{results['pca_sweep'][str(k)]['f1_pa']:.3f}' for k in PCA_COMPONENTS]}")
    b = results['bootstrap']['pca10_50pct_subsample']
    print(f"Bootstrap PCA-10: PA-F1 {b['f1_pa_mean']:.3f} +/- {b['f1_pa_std']:.3f}")
    print(f"Lead time: continuation {lt['continuation_frac']:.1%} / "
          f"true prediction {lt['true_prediction_frac']:.1%}")
    print(f"MTS-JEPA PA-F1 ref: 0.336 / L1 ref: 0.219")


if __name__ == '__main__':
    main()
