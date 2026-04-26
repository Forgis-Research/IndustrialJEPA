"""
V18 Phase 4d: Random-initialisation Mahalanobis control.

Reviewer 2 round-2: "What is the PA-F1 of Mahalanobis(PCA-10) on h_past from a
randomly initialised encoder? This is the critical control that separates
'JEPA representations carry SMAP anomaly structure' from 'Mahalanobis on any
feature space of SMAP works'."

Setup: random-init TrajectoryJEPA with the same architecture as v17 SMAP
(n_sensors=25, d=256, d_ff=1024, L=2), NO pretraining. Encode SMAP train+test
and apply Mahalanobis(PCA-10). Compare PA-F1 to v17-pretrained baseline.

Output: experiments/v18/phase4d_random_init_control.json
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

WINDOW = 100; STRIDE = 10; BATCH = 256
THRESH_PCTL = 95
PCA_K = 10
RANDOM_SEEDS = [0, 1, 2]


def build_random_model(C, seed):
    torch.manual_seed(seed)
    model = TrajectoryJEPA(
        n_sensors=C, patch_length=1,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=D_FF,
    ).to(DEVICE)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    return model


def build_pretrained_model(C):
    model = TrajectoryJEPA(
        n_sensors=C, patch_length=1,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=D_FF,
    ).to(DEVICE)
    sd = torch.load(V17 / 'ckpts' / 'v17_smap_seed42.pt',
                    map_location=DEVICE, weights_only=False)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
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


def mahal_score(H_test, starts, H_train, n_comp, T_len):
    from sklearn.decomposition import PCA
    mu = H_train.mean(axis=0); Htc = H_train - mu
    k = min(n_comp, Htc.shape[1], Htc.shape[0] - 1)
    pca = PCA(n_components=k); pca.fit(Htc)
    var = pca.explained_variance_
    Zte = pca.transform(H_test - mu)
    m2 = (Zte ** 2 / (var + 1e-8)).sum(axis=1)

    scores = np.zeros(T_len, dtype=np.float32)
    counts = np.zeros(T_len, dtype=np.float32)
    for i, s in enumerate(starts):
        end = s + WINDOW
        scores[end: min(end + STRIDE, T_len)] += m2[i]
        counts[end: min(end + STRIDE, T_len)] += 1
    valid = counts > 0
    scores[valid] /= counts[valid]
    if (~valid).any() and valid.any():
        scores[~valid] = float(np.median(scores[valid]))
    return scores


def eval_metrics(scores, labels):
    n_norm = int(0.1 * len(scores))
    thr = float(np.percentile(scores[:n_norm], THRESH_PCTL))
    m = _anomaly_metrics(scores, labels, threshold=thr)
    anom = float(scores[labels == 1].mean()) if (labels == 1).any() else 0.0
    norm = float(scores[labels == 0].mean()) if (labels == 0).any() else 0.0
    return {
        'f1_non_pa': float(m['f1_non_pa']),
        'f1_pa': float(m['f1_pa']),
        'auc_pr': float(m['auc_pr']),
        'threshold': thr,
        'score_gap': anom - norm,
    }


def run_once(model, data, T_len):
    _, H_train = encode_windows(model, data['train'])
    starts, H_test = encode_windows(model, data['test'])
    scores = mahal_score(H_test, starts, H_train, PCA_K, T_len)
    return eval_metrics(scores, data['labels'])


def main():
    V18.mkdir(exist_ok=True)
    print("Loading SMAP...", flush=True)
    data = load_smap()
    C = data['n_channels']; T_len = len(data['test'])
    print(f"  SMAP: anomaly_rate={data['anomaly_rate']:.3f}", flush=True)

    t0 = time.time()

    # Pretrained baseline (sanity — should match Phase 4b)
    print("\n[pretrained v17] running baseline...", flush=True)
    model_pre = build_pretrained_model(C)
    m_pre = run_once(model_pre, data, T_len)
    print(f"  pretrained: non-PA F1={m_pre['f1_non_pa']:.3f} PA-F1={m_pre['f1_pa']:.3f} "
          f"gap={m_pre['score_gap']:+.3f}", flush=True)
    del model_pre; torch.cuda.empty_cache()

    # Random-init controls, 3 seeds
    print("\n[random-init controls]", flush=True)
    rand_results = []
    for seed in RANDOM_SEEDS:
        model_rand = build_random_model(C, seed)
        m_rand = run_once(model_rand, data, T_len)
        rand_results.append({'seed': seed, **m_rand})
        print(f"  random seed={seed}: non-PA F1={m_rand['f1_non_pa']:.3f} "
              f"PA-F1={m_rand['f1_pa']:.3f} gap={m_rand['score_gap']:+.3f}", flush=True)
        del model_rand; torch.cuda.empty_cache()

    rand_pa_mean = float(np.mean([r['f1_pa'] for r in rand_results]))
    rand_pa_std = float(np.std([r['f1_pa'] for r in rand_results]))
    rand_nonpa_mean = float(np.mean([r['f1_non_pa'] for r in rand_results]))

    summary = {
        'config': 'v18_phase4d_random_init_control',
        'pca_k': PCA_K,
        'window': WINDOW, 'stride': STRIDE, 'thresh_pctl': THRESH_PCTL,
        'pretrained_v17': m_pre,
        'random_init': {
            'per_seed': rand_results,
            'pa_f1_mean': rand_pa_mean,
            'pa_f1_std': rand_pa_std,
            'non_pa_f1_mean': rand_nonpa_mean,
        },
        'delta_pretrained_minus_random': {
            'pa_f1': m_pre['f1_pa'] - rand_pa_mean,
            'non_pa_f1': m_pre['f1_non_pa'] - rand_nonpa_mean,
        },
        'mts_jepa_ref': {'smap_pa_f1': 0.336},
        'runtime_min': (time.time() - t0) / 60,
    }
    with open(V18 / 'phase4d_random_init_control.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n" + "=" * 60)
    print("V18 Phase 4d: Random-Init Control")
    print("=" * 60)
    print(f"  Pretrained v17:   PA-F1 {m_pre['f1_pa']:.3f}  non-PA {m_pre['f1_non_pa']:.3f}")
    print(f"  Random-init (3s): PA-F1 {rand_pa_mean:.3f} +/- {rand_pa_std:.3f}  "
          f"non-PA {rand_nonpa_mean:.3f}")
    print(f"  Delta (pre - rand): PA-F1 {summary['delta_pretrained_minus_random']['pa_f1']:+.3f}")
    print(f"  MTS-JEPA ref:     PA-F1 {0.336:.3f}")
    print(f"Runtime: {summary['runtime_min']:.1f} min")


if __name__ == '__main__':
    main()
