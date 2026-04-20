"""
V18 Phase 4h: Re-evaluate SMAP seeds 123 and 456 at Mahalanobis k=50.

Phase 4g discovered that the MSL Mahalanobis "failure" at k=10 was actually a
PCA-dimension problem: k=50 gives MSL PA-F1 0.601 (at 150 epochs). Phase 4c
already showed SMAP seed 42 at k=50: PA-F1 0.796.

For the honest multi-seed SMAP headline at the BEST k, we need SMAP seeds
123 and 456 at k=50. This phase runs them quickly (~1 min each).

Uses existing v18/ckpts/v18_smap_seed{123,456}.pt from Phase 4f.

Output: experiments/v18/phase4h_smap_k50_seeds.json
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


def load_ckpt(path, C):
    model = TrajectoryJEPA(
        n_sensors=C, patch_length=1,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=D_FF,
    ).to(DEVICE)
    sd = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(sd, dict) and 'model_state_dict' in sd: sd = sd['model_state_dict']
    model.load_state_dict(sd)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    return model


@torch.no_grad()
def encode_windows(model, arr):
    T, C = arr.shape
    starts = list(range(0, T - WINDOW, STRIDE))
    H_list = []
    for b in range(0, len(starts), BATCH):
        batch = starts[b:b+BATCH]
        B = len(batch)
        x = np.stack([arr[s:s+WINDOW] for s in batch])
        x_t = torch.from_numpy(x).float().to(DEVICE)
        pad = torch.zeros(B, WINDOW, dtype=torch.bool, device=DEVICE)
        h = model.encode_past(x_t, pad)
        H_list.append(h.cpu().numpy())
    return np.array(starts), np.concatenate(H_list, axis=0)


def mahal(H_test, starts, H_train, n_comp, T_len):
    from sklearn.decomposition import PCA
    mu = H_train.mean(axis=0); Htc = H_train - mu
    k = min(n_comp, Htc.shape[1], Htc.shape[0] - 1)
    pca = PCA(n_components=k); pca.fit(Htc)
    var = pca.explained_variance_
    Zte = pca.transform(H_test - mu)
    m2 = (Zte ** 2 / (var + 1e-8)).sum(axis=1)
    scores = np.zeros(T_len, dtype=np.float32); counts = np.zeros(T_len, dtype=np.float32)
    for i, s in enumerate(starts):
        end = s + WINDOW
        scores[end: min(end + STRIDE, T_len)] += m2[i]
        counts[end: min(end + STRIDE, T_len)] += 1
    valid = counts > 0
    scores[valid] /= counts[valid]
    return scores


def eval_metrics(scores, labels):
    n_norm = int(0.1 * len(scores))
    thr = float(np.percentile(scores[:n_norm], THRESH_PCTL))
    m = _anomaly_metrics(scores, labels, threshold=thr)
    return {
        'f1_non_pa': float(m['f1_non_pa']),
        'f1_pa': float(m['f1_pa']),
        'auc_pr': float(m['auc_pr']),
    }


def main():
    print("Loading SMAP...", flush=True)
    data = load_smap()
    C = data['n_channels']; T_len = len(data['test'])
    t0 = time.time()

    results = {}
    for seed_spec in [(42, V17/'ckpts'/'v17_smap_seed42.pt'),
                       (123, V18/'ckpts'/'v18_smap_seed123.pt'),
                       (456, V18/'ckpts'/'v18_smap_seed456.pt')]:
        seed, ckpt = seed_spec
        print(f"\n=== SMAP seed {seed} at k=50 ===", flush=True)
        model = load_ckpt(ckpt, C)
        _, H_tr = encode_windows(model, data['train'])
        starts, H_te = encode_windows(model, data['test'])
        for k_pca in [10, 20, 50, 100]:
            scores = mahal(H_te, starts, H_tr, k_pca, T_len)
            m = eval_metrics(scores, data['labels'])
            results[f'seed{seed}_k{k_pca}'] = m
            print(f"  k={k_pca:>3d}: non-PA {m['f1_non_pa']:.3f} "
                  f"PA {m['f1_pa']:.3f} AUC-PR {m['auc_pr']:.3f}", flush=True)
        del model; torch.cuda.empty_cache()

    # Aggregate PA-F1 across seeds at each k
    summary = {'config': 'v18_phase4h_smap_k50_seeds', 'per_eval': results, 'aggregate': {}}
    for k_pca in [10, 20, 50, 100]:
        pa_vals = [results[f'seed{s}_k{k_pca}']['f1_pa'] for s in [42, 123, 456]]
        nonpa_vals = [results[f'seed{s}_k{k_pca}']['f1_non_pa'] for s in [42, 123, 456]]
        summary['aggregate'][f'k{k_pca}'] = {
            'pa_per_seed': pa_vals,
            'pa_mean': float(np.mean(pa_vals)),
            'pa_std': float(np.std(pa_vals)),
            'non_pa_mean': float(np.mean(nonpa_vals)),
            'non_pa_std': float(np.std(nonpa_vals)),
        }
    summary['runtime_min'] = (time.time() - t0) / 60

    with open(V18 / 'phase4h_smap_k50_seeds.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n" + "=" * 60)
    print("V18 Phase 4h: SMAP Mahalanobis across seeds and PCA-k")
    print("=" * 60)
    print(f"{'k':>4s} | {'seed 42':>8s} | {'seed 123':>8s} | {'seed 456':>8s} | {'mean +/- std':>16s}")
    for k_pca in [10, 20, 50, 100]:
        a = summary['aggregate'][f'k{k_pca}']
        print(f"{k_pca:>4d} | {a['pa_per_seed'][0]:>8.3f} | "
              f"{a['pa_per_seed'][1]:>8.3f} | {a['pa_per_seed'][2]:>8.3f} | "
              f"{a['pa_mean']:>6.3f} +/- {a['pa_std']:>5.3f}")
    print(f"Runtime: {summary['runtime_min']:.1f} min")


if __name__ == '__main__':
    main()
