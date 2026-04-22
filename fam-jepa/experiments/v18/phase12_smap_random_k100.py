"""
V18 Phase 12: Fill the missing SMAP random-init k=100 cell.

Round-5 reviewer Major-2: "The +0.205 SMAP delta is k=100 pretrained minus
k=10 random-init, which is NOT matched rank. MSL has both rows; SMAP
random-init k=100 is missing."

Fix: run random-init (Kaiming) SMAP encoder, Mahalanobis k=100, 3 seeds.
If it matches MSL pattern (~0.62), then matched-k SMAP delta is still ~+0.17.
"""

import sys, json, time, gc
from pathlib import Path
import numpy as np
import torch

V11 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
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
SEEDS = [0, 1, 2]


def build_random(C, seed):
    torch.manual_seed(seed)
    model = TrajectoryJEPA(
        n_sensors=C, patch_length=1,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=D_FF,
    ).to(DEVICE)
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
    return {'f1_non_pa': float(m['f1_non_pa']),
            'f1_pa': float(m['f1_pa']),
            'auc_pr': float(m['auc_pr'])}


def main():
    data = load_smap()
    C = data['n_channels']; T_len = len(data['test'])
    print(f"SMAP: anom_rate {data['anomaly_rate']:.3f}", flush=True)

    t0 = time.time()
    rs = []
    for seed in SEEDS:
        model = build_random(C, seed)
        _, H_tr = encode_windows(model, data['train'])
        starts, H_te = encode_windows(model, data['test'])
        rec = {'seed': seed}
        for k_pca in [10, 100]:
            scores = mahal(H_te, starts, H_tr, k_pca, T_len)
            m = eval_metrics(scores, data['labels'])
            rec[f'k{k_pca}'] = m
            print(f"  rand SMAP seed={seed} k={k_pca}: "
                  f"non-PA={m['f1_non_pa']:.3f} PA={m['f1_pa']:.3f}", flush=True)
        rs.append(rec)
        del model; gc.collect(); torch.cuda.empty_cache()

    agg = {}
    for k_pca in [10, 100]:
        pa = [r[f'k{k_pca}']['f1_pa'] for r in rs]
        non_pa = [r[f'k{k_pca}']['f1_non_pa'] for r in rs]
        agg[f'k{k_pca}'] = {
            'pa_per_seed': pa,
            'pa_mean': float(np.mean(pa)),
            'pa_std': float(np.std(pa)),
            'non_pa_mean': float(np.mean(non_pa)),
        }
    summary = {
        'config': 'v18_phase12_smap_random_init_k100',
        'per_seed': rs,
        'aggregate': agg,
        'pretrained_ref_k100': 0.793,
        'matched_k100_delta_smap': float(0.793 - agg['k100']['pa_mean']),
        'runtime_min': (time.time() - t0) / 60,
    }
    with open(V18/'phase12_smap_random_k100.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n" + "=" * 60)
    print("V18 Phase 12: SMAP random-init Mahalanobis (matched-k)")
    print("=" * 60)
    print(f"  k=10:  PA-F1 {agg['k10']['pa_mean']:.3f} +/- {agg['k10']['pa_std']:.3f}")
    print(f"  k=100: PA-F1 {agg['k100']['pa_mean']:.3f} +/- {agg['k100']['pa_std']:.3f}")
    print(f"\nPretrained k=100: 0.793")
    print(f"Matched-k100 pretraining delta: {summary['matched_k100_delta_smap']:+.3f}")
    print(f"Previously reported k=100 pretr vs k=10 rand: +0.205 (NOT matched)")


if __name__ == '__main__':
    main()
