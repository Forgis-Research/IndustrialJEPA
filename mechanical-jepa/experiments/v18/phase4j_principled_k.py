"""
V18 Phase 4j: Principled PCA-k selection + random-init MSL control.

Round-3 reviewer said a label-free k-selection method is ONE of three paths
to a 6. The natural approach: pick the smallest k such that cumulative PCA
variance retention >= some threshold (e.g. 95% or 99%).

For each pretrained v17 SMAP/MSL encoder:
  1. Encode training h_past into (N, 256) matrix
  2. Fit full-rank PCA
  3. Report k s.t. cumulative var ratio >= 0.90, 0.95, 0.99
  4. Re-evaluate Mahalanobis at those k values to check PA-F1 matches our
     heuristic k=100 headline

Also runs Mahalanobis(PCA-10) on random-init MSL (addresses Q5 from round-3
reviewer: "What is random-init MSL? Is it also above MTS-JEPA 0.336?")

Output: experiments/v18/phase4j_principled_k.json
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
from data.smap_msl import load_smap, load_msl
from evaluation.grey_swan_metrics import anomaly_metrics as _anomaly_metrics

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D_MODEL = 256; N_HEADS = 4; N_LAYERS = 2; D_FF = 4 * D_MODEL
EMA_MOMENTUM = 0.99
WINDOW = 100; STRIDE = 10; BATCH = 256
THRESH_PCTL = 95
VAR_THRESHOLDS = [0.90, 0.95, 0.99]


def build(C, seed=None):
    if seed is not None: torch.manual_seed(seed)
    model = TrajectoryJEPA(
        n_sensors=C, patch_length=1,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=D_FF,
    ).to(DEVICE)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    return model


def load_ckpt(C, path):
    model = build(C)
    sd = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(sd, dict) and 'model_state_dict' in sd: sd = sd['model_state_dict']
    model.load_state_dict(sd)
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


def find_k_for_variance(H_train, threshold):
    """Return smallest k such that cumulative PCA variance >= threshold."""
    from sklearn.decomposition import PCA
    mu = H_train.mean(axis=0)
    Htc = H_train - mu
    k_max = min(Htc.shape[1], Htc.shape[0] - 1)
    pca = PCA(n_components=k_max)
    pca.fit(Htc)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cum, threshold) + 1)
    return k, cum


def mahal_at_k(H_test, starts, H_train, n_comp, T_len):
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
    V18.mkdir(exist_ok=True)
    t0 = time.time()
    results = {}

    # --- SMAP principled k (3 seeds) ---
    print("\n=== SMAP: principled k selection ===", flush=True)
    smap_data = load_smap()
    C_smap = smap_data['n_channels']; T_smap = len(smap_data['test'])
    smap_ckpts = [
        (42, V17/'ckpts'/'v17_smap_seed42.pt'),
        (123, V18/'ckpts'/'v18_smap_seed123.pt'),
        (456, V18/'ckpts'/'v18_smap_seed456.pt'),
    ]
    results['smap'] = {}
    for seed, ckpt in smap_ckpts:
        if not ckpt.exists():
            print(f"  seed {seed}: ckpt missing", flush=True)
            continue
        model = load_ckpt(C_smap, ckpt)
        _, H_tr = encode_windows(model, smap_data['train'])
        starts, H_te = encode_windows(model, smap_data['test'])
        var_ks = {}
        for thr in VAR_THRESHOLDS:
            k_sel, _ = find_k_for_variance(H_tr, thr)
            var_ks[str(thr)] = k_sel
        # Eval at selected k
        metrics_at_var_k = {}
        for thr, k_sel in var_ks.items():
            scores = mahal_at_k(H_te, starts, H_tr, k_sel, T_smap)
            m = eval_metrics(scores, smap_data['labels'])
            metrics_at_var_k[f'var{thr}_k{k_sel}'] = m
        results['smap'][f'seed{seed}'] = {
            'var_thresholds_to_k': var_ks,
            'metrics': metrics_at_var_k,
        }
        print(f"  seed {seed}: k@90%={var_ks['0.9']} k@95%={var_ks['0.95']} "
              f"k@99%={var_ks['0.99']}", flush=True)
        for key, m in metrics_at_var_k.items():
            print(f"    {key}: PA-F1={m['f1_pa']:.3f}", flush=True)
        del model; torch.cuda.empty_cache()

    # Aggregate SMAP at each variance level
    smap_agg = {}
    for thr in VAR_THRESHOLDS:
        thr_s = str(thr)
        ks = [results['smap'][f'seed{s}']['var_thresholds_to_k'][thr_s] for s in [42, 123, 456]
              if f'seed{s}' in results['smap']]
        smap_agg[f'var{thr}'] = {
            'k_values': ks,
            'k_mean': float(np.mean(ks)) if ks else None,
        }
    results['smap']['aggregate'] = smap_agg

    # --- MSL principled k (3 seeds) ---
    print("\n=== MSL: principled k selection ===", flush=True)
    msl_data = load_msl()
    C_msl = msl_data['n_channels']; T_msl = len(msl_data['test'])
    msl_ckpts = [
        (42, V18/'ckpts'/'v18_msl_seed42_ep150.pt'),
        (123, V18/'ckpts'/'v18_msl_seed123_ep150.pt'),
        (456, V18/'ckpts'/'v18_msl_seed456_ep150.pt'),
    ]
    results['msl'] = {}
    for seed, ckpt in msl_ckpts:
        if not ckpt.exists():
            print(f"  seed {seed}: ckpt missing at {ckpt}", flush=True)
            continue
        model = load_ckpt(C_msl, ckpt)
        _, H_tr = encode_windows(model, msl_data['train'])
        starts, H_te = encode_windows(model, msl_data['test'])
        var_ks = {}
        for thr in VAR_THRESHOLDS:
            k_sel, _ = find_k_for_variance(H_tr, thr)
            var_ks[str(thr)] = k_sel
        metrics_at_var_k = {}
        for thr, k_sel in var_ks.items():
            scores = mahal_at_k(H_te, starts, H_tr, k_sel, T_msl)
            m = eval_metrics(scores, msl_data['labels'])
            metrics_at_var_k[f'var{thr}_k{k_sel}'] = m
        results['msl'][f'seed{seed}'] = {
            'var_thresholds_to_k': var_ks,
            'metrics': metrics_at_var_k,
        }
        print(f"  seed {seed}: k@90%={var_ks['0.9']} k@95%={var_ks['0.95']} "
              f"k@99%={var_ks['0.99']}", flush=True)
        for key, m in metrics_at_var_k.items():
            print(f"    {key}: PA-F1={m['f1_pa']:.3f}", flush=True)
        del model; torch.cuda.empty_cache()

    msl_agg = {}
    for thr in VAR_THRESHOLDS:
        thr_s = str(thr)
        ks = [results['msl'][f'seed{s}']['var_thresholds_to_k'][thr_s]
              for s in [42, 123, 456] if f'seed{s}' in results['msl']]
        msl_agg[f'var{thr}'] = {
            'k_values': ks,
            'k_mean': float(np.mean(ks)) if ks else None,
        }
    results['msl']['aggregate'] = msl_agg

    # --- Random-init MSL control (addresses round-3 Q5) ---
    print("\n=== Random-init MSL Mahalanobis (addresses round-3 Q5) ===", flush=True)
    rand_msl_results = []
    for seed in [0, 1, 2]:
        model = build(C_msl, seed=seed)
        _, H_tr = encode_windows(model, msl_data['train'])
        starts, H_te = encode_windows(model, msl_data['test'])
        scores_k10 = mahal_at_k(H_te, starts, H_tr, 10, T_msl)
        scores_k100 = mahal_at_k(H_te, starts, H_tr, 100, T_msl)
        m10 = eval_metrics(scores_k10, msl_data['labels'])
        m100 = eval_metrics(scores_k100, msl_data['labels'])
        rand_msl_results.append({
            'seed': seed, 'k10': m10, 'k100': m100,
        })
        print(f"  rand MSL seed={seed}: k=10 PA={m10['f1_pa']:.3f} | "
              f"k=100 PA={m100['f1_pa']:.3f}", flush=True)
        del model; torch.cuda.empty_cache()

    results['random_init_msl'] = {
        'per_seed': rand_msl_results,
        'k10_pa_mean': float(np.mean([r['k10']['f1_pa'] for r in rand_msl_results])),
        'k10_pa_std': float(np.std([r['k10']['f1_pa'] for r in rand_msl_results])),
        'k100_pa_mean': float(np.mean([r['k100']['f1_pa'] for r in rand_msl_results])),
        'k100_pa_std': float(np.std([r['k100']['f1_pa'] for r in rand_msl_results])),
    }

    summary = {
        'config': 'v18_phase4j_principled_k_plus_rand_msl',
        'variance_thresholds': VAR_THRESHOLDS,
        'results': results,
        'runtime_min': (time.time() - t0) / 60,
    }
    with open(V18 / 'phase4j_principled_k.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    # Print summary
    print("\n" + "=" * 70)
    print("V18 Phase 4j: Principled k-selection + random-init MSL")
    print("=" * 70)
    print("\nPrincipled k (variance-threshold heuristic):")
    for ds in ['smap', 'msl']:
        print(f"  {ds.upper()}:")
        for thr in VAR_THRESHOLDS:
            ks = results[ds]['aggregate'][f'var{thr}']['k_values']
            if ks:
                print(f"    var>={thr}: k = {ks} (mean {np.mean(ks):.1f})")
    print(f"\nRandom-init MSL control:")
    r = results['random_init_msl']
    print(f"  k=10 PA-F1: {r['k10_pa_mean']:.3f} +/- {r['k10_pa_std']:.3f}")
    print(f"  k=100 PA-F1: {r['k100_pa_mean']:.3f} +/- {r['k100_pa_std']:.3f}")
    print(f"  MTS-JEPA MSL ref: 0.336 (SMAP), MSL paper number varies")
    print(f"\nRuntime: {summary['runtime_min']:.1f} min")


if __name__ == '__main__':
    main()
