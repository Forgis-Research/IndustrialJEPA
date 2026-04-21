"""
V19 Phase 4b: SMD per-machine Mahalanobis (one encoder, per-machine PCA fit).

Phase 4 (one encoder + ONE global Mahalanobis on union of 28 machines) was
weak: PA-F1 0.26. Diagnosis: per-machine distributions are heterogeneous so
a global fit averages over them.

Phase 4b tests the right middle: same encoder, but fit Mahalanobis covariance
separately per machine (using that machine's own train h distribution), then
score that machine's test with it. Aggregate PA-F1 across all machines.

This keeps the "same recipe" spirit (one encoder) while matching per-machine
scoring conventions of TranAD/MTS-JEPA.

Reuses v19_smd_seed{42,123,456}.pt from Phase 4.

Output: experiments/v19/phase4b_smd_per_machine.json
"""

import sys, json, time, gc, glob, os
from pathlib import Path
import numpy as np
import torch

V11 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V17 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v17')
V19 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v19')
ROOT = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')
sys.path.insert(0, str(V11)); sys.path.insert(0, str(V17)); sys.path.insert(0, str(ROOT))

from models import TrajectoryJEPA
from evaluation.grey_swan_metrics import anomaly_metrics as _anomaly_metrics
from phase5_smap_anomaly import (
    D_MODEL, N_HEADS, N_LAYERS, D_FF, EMA_MOMENTUM,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_DIR = V19 / 'ckpts'
WINDOW = 100; STRIDE = 10; BATCH_ENC = 256
THRESH_PCTL = 95
SEEDS = [42, 123, 456]

SMD_PATH = Path('/home/sagemaker-user/IndustrialJEPA/paper-replications/mts-jepa/data/tranad_repo/data/SMD')


def load_machine(name):
    tr = np.loadtxt(SMD_PATH/'train'/name, delimiter=',').astype(np.float32)
    te = np.loadtxt(SMD_PATH/'test'/name, delimiter=',').astype(np.float32)
    lb = np.loadtxt(SMD_PATH/'labels'/name, delimiter=',').astype(np.int32)
    return tr, te, lb


@torch.no_grad()
def encode_windows(model, arr):
    T, C = arr.shape
    starts = list(range(0, T - WINDOW, STRIDE))
    H = []
    for b in range(0, len(starts), BATCH_ENC):
        batch = starts[b:b+BATCH_ENC]
        B = len(batch)
        x = np.stack([arr[s:s+WINDOW] for s in batch])
        x_t = torch.from_numpy(x).float().to(DEVICE)
        pad = torch.zeros(B, WINDOW, dtype=torch.bool, device=DEVICE)
        h = model.encode_past(x_t, pad)
        H.append(h.cpu().numpy())
    return np.array(starts), np.concatenate(H, axis=0)


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


def find_k_var(H_train, threshold=0.99):
    from sklearn.decomposition import PCA
    mu = H_train.mean(axis=0); Htc = H_train - mu
    k_max = min(Htc.shape[1], Htc.shape[0] - 1)
    pca = PCA(n_components=k_max); pca.fit(Htc)
    cum = np.cumsum(pca.explained_variance_ratio_)
    return int(np.searchsorted(cum, threshold) + 1)


def eval_metrics(scores, labels):
    n_norm = int(0.1 * len(scores))
    thr = float(np.percentile(scores[:n_norm], THRESH_PCTL))
    m = _anomaly_metrics(scores, labels, threshold=thr)
    return {'f1_non_pa': float(m['f1_non_pa']),
            'f1_pa': float(m['f1_pa']),
            'auc_pr': float(m['auc_pr'])}


def run_seed(seed, machine_names):
    print(f"\n=== SMD per-machine seed {seed} ===", flush=True)
    ckpt = CKPT_DIR / f'v19_smd_seed{seed}.pt'
    model = TrajectoryJEPA(
        n_sensors=38, patch_length=1,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=D_FF,
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    model.eval()
    for p in model.parameters(): p.requires_grad = False

    per_machine = {}
    all_scores = []
    all_labels = []
    for name in machine_names:
        tr, te, lb = load_machine(name)
        _, H_tr = encode_windows(model, tr)
        starts, H_te = encode_windows(model, te)
        T_len = len(te)
        k_auto = find_k_var(H_tr, 0.99)
        per_k = {}
        for k_pca in [10, 20, 50, 100, k_auto]:
            scores = mahal(H_te, starts, H_tr, k_pca, T_len)
            m = eval_metrics(scores, lb)
            per_k[f'k{k_pca}'] = m
        per_machine[name] = {'k_auto': k_auto, **per_k}

    # Aggregate per-seed mean across machines
    agg = {}
    for k in ['k10', 'k20', 'k50', 'k100']:
        pa = [per_machine[n][k]['f1_pa'] for n in machine_names]
        nonpa = [per_machine[n][k]['f1_non_pa'] for n in machine_names]
        agg[k] = {'pa_mean': float(np.mean(pa)),
                  'pa_median': float(np.median(pa)),
                  'non_pa_mean': float(np.mean(nonpa))}
    # Auto k per-machine
    pa_auto = [per_machine[n][f'k{per_machine[n]["k_auto"]}']['f1_pa']
               for n in machine_names]
    agg['k_auto_per_machine'] = {
        'pa_mean': float(np.mean(pa_auto)),
        'pa_median': float(np.median(pa_auto)),
    }
    for k in ['k10', 'k20', 'k50', 'k100']:
        print(f"  {k}: PA-F1 mean {agg[k]['pa_mean']:.3f}  "
              f"median {agg[k]['pa_median']:.3f}", flush=True)

    del model; gc.collect(); torch.cuda.empty_cache()
    return {'seed': seed, 'per_machine': per_machine, 'aggregate_across_machines': agg}


def main():
    t0 = time.time()
    machine_names = sorted([os.path.basename(p)
                             for p in glob.glob(str(SMD_PATH/'train'/'*.txt'))])
    print(f"Running per-machine SMD Mahalanobis on {len(machine_names)} machines, "
          f"{len(SEEDS)} seeds...", flush=True)

    results = []
    for seed in SEEDS:
        r = run_seed(seed, machine_names)
        results.append(r)
        with open(V19/'phase4b_smd_per_machine.json', 'w') as f:
            json.dump({'config': 'v19_phase4b_smd_per_machine',
                       'n_machines': len(machine_names),
                       'results': results,
                       'runtime_min': (time.time()-t0)/60}, f, indent=2, default=float)

    # Final aggregate across seeds
    final_agg = {}
    for k in ['k10', 'k20', 'k50', 'k100']:
        pa_means = [r['aggregate_across_machines'][k]['pa_mean'] for r in results]
        pa_medians = [r['aggregate_across_machines'][k]['pa_median'] for r in results]
        final_agg[k] = {
            'pa_mean_mean': float(np.mean(pa_means)),
            'pa_mean_std': float(np.std(pa_means)),
            'pa_median_mean': float(np.mean(pa_medians)),
        }
    summary = {
        'config': 'v19_phase4b_smd_per_machine',
        'n_machines': len(machine_names),
        'results': results,
        'final_aggregate_across_seeds': final_agg,
        'runtime_min': (time.time()-t0)/60,
    }
    with open(V19/'phase4b_smd_per_machine.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n" + "=" * 60)
    print("V19 Phase 4b: SMD per-machine Mahalanobis")
    print("=" * 60)
    for k in ['k10', 'k20', 'k50', 'k100']:
        a = final_agg[k]
        print(f"  {k}: PA-F1 mean-of-machine-means {a['pa_mean_mean']:.3f} "
              f"+/- {a['pa_mean_std']:.3f}  median {a['pa_median_mean']:.3f}")
    print(f"\nPhase 4 (aggregate scoring) was: 0.26 PA-F1 at k=20")
    print(f"Runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
