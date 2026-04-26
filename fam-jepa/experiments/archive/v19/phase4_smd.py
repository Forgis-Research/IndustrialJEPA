"""
V19 Phase 4: SMD (Server Machine Dataset) - 28 machines aggregated.

Standard SMD evaluation has 28 separate machines; MTS-JEPA and TranAD train
per-machine. For the "same recipe across domains" narrative, we train ONE FAM
on the UNION of all 28 train sets, evaluate on union of test sets with
per-sample binary labels. Anomaly rate ~9.5%.

If FAM + Mahalanobis gives reasonable PA-F1 on this aggregate, that's a strong
"one model generalises across 28 different machines" claim.

Output: experiments/v19/phase4_smd_results.json
"""

import sys, json, time, gc, glob, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

V11 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V17 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v17')
V19 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v19')
ROOT = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')
sys.path.insert(0, str(V11)); sys.path.insert(0, str(V17)); sys.path.insert(0, str(ROOT))

from models import TrajectoryJEPA
from evaluation.grey_swan_metrics import anomaly_metrics as _anomaly_metrics
from phase5_smap_anomaly import (
    SMAPV17Dataset, collate_smap, v17_loss,
    D_MODEL, N_HEADS, N_LAYERS, D_FF, BATCH_SIZE, LR,
    WEIGHT_DECAY, EMA_MOMENTUM, N_SAMPLES,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_DIR = V19 / 'ckpts'
CKPT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 100; STRIDE = 10; BATCH_ENC = 256
THRESH_PCTL = 95
SEEDS = [42, 123, 456]

SMD_PATH = Path('/home/sagemaker-user/IndustrialJEPA/paper-replications/mts-jepa/data/tranad_repo/data/SMD')


def load_smd_aggregate():
    """Concatenate all 28 machines' train/test/labels into single arrays."""
    tr_files = sorted(glob.glob(str(SMD_PATH / 'train' / '*.txt')))
    trains, tests, labels = [], [], []
    offsets = [0]
    for tp in tr_files:
        name = os.path.basename(tp)
        tr = np.loadtxt(tp, delimiter=',').astype(np.float32)
        te = np.loadtxt(str(SMD_PATH / 'test' / name), delimiter=',').astype(np.float32)
        lb = np.loadtxt(str(SMD_PATH / 'labels' / name), delimiter=',').astype(np.int32)
        trains.append(tr); tests.append(te); labels.append(lb)
        offsets.append(offsets[-1] + te.shape[0])
    train = np.concatenate(trains, axis=0)
    test = np.concatenate(tests, axis=0)
    labels_cat = np.concatenate(labels, axis=0)
    return {'train': train, 'test': test, 'labels': labels_cat,
            'n_channels': train.shape[1], 'name': 'SMD',
            'anomaly_rate': float(labels_cat.mean()),
            'machine_offsets': offsets,
            'n_machines': len(tr_files)}


def pretrain(data, seed, ckpt_path, n_epochs=50):
    torch.manual_seed(seed); np.random.seed(seed)
    C = data['n_channels']
    model = TrajectoryJEPA(
        n_sensors=C, patch_length=1,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=D_FF,
    ).to(DEVICE)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n_epochs)
    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        ds = SMAPV17Dataset(data['train'], n_samples=N_SAMPLES,
                            seed=seed * 1000 + epoch)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_smap, num_workers=0)
        model.train()
        tot = 0.0; n = 0
        for x_past, past_mask, x_fut, fut_mask, k in loader:
            x_past, past_mask = x_past.to(DEVICE), past_mask.to(DEVICE)
            x_fut, fut_mask = x_fut.to(DEVICE), fut_mask.to(DEVICE)
            k = k.to(DEVICE)
            optim.zero_grad()
            pred, targ, _ = model.forward_pretrain(x_past, past_mask, x_fut, fut_mask, k)
            loss, _, _ = v17_loss(pred, targ, lambda_var=0.04)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); model.update_ema()
            B = x_past.shape[0]; tot += loss.item() * B; n += B
        sched.step()
        if epoch % 10 == 0 or epoch in (1, n_epochs):
            print(f"  ep {epoch:3d} | L={tot/n:.4f}", flush=True)
    print(f"  pretrain done in {(time.time()-t0)/60:.1f} min", flush=True)
    torch.save(model.state_dict(), ckpt_path)
    return model


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


def run_seed(seed, data):
    print(f"\n=== SMD seed {seed} ===", flush=True)
    C = data['n_channels']; T_len = len(data['test'])
    ckpt = CKPT_DIR / f'v19_smd_seed{seed}.pt'
    if ckpt.exists():
        print(f"  ckpt exists, loading", flush=True)
        model = TrajectoryJEPA(
            n_sensors=C, patch_length=1,
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
            dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=D_FF,
        ).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    else:
        model = pretrain(data, seed, ckpt, n_epochs=50)
    model.eval()
    for p in model.parameters(): p.requires_grad = False

    _, H_tr = encode_windows(model, data['train'])
    starts, H_te = encode_windows(model, data['test'])
    k_auto = find_k_var(H_tr, 0.99)
    print(f"  label-free k (var>=0.99): {k_auto}", flush=True)

    result = {'seed': seed, 'k_auto': k_auto}
    for k_pca in [10, 20, 50, 100, k_auto]:
        scores = mahal(H_te, starts, H_tr, k_pca, T_len)
        m = eval_metrics(scores, data['labels'])
        result[f'k{k_pca}'] = m
        print(f"  k={k_pca:>3d}: non-PA={m['f1_non_pa']:.3f} "
              f"PA={m['f1_pa']:.3f} AUC-PR={m['auc_pr']:.3f}", flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    V19.mkdir(exist_ok=True)
    print("Loading SMD (union of 28 machines)...", flush=True)
    data = load_smd_aggregate()
    print(f"  train: {data['train'].shape}  test: {data['test'].shape}  "
          f"anom rate: {data['anomaly_rate']:.3f}  n_machines: {data['n_machines']}",
          flush=True)

    t0 = time.time()
    results = []
    for seed in SEEDS:
        r = run_seed(seed, data)
        results.append(r)
        with open(V19/'phase4_smd_results.json', 'w') as f:
            json.dump({'config': 'v19_phase4_smd', 'results': results,
                       'runtime_min': (time.time()-t0)/60}, f, indent=2, default=float)

    agg = {}
    for k in ['k10', 'k20', 'k50', 'k100']:
        pa = [r[k]['f1_pa'] for r in results]
        nonpa = [r[k]['f1_non_pa'] for r in results]
        agg[k] = {
            'pa_mean': float(np.mean(pa)), 'pa_std': float(np.std(pa)),
            'non_pa_mean': float(np.mean(nonpa)),
        }
    pa_auto = [r[f'k{r["k_auto"]}']['f1_pa'] for r in results]
    agg['k_auto'] = {'pa_mean': float(np.mean(pa_auto)),
                      'pa_std': float(np.std(pa_auto)),
                      'k_values': [r['k_auto'] for r in results]}

    summary = {'config': 'v19_phase4_smd',
               'n_machines': data['n_machines'],
               'results': results,
               'aggregate': agg,
               'runtime_min': (time.time()-t0)/60}
    with open(V19/'phase4_smd_results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n" + "=" * 60)
    print("V19 Phase 4: SMD (28 machines aggregated) Mahalanobis")
    print("=" * 60)
    for k in ['k10', 'k20', 'k50', 'k100', 'k_auto']:
        a = agg[k]
        extra = f" (k={a['k_values']})" if k == 'k_auto' else ''
        print(f"  {k:<8}: PA-F1 {a['pa_mean']:.3f} +/- {a['pa_std']:.3f}  "
              f"non-PA {a.get('non_pa_mean', 'n/a')}{extra}")


if __name__ == '__main__':
    main()
