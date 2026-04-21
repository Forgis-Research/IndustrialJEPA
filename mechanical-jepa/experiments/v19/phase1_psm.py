"""
V19 Phase 1: PSM (server metrics, MTS-JEPA's 4th benchmark).

Pretrain FAM (v17 arch) on PSM train (132481 steps, 25 channels, 27.8% anomaly
rate in test). Then evaluate with Mahalanobis(PCA-k) anomaly scoring. Use
label-free k via cumulative variance >= 0.99 (v18 Phase 4j recipe).

Also evaluates at fixed k in {10, 20, 50, 100} for sweep comparison.

3 pretraining seeds. 50 epochs initially; switch to 150 if loss doesn't
converge (MSL needed this).

Output: experiments/v19/phase1_psm_results.json
"""

import sys, json, time, gc
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

V11 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V17 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v17')
V18 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v18')
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
N_EPOCHS_SHORT = 50
N_EPOCHS_LONG = 150
SEEDS = [42, 123, 456]

PSM_PATH = Path('/home/sagemaker-user/IndustrialJEPA/paper-replications/mts-jepa/data/PSM')


def load_psm():
    tr = np.load(PSM_PATH / 'train.npy').astype(np.float32)
    te = np.load(PSM_PATH / 'test.npy').astype(np.float32)
    lb = np.load(PSM_PATH / 'test_labels.npy').astype(np.int32)
    # Already normalised; do not re-normalise
    return {'train': tr, 'test': te, 'labels': lb,
            'n_channels': tr.shape[1], 'name': 'PSM',
            'anomaly_rate': float(lb.mean())}


def pretrain(data, seed, ckpt_path, n_epochs):
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

    history = {'loss': []}
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
            B = x_past.shape[0]
            tot += loss.item() * B; n += B
        sched.step()
        history['loss'].append(tot / n)
        if epoch % 10 == 0 or epoch in (1, n_epochs):
            print(f"  ep {epoch:3d} | L={tot/n:.4f}", flush=True)

    elapsed = (time.time() - t0) / 60
    print(f"  pretrain done in {elapsed:.1f} min", flush=True)
    torch.save(model.state_dict(), ckpt_path)
    return model, history


@torch.no_grad()
def encode_windows(model, arr):
    T, C = arr.shape
    starts = list(range(0, T - WINDOW, STRIDE))
    H_list = []
    for b in range(0, len(starts), BATCH_ENC):
        batch = starts[b:b+BATCH_ENC]
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


def find_k_variance(H_train, threshold=0.99):
    from sklearn.decomposition import PCA
    mu = H_train.mean(axis=0)
    Htc = H_train - mu
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


def run_seed(seed, data, n_epochs):
    print(f"\n=== PSM seed {seed} ({n_epochs} epochs) ===", flush=True)
    C = data['n_channels']; T_len = len(data['test'])
    ckpt = CKPT_DIR / f'v19_psm_seed{seed}_ep{n_epochs}.pt'
    if ckpt.exists():
        print(f"  ckpt exists, loading", flush=True)
        model = TrajectoryJEPA(
            n_sensors=C, patch_length=1,
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
            dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=D_FF,
        ).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    else:
        model, _ = pretrain(data, seed, ckpt, n_epochs)
    model.eval()
    for p in model.parameters(): p.requires_grad = False

    _, H_tr = encode_windows(model, data['train'])
    starts, H_te = encode_windows(model, data['test'])
    k_auto = find_k_variance(H_tr, 0.99)
    print(f"  label-free k (var>=0.99): {k_auto}", flush=True)

    result = {'seed': seed, 'n_epochs': n_epochs, 'k_auto': k_auto}
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
    data = load_psm()
    print(f"PSM: {data['train'].shape} train / {data['test'].shape} test, "
          f"anom rate {data['anomaly_rate']:.3f}", flush=True)

    t0 = time.time()
    # Start with 50 epochs, check loss, extend to 150 if needed
    results_50 = []
    for seed in SEEDS:
        r = run_seed(seed, data, N_EPOCHS_SHORT)
        results_50.append(r)
        with open(V19 / 'phase1_psm_results.json', 'w') as f:
            json.dump({'config': 'v19_phase1_psm',
                       'ep50': results_50,
                       'mts_jepa_ref_psm_pa_f1': 0.616,
                       'runtime_min': (time.time() - t0) / 60}, f,
                      indent=2, default=float)

    # Aggregate
    agg_50 = {}
    for k in ['k10', 'k20', 'k50', 'k100']:
        pa = [r[k]['f1_pa'] for r in results_50]
        nonpa = [r[k]['f1_non_pa'] for r in results_50]
        aucpr = [r[k]['auc_pr'] for r in results_50]
        agg_50[k] = {
            'pa_mean': float(np.mean(pa)), 'pa_std': float(np.std(pa)),
            'non_pa_mean': float(np.mean(nonpa)), 'non_pa_std': float(np.std(nonpa)),
            'auc_pr_mean': float(np.mean(aucpr)),
        }
    # Also aggregate label-free k
    pa_auto = [r[f'k{r["k_auto"]}']['f1_pa'] for r in results_50]
    nonpa_auto = [r[f'k{r["k_auto"]}']['f1_non_pa'] for r in results_50]
    agg_50['k_auto'] = {
        'pa_mean': float(np.mean(pa_auto)), 'pa_std': float(np.std(pa_auto)),
        'non_pa_mean': float(np.mean(nonpa_auto)),
        'k_values': [r['k_auto'] for r in results_50],
    }

    summary = {
        'config': 'v19_phase1_psm',
        'ep50': results_50,
        'ep50_aggregate': agg_50,
        'mts_jepa_ref_psm_pa_f1': 0.616,
        'runtime_min': (time.time() - t0) / 60,
    }
    with open(V19 / 'phase1_psm_results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n" + "=" * 70)
    print("V19 Phase 1: PSM Mahalanobis multi-seed")
    print("=" * 70)
    for k in ['k10', 'k20', 'k50', 'k100', 'k_auto']:
        a = agg_50[k]
        label_free = f" (k={a['k_values']})" if k == 'k_auto' else ''
        print(f"  {k:<8}: PA-F1 {a['pa_mean']:.3f} +/- {a['pa_std']:.3f}  "
              f"non-PA {a['non_pa_mean']:.3f}{label_free}")
    print(f"\nMTS-JEPA PSM PA-F1 reference: 0.616")
    print(f"Runtime: {summary['runtime_min']:.1f} min")


if __name__ == '__main__':
    main()
