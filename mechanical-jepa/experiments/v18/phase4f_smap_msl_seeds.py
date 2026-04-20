"""
V18 Phase 4f: Multi-seed SMAP + MSL Mahalanobis evaluation.

Reviewer round-2 Q3: "Does PA-F1 0.73 replicate on your other v17 pretraining
seeds?" - single seed was the paper's biggest remaining vulnerability on the
SMAP headline.

Reviewer round-2 W2: "The MSL 43.3 number is unsupported; run Mahalanobis on
MSL before submission."

This phase:
  1. Pretrain v17 SMAP encoder with seed 123 (~35 min).
  2. Pretrain v17 SMAP encoder with seed 456 (~35 min).
  3. Pretrain v17 MSL encoder with seed 42 (~35 min, 55 channels).
  4. For each checkpoint, compute Mahalanobis(PCA-10) on the corresponding
     held-out test set, report PA-F1 / non-PA F1 / AUC-PR.

Saves checkpoints to v18/ckpts/ and results JSON.
"""

import sys, math, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

V11 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V17 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v17')
V18 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v18')
ROOT = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')
sys.path.insert(0, str(V11)); sys.path.insert(0, str(V17)); sys.path.insert(0, str(ROOT))

from models import TrajectoryJEPA
from data.smap_msl import load_smap, load_msl
from evaluation.grey_swan_metrics import anomaly_metrics as _anomaly_metrics

# Reuse v17 phase5 dataset + collate
from phase5_smap_anomaly import (
    SMAPV17Dataset, collate_smap, v17_loss,
    D_MODEL, N_HEADS, N_LAYERS, D_FF, BATCH_SIZE, N_EPOCHS, LR,
    WEIGHT_DECAY, EMA_MOMENTUM, N_SAMPLES,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CKPT_DIR = V18 / 'ckpts'
CKPT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 100; STRIDE = 10; BATCH_ENC = 256
THRESH_PCTL = 95
PCA_K = 10


def pretrain(data, seed, ckpt_path, n_epochs=N_EPOCHS):
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
        tot = lp = 0.0; n = 0
        for x_past, past_mask, x_fut, fut_mask, k in loader:
            x_past, past_mask = x_past.to(DEVICE), past_mask.to(DEVICE)
            x_fut, fut_mask = x_fut.to(DEVICE), fut_mask.to(DEVICE)
            k = k.to(DEVICE)
            optim.zero_grad()
            pred, targ, _ = model.forward_pretrain(
                x_past, past_mask, x_fut, fut_mask, k)
            loss, lp_, _ = v17_loss(pred, targ, lambda_var=0.04)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); model.update_ema()
            B = x_past.shape[0]
            tot += loss.item() * B; lp += lp_.item() * B; n += B
        sched.step()
        if epoch % 10 == 0 or epoch == 1 or epoch == n_epochs:
            print(f"  ep {epoch:3d} | L={tot/n:.4f} pred={lp/n:.4f}", flush=True)

    elapsed = (time.time() - t0) / 60
    print(f"  pretrain done in {elapsed:.1f} min", flush=True)
    torch.save(model.state_dict(), ckpt_path)
    return model


@torch.no_grad()
def encode_windows(model, arr, window=WINDOW, stride=STRIDE):
    T, C = arr.shape
    starts = list(range(0, T - window, stride))
    H_list = []
    for b in range(0, len(starts), BATCH_ENC):
        batch = starts[b:b+BATCH_ENC]
        B = len(batch)
        x = np.stack([arr[s:s+window] for s in batch])
        x_t = torch.from_numpy(x).float().to(DEVICE)
        pad = torch.zeros(B, window, dtype=torch.bool, device=DEVICE)
        h = model.encode_past(x_t, pad)
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


def evaluate_mahal(model, data):
    T_len = len(data['test'])
    _, H_train = encode_windows(model, data['train'])
    starts, H_test = encode_windows(model, data['test'])
    scores = mahal_score(H_test, starts, H_train, PCA_K, T_len)
    anom = float(scores[data['labels'] == 1].mean()) if (data['labels'] == 1).any() else 0.0
    norm = float(scores[data['labels'] == 0].mean()) if (data['labels'] == 0).any() else 0.0
    n_norm = int(0.1 * T_len)
    thr = float(np.percentile(scores[:n_norm], THRESH_PCTL))
    m = _anomaly_metrics(scores, data['labels'], threshold=thr)
    return {
        'f1_non_pa': float(m['f1_non_pa']),
        'f1_pa': float(m['f1_pa']),
        'auc_pr': float(m['auc_pr']),
        'threshold': thr,
        'score_gap': anom - norm,
    }


def run_smap_seed(seed):
    print(f"\n=== SMAP seed={seed} ===", flush=True)
    data = load_smap()
    ckpt = CKPT_DIR / f'v18_smap_seed{seed}.pt'
    if ckpt.exists():
        print(f"  ckpt exists: {ckpt}, skipping pretrain", flush=True)
        from copy import deepcopy
        model = TrajectoryJEPA(
            n_sensors=data['n_channels'], patch_length=1,
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
            dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=D_FF,
        ).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
        model.eval()
        for p in model.parameters(): p.requires_grad = False
    else:
        model = pretrain(data, seed, ckpt)
        model.eval()
        for p in model.parameters(): p.requires_grad = False
    metrics = evaluate_mahal(model, data)
    print(f"  SMAP seed={seed}: PA-F1={metrics['f1_pa']:.3f} "
          f"non-PA={metrics['f1_non_pa']:.3f} AUC-PR={metrics['auc_pr']:.3f}", flush=True)
    del model; torch.cuda.empty_cache()
    return metrics


def run_msl_seed(seed):
    print(f"\n=== MSL seed={seed} ===", flush=True)
    data = load_msl()
    print(f"  MSL: train={data['train'].shape} test={data['test'].shape} "
          f"anomaly_rate={data['anomaly_rate']:.3f}", flush=True)
    ckpt = CKPT_DIR / f'v18_msl_seed{seed}.pt'
    if ckpt.exists():
        print(f"  ckpt exists: {ckpt}, skipping pretrain", flush=True)
        model = TrajectoryJEPA(
            n_sensors=data['n_channels'], patch_length=1,
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
            dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=D_FF,
        ).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
        model.eval()
        for p in model.parameters(): p.requires_grad = False
    else:
        model = pretrain(data, seed, ckpt)
        model.eval()
        for p in model.parameters(): p.requires_grad = False
    metrics = evaluate_mahal(model, data)
    print(f"  MSL seed={seed}: PA-F1={metrics['f1_pa']:.3f} "
          f"non-PA={metrics['f1_non_pa']:.3f} AUC-PR={metrics['auc_pr']:.3f}", flush=True)
    del model; torch.cuda.empty_cache()
    return metrics


def main():
    V18.mkdir(exist_ok=True)
    t0 = time.time()

    summary = {'config': 'v18_phase4f_multi_seed_mahal', 'smap': {}, 'msl': {}}

    # SMAP seeds 123 and 456
    for s in [123, 456]:
        summary['smap'][str(s)] = run_smap_seed(s)
        with open(V18 / 'phase4f_smap_msl_seeds.json', 'w') as f:
            json.dump(summary, f, indent=2, default=float)

    # MSL seed 42
    summary['msl']['42'] = run_msl_seed(42)
    with open(V18 / 'phase4f_smap_msl_seeds.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    # Aggregate SMAP across seeds (incl v17_smap_seed42 ref = 0.733)
    smap_pa = [0.733] + [summary['smap'][s]['f1_pa'] for s in ['123', '456']]
    summary['smap']['aggregate'] = {
        'seeds': [42, 123, 456],
        'pa_f1_per_seed': smap_pa,
        'pa_f1_mean': float(np.mean(smap_pa)),
        'pa_f1_std': float(np.std(smap_pa)),
    }

    summary['runtime_min'] = (time.time() - t0) / 60
    with open(V18 / 'phase4f_smap_msl_seeds.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n" + "=" * 60)
    print("V18 Phase 4f: Multi-seed SMAP + MSL")
    print("=" * 60)
    print(f"SMAP PA-F1 (seed 42 ref + 123 + 456):")
    for s, p in zip([42, 123, 456], smap_pa):
        print(f"  seed {s}: {p:.3f}")
    print(f"  mean: {summary['smap']['aggregate']['pa_f1_mean']:.3f} "
          f"+/- {summary['smap']['aggregate']['pa_f1_std']:.3f}")
    msl = summary['msl']['42']
    print(f"MSL seed 42: PA-F1={msl['f1_pa']:.3f} non-PA={msl['f1_non_pa']:.3f}")
    print(f"MTS-JEPA ref: SMAP 0.336, MSL ?")
    print(f"Runtime: {summary['runtime_min']:.1f} min")


if __name__ == '__main__':
    main()
