"""
V18 Phase 4i: MSL multi-seed at 150 epochs + k=50 (following Phase 4g finding).

Phase 4g found MSL Mahalanobis works with k=50 and 150 epochs (PA-F1 0.601,
seed 42). For a proper multi-seed headline we pretrain seeds 123 and 456 at
150 epochs and evaluate at k=50 (same protocol as the best seed-42 setting).

Uses SMAP pretraining recipe from v17/phase5_smap_anomaly.py, loads MSL data
from data.smap_msl.

Output: experiments/v18/phase4i_msl_multiseed.json
"""

import sys, math, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

V11 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V17 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v17')
V18 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v18')
ROOT = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')
sys.path.insert(0, str(V11)); sys.path.insert(0, str(V17)); sys.path.insert(0, str(ROOT))

from models import TrajectoryJEPA
from data.smap_msl import load_msl
from evaluation.grey_swan_metrics import anomaly_metrics as _anomaly_metrics
from phase5_smap_anomaly import (
    SMAPV17Dataset, collate_smap, v17_loss,
    D_MODEL, N_HEADS, N_LAYERS, D_FF, BATCH_SIZE, LR,
    WEIGHT_DECAY, EMA_MOMENTUM, N_SAMPLES,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CKPT_DIR = V18 / 'ckpts'
CKPT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 100; STRIDE = 10; BATCH_ENC = 256
THRESH_PCTL = 95
N_EPOCHS_LONG = 150
PCA_K = 50


def pretrain(data, seed, ckpt_path, n_epochs=N_EPOCHS_LONG):
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
        if epoch % 20 == 0 or epoch == 1 or epoch == n_epochs:
            print(f"  ep {epoch:3d} | L={tot/n:.4f} pred={lp/n:.4f}", flush=True)
    elapsed = (time.time() - t0) / 60
    print(f"  pretrain done in {elapsed:.1f} min", flush=True)
    torch.save(model.state_dict(), ckpt_path)
    return model


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


def eval_metrics(scores, labels):
    n_norm = int(0.1 * len(scores))
    thr = float(np.percentile(scores[:n_norm], THRESH_PCTL))
    m = _anomaly_metrics(scores, labels, threshold=thr)
    return {'f1_non_pa': float(m['f1_non_pa']),
            'f1_pa': float(m['f1_pa']),
            'auc_pr': float(m['auc_pr'])}


def run_seed(seed, data, T_len):
    print(f"\n=== MSL seed {seed} ===", flush=True)
    ckpt = CKPT_DIR / f'v18_msl_seed{seed}_ep150.pt'
    if ckpt.exists():
        print(f"  ckpt exists, skipping pretrain", flush=True)
        model = TrajectoryJEPA(
            n_sensors=data['n_channels'], patch_length=1,
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
            dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=D_FF,
        ).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    else:
        model = pretrain(data, seed, ckpt)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    _, H_tr = encode_windows(model, data['train'])
    starts, H_te = encode_windows(model, data['test'])
    result = {}
    for k_pca in [10, 20, 50, 100]:
        scores = mahal(H_te, starts, H_tr, k_pca, T_len)
        m = eval_metrics(scores, data['labels'])
        result[f'k{k_pca}'] = m
        print(f"  k={k_pca:>3d}: non-PA {m['f1_non_pa']:.3f} "
              f"PA {m['f1_pa']:.3f} AUC-PR {m['auc_pr']:.3f}", flush=True)
    del model; torch.cuda.empty_cache()
    return result


def main():
    data = load_msl()
    T_len = len(data['test'])
    print(f"MSL: {data['train'].shape} train / {data['test'].shape} test, "
          f"anom {data['anomaly_rate']:.3f}", flush=True)
    t0 = time.time()

    results = {}
    # Seed 42 already has ep150 ckpt from Phase 4g
    for seed in [42, 123, 456]:
        results[str(seed)] = run_seed(seed, data, T_len)
        with open(V18 / 'phase4i_msl_multiseed.json', 'w') as f:
            json.dump({'config': 'v18_phase4i_msl_multiseed',
                       'n_epochs': N_EPOCHS_LONG,
                       'seeds': [42, 123, 456],
                       'results': results,
                       'runtime_min': (time.time() - t0) / 60}, f, indent=2)

    # Aggregate
    summary_agg = {}
    for k_pca in [10, 20, 50, 100]:
        pa = [results[str(s)][f'k{k_pca}']['f1_pa'] for s in [42, 123, 456]]
        non_pa = [results[str(s)][f'k{k_pca}']['f1_non_pa'] for s in [42, 123, 456]]
        summary_agg[f'k{k_pca}'] = {
            'pa_per_seed': pa,
            'pa_mean': float(np.mean(pa)),
            'pa_std': float(np.std(pa)),
            'non_pa_mean': float(np.mean(non_pa)),
        }

    final = {'config': 'v18_phase4i_msl_multiseed',
             'n_epochs': N_EPOCHS_LONG,
             'seeds': [42, 123, 456],
             'results': results,
             'aggregate': summary_agg,
             'runtime_min': (time.time() - t0) / 60}
    with open(V18 / 'phase4i_msl_multiseed.json', 'w') as f:
        json.dump(final, f, indent=2, default=float)

    print("\n" + "=" * 60)
    print("V18 Phase 4i: MSL Mahalanobis across 3 seeds (ep150)")
    print("=" * 60)
    print(f"{'k':>4s} | {'seed 42':>8s} | {'seed 123':>8s} | {'seed 456':>8s} | {'mean +/- std':>16s}")
    for k_pca in [10, 20, 50, 100]:
        a = summary_agg[f'k{k_pca}']
        print(f"{k_pca:>4d} | {a['pa_per_seed'][0]:>8.3f} | {a['pa_per_seed'][1]:>8.3f}"
              f" | {a['pa_per_seed'][2]:>8.3f} | {a['pa_mean']:>6.3f} +/- {a['pa_std']:>5.3f}")
    print(f"Runtime: {final['runtime_min']:.1f} min")


if __name__ == '__main__':
    main()
