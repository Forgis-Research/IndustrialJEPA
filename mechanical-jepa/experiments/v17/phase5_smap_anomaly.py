"""
V17 Phase 5: SMAP anomaly detection.

Fixes over V16 Phase 3 (see SMAP_FIX.md):
  1. anomaly_metrics argument order: (scores, y_true, threshold)  [was swapped]
  2. For scoring: use EMA target encoder (V16 incorrectly used context_encoder)
  3. Don't overtrain: 50 epochs (V16 used 100 - anomalies become predictable)

Pretraining:
  V17 on SMAP train. k ~ LogUniform[1, 500], w=10, EMA mode.
  50 epochs. Encoder = causal transformer, 2L, d_model=256, n_sensors=25.

Anomaly scoring:
  score(t) = average over k in {5, 10, 20, 50} of
             || predictor(encoder(x_{t-W:t}), k) - ema_target_encoder(x_{t+k:t+k+w}) ||_1
  Threshold: 95th percentile of first 10% of test (heuristic for normal baseline).

Metrics: non-PA F1 (primary), PA-F1, AUC-PR, TaPR-F1.
MTS-JEPA reference: SMAP PA-F1 = 33.6%.
"""

import sys, math, json, time, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

V11 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V17 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v17')
sys.path.insert(0, str(V11))
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')

from models import TrajectoryJEPA
from data.smap_msl import load_smap
from evaluation.grey_swan_metrics import anomaly_metrics as _anomaly_metrics

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Config ---
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 2
D_FF = 4 * D_MODEL
BATCH_SIZE = 64
N_EPOCHS = 50           # shorter than C-MAPSS (overtraining inverts signal - see V16)
LR = 3e-4
WEIGHT_DECAY = 0.01
EMA_MOMENTUM = 0.99
K_MAX = 500
W_WIN = 10
MIN_CTX = 50
MAX_CTX = 100
N_SAMPLES = 50000
CKPT_DIR = V17 / 'ckpts'

# Scoring
WINDOW_SIZE = 100
SCORE_K_SET = [5, 10, 20, 50]   # multi-k scoring
SCORE_STRIDE = 10                # stride-1 full density is too slow for 400K
THRESH_PERCENTILE = 95


# ============================================================
# SMAP pretraining dataset (LogUniform k, fixed-w target)
# ============================================================

class SMAPV17Dataset(Dataset):
    """Variable-length context, LogUniform k, fixed w-length target."""

    def __init__(self, data_arr, n_samples=N_SAMPLES,
                 min_ctx=MIN_CTX, max_ctx=MAX_CTX, K_max=K_MAX, w=W_WIN,
                 seed=42):
        self.data = data_arr  # (T, C)
        self.w = w
        rng = np.random.RandomState(seed)
        T = len(data_arr)
        self.samples = []
        for _ in range(n_samples):
            ctx = int(rng.randint(min_ctx, max_ctx + 1))
            # LogUniform k
            k = int(math.exp(rng.uniform(0.0, math.log(K_max))))
            k = max(1, min(k, K_max))
            t_min = ctx
            t_max = T - k - w - 1
            if t_max <= t_min:
                continue
            t = int(rng.randint(t_min, t_max + 1))
            self.samples.append((t, ctx, k))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t, ctx, k = self.samples[idx]
        past = self.data[t - ctx: t]                     # (ctx, C)
        future = self.data[t + k: t + k + self.w]        # (w, C)
        return (torch.from_numpy(past).float(),
                torch.from_numpy(future).float(), k)


def collate_smap(batch):
    pasts, futures, ks = zip(*batch)
    T_max = max(p.shape[0] for p in pasts)
    B = len(pasts); C = pasts[0].shape[1]
    x_past = torch.zeros(B, T_max, C)
    past_mask = torch.ones(B, T_max, dtype=torch.bool)
    for i, p in enumerate(pasts):
        x_past[i, :p.shape[0]] = p
        past_mask[i, :p.shape[0]] = False
    x_fut = torch.stack(list(futures), dim=0)
    fut_mask = torch.zeros(B, x_fut.shape[1], dtype=torch.bool)
    k_t = torch.tensor(ks, dtype=torch.long)
    return x_past, past_mask, x_fut, fut_mask, k_t


# ============================================================
# Pretraining
# ============================================================

def v17_loss(pred, target, lambda_var=0.04):
    pred_n = F.normalize(pred, dim=-1)
    targ_n = F.normalize(target.detach(), dim=-1)
    l_pred = F.l1_loss(pred_n, targ_n)
    pred_std = pred.std(dim=0)
    l_var = F.relu(1.0 - pred_std).mean()
    return l_pred + lambda_var * l_var, l_pred, l_var


def pretrain(data, seed=42, n_epochs=N_EPOCHS):
    torch.manual_seed(seed); np.random.seed(seed)
    C = data['n_channels']
    model = TrajectoryJEPA(
        n_sensors=C, patch_length=1,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=D_FF,
    ).to(DEVICE)
    print(f"  SMAP V17 model: C={C}, params="
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
          flush=True)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n_epochs)

    history = {'loss': [], 'pred': [], 'var': []}
    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        ds = SMAPV17Dataset(data['train'], n_samples=N_SAMPLES,
                             seed=seed * 1000 + epoch)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_smap, num_workers=0)
        model.train()
        tot = lp = lv = 0.0; n = 0
        for x_past, past_mask, x_fut, fut_mask, k in loader:
            x_past, past_mask = x_past.to(DEVICE), past_mask.to(DEVICE)
            x_fut, fut_mask = x_fut.to(DEVICE), fut_mask.to(DEVICE)
            k = k.to(DEVICE)
            optim.zero_grad()
            pred, targ, _ = model.forward_pretrain(
                x_past, past_mask, x_fut, fut_mask, k,
            )
            loss, lp_, lv_ = v17_loss(pred, targ, lambda_var=0.04)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            model.update_ema()
            B = x_past.shape[0]
            tot += loss.item() * B; lp += lp_.item() * B; lv += lv_.item() * B
            n += B
        sched.step()
        history['loss'].append(tot / n)
        history['pred'].append(lp / n)
        history['var'].append(lv / n)
        print(f"  Ep {epoch:3d} | L={tot/n:.4f} pred={lp/n:.4f} var={lv/n:.4f}",
              flush=True)

    elapsed = (time.time() - t0) / 60
    print(f"  pretrain done in {elapsed:.1f} min", flush=True)
    torch.save(model.state_dict(), CKPT_DIR / f'v17_smap_seed{seed}.pt')
    return model, history


# ============================================================
# Multi-k scoring using EMA target_encoder (bug-fixed)
# ============================================================

@torch.no_grad()
def score_multi_k(model, test_arr, window=WINDOW_SIZE, stride=SCORE_STRIDE,
                  k_set=SCORE_K_SET, w=W_WIN, batch_size=128):
    """
    For each window starting at `start`, compute the average (over k_set)
    of L1 prediction error:
       err_k = || predictor(encoder(x_{start:start+window}), k)
                  - target_encoder(x_{start+window+k:start+window+k+w}) ||_1

    Map per-window scores onto per-timestep scores by assigning score[start:start+window].

    Returns: scores (T,)
    """
    model.eval()
    T, C = test_arr.shape
    scores = np.zeros(T, dtype=np.float32)
    counts = np.zeros(T, dtype=np.float32)

    for k in k_set:
        # Valid window starts: need t = start+window, and t+k+w <= T
        starts = list(range(0, T - window - k - w, stride))
        if not starts:
            continue

        for b_start in range(0, len(starts), batch_size):
            batch = starts[b_start: b_start + batch_size]
            B = len(batch)
            x_past = np.stack([test_arr[s: s + window] for s in batch])  # (B, W, C)
            x_fut = np.stack([test_arr[s + window + k: s + window + k + w]
                              for s in batch])                            # (B, w, C)
            x_past_t = torch.from_numpy(x_past).float().to(DEVICE)
            x_fut_t = torch.from_numpy(x_fut).float().to(DEVICE)
            past_mask = torch.zeros(B, window, dtype=torch.bool, device=DEVICE)
            fut_mask = torch.zeros(B, w, dtype=torch.bool, device=DEVICE)
            k_t = torch.full((B,), k, dtype=torch.long, device=DEVICE)

            h = model.encode_past(x_past_t, past_mask)
            pred = model.predictor(h, k_t)
            # BUG FIX: use EMA target_encoder for target, NOT context_encoder
            targ = model.target_encoder(x_fut_t, fut_mask)
            err = (pred - targ).abs().mean(dim=-1).cpu().numpy()  # (B,)

            for i, s in enumerate(batch):
                scores[s: s + window] += err[i]
                counts[s: s + window] += 1

    valid = counts > 0
    scores[valid] /= counts[valid]
    # Fill missing with median (rare edge cases)
    if (~valid).any() and valid.any():
        scores[~valid] = float(np.median(scores[valid]))
    return scores


def evaluate_smap_anomaly(model, data, threshold_percentile=THRESH_PERCENTILE):
    """Returns metrics + score summary."""
    scores = score_multi_k(model, data['test'])
    labels = data['labels']

    # Score sanity: anomaly windows should score HIGHER than normal
    anom_mean = float(scores[labels == 1].mean()) if (labels == 1).any() else 0.0
    norm_mean = float(scores[labels == 0].mean()) if (labels == 0).any() else 0.0
    print(f"  score gap: anomaly={anom_mean:.4f} vs normal={norm_mean:.4f} "
          f"(gap={anom_mean - norm_mean:+.4f})", flush=True)

    # Threshold: 95th percentile of first 10% of test (heuristic normal)
    n_normal = int(0.1 * len(scores))
    threshold = float(np.percentile(scores[:n_normal], threshold_percentile))
    # FIX: correct argument order (scores, y_true, threshold)
    metrics = _anomaly_metrics(scores, labels, threshold=threshold)
    metrics['dataset'] = 'SMAP'
    metrics['threshold'] = threshold
    metrics['anom_score_mean'] = anom_mean
    metrics['norm_score_mean'] = norm_mean
    metrics['score_gap'] = anom_mean - norm_mean
    return metrics, scores


def main():
    V17.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading SMAP...", flush=True)
    data = load_smap()
    print(f"  SMAP: train={data['train'].shape} test={data['test'].shape} "
          f"anomaly_rate={data['anomaly_rate']:.3f}", flush=True)

    seed = 42
    t0 = time.time()
    model, history = pretrain(data, seed=seed, n_epochs=N_EPOCHS)
    print(f"\n=== Evaluating multi-k anomaly scores ===", flush=True)
    metrics, scores = evaluate_smap_anomaly(model, data)
    print(f"  non-PA F1 = {metrics['f1_non_pa']:.4f}  (MTS-JEPA PA-F1=0.336 "
          f"ref is *PA*, not non-PA)", flush=True)
    print(f"  PA-F1     = {metrics['f1_pa']:.4f}", flush=True)
    print(f"  AUC-PR    = {metrics['auc_pr']:.4f}", flush=True)
    print(f"  TaPR-F1   = {metrics['tapr_f1']:.4f}", flush=True)

    out = {
        'config': 'v17_smap_phase5',
        'seed': seed,
        'n_epochs': N_EPOCHS, 'K_max': K_MAX, 'w': W_WIN,
        'score_k_set': SCORE_K_SET, 'score_stride': SCORE_STRIDE,
        'threshold_percentile': THRESH_PERCENTILE,
        'history': history,
        'metrics': metrics,
        'mts_jepa_reference': {'smap_pa_f1': 0.336, 'swat_pa_f1': 0.729},
        'runtime_minutes': (time.time() - t0) / 60,
    }
    with open(V17 / 'phase5_smap_results.json', 'w') as f:
        json.dump(out, f, indent=2, default=float)

    print(f"\nSaved phase5_smap_results.json, runtime={out['runtime_minutes']:.1f} min")


if __name__ == '__main__':
    main()
