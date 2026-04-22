"""
V16a: Bidirectional Context + Causal Target (Fix for V15-EMA Collapse).

Root cause of V15-EMA collapse:
  Context: x_{0:t}, Target: x_{0:t+k} -> share prefix x_{0:t}
  Predictor learns to copy, not predict future. EMA cannot prevent this.

V16a fix:
  Context encoder: BidiTransformerEncoder(x_{0:t})  [bidirectional - new]
  Target encoder: EMA copy of TargetEncoder(x_{t+1:t+k})  [causal target - no shared prefix]
  Predictor: concat(h_context, PE(k)) -> h_future_hat  [unchanged]

This preserves the NON-TRIVIAL prediction task (target has no overlap with context)
while testing whether bidirectional context encoding helps vs V2's causal context.

V2 baseline reference: frozen=17.81+/-1.7, E2E=14.23+/-0.39 (5 seeds)

Output:
  experiments/v16/phase1_v16a_results.json
  experiments/v16/best_v16a_seed{seed}.pt
"""

import sys, json, time, copy, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V16_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v16')
sys.path.insert(0, str(V11_DIR))

from data_utils import (
    load_cmapss_subset, N_SENSORS, RUL_CAP, SELECTED_SENSORS,
    CMAPSSFinetuneDataset, CMAPSSTestDataset,
    collate_finetune, collate_test, compute_rul_labels,
)
from models import RULProbe

try:
    import wandb
    HAS_WANDB = True
except Exception:
    HAS_WANDB = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- Hyperparameters ----
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 2
N_CUTS = 30
BATCH_SIZE = 64
N_EPOCHS = 200
PROBE_EVERY = 10
PATIENCE = 20
EMA_MOMENTUM = 0.99
LR = 3e-4
SEEDS = [42, 123, 456]
V16_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# Positional encoding utilities
# ===========================================================================

def sinusoidal_pe(length: int, d_model: int, device) -> torch.Tensor:
    """Returns (1, length, d_model) sinusoidal PE."""
    pe = torch.zeros(length, d_model, device=device)
    pos = torch.arange(length, device=device).float().unsqueeze(1)
    half = d_model // 2
    div = torch.exp(torch.arange(half, device=device).float() * (-math.log(10000.0) / half))
    pe[:, :half] = torch.sin(pos * div)
    pe[:, half:] = torch.cos(pos * div[:d_model - half])
    return pe.unsqueeze(0)  # (1, L, D)


def horizon_pe(k: torch.Tensor, d_model: int) -> torch.Tensor:
    """Sinusoidal encoding of horizon k. Returns (B, D)."""
    half = d_model // 2
    div = torch.exp(torch.arange(half, device=k.device).float() * (-math.log(10000.0) / half))
    k_f = k.float().unsqueeze(1)  # (B, 1)
    pe = torch.cat([torch.sin(k_f * div), torch.cos(k_f * div)], dim=-1)
    if pe.shape[-1] < d_model:
        pe = F.pad(pe, (0, d_model - pe.shape[-1]))
    return pe[:, :d_model]


# ===========================================================================
# Context Encoder: Bidirectional Transformer (V16a - key change vs V2)
# ===========================================================================

class BidiContextEncoder(nn.Module):
    """
    Bidirectional transformer encoder for context x_{0:t}.
    Uses attention pooling to produce a single D-dim vector.

    Key difference from V2 ContextEncoder: NO causal mask.
    All positions in the past context attend to all other positions.
    This is valid because we only see x_{0:t} (no future leakage).
    """

    def __init__(self, n_sensors: int = 14, d_model: int = 256,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(n_sensors, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Attention pooling: learned query aggregates sequence to single vector
        self.attn_pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn_pool = nn.MultiheadAttention(d_model, n_heads,
                                               dropout=dropout, batch_first=True)
        self.pool_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, T, n_sensors)
        key_padding_mask: (B, T) bool, True = padding
        Returns: (B, d_model)
        """
        B, T, _ = x.shape
        h = self.input_proj(x)  # (B, T, D)
        pe = sinusoidal_pe(T, self.d_model, x.device)
        h = h + pe  # (B, T, D)

        # Bidirectional (no causal mask) - valid since x is purely past
        h = self.transformer(h, src_key_padding_mask=key_padding_mask)

        # Attention pooling
        query = self.attn_pool_query.expand(B, -1, -1)  # (B, 1, D)
        pooled, _ = self.attn_pool(query, h, h, key_padding_mask=key_padding_mask)
        return self.pool_norm(pooled.squeeze(1))  # (B, D)


# ===========================================================================
# Target Encoder: Bidirectional over x_{t+1:t+k} (NO shared prefix with context)
# ===========================================================================

class FutureTargetEncoder(nn.Module):
    """
    Bidirectional encoder over x_{t+1:t+k} (the FUTURE only).
    This is the critical design: target has NO overlap with context x_{0:t}.
    Makes prediction non-trivial: predictor must anticipate future state.

    Architecture mirrors V11 TargetEncoder but with attention pooling.
    EMA copy prevents collapse (well-posed since task is genuinely hard).
    """

    def __init__(self, n_sensors: int = 14, d_model: int = 256,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(n_sensors, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.attn_pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn_pool = nn.MultiheadAttention(d_model, n_heads,
                                               dropout=0.0, batch_first=True)
        self.pool_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, K, n_sensors) - future window x_{t+1:t+k}
        key_padding_mask: (B, K) bool, True = padding
        Returns: (B, d_model)
        """
        B, K, _ = x.shape
        h = self.input_proj(x)  # (B, K, D)
        pe = sinusoidal_pe(K, self.d_model, x.device)
        h = h + pe

        h = self.transformer(h, src_key_padding_mask=key_padding_mask)

        query = self.attn_pool_query.expand(B, -1, -1)
        pooled, _ = self.attn_pool(query, h, h, key_padding_mask=key_padding_mask)
        return self.pool_norm(pooled.squeeze(1))


# ===========================================================================
# Predictor
# ===========================================================================

class V16aPredictor(nn.Module):
    """Horizon-aware predictor: concat(h_context, PE(k)) -> h_future_hat."""

    def __init__(self, d_model: int = 256, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, h_ctx: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        pe_k = horizon_pe(k, h_ctx.shape[-1])
        return self.net(torch.cat([h_ctx, pe_k], dim=-1))


# ===========================================================================
# V16a Full Model
# ===========================================================================

class V16aJEPA(nn.Module):
    """
    V16a JEPA:
      - Context: BidiContextEncoder(x_{0:t})  [bidirectional, no causal mask]
      - Target: EMA copy of FutureTargetEncoder(x_{t+1:t+k})  [no shared prefix]
      - Predictor: horizon-aware MLP

    Fix vs V15-EMA: target sees ONLY x_{t+1:t+k}, not x_{0:t+k}.
    The prediction task is non-trivial: predictor cannot just copy context.
    EMA target prevents collapse (same reason V2 works: hard task + EMA).

    Fix vs V2: context encoder is bidirectional (not causal).
    All past observations attend to each other, richer context representation.
    """

    def __init__(self, n_sensors: int = 14, d_model: int = 256,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1,
                 ema_momentum: float = 0.99):
        super().__init__()
        self.ema_momentum = ema_momentum

        # Bidirectional context encoder (online, gradient)
        self.context_encoder = BidiContextEncoder(n_sensors, d_model, n_heads, n_layers, dropout)

        # Future target encoder (EMA, no gradient)
        self.target_encoder = FutureTargetEncoder(n_sensors, d_model, n_heads, n_layers, dropout)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = V16aPredictor(d_model)

    def update_ema(self):
        tau = self.ema_momentum
        for p_ctx, p_tgt in zip(self.context_encoder.parameters(),
                                 self.target_encoder.parameters()):
            p_tgt.data = tau * p_tgt.data + (1 - tau) * p_ctx.data

    def encode_context(self, x_past: torch.Tensor,
                       mask: torch.Tensor = None) -> torch.Tensor:
        """Encode past sequence for downstream tasks."""
        return self.context_encoder(x_past, key_padding_mask=mask)

    def forward_pretrain(self, x_past: torch.Tensor, past_mask: torch.Tensor,
                         x_future: torch.Tensor, future_mask: torch.Tensor,
                         k: torch.Tensor):
        """
        Training forward.

        x_past: (B, T, n_sensors) - context x_{0:t}
        past_mask: (B, T) True=padding
        x_future: (B, K, n_sensors) - future x_{t+1:t+k}
        future_mask: (B, K) True=padding
        k: (B,) horizon
        """
        # Context embedding (bidirectional over past only)
        h_ctx = self.context_encoder(x_past, key_padding_mask=past_mask)  # (B, D)

        # Target embedding (EMA over future only - no shared prefix)
        with torch.no_grad():
            h_tgt = self.target_encoder(x_future, key_padding_mask=future_mask)  # (B, D)

        # Predict future embedding from context + horizon
        h_hat = self.predictor(h_ctx, k)  # (B, D)

        # L1 loss on normalized embeddings (same as V2)
        h_hat_n = F.normalize(h_hat, dim=-1)
        h_tgt_n = F.normalize(h_tgt, dim=-1)
        l_pred = F.l1_loss(h_hat_n, h_tgt_n)

        # Variance regularizer on predictions (standard anti-collapse for EMA models)
        pred_std = h_hat.std(dim=0)  # (D,)
        l_var = F.relu(1.0 - pred_std).mean()

        loss = l_pred + 0.04 * l_var  # lambda_var=0.04 (same as V2)
        return loss, h_ctx, h_tgt


# ===========================================================================
# Dataset: returns (past=x_{0:t}, future=x_{t+1:t+k}, k)
# ===========================================================================

class V16aPretrainDataset(Dataset):
    """
    Generates (context, future, k) triples for V16a pretraining.

    context = x_{0:t}  - the past
    future  = x_{t+1:t+k} - the future (NO overlap with context)

    Normalization: per-engine z-score using context statistics only.
    """

    def __init__(self, engines: dict, n_cuts: int = 30,
                 min_past: int = 10, min_horizon: int = 5,
                 max_horizon: int = 30, seed: int = 42):
        self.engines = engines
        rng = np.random.RandomState(seed)
        self.samples = []
        for eid, arr in engines.items():
            T = len(arr)
            for _ in range(n_cuts):
                # Need: t >= min_past, t+k <= T-1, k in [min_horizon, max_horizon]
                k = int(rng.randint(min_horizon, max_horizon + 1))
                t_max = T - k - 1
                if t_max < min_past:
                    continue
                t = int(rng.randint(min_past, t_max + 1))
                self.samples.append((eid, t, k))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        eid, t, k = self.samples[idx]
        arr = self.engines[eid]  # (T, N_sensors)

        # Normalize using context statistics (no leakage)
        past = arr[:t]
        mu = past.mean(axis=0, keepdims=True)
        std = past.std(axis=0, keepdims=True) + 1e-6

        past_norm = (past - mu) / std  # (t, N)
        future_norm = (arr[t:t + k] - mu) / std  # (k, N) - future only, same normalization

        return past_norm, future_norm, k


def collate_v16a(batch):
    """Collate (past, future, k) into padded tensors."""
    pasts, futures, ks = zip(*batch)
    B = len(pasts)
    N = pasts[0].shape[1]

    T_max = max(p.shape[0] for p in pasts)
    K_max = max(f.shape[0] for f in futures)

    x_past = torch.zeros(B, T_max, N)
    past_mask = torch.ones(B, T_max, dtype=torch.bool)  # True = padding
    x_future = torch.zeros(B, K_max, N)
    future_mask = torch.ones(B, K_max, dtype=torch.bool)

    for i, (p, f, k) in enumerate(zip(pasts, futures, ks)):
        x_past[i, :p.shape[0]] = torch.from_numpy(p).float()
        past_mask[i, :p.shape[0]] = False
        x_future[i, :f.shape[0]] = torch.from_numpy(f).float()
        future_mask[i, :f.shape[0]] = False

    k_tensor = torch.tensor(ks, dtype=torch.long)
    return x_past, past_mask, x_future, future_mask, k_tensor


# ===========================================================================
# Probe evaluation
# ===========================================================================

def eval_probe_rmse(encode_fn, train_engines, val_engines, d_model=D_MODEL, seed=42):
    """Train linear probe on frozen encoder. Returns val RMSE."""
    torch.manual_seed(seed)
    probe = nn.Linear(d_model, 1).to(DEVICE)
    optim = torch.optim.Adam(probe.parameters(), lr=1e-3)

    tr_ds = CMAPSSFinetuneDataset(train_engines, n_cuts_per_engine=5, seed=seed)
    va_ds = CMAPSSFinetuneDataset(val_engines, use_last_only=True)
    tr = DataLoader(tr_ds, batch_size=32, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(va_ds, batch_size=32, shuffle=False, collate_fn=collate_finetune)

    best_val = float('inf')
    no_impr = 0
    for ep in range(100):
        probe.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            optim.zero_grad()
            with torch.no_grad():
                h = encode_fn(past, mask)
            loss = F.mse_loss(probe(h).squeeze(-1), rul)
            loss.backward()
            optim.step()

        probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                pv.append(probe(encode_fn(past, mask)).squeeze(-1).cpu().numpy())
                tv.append(rul.numpy())
        val_rmse = float(np.sqrt(np.mean(
            (np.concatenate(pv) * RUL_CAP - np.concatenate(tv) * RUL_CAP) ** 2)))
        if val_rmse < best_val:
            best_val = val_rmse
            no_impr = 0
        else:
            no_impr += 1
            if no_impr >= 10:
                break

    return best_val


# ===========================================================================
# Pretraining loop
# ===========================================================================

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pretrain_v16a(data, seed=42, ckpt_path=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = V16aJEPA(
        n_sensors=N_SENSORS, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, dropout=0.1, ema_momentum=EMA_MOMENTUM,
    ).to(DEVICE)
    n_params = count_params(model)
    print(f"  [v16a] params={n_params:,}")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, N_EPOCHS)

    history = {'loss': [], 'probe_rmse': [], 'probe_epochs': [], 'seed': seed}
    best_probe = float('inf')
    no_impr = 0

    run = None
    if HAS_WANDB:
        try:
            run = wandb.init(project='industrialjepa',
                             name=f'v16a-s{seed}',
                             tags=['v16', 'v16a', 'bidi-context-causal-target'],
                             config={'seed': seed, 'd_model': D_MODEL,
                                     'n_params': n_params, 'ema_momentum': EMA_MOMENTUM,
                                     'architecture': 'bidi_context+causal_target'},
                             reinit=True)
        except Exception as e:
            print(f"  wandb init failed: {e}")

    t0 = time.time()
    for epoch in range(1, N_EPOCHS + 1):
        ds = V16aPretrainDataset(data['train_engines'], n_cuts=N_CUTS,
                                  min_past=10, min_horizon=5, max_horizon=30,
                                  seed=epoch + seed)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_v16a, num_workers=0)

        model.train()
        total_loss = 0.0
        nbatch = 0
        for x_past, past_mask, x_future, future_mask, k in loader:
            x_past = x_past.to(DEVICE)
            past_mask = past_mask.to(DEVICE)
            x_future = x_future.to(DEVICE)
            future_mask = future_mask.to(DEVICE)
            k = k.to(DEVICE)

            optim.zero_grad()
            loss, _, _ = model.forward_pretrain(x_past, past_mask, x_future, future_mask, k)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            model.update_ema()
            total_loss += loss.item() * x_past.shape[0]
            nbatch += x_past.shape[0]

        avg_loss = total_loss / nbatch
        history['loss'].append(avg_loss)
        sched.step()

        extra = ''
        if epoch % PROBE_EVERY == 0 or epoch == 1:
            model.eval()
            encode_fn = lambda past, mask: model.encode_context(past, mask)
            probe_rmse = eval_probe_rmse(encode_fn, data['train_engines'],
                                          data['val_engines'])
            history['probe_rmse'].append(probe_rmse)
            history['probe_epochs'].append(epoch)

            if probe_rmse < best_probe:
                best_probe = probe_rmse
                no_impr = 0
                if ckpt_path is not None:
                    torch.save(model.state_dict(), ckpt_path)
            else:
                no_impr += 1

            extra = f" | probe={probe_rmse:.2f} (best={best_probe:.2f})"

            if run is not None:
                run.log({'epoch': epoch, 'train_loss': avg_loss,
                         'probe_rmse': probe_rmse, 'best_probe': best_probe})

        elapsed = (time.time() - t0) / 60
        print(f"  Ep {epoch:3d} | loss={avg_loss:.4f}{extra}", flush=True)

        if no_impr >= PATIENCE:
            print(f"  Early stop at epoch {epoch} (patience={PATIENCE})")
            break

    if run is not None:
        run.finish()

    elapsed = (time.time() - t0) / 60
    print(f"  done in {elapsed:.1f} min, best_probe={best_probe:.2f}")
    return model, history, best_probe


# ===========================================================================
# End-to-end fine-tuning evaluation
# ===========================================================================

def eval_e2e(model, data, seed=42, lr=3e-4, n_epochs=100):
    """Fine-tune entire model end-to-end. Returns test RMSE."""
    torch.manual_seed(seed)
    model_ft = copy.deepcopy(model)
    probe = nn.Linear(D_MODEL, 1).to(DEVICE)

    all_params = list(model_ft.parameters()) + list(probe.parameters())
    optim = torch.optim.AdamW(all_params, lr=lr, weight_decay=0.01)

    tr_ds = CMAPSSFinetuneDataset(data['train_engines'], n_cuts_per_engine=10, seed=seed)
    va_ds = CMAPSSFinetuneDataset(data['val_engines'], use_last_only=True)
    te_ds = CMAPSSTestDataset(data['test_engines'], data['test_rul'])
    tr = DataLoader(tr_ds, batch_size=32, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(va_ds, batch_size=32, shuffle=False, collate_fn=collate_finetune)
    te = DataLoader(te_ds, batch_size=32, shuffle=False, collate_fn=collate_test)

    best_val = float('inf')
    best_state = copy.deepcopy({'model': model_ft.state_dict(), 'probe': probe.state_dict()})
    patience = 15
    no_impr = 0

    for ep in range(n_epochs):
        model_ft.train(); probe.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            optim.zero_grad()
            h = model_ft.encode_context(past, mask)
            loss = F.mse_loss(probe(h).squeeze(-1), rul)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optim.step()

        model_ft.eval(); probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                pv.append(probe(model_ft.encode_context(past, mask)).squeeze(-1).cpu().numpy())
                tv.append(rul.numpy())
        val_rmse = float(np.sqrt(np.mean(
            (np.concatenate(pv) * RUL_CAP - np.concatenate(tv) * RUL_CAP) ** 2)))
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = copy.deepcopy({'model': model_ft.state_dict(), 'probe': probe.state_dict()})
            no_impr = 0
        else:
            no_impr += 1
            if no_impr >= patience:
                break

    # Load best model and evaluate on test
    model_ft.load_state_dict(best_state['model'])
    probe.load_state_dict(best_state['probe'])
    model_ft.eval(); probe.eval()
    preds, targets = [], []
    with torch.no_grad():
        for past, mask, rul in te:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            preds.append(probe(model_ft.encode_context(past, mask)).squeeze(-1).cpu().numpy())
            targets.append(rul.numpy())
    preds = np.concatenate(preds) * RUL_CAP
    targets = np.concatenate(targets)  # CMAPSSTestDataset returns raw cycles (not normalized)
    return float(np.sqrt(np.mean((preds - targets) ** 2)))


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    print("=" * 60)
    print("V16a: Bidirectional Context + Causal Target")
    print("Fix: target=x_{t+1:t+k} (no shared prefix with context)")
    print(f"Seeds: {SEEDS}, N_epochs: {N_EPOCHS}")
    print("Comparison: V2 frozen=17.81+/-1.7, V14-full-seq frozen=15.70")
    print("=" * 60)

    print("\nLoading FD001 data...")
    data = load_cmapss_subset('FD001')
    print(f"  Train engines: {len(data['train_engines'])}")
    print(f"  Val engines: {len(data['val_engines'])}")
    print(f"  Test engines: {len(data['test_engines'])}")

    results = {
        'config': 'v16a',
        'architecture': 'bidi_context_encoder + causal_target_encoder (EMA)',
        'fix': 'target=x_{t+1:t+k} not x_{0:t+k} - no shared prefix',
        'seeds': SEEDS,
        'n_epochs': N_EPOCHS,
        'frozen_probe_rmse_per_seed': [],
        'e2e_rmse_per_seed': [],
        'probe_histories': [],
        'baselines': {
            'v2_frozen': {'mean': 17.81, 'std': 1.7},
            'v2_e2e': {'mean': 14.23, 'std': 0.39},
            'v14_full_seq_frozen': {'mean': 15.70, 'std': 0.22},
        }
    }

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        ckpt = V16_DIR / f'best_v16a_seed{seed}.pt'
        model, history, best_probe = pretrain_v16a(data, seed=seed, ckpt_path=str(ckpt))

        print(f"  Running E2E fine-tuning (seed={seed})...")
        e2e_rmse = eval_e2e(model, data, seed=seed)
        print(f"  Seed {seed}: frozen={best_probe:.2f}, E2E={e2e_rmse:.2f}")

        results['frozen_probe_rmse_per_seed'].append(best_probe)
        results['e2e_rmse_per_seed'].append(e2e_rmse)
        results['probe_histories'].append({
            'seed': seed, 'loss': history['loss'],
            'probe_rmse': history['probe_rmse'],
            'probe_epochs': history['probe_epochs'],
        })

        # Save intermediate results
        with open(V16_DIR / 'phase1_v16a_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    # Summary
    frozen_vals = results['frozen_probe_rmse_per_seed']
    e2e_vals = results['e2e_rmse_per_seed']
    frozen_mean = float(np.mean(frozen_vals))
    frozen_std = float(np.std(frozen_vals))
    e2e_mean = float(np.mean(e2e_vals))
    e2e_std = float(np.std(e2e_vals))

    results['frozen_probe_mean'] = frozen_mean
    results['frozen_probe_std'] = frozen_std
    results['e2e_mean'] = e2e_mean
    results['e2e_std'] = e2e_std

    print("\n" + "=" * 60)
    print("V16a Summary")
    print("=" * 60)
    print(f"  Frozen probe: {frozen_mean:.2f} +/- {frozen_std:.2f}")
    print(f"  E2E RMSE:     {e2e_mean:.2f} +/- {e2e_std:.2f}")
    print(f"  V2 baseline:  frozen=17.81 +/- 1.7, E2E=14.23 +/- 0.39")
    if frozen_mean < 17.81:
        print(f"  IMPROVEMENT: +{17.81 - frozen_mean:.2f} cycles vs V2 frozen")
    else:
        print(f"  NO IMPROVEMENT vs V2 frozen ({frozen_mean - 17.81:+.2f})")

    with open(V16_DIR / 'phase1_v16a_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {V16_DIR / 'phase1_v16a_results.json'}")
