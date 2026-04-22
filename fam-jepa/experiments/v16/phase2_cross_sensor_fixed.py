"""
V16 Phase 2: Cross-Sensor Encoder WITHOUT Sensor ID Embeddings (Fix).

Problem with V15 Phase 2:
  Sensor ID embeddings caused shortcut learning. The model predicted future
  sensor values from sensor identity statistics (mean/range per sensor) rather
  than temporal degradation context. Training loss -> 0.0014 (trivial) while
  probe RMSE INCREASED from 69.62 to 75.41 (getting worse!).

V16 fix:
  Remove sensor ID embeddings entirely. Use only:
  1. Per-sensor linear projection (each sensor projects independently)
  2. Time-based positional encoding
  3. Sensor position encoding (index-based, not learnable ID)
  4. Sensor dropout 20% (kept - valid regularization, just no identity shortcut)

This forces the cross-sensor attention to learn from TEMPORAL PATTERNS
and sensor co-activation, not from sensor-specific statistics.

Architecture:
  h[t,s] = proj_s(x[t,s]) + PE_time[t] + PE_sensor_pos[s]  (no learnable ID embed)
  Cross-sensor attention: each timestep sees all sensors (bidirectional in sensor dim)
  Temporal attention: causal in time dimension

Compare against:
  V14 cross-sensor: frozen=14.98 +/- 0.22 (3 seeds, no ID embeds)
  V14 cross-sensor (FD003): frozen=17.75 +/- 0.58

Run: 3 seeds, 200 epochs, FD001.

Output:
  experiments/v16/phase2_cross_sensor_results.json
"""

import sys, json, time, copy, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V16_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v16')
sys.path.insert(0, str(V11_DIR))

from data_utils import (
    load_cmapss_subset, N_SENSORS, RUL_CAP, SELECTED_SENSORS,
    CMAPSSPretrainDataset, CMAPSSFinetuneDataset, CMAPSSTestDataset,
    collate_pretrain, collate_finetune, collate_test,
)

try:
    import wandb
    HAS_WANDB = True
except Exception:
    HAS_WANDB = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
V16_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
D_MODEL = 128  # same as V14 cross-sensor
N_HEADS = 4
N_PAIRS = 2
N_CUTS = 30
BATCH_SIZE = 32
N_EPOCHS = 200
PROBE_EVERY = 10
PATIENCE = 20
LAMBDA_VAR = 0.04
EMA_MOMENTUM = 0.99
LR = 3e-4
SEEDS = [42, 123, 456]
SENSOR_DROPOUT = 0.2


def sinusoidal_pe_1d(positions: torch.Tensor, d_model: int) -> torch.Tensor:
    """Returns (len, d_model) sinusoidal PE for a sequence of positions."""
    positions = positions.float()
    half_d = d_model // 2
    div_term = torch.exp(
        torch.arange(half_d, device=positions.device).float()
        * -(math.log(10000.0) / half_d)
    )
    sin_enc = torch.sin(positions.unsqueeze(-1) * div_term)
    cos_enc = torch.cos(positions.unsqueeze(-1) * div_term)
    pe = torch.cat([sin_enc, cos_enc], dim=-1)
    if d_model % 2 == 1:
        pe = pe[..., :d_model]
    return pe


class TBlock(nn.Module):
    """Standard pre-norm transformer block with attention weight extraction."""

    def __init__(self, d, heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, attn_mask=None, need_weights=False):
        x_ln = self.ln1(x)
        out, attn_w = self.attn(x_ln, x_ln, x_ln,
                                key_padding_mask=key_padding_mask,
                                attn_mask=attn_mask,
                                need_weights=need_weights,
                                average_attn_weights=True)
        x = x + self.drop(out)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x, attn_w


class CrossSensorEncoderFixed(nn.Module):
    """
    Cross-sensor encoder WITHOUT learnable sensor ID embeddings.

    Key fix vs V15: sensor_id_embed is REMOVED.
    Sensors are distinguished by:
      1. Separate per-sensor projection weights (each sensor has its own W_proj)
      2. Fixed sinusoidal positional encoding by sensor index (not learnable)

    This prevents shortcut learning: model cannot predict future by memorizing
    sensor-specific statistics. Must learn temporal co-activation patterns.

    Architecture:
      For each sensor s at each time t:
        h[t,s] = W_s * x[t,s] + PE_time[t] + PE_sensor[s]
      Temporal attention (causal): time axis, within each sensor
      Cross-sensor attention (bidi): sensor axis, within each time step
      Output: attention-pool the 2D (T, S) token grid -> D-dim vector
    """

    def __init__(self, n_sensors: int = N_SENSORS, d_model: int = D_MODEL,
                 n_heads: int = N_HEADS, n_pairs: int = N_PAIRS, dropout: float = 0.1,
                 sensor_dropout: float = SENSOR_DROPOUT):
        super().__init__()
        self.n_sensors = n_sensors
        self.d_model = d_model
        self.sensor_dropout = sensor_dropout

        # Per-sensor projection (each sensor has independent weights)
        # This is NOT a shortcut: different weight matrices don't encode statistics
        self.sensor_proj = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_sensors)
        ])

        # Fixed sinusoidal sensor position encoding (by index, not learnable)
        # Registered as buffer - not trained
        sensor_pos = torch.arange(n_sensors).float()
        sensor_pe = sinusoidal_pe_1d(sensor_pos, d_model)  # (S, D)
        self.register_buffer('sensor_pe', sensor_pe)

        # Alternating temporal + cross-sensor blocks
        self.temporal_blocks = nn.ModuleList([
            TBlock(d_model, n_heads, dropout) for _ in range(n_pairs)
        ])
        self.cross_blocks = nn.ModuleList([
            TBlock(d_model, n_heads, dropout) for _ in range(n_pairs)
        ])

        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None,
                sensor_dropout_mask: torch.Tensor = None,
                need_weights: bool = False):
        """
        x: (B, T, S) sensor readings
        key_padding_mask: (B, T) bool, True = padded timestep
        sensor_dropout_mask: (B, S) bool, True = dropped sensor
        Returns: h (B, d_model), [attn_weights if need_weights]
        """
        B, T, S = x.shape
        device = x.device

        # Time positional encoding
        t_pos = torch.arange(T, device=device)
        time_pe = sinusoidal_pe_1d(t_pos, self.d_model)  # (T, D)

        # Build token grid: (B, T, S, D)
        tokens = torch.zeros(B, T, S, self.d_model, device=device)
        for s in range(S):
            xs = x[:, :, s:s+1]  # (B, T, 1)
            hs = self.sensor_proj[s](xs)  # (B, T, D)
            hs = hs + time_pe.unsqueeze(0)  # add time PE
            hs = hs + self.sensor_pe[s].unsqueeze(0).unsqueeze(0)  # add fixed sensor PE
            tokens[:, :, s] = hs

        # Flatten to sequence for attention: each token is a (t, s) pair
        # tokens: (B, T, S, D) -> (B, T*S, D) for temporal pass
        # Temporal: (B*S, T, D) - within each sensor, across time
        # Cross-sensor: (B*T, S, D) - within each timestep, across sensors

        # Build causal mask for temporal attention
        causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

        # Build temporal padding mask: (B*S, T)
        if key_padding_mask is not None:
            # key_padding_mask: (B, T) -> replicate for each sensor -> (B*S, T)
            temp_mask = key_padding_mask.unsqueeze(1).expand(B, S, T).reshape(B * S, T)
        else:
            temp_mask = None

        # Build cross-sensor mask: (B*T, S) for sensor dropout
        if sensor_dropout_mask is not None:
            # sensor_dropout_mask: (B, S) -> replicate for each timestep -> (B*T, S)
            cross_mask = sensor_dropout_mask.unsqueeze(1).expand(B, T, S).reshape(B * T, S)
        else:
            cross_mask = None

        all_cross_attn = []

        for i in range(len(self.temporal_blocks)):
            # --- Temporal attention: causal, within each sensor ---
            # tokens: (B, T, S, D) -> (B*S, T, D)
            t_inp = tokens.permute(0, 2, 1, 3).reshape(B * S, T, self.d_model)
            t_out, _ = self.temporal_blocks[i](
                t_inp, key_padding_mask=temp_mask, attn_mask=causal_mask)
            tokens = t_out.reshape(B, S, T, self.d_model).permute(0, 2, 1, 3)  # (B, T, S, D)

            # --- Cross-sensor attention: bidirectional, within each timestep ---
            # tokens: (B, T, S, D) -> (B*T, S, D)
            c_inp = tokens.reshape(B * T, S, self.d_model)
            c_out, c_attn = self.cross_blocks[i](
                c_inp, key_padding_mask=cross_mask,
                need_weights=(need_weights and i == len(self.temporal_blocks) - 1))
            tokens = c_out.reshape(B, T, S, self.d_model)
            if c_attn is not None:
                all_cross_attn.append(c_attn.reshape(B, T, S, S))

        # Pool: average over sensors, then last valid timestep
        # tokens: (B, T, S, D) -> mean over S -> (B, T, D)
        pooled_sensors = tokens.mean(dim=2)  # (B, T, D)
        out_norm = self.out_norm(pooled_sensors)  # (B, T, D)

        # Take last valid timestep per batch
        if key_padding_mask is not None:
            valid = (~key_padding_mask).long()  # (B, T)
            last_idx = (valid * torch.arange(T, device=device).unsqueeze(0)).argmax(dim=1)
            h = out_norm[torch.arange(B, device=device), last_idx]  # (B, D)
        else:
            h = out_norm[:, -1]  # (B, D)

        if need_weights and len(all_cross_attn) > 0:
            return h, all_cross_attn[-1]
        return h


# ===========================================================================
# V2-style JEPA model using CrossSensorEncoderFixed
# ===========================================================================

class CrossSensorJEPAFixed(nn.Module):
    """
    JEPA with cross-sensor encoder (no ID embeddings).
    Context and target encoders are separate cross-sensor encoders.
    Target encoder is EMA copy of context encoder.
    """

    def __init__(self, n_sensors=N_SENSORS, d_model=D_MODEL, n_heads=N_HEADS,
                 n_pairs=N_PAIRS, dropout=0.1, sensor_dropout=SENSOR_DROPOUT,
                 ema_momentum=EMA_MOMENTUM, lambda_var=LAMBDA_VAR):
        super().__init__()
        self.ema_momentum = ema_momentum
        self.lambda_var = lambda_var
        self.sensor_dropout = sensor_dropout

        self.context_encoder = CrossSensorEncoderFixed(
            n_sensors, d_model, n_heads, n_pairs, dropout, sensor_dropout)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor: concat(h_context, PE(k)) -> h_future
        half_d = d_model // 2
        self.predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def update_ema(self):
        tau = self.ema_momentum
        for p_ctx, p_tgt in zip(self.context_encoder.parameters(),
                                 self.target_encoder.parameters()):
            p_tgt.data = tau * p_tgt.data + (1 - tau) * p_ctx.data

    def sinusoidal_horizon_pe(self, k, d_model):
        half = d_model // 2
        div = torch.exp(torch.arange(half, device=k.device).float() *
                        -(math.log(10000.0) / half))
        k_f = k.float().unsqueeze(1)
        pe = torch.cat([torch.sin(k_f * div), torch.cos(k_f * div)], dim=-1)
        if pe.shape[-1] < d_model:
            pe = F.pad(pe, (0, d_model - pe.shape[-1]))
        return pe[:, :d_model]

    def forward_pretrain(self, past, past_mask, future, future_mask, k):
        """
        past: (B, T, S) - context
        future: (B, K, S) - target window x_{t+1:t+k}
        k: (B,) horizon
        """
        B, T, S = past.shape

        # Sensor dropout during training (on context only)
        sensor_mask = None
        if self.training and self.sensor_dropout > 0:
            sensor_mask = torch.rand(B, S, device=past.device) < self.sensor_dropout

        # Context encoding
        h_ctx = self.context_encoder(past, key_padding_mask=past_mask,
                                     sensor_dropout_mask=sensor_mask)  # (B, D)

        # Target encoding (EMA, no dropout, no grad)
        with torch.no_grad():
            h_tgt = self.target_encoder(future, key_padding_mask=future_mask)  # (B, D)

        # Predict
        pe_k = self.sinusoidal_horizon_pe(k, h_ctx.shape[-1])
        h_hat = self.predictor(torch.cat([h_ctx, pe_k], dim=-1))  # (B, D)

        # L1 loss on normalized embeddings
        h_hat_n = F.normalize(h_hat, dim=-1)
        h_tgt_n = F.normalize(h_tgt, dim=-1)
        l_pred = F.l1_loss(h_hat_n, h_tgt_n)

        # Variance regularizer
        pred_std = h_hat.std(dim=0)
        l_var = F.relu(1.0 - pred_std).mean()

        loss = l_pred + self.lambda_var * l_var
        return loss, h_ctx, h_tgt

    def encode_context(self, x, mask=None):
        return self.context_encoder(x, key_padding_mask=mask)


# ===========================================================================
# Dataset: past = x_{0:t}, future = x_{t+1:t+k}
# ===========================================================================

class CrossSensorPretrainDataset:
    """Generates (past, future, k) for cross-sensor pretraining."""

    def __init__(self, engines, n_cuts=N_CUTS, min_past=10, min_horizon=5,
                 max_horizon=30, seed=42):
        self.engines = engines
        rng = np.random.RandomState(seed)
        self.samples = []
        for eid, arr in engines.items():
            T = len(arr)
            for _ in range(n_cuts):
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
        arr = self.engines[eid]
        past = arr[:t]
        mu = past.mean(axis=0, keepdims=True)
        std = past.std(axis=0, keepdims=True) + 1e-6
        past_norm = (past - mu) / std
        future_norm = (arr[t:t + k] - mu) / std
        return past_norm, future_norm, k


def collate_cross_sensor(batch):
    pasts, futures, ks = zip(*batch)
    B = len(pasts)
    N = pasts[0].shape[1]
    T_max = max(p.shape[0] for p in pasts)
    K_max = max(f.shape[0] for f in futures)

    x_past = torch.zeros(B, T_max, N)
    past_mask = torch.ones(B, T_max, dtype=torch.bool)
    x_future = torch.zeros(B, K_max, N)
    future_mask = torch.ones(B, K_max, dtype=torch.bool)

    for i, (p, f, k) in enumerate(zip(pasts, futures, ks)):
        x_past[i, :p.shape[0]] = torch.from_numpy(p).float()
        past_mask[i, :p.shape[0]] = False
        x_future[i, :f.shape[0]] = torch.from_numpy(f).float()
        future_mask[i, :f.shape[0]] = False

    return x_past, past_mask, x_future, future_mask, torch.tensor(ks, dtype=torch.long)


# ===========================================================================
# Probe evaluation and training
# ===========================================================================

def eval_probe_rmse(encode_fn, train_engines, val_engines, d_model=D_MODEL, seed=42):
    """Train linear probe on frozen encoder embeddings."""
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


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pretrain_cross_sensor_fixed(data, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = CrossSensorJEPAFixed(n_sensors=N_SENSORS).to(DEVICE)
    n_params = count_params(model)
    print(f"  [cross_sensor_fixed] params={n_params:,}")

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
                             name=f'v16-cross-sensor-fixed-s{seed}',
                             tags=['v16', 'cross_sensor', 'no_id_embed'],
                             config={'seed': seed, 'n_params': n_params,
                                     'sensor_dropout': SENSOR_DROPOUT,
                                     'sensor_id_embed': False},
                             reinit=True)
        except Exception as e:
            print(f"  wandb init failed: {e}")

    t0 = time.time()
    for epoch in range(1, N_EPOCHS + 1):
        ds = CrossSensorPretrainDataset(data['train_engines'], n_cuts=N_CUTS,
                                        seed=epoch + seed)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_cross_sensor, num_workers=0)

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
            else:
                no_impr += 1

            extra = f" | probe={probe_rmse:.2f} (best={best_probe:.2f})"

            if run is not None:
                run.log({'epoch': epoch, 'train_loss': avg_loss, 'probe_rmse': probe_rmse})

        elapsed = (time.time() - t0) / 60
        print(f"  Ep {epoch:3d} | loss={avg_loss:.4f}{extra}", flush=True)

        if no_impr >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    if run is not None:
        run.finish()

    elapsed = (time.time() - t0) / 60
    print(f"  done in {elapsed:.1f} min, best_probe={best_probe:.2f}")
    return model, history, best_probe


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    print("=" * 60)
    print("V16 Phase 2: Cross-Sensor Encoder (No Sensor ID Embeddings)")
    print("Fix: removed learnable sensor_id_embed -> prevents shortcut learning")
    print(f"Seeds: {SEEDS}, N_epochs: {N_EPOCHS}")
    print("Comparison: V14 cross-sensor frozen=14.98 +/- 0.22")
    print("=" * 60)

    print("\nLoading FD001 data...")
    data = load_cmapss_subset('FD001')

    results = {
        'config': 'cross_sensor_fixed',
        'fix': 'removed sensor_id_embed, kept per-sensor projection + fixed sensor PE',
        'seeds': SEEDS,
        'frozen_probe_rmse_per_seed': [],
        'probe_histories': [],
        'baselines': {
            'v14_cross_sensor': {'mean': 14.98, 'std': 0.22},
            'v15_shortcut': 'ABORTED (probe=75.41 at epoch 20, loss=0.0014)',
        }
    }

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        model, history, best_probe = pretrain_cross_sensor_fixed(data, seed=seed)
        results['frozen_probe_rmse_per_seed'].append(best_probe)
        results['probe_histories'].append({
            'seed': seed, 'loss': history['loss'],
            'probe_rmse': history['probe_rmse'],
            'probe_epochs': history['probe_epochs'],
        })
        print(f"  Seed {seed}: best_probe={best_probe:.2f}")

        with open(V16_DIR / 'phase2_cross_sensor_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    vals = results['frozen_probe_rmse_per_seed']
    mean_v = float(np.mean(vals))
    std_v = float(np.std(vals))
    results['frozen_probe_mean'] = mean_v
    results['frozen_probe_std'] = std_v

    print("\n" + "=" * 60)
    print("V16 Phase 2 Summary")
    print("=" * 60)
    print(f"  Frozen probe: {mean_v:.2f} +/- {std_v:.2f}")
    print(f"  V14 baseline: 14.98 +/- 0.22")
    if mean_v < 14.98:
        print(f"  IMPROVEMENT: +{14.98 - mean_v:.2f} cycles vs V14")
    else:
        print(f"  NO IMPROVEMENT vs V14 ({mean_v - 14.98:+.2f})")

    with open(V16_DIR / 'phase2_cross_sensor_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {V16_DIR / 'phase2_cross_sensor_results.json'}")
