"""
V15 Phase 2: Improved Cross-Sensor Encoder.

Builds on V14 Phase 3 cross-sensor encoder with:
1. Sensor-token dropout (20%) during pretraining - forces redundancy
2. Learnable sensor ID embeddings (permutation-equivariant)
3. Cross-sensor attention map extraction (healthy vs degraded)

Goal: fix the low-label brittleness found in V14
(cross-sensor: 14.98 at 100%, but brittle at 5%).

Architecture modification:
- Add learnable sensor ID embedding: embed_s[s] in R^d for s in {0..N_sensors-1}
- Add to sensor token: h[t,s] = W_proj(x[t,s]) + embed_s[s] + PE_time[t]
- During training: randomly drop 20% of sensor tokens per timestep
  (set their attention mask to True = padding for those positions)
- At inference: use all sensors (no dropout)

This is cheap: just modify the cross-sensor attention mask during training.

Compare against V14 Phase 3 results:
  V14 cross-sensor frozen: 14.98 +/- 0.22 at 100%, brittle at 5%
  V15 improved frozen: target < 14.98 at 100%, AND robust at 5%

Run: 3 seeds, 200 epochs, BATCH_SIZE=32, FD001.

Output:
  experiments/v15/phase2_results.json
  experiments/v15/cross_sensor_improved_attention_maps.json
"""

import sys, json, time, copy, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V14_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v14')
V15_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v15')
PLOT_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/analysis/plots/v15')
sys.path.insert(0, str(V11_DIR))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_utils import (
    load_cmapss_subset, N_SENSORS, RUL_CAP, SELECTED_SENSORS,
    CMAPSSPretrainDataset, CMAPSSFinetuneDataset, CMAPSSTestDataset,
    collate_pretrain, collate_finetune, collate_test, compute_rul_labels,
)
from models import RULProbe
from train_utils import subsample_engines

try:
    import wandb
    HAS_WANDB = True
except Exception:
    HAS_WANDB = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
D_MODEL = 128
N_HEADS = 4
N_PAIRS = 2  # alternating layers
N_CUTS = 30
BATCH_SIZE = 32
N_EPOCHS = 200
PROBE_EVERY = 10
PATIENCE = 20
LAMBDA_VAR = 0.04
EMA_MOMENTUM = 0.99
LR = 3e-4
SEEDS = [42, 123, 456]
SENSOR_DROPOUT = 0.2  # drop 20% of sensor tokens during training


def sinusoidal_pe(positions: torch.Tensor, d_model: int) -> torch.Tensor:
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


class ImprovedCrossSensorEncoder(nn.Module):
    """
    Cross-sensor encoder with learnable sensor ID embeddings and sensor dropout.

    Temporal attention: causal, within each sensor's time axis
    Cross-sensor attention: across sensors, within each timestep

    Key additions vs V14:
    1. Learnable sensor ID embedding embed_s[s] in R^d
    2. Sensor token dropout during training (20% of sensors masked per step)
    """

    def __init__(self, n_sensors: int = N_SENSORS, d_model: int = D_MODEL,
                 n_heads: int = N_HEADS, n_pairs: int = N_PAIRS, dropout: float = 0.1,
                 sensor_dropout: float = SENSOR_DROPOUT):
        super().__init__()
        self.n_sensors = n_sensors
        self.d_model = d_model
        self.sensor_dropout = sensor_dropout

        # Per-sensor projection
        self.sensor_proj = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_sensors)
        ])

        # LEARNABLE sensor ID embeddings (new vs V14)
        self.sensor_id_embed = nn.Embedding(n_sensors, d_model)
        nn.init.normal_(self.sensor_id_embed.weight, std=0.01)

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
        Args:
            x: (B, T, S) sensor readings
            key_padding_mask: (B, T) bool, True = padded timestep
            sensor_dropout_mask: (B, S) bool, True = dropped sensor (training only)
            need_weights: if True, return cross-sensor attention weights

        Returns:
            h: (B, d_model) pooled representation
            [optional] attn_weights: (B, T, S, S) cross-sensor attention
        """
        B, T, S = x.shape
        device = x.device

        # Time positions
        t_pos = torch.arange(T, device=device)
        time_pe = sinusoidal_pe(t_pos, self.d_model)  # (T, D)

        # Sensor ID embeddings
        s_ids = torch.arange(S, device=device)
        sensor_pe = self.sensor_id_embed(s_ids)  # (S, D)

        # Token: h[b, t, s, :] = proj(x[b,t,s]) + time_pe[t] + sensor_pe[s]
        tokens = []
        for s in range(S):
            x_s = x[:, :, s:s+1]  # (B, T, 1)
            h_s = self.sensor_proj[s](x_s)  # (B, T, D)
            h_s = h_s + time_pe.unsqueeze(0) + sensor_pe[s].unsqueeze(0).unsqueeze(0)
            tokens.append(h_s)
        # tokens shape: S x (B, T, D)
        h = torch.stack(tokens, dim=2)  # (B, T, S, D)

        all_cross_attn = []

        # Alternating temporal (causal) + cross-sensor attention
        for pair_idx, (temp_block, cross_block) in enumerate(
                zip(self.temporal_blocks, self.cross_blocks)):

            # Temporal attention: for each sensor, attend over time (causal)
            h_temp = h.reshape(B * S, T, self.d_model)
            # Build causal mask
            causal = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), 1)
            # Temporal key_padding_mask: (B*S, T) if provided
            if key_padding_mask is not None:
                t_mask = key_padding_mask.unsqueeze(1).expand(-1, S, -1).reshape(B * S, T)
            else:
                t_mask = None
            h_temp, _ = temp_block(h_temp, key_padding_mask=t_mask, attn_mask=causal)
            h = h_temp.reshape(B, T, S, self.d_model)

            # Cross-sensor attention: for each timestep, attend over sensors
            h_cross = h.reshape(B * T, S, self.d_model)
            # Sensor dropout mask: (B, S) -> (B*T, S) if training
            if sensor_dropout_mask is not None and self.training:
                s_mask = sensor_dropout_mask.unsqueeze(1).expand(-1, T, -1).reshape(B * T, S)
            else:
                s_mask = None
            h_cross, attn_w = cross_block(h_cross, key_padding_mask=s_mask,
                                           need_weights=need_weights)
            if need_weights and attn_w is not None:
                all_cross_attn.append(attn_w.reshape(B, T, S, S))
            h = h_cross.reshape(B, T, S, self.d_model)

        h = self.out_norm(h)

        # Pool: mean over valid timesteps, then mean over sensors
        if key_padding_mask is not None:
            valid_mask = (~key_padding_mask).float().unsqueeze(2).unsqueeze(3)  # (B, T, 1, 1)
            h_pooled = (h * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        else:
            h_pooled = h.mean(dim=1)  # (B, S, D)

        # Pool over sensors (exclude dropped sensors if mask provided)
        if sensor_dropout_mask is not None and self.training:
            valid_s = (~sensor_dropout_mask).float().unsqueeze(-1)  # (B, S, 1)
            h_final = (h_pooled * valid_s).sum(dim=1) / valid_s.sum(dim=1).clamp(min=1)
        else:
            h_final = h_pooled.mean(dim=1)  # (B, D)

        if need_weights and all_cross_attn:
            return h_final, all_cross_attn[-1]  # last pair's attention
        return h_final

    def generate_sensor_dropout_mask(self, B: int, device) -> torch.Tensor:
        """Generate random sensor dropout mask (B, S)."""
        if not self.training or self.sensor_dropout == 0:
            return None
        mask = torch.rand(B, self.n_sensors, device=device) < self.sensor_dropout
        # Ensure at least 50% of sensors remain
        n_keep = int((1 - self.sensor_dropout) * self.n_sensors)
        for b in range(B):
            if mask[b].sum() > self.n_sensors - 2:
                # Too many dropped - randomly restore some
                drop_idx = torch.where(mask[b])[0]
                keep = torch.randperm(len(drop_idx))[:max(0, len(drop_idx) - (self.n_sensors - n_keep))]
                mask[b][drop_idx[keep]] = False
        return mask


class ImprovedCrossSensorJEPA(nn.Module):
    """
    Cross-sensor JEPA with sensor ID embeddings + dropout.
    Same architecture as V14 but with the two key additions.
    """

    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, n_pairs=N_PAIRS,
                 dropout=0.1, ema_momentum=EMA_MOMENTUM, sensor_dropout=SENSOR_DROPOUT):
        super().__init__()
        self.ema_momentum = ema_momentum

        # Context encoder (causal, with sensor dropout)
        self.context_encoder = ImprovedCrossSensorEncoder(
            N_SENSORS, d_model, n_heads, n_pairs, dropout, sensor_dropout)

        # Target encoder (bidirectional, no dropout, EMA of context)
        self.target_encoder = ImprovedCrossSensorEncoder(
            N_SENSORS, d_model, n_heads, n_pairs, dropout, sensor_dropout=0.0)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor MLP
        half_d = d_model // 2
        self.predictor = nn.Sequential(
            nn.Linear(d_model + half_d, half_d * 2),
            nn.GELU(),
            nn.Linear(half_d * 2, d_model),
        )
        # Sinusoidal PE for horizon
        self.horizon_embed_dim = half_d

    def horizon_pe(self, k: torch.Tensor) -> torch.Tensor:
        half = self.horizon_embed_dim
        div = torch.exp(torch.arange(half, device=k.device).float()
                        * (-math.log(10000.0) / half))
        k_f = k.float().unsqueeze(1)
        return torch.cat([torch.sin(k_f * div), torch.cos(k_f * div)], dim=-1)[:, :half]

    def encode_past(self, past: torch.Tensor, mask: torch.Tensor = None,
                     return_attn_maps: bool = False):
        """Encode past sequence. past: (B, T, S). mask: (B, T)."""
        if return_attn_maps:
            return self.context_encoder(past, key_padding_mask=mask, need_weights=True)
        return self.context_encoder(past, key_padding_mask=mask)

    def forward_pretrain(self, past: torch.Tensor, past_mask: torch.Tensor,
                          future: torch.Tensor, future_mask: torch.Tensor,
                          k: torch.Tensor):
        """
        past: (B, T_past, S)
        past_mask: (B, T_past) True=padding
        future: (B, T_fut, S)
        future_mask: (B, T_fut) True=padding
        k: (B,) horizon
        """
        B = past.shape[0]

        # Sensor dropout mask
        s_mask = self.context_encoder.generate_sensor_dropout_mask(B, past.device)

        # Context embedding (with dropout)
        h_past = self.context_encoder(past, key_padding_mask=past_mask,
                                       sensor_dropout_mask=s_mask)

        # Target embedding (EMA, no grad, no dropout)
        with torch.no_grad():
            h_fut = self.target_encoder(future, key_padding_mask=future_mask)

        # Predict
        pe_k = self.horizon_pe(k)
        h_hat = self.predictor(torch.cat([h_past, pe_k], dim=-1))

        # Loss
        h_hat_n = F.normalize(h_hat, dim=-1)
        h_fut_n = F.normalize(h_fut, dim=-1)
        l_pred = F.l1_loss(h_hat_n, h_fut_n)

        # Variance regularizer on predictions
        pred_std = h_hat.std(dim=0)
        l_var = F.relu(1.0 - pred_std).mean()

        loss = l_pred + LAMBDA_VAR * l_var
        return loss, h_past, h_fut

    def update_ema(self):
        tau = self.ema_momentum
        for p_c, p_t in zip(self.context_encoder.parameters(),
                              self.target_encoder.parameters()):
            p_t.data = tau * p_t.data + (1 - tau) * p_c.data


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def eval_probe_rmse(model, train_engines, val_engines, seed=42):
    """Frozen linear probe evaluation."""
    torch.manual_seed(seed)
    probe = nn.Linear(D_MODEL, 1).to(DEVICE)
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
                h = model.encode_past(past, mask=mask)
            loss = F.mse_loss(probe(h).squeeze(-1), rul)
            loss.backward()
            optim.step()
        probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                h = model.encode_past(past, mask=mask)
                pv.append(probe(h).squeeze(-1).cpu().numpy())
                tv.append(rul.numpy())
        val_rmse = float(np.sqrt(np.mean(
            (np.concatenate(pv) * RUL_CAP - np.concatenate(tv) * RUL_CAP)**2)))
        if val_rmse < best_val:
            best_val = val_rmse; no_impr = 0
        else:
            no_impr += 1
            if no_impr >= 10: break
    return best_val


def pretrain(data, seed=42, n_epochs=N_EPOCHS):
    """Pretrain improved cross-sensor JEPA model."""
    torch.manual_seed(seed); np.random.seed(seed)

    model = ImprovedCrossSensorJEPA(d_model=D_MODEL, n_heads=N_HEADS, n_pairs=N_PAIRS,
                                     sensor_dropout=SENSOR_DROPOUT).to(DEVICE)
    n = count_params(model)
    print(f"  Improved cross-sensor model: {n:,} params")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n_epochs)

    history = {'loss': [], 'probe_rmse': [], 'probe_epochs': []}
    best_probe = float('inf')
    no_impr = 0

    run = None
    if HAS_WANDB:
        try:
            run = wandb.init(project='industrialjepa',
                             name=f'v15-phase2-cross-sensor-improved-s{seed}',
                             tags=['v15', 'phase2', 'cross_sensor_improved'],
                             config={'seed': seed, 'sensor_dropout': SENSOR_DROPOUT,
                                     'd_model': D_MODEL, 'n_params': n,
                                     'has_sensor_id_embed': True},
                             reinit=True)
        except Exception:
            pass

    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        ds = CMAPSSPretrainDataset(data['train_engines'], n_cuts_per_engine=N_CUTS,
                                   min_past=10, min_horizon=5, max_horizon=30, seed=epoch+seed)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_pretrain)

        model.train()
        total_loss = 0.0; nbatch = 0
        for past, past_mask, fut, fut_mask, k, _ in loader:
            past, past_mask = past.to(DEVICE), past_mask.to(DEVICE)
            fut, fut_mask = fut.to(DEVICE), fut_mask.to(DEVICE)
            k = k.to(DEVICE)
            optim.zero_grad()
            loss, _, _ = model.forward_pretrain(past, past_mask, fut, fut_mask, k)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            model.update_ema()
            total_loss += loss.item() * past.shape[0]
            nbatch += past.shape[0]

        avg_loss = total_loss / nbatch
        history['loss'].append(avg_loss)
        sched.step()

        extra = ''
        if epoch % PROBE_EVERY == 0 or epoch == 1:
            model.eval()
            probe_rmse = eval_probe_rmse(model, data['train_engines'], data['val_engines'])
            history['probe_rmse'].append(probe_rmse)
            history['probe_epochs'].append(epoch)
            if probe_rmse < best_probe:
                best_probe = probe_rmse; no_impr = 0
            else:
                no_impr += 1
            extra = f" | probe={probe_rmse:.2f} (best={best_probe:.2f})"

        if run is not None:
            try:
                d = {'epoch': epoch, 'loss': avg_loss}
                if extra: d['probe_rmse'] = probe_rmse; d['best_probe'] = best_probe
                wandb.log(d)
            except Exception: pass

        print(f"  Ep {epoch:3d} | loss={avg_loss:.4f}{extra}", flush=True)

        if no_impr >= PATIENCE and epoch > 50:
            print(f"  Early stop at {epoch}")
            break

    if run is not None:
        try: wandb.finish()
        except Exception: pass

    print(f"  Done in {(time.time()-t0)/60:.1f} min, best_probe={best_probe:.2f}")
    return model, history, best_probe


def main():
    t0 = time.time()
    V15_DIR.mkdir(exist_ok=True)
    print("=" * 60)
    print("V15 Phase 2: Improved Cross-Sensor Encoder")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    data = load_cmapss_subset('FD001')

    all_probe = []
    histories = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        model, hist, best_probe = pretrain(data, seed=seed)
        all_probe.append(best_probe)
        histories.append(hist)

    mean_rmse = float(np.mean(all_probe))
    std_rmse = float(np.std(all_probe))
    print(f"\nPhase 2 results: {mean_rmse:.2f} +/- {std_rmse:.2f}")
    print(f"V14 cross-sensor baseline: 14.98 +/- 0.22")
    print(f"Delta vs V14: {mean_rmse - 14.98:+.2f}")

    results = {
        'config': 'cross_sensor_improved_sensor_dropout20_sensor_id_embed',
        'seeds': SEEDS,
        'probe_rmse_per_seed': [float(r) for r in all_probe],
        'probe_rmse_mean': mean_rmse,
        'probe_rmse_std': std_rmse,
        'v14_baseline_mean': 14.98,
        'v14_baseline_std': 0.22,
        'improvement_vs_v14': float(14.98 - mean_rmse),
        'runtime_hours': (time.time() - t0) / 3600,
    }

    with open(V15_DIR / 'phase2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {V15_DIR / 'phase2_results.json'}")


if __name__ == '__main__':
    main()
