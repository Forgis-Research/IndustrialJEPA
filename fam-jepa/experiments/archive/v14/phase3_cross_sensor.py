"""
V14 Phase 3: Cross-sensor (iTransformer-inspired) attention encoder.

Encoder alternates temporal attention (causal, within each sensor's cycle
stream) and cross-sensor attention (within each cycle, across the 14
sensors). This is closer to STAR's two-stage design.

Model:
  token[b, t, s, :] = W_proj * x[b, t, s] + sensor_embed[s] + time_PE[t]
  for layer in alt_layers:
      if temporal:  token = attn over t (causal, within each sensor)
      if cross:     token = attn over s (within each cycle)
  h_past = pool_s(token[b, last_non_pad_t, :, :])

Target encoder: same arch but bidirectional.
Predictor: same as V2 (MLP).
Loss: same as V2 (L1 on L2-normalized latents + variance reg).

Pretrain on FD001 (original future-only target for direct V2 comparison),
3-seed frozen + E2E at 100% labels, then 5-seed frozen at 5/10/20%.

Output:
  experiments/v14/cross_sensor_results.json
  experiments/v14/best_pretrain_cross_sensor.pt
  experiments/v14/cross_sensor_attention_maps.json
  analysis/plots/v14/cross_sensor_attention_*.png
"""

import sys, json, time, copy, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V14_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v14')
PLOT_PNG = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/analysis/plots/v14')
sys.path.insert(0, str(V11_DIR))

from data_utils import (
    load_cmapss_subset, N_SENSORS, RUL_CAP, SELECTED_SENSORS,
    CMAPSSPretrainDataset, CMAPSSFinetuneDataset, CMAPSSTestDataset,
    collate_pretrain, collate_finetune, collate_test, compute_rul_labels,
)
from models import RULProbe, trajectory_jepa_loss
from train_utils import subsample_engines

try:
    import wandb
    HAS_WANDB = True
except Exception:
    HAS_WANDB = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# Sinusoidal PE (copied to avoid circular import)
# ============================================================

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


# ============================================================
# Transformer layer (with optional causal mask, returns attn weights)
# ============================================================

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


# ============================================================
# Cross-sensor encoder
# ============================================================

class CrossSensorEncoder(nn.Module):
    """
    Token at (t, s) = Linear([x_{t,s}]) + sensor_embed[s] + time_PE[t]
    Layers alternate temporal causal attention (reshape to (B*S, T, d)) and
    cross-sensor attention (reshape to (B*T, S, d)).
    """

    def __init__(self, n_sensors=N_SENSORS, d_model=128, n_heads=4,
                 n_pairs=2, dropout=0.1, causal_temporal=True):
        super().__init__()
        self.n_sensors = n_sensors
        self.d = d_model
        self.causal_temporal = causal_temporal
        self.proj = nn.Linear(1, d_model)
        self.sensor_embed = nn.Parameter(torch.randn(n_sensors, d_model) * 0.02)
        self.n_pairs = n_pairs
        self.temporal_blocks = nn.ModuleList(
            [TBlock(d_model, n_heads, dropout) for _ in range(n_pairs)]
        )
        self.cross_blocks = nn.ModuleList(
            [TBlock(d_model, n_heads, dropout) for _ in range(n_pairs)]
        )
        self.norm_out = nn.LayerNorm(d_model)
        # Pool across sensors via a learnable query
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0,
                                                batch_first=True)

    def forward(self, x, key_padding_mask=None,
                return_attn_maps=False):
        """
        x: (B, T, S)
        key_padding_mask: (B, T) True at padding positions
        return_attn_maps: if True, also return per-cycle cross-sensor attention
                          maps of shape (B, T, S, S).
        Returns:
          h: (B, d_model) representation (pool across sensors at last non-pad t).
        """
        B, T, S = x.shape
        assert S == self.n_sensors, f"{S} vs {self.n_sensors}"

        # Project each scalar to d_model
        tok = self.proj(x.unsqueeze(-1))  # (B, T, S, d)
        tok = tok + self.sensor_embed.view(1, 1, S, self.d)

        # Time PE
        t_pe = sinusoidal_pe(torch.arange(T, device=x.device), self.d)  # (T, d)
        tok = tok + t_pe.view(1, T, 1, self.d)

        cross_attn_maps = None  # collect last cross-sensor layer's attn

        for p in range(self.n_pairs):
            # ---- Temporal attention: reshape to (B*S, T, d) ----
            temp_in = tok.permute(0, 2, 1, 3).reshape(B * S, T, self.d)  # (B*S, T, d)
            temp_mask = None
            if key_padding_mask is not None:
                # Expand: (B, T) -> (B, S, T) -> (B*S, T)
                temp_mask = (key_padding_mask.unsqueeze(1).expand(-1, S, -1)
                              .reshape(B * S, T))
            attn_mask = None
            if self.causal_temporal:
                attn_mask = torch.triu(torch.full((T, T), float('-inf'),
                                                  device=x.device), diagonal=1)
            temp_out, _ = self.temporal_blocks[p](temp_in,
                                                   key_padding_mask=temp_mask,
                                                   attn_mask=attn_mask)
            tok = temp_out.reshape(B, S, T, self.d).permute(0, 2, 1, 3)  # (B, T, S, d)

            # ---- Cross-sensor attention: reshape to (B*T, S, d) ----
            cross_in = tok.reshape(B * T, S, self.d)  # (B*T, S, d)
            # Don't pass key_padding_mask here - sensor dimension has no padding
            cross_out, cross_w = self.cross_blocks[p](cross_in,
                                                      need_weights=return_attn_maps and p == self.n_pairs - 1)
            tok = cross_out.reshape(B, T, S, self.d)
            if return_attn_maps and p == self.n_pairs - 1 and cross_w is not None:
                cross_attn_maps = cross_w.reshape(B, T, S, S)

        tok = self.norm_out(tok)

        # Find last non-pad t per batch
        if key_padding_mask is not None:
            valid = (~key_padding_mask).long()  # (B, T)
            last_idx = (valid * torch.arange(T, device=x.device).unsqueeze(0)).argmax(dim=1)
        else:
            last_idx = torch.full((B,), T - 1, device=x.device, dtype=torch.long)

        # Take token at last t: (B, S, d)
        last_tok = tok[torch.arange(B, device=x.device), last_idx]  # (B, S, d)

        # Pool across sensors using a learnable query
        q = self.pool_query.expand(B, -1, -1)  # (B, 1, d)
        h, _ = self.pool_attn(q, last_tok, last_tok, need_weights=False)
        h = h.squeeze(1)  # (B, d)

        if return_attn_maps:
            return h, cross_attn_maps
        return h


class CrossSensorTargetEncoder(nn.Module):
    """Bidirectional counterpart: same arch but with causal_temporal=False,
    and pools the whole sequence (not just last t)."""

    def __init__(self, n_sensors=N_SENSORS, d_model=128, n_heads=4,
                 n_pairs=2, dropout=0.1):
        super().__init__()
        self.enc = CrossSensorEncoder(n_sensors, d_model, n_heads, n_pairs, dropout,
                                       causal_temporal=False)
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0,
                                                batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        B = x.shape[0]
        # Reuse CrossSensorEncoder for the per-cycle tokens, but we want the
        # full trajectory not just the last cycle's tokens. Easiest: call
        # self.enc internals manually by replicating with a full-pool.
        # Simpler shortcut: call enc which gives (B, d) from the last t;
        # for a bidirectional target we want to pool across ALL cycles.
        # Re-implement minimally for the target:
        with torch.enable_grad():
            B, T, S = x.shape
            tok = self.enc.proj(x.unsqueeze(-1))
            tok = tok + self.enc.sensor_embed.view(1, 1, S, self.enc.d)
            t_pe = sinusoidal_pe(torch.arange(T, device=x.device), self.enc.d)
            tok = tok + t_pe.view(1, T, 1, self.enc.d)
            for p in range(self.enc.n_pairs):
                temp_in = tok.permute(0, 2, 1, 3).reshape(B * S, T, self.enc.d)
                temp_mask = None
                if key_padding_mask is not None:
                    temp_mask = (key_padding_mask.unsqueeze(1).expand(-1, S, -1)
                                  .reshape(B * S, T))
                temp_out, _ = self.enc.temporal_blocks[p](temp_in,
                                                           key_padding_mask=temp_mask,
                                                           attn_mask=None)
                tok = temp_out.reshape(B, S, T, self.enc.d).permute(0, 2, 1, 3)
                cross_in = tok.reshape(B * T, S, self.enc.d)
                cross_out, _ = self.enc.cross_blocks[p](cross_in)
                tok = cross_out.reshape(B, T, S, self.enc.d)
            tok = self.enc.norm_out(tok)
            # Flatten T*S tokens, pool via query
            flat = tok.reshape(B, T * S, self.enc.d)
            flat_mask = None
            if key_padding_mask is not None:
                # Expand (B, T) to (B, T*S)
                flat_mask = key_padding_mask.unsqueeze(-1).expand(-1, -1, S).reshape(B, T * S)
            q = self.pool_query.expand(B, -1, -1)
            pooled, _ = self.pool_attn(q, flat, flat,
                                        key_padding_mask=flat_mask, need_weights=False)
            return self.norm(pooled.squeeze(1))


# ============================================================
# Full model (CrossSensorJEPA)
# ============================================================

class Predictor(nn.Module):
    def __init__(self, d=128, d_hidden=256):
        super().__init__()
        self.d = d
        self.net = nn.Sequential(
            nn.Linear(2 * d, d_hidden), nn.ReLU(), nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d),
        )

    def forward(self, h_past, k):
        k_pe = sinusoidal_pe(k, self.d)
        return self.net(torch.cat([h_past, k_pe], dim=-1))


class CrossSensorJEPA(nn.Module):
    def __init__(self, n_sensors=N_SENSORS, d_model=128, n_heads=4,
                 n_pairs=2, dropout=0.1, ema_momentum=0.99):
        super().__init__()
        self.d = d_model
        self.ema = ema_momentum
        self.context_encoder = CrossSensorEncoder(
            n_sensors, d_model, n_heads, n_pairs, dropout, causal_temporal=True)
        self.target_encoder = CrossSensorTargetEncoder(
            n_sensors, d_model, n_heads, n_pairs, dropout)
        self.predictor = Predictor(d_model, 2 * d_model)
        # Target encoder does not receive gradient
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        # Initialize target shared components from context
        self._init_target()

    def _init_target(self):
        ctx = self.context_encoder.state_dict()
        tgt = self.target_encoder.state_dict()
        # Target encoder has enc.* keys matching context's keys
        for key in list(tgt.keys()):
            if key.startswith('enc.'):
                src_key = key[len('enc.'):]
                if src_key in ctx:
                    tgt[key] = ctx[src_key].clone()
        self.target_encoder.load_state_dict(tgt)

    @torch.no_grad()
    def update_ema(self):
        m = self.ema
        ctx = dict(self.context_encoder.named_parameters())
        tgt = dict(self.target_encoder.named_parameters())
        for key in tgt:
            if key.startswith('enc.'):
                src_key = key[len('enc.'):]
                if src_key in ctx:
                    tgt[key].data.mul_(m).add_(ctx[src_key].data, alpha=1 - m)

    def forward_pretrain(self, past, past_mask, future, future_mask, k):
        h_past = self.context_encoder(past, key_padding_mask=past_mask)
        with torch.no_grad():
            h_future = self.target_encoder(future, key_padding_mask=future_mask)
        pred = self.predictor(h_past, k)
        return pred, h_future, h_past

    def encode_past(self, past, past_mask=None, return_attn_maps=False):
        return self.context_encoder(past, key_padding_mask=past_mask,
                                     return_attn_maps=return_attn_maps)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# Probe helper (mirror phase2 eval_probe_rmse)
# ============================================================

def eval_probe_rmse(model, train_eng, val_eng, d_model=128, n_probe_epochs=30):
    model.eval()
    probe = RULProbe(d_model).to(DEVICE)
    optim = torch.optim.Adam(probe.parameters(), lr=1e-3)
    tr = DataLoader(CMAPSSFinetuneDataset(train_eng, n_cuts_per_engine=3),
                    batch_size=32, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(CMAPSSFinetuneDataset(val_eng, use_last_only=False, n_cuts_per_engine=10),
                    batch_size=32, shuffle=False, collate_fn=collate_finetune)
    for _ in range(n_probe_epochs):
        probe.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            with torch.no_grad():
                h = model.encode_past(past, mask)
            loss = F.mse_loss(probe(h), rul)
            optim.zero_grad(); loss.backward(); optim.step()
    probe.eval()
    preds, targets = [], []
    with torch.no_grad():
        for past, mask, rul in va:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = model.encode_past(past, mask)
            preds.append(probe(h).cpu().numpy()); targets.append(rul.numpy())
    preds = np.concatenate(preds) * RUL_CAP
    targets = np.concatenate(targets) * RUL_CAP
    return float(np.sqrt(np.mean((preds - targets) ** 2)))


# ============================================================
# Pretraining
# ============================================================

D_MODEL = 128
N_PAIRS = 2
N_HEADS = 4
N_EPOCHS = 150
BATCH_SIZE = 4
N_CUTS = 20
LAMBDA_VAR = 0.01
PROBE_EVERY = 5
PATIENCE_PROBE = 10


def pretrain(data, ckpt_path, log_path, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = CrossSensorJEPA(d_model=D_MODEL, n_heads=N_HEADS, n_pairs=N_PAIRS,
                             dropout=0.1, ema_momentum=0.99).to(DEVICE)
    n = count_params(model)
    print(f"Model params: {n:,}")

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, N_EPOCHS)
    history = {'loss': [], 'probe_rmse': [], 'probe_epochs': []}
    best_probe = float('inf'); best_state = None; no_improve = 0

    run = None
    if HAS_WANDB:
        try:
            run = wandb.init(project='industrialjepa',
                             name=f'v14-phase3-crosssensor-s{seed}',
                             tags=['v14-phase3-cross-sensor'],
                             config={'d_model': D_MODEL, 'n_pairs': N_PAIRS,
                                     'n_params': n, 'seed': seed},
                             reinit=True)
        except Exception:
            pass

    t0 = time.time()
    with open(log_path, 'w') as f:
        f.write(f"V14 Phase 3 cross-sensor. Device={DEVICE}. Params={n:,}\n")

    for epoch in range(1, N_EPOCHS + 1):
        ds = CMAPSSPretrainDataset(data['train_engines'], n_cuts_per_engine=N_CUTS,
                                    min_past=10, min_horizon=5, max_horizon=30, seed=epoch)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_pretrain)
        model.train()
        total_loss = 0.0; nbatch = 0
        for past, past_mask, fut, fut_mask, k, _ in loader:
            past, past_mask = past.to(DEVICE), past_mask.to(DEVICE)
            fut, fut_mask = fut.to(DEVICE), fut_mask.to(DEVICE)
            k = k.to(DEVICE)
            optim.zero_grad()
            pred, h_fut, _ = model.forward_pretrain(past, past_mask, fut, fut_mask, k)
            loss, _, _ = trajectory_jepa_loss(pred, h_fut, LAMBDA_VAR)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            model.update_ema()
            total_loss += loss.item() * past.shape[0]; nbatch += past.shape[0]
        avg_loss = total_loss / nbatch
        history['loss'].append(avg_loss)
        sched.step()

        extra = ''
        if epoch % PROBE_EVERY == 0 or epoch == 1:
            probe_rmse = eval_probe_rmse(model, data['train_engines'],
                                          data['val_engines'], d_model=D_MODEL)
            history['probe_rmse'].append(probe_rmse)
            history['probe_epochs'].append(epoch)
            if probe_rmse < best_probe:
                best_probe = probe_rmse; best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, ckpt_path); no_improve = 0
            else:
                no_improve += 1
            extra = f" | probe={probe_rmse:.2f} (best={best_probe:.2f}, ni={no_improve})"

        if run is not None:
            try:
                log_dict = {'epoch': epoch, 'loss': avg_loss}
                if epoch % PROBE_EVERY == 0 or epoch == 1:
                    log_dict['probe_rmse'] = probe_rmse
                    log_dict['best_probe_rmse'] = best_probe
                wandb.log(log_dict)
            except Exception:
                pass

        line = f"Ep {epoch:3d} | loss={avg_loss:.4f}{extra}"
        print(line, flush=True)
        with open(log_path, 'a') as f:
            f.write(line + '\n')

        if no_improve >= PATIENCE_PROBE:
            print(f"Early stop at {epoch}")
            break

    if run is not None:
        try: wandb.finish()
        except Exception: pass

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Pretrain done in {(time.time()-t0)/60:.1f} min, best probe={best_probe:.2f}")
    return model, history, best_probe


# ============================================================
# Fine-tune (frozen/E2E at any budget)
# ============================================================

def run_finetune(ckpt_path, data, mode, seed, budget=1.0):
    model = CrossSensorJEPA(d_model=D_MODEL, n_heads=N_HEADS, n_pairs=N_PAIRS,
                             dropout=0.1).to(DEVICE)
    model.load_state_dict(torch.load(str(ckpt_path), map_location=DEVICE))

    train_eng = data['train_engines']
    if budget < 1.0:
        train_eng = subsample_engines(train_eng, budget, seed=seed)

    probe = RULProbe(D_MODEL).to(DEVICE)
    torch.manual_seed(seed); np.random.seed(seed)

    tr_ds = CMAPSSFinetuneDataset(train_eng, n_cuts_per_engine=5, seed=seed)
    va_ds = CMAPSSFinetuneDataset(data['val_engines'], use_last_only=True)
    te_ds = CMAPSSTestDataset(data['test_engines'], data['test_rul'])
    tr = DataLoader(tr_ds, batch_size=16, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(va_ds, batch_size=16, shuffle=False, collate_fn=collate_finetune)
    te = DataLoader(te_ds, batch_size=16, shuffle=False, collate_fn=collate_test)

    if mode == 'frozen':
        for p in model.parameters(): p.requires_grad = False
        optim = torch.optim.Adam(probe.parameters(), lr=1e-3)
    else:
        for p in model.context_encoder.parameters(): p.requires_grad = True
        optim = torch.optim.Adam(
            list(model.context_encoder.parameters()) + list(probe.parameters()), lr=1e-4)

    best_val = float('inf'); best_ps = None; best_es = None; no_impr = 0
    for ep in range(100):
        if mode == 'frozen': model.eval()
        else: model.train()
        probe.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            optim.zero_grad()
            if mode == 'frozen':
                with torch.no_grad(): h = model.encode_past(past, mask)
            else:
                h = model.encode_past(past, mask)
            loss = F.mse_loss(probe(h), rul)
            loss.backward(); optim.step()
        model.eval(); probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                h = model.encode_past(past, mask)
                pv.append(probe(h).cpu().numpy()); tv.append(rul.numpy())
        val_rmse = float(np.sqrt(np.mean(
            (np.concatenate(pv)*RUL_CAP - np.concatenate(tv)*RUL_CAP)**2)))
        if val_rmse < best_val:
            best_val = val_rmse; best_ps = copy.deepcopy(probe.state_dict())
            if mode == 'e2e':
                best_es = copy.deepcopy(model.context_encoder.state_dict())
            no_impr = 0
        else:
            no_impr += 1
            if no_impr >= 20: break

    probe.load_state_dict(best_ps)
    if mode == 'e2e' and best_es is not None:
        model.context_encoder.load_state_dict(best_es)
    model.eval(); probe.eval()
    pt, tt = [], []
    with torch.no_grad():
        for past, mask, rul_gt in te:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = model.encode_past(past, mask)
            pt.append(probe(h).cpu().numpy() * RUL_CAP); tt.append(rul_gt.numpy())
    return float(np.sqrt(np.mean((np.concatenate(pt) - np.concatenate(tt))**2))), best_val


# ============================================================
# Attention map extraction (averaged across cycles, by phase)
# ============================================================

@torch.no_grad()
def extract_attention_maps(model, engines, key_sensors=None):
    """For each engine, compute cross-sensor attention maps at every cycle.
    Average across: all cycles, healthy phase (first 60%), degradation phase (last 40%).
    """
    model.eval()
    SENSOR_LABELS = [f's{s}' for s in SELECTED_SENSORS]
    all_maps, healthy_maps, degrad_maps = [], [], []
    for eid, seq in engines.items():
        T = len(seq)
        if T < 60: continue
        past = torch.from_numpy(seq).unsqueeze(0).to(DEVICE)  # (1, T, S)
        _, attn = model.encode_past(past, return_attn_maps=True)  # (1, T, S, S)
        attn_np = attn.squeeze(0).cpu().numpy()  # (T, S, S)
        healthy_end = int(T * 0.6)
        all_maps.append(attn_np.mean(axis=0))
        healthy_maps.append(attn_np[:healthy_end].mean(axis=0))
        degrad_maps.append(attn_np[healthy_end:].mean(axis=0))
    return {
        'all_mean': np.mean(all_maps, axis=0),
        'healthy_mean': np.mean(healthy_maps, axis=0),
        'degradation_mean': np.mean(degrad_maps, axis=0),
        'sensor_labels': SENSOR_LABELS,
    }


def plot_attention_maps(maps, out_prefix):
    labels = maps['sensor_labels']
    for phase in ['all_mean', 'healthy_mean', 'degradation_mean']:
        fig, ax = plt.subplots(figsize=(6, 5))
        M = maps[phase]
        im = ax.imshow(M, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(f'Cross-sensor attention ({phase})', fontsize=10)
        ax.set_xlabel('key sensor'); ax.set_ylabel('query sensor')
        plt.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(f'{out_prefix}_{phase}.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
    # Diff: degradation - healthy
    diff = maps['degradation_mean'] - maps['healthy_mean']
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = np.abs(diff).max()
    im = ax.imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title('Attention shift: degradation - healthy', fontsize=10)
    ax.set_xlabel('key sensor'); ax.set_ylabel('query sensor')
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(f'{out_prefix}_diff.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    V14_DIR.mkdir(exist_ok=True)
    ckpt_path = V14_DIR / 'best_pretrain_cross_sensor.pt'
    log_path = V14_DIR / 'phase3_output.log'
    out_path = V14_DIR / 'cross_sensor_results.json'

    print(f"V14 Phase 3: cross-sensor attention encoder")
    print(f"Device: {DEVICE}")
    t0 = time.time()

    data = load_cmapss_subset('FD001')
    model, history, best_probe = pretrain(data, ckpt_path, log_path, seed=42)

    print("\n==== FINE-TUNING @ 100% ====")
    seeds_100 = [42, 123, 456]
    results = {'pretrain_best_probe': best_probe, 'pretrain_history': history,
               'seeds_100': seeds_100, 'frozen_100': [], 'e2e_100': []}
    for seed in seeds_100:
        for mode in ['frozen', 'e2e']:
            rmse, val = run_finetune(ckpt_path, data, mode, seed, budget=1.0)
            print(f"  seed={seed} {mode:6s} test={rmse:.3f} val={val:.3f}")
            results[f'{mode}_100'].append({'seed': seed, 'test_rmse': rmse, 'val_rmse': val})

    for mode in ['frozen_100', 'e2e_100']:
        vals = [r['test_rmse'] for r in results[mode]]
        results[f'{mode}_mean'] = float(np.mean(vals))
        results[f'{mode}_std'] = float(np.std(vals))

    # Low-label frozen 5 seeds @ 20/10/5
    print("\n==== FINE-TUNING @ LOW LABELS (frozen only) ====")
    seeds_low = [42, 123, 456, 789, 1024]
    for budget in [0.20, 0.10, 0.05]:
        key = f"frozen_{int(budget*100)}"
        results[key] = []
        for seed in seeds_low:
            rmse, val = run_finetune(ckpt_path, data, 'frozen', seed, budget=budget)
            print(f"  budget={int(budget*100)}% seed={seed} test={rmse:.3f}")
            results[key].append({'seed': seed, 'test_rmse': rmse, 'val_rmse': val})
        vals = [r['test_rmse'] for r in results[key]]
        results[f'{key}_mean'] = float(np.mean(vals))
        results[f'{key}_std'] = float(np.std(vals))

    # Attention maps
    print("\n==== EXTRACTING ATTENTION MAPS ====")
    model.load_state_dict(torch.load(str(ckpt_path), map_location=DEVICE))
    model.eval()
    maps = extract_attention_maps(model, {**data['train_engines'], **data['val_engines']})
    plot_attention_maps(maps, str(PLOT_PNG / 'cross_sensor_attention'))
    # Save raw maps
    with open(V14_DIR / 'cross_sensor_attention_maps.json', 'w') as f:
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in maps.items()}, f, indent=2)
    print(f"Saved attention maps")

    # Summary
    results['wall_time_s'] = time.time() - t0
    # V2 baselines
    results['v2_baselines'] = {
        'frozen_100': (17.81, 1.7), 'e2e_100': (14.23, 0.39),
        'frozen_20': (19.83, 0.30), 'frozen_10': (19.93, 0.9),
        'frozen_5': (21.53, 2.0),
    }
    print(f"\n=== SUMMARY ===")
    print(f"@100%: frozen={results['frozen_100_mean']:.2f} (V2: 17.81)  "
          f"e2e={results['e2e_100_mean']:.2f} (V2: 14.23)")
    print(f"@20%:  frozen={results['frozen_20_mean']:.2f} (V2: 19.83)")
    print(f"@10%:  frozen={results['frozen_10_mean']:.2f} (V2: 19.93)")
    print(f"@5%:   frozen={results['frozen_5_mean']:.2f} (V2: 21.53)")

    if results['frozen_100_mean'] > 18.5:
        verdict = 'KILL - frozen@100% > 18.5'
    elif results['frozen_100_mean'] < 17.0:
        verdict = 'POSITIVE - cross-sensor improves frozen@100%'
    else:
        verdict = 'NEUTRAL - within V2 noise'
    results['verdict'] = verdict
    print(f"Verdict: {verdict}")
    print(f"Wall time: {(time.time()-t0)/60:.1f} min")

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
