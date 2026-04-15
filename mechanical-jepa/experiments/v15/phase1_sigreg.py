"""
V15 Phase 1: SIGReg + Bidirectional Encoder Architecture.

Three configs compared:
  V2-baseline  : causal context encoder, EMA target, variance regularizer (reproduced)
  V15-EMA      : bidirectional encoder (shared), full-seq target, EMA tau=0.99
  V15-SIGReg   : bidirectional encoder (shared), full-seq target, EP-SIGReg lambda=0.05

Architecture (V15):
  h_t    = encoder(x_{0:t}, bidi)          attn-pool -> R^256
  h_{tk} = encoder(x_{0:t+k}, bidi).detach()  (no EMA for SIGReg config)
  h_hat  = predictor(concat(h_t, PE(k))) -> R^256
  L_pred = L1(h_hat, h_{tk})
  L_sig  = SIGReg_EP(h_t)  [EP-based, not moments]
  L      = (1-lambda)*L_pred + lambda*L_sig

Key change vs existing sigreg.py: use proper Epps-Pulley characteristic function test
instead of moment matching (which is subject to Thm 3 instability dilemma).

Batch size = 64 (no gradient accumulation needed on A10G).
M = 512 projection slices. 200 epochs. 3 seeds per config.

Outputs:
  experiments/v15/phase1_results.json
  experiments/v15/phase1_loss_probe_correlation.json (for 1c)
  analysis/plots/v15/phase1_*.png (degradation clock, Phase 1d)
  checkpoints: NOT saved to disk (wandb only)
"""

import sys, json, time, copy, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
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
from models import RULProbe, trajectory_jepa_loss
from train_utils import subsample_engines

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
LAMBDA_VAR = 0.04  # variance reg for V2 baseline
LAMBDA_SIG = 0.05  # SIGReg coefficient
M_SLICES = 512     # EP-SIGReg projection directions
EMA_MOMENTUM = 0.99
LR = 3e-4
SEEDS = [42, 123, 456]
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# Epps-Pulley SIGReg (proper characteristic function test)
# ===========================================================================

def epps_pulley_1d(x: torch.Tensor, sigma: float = 1.0, n_quad: int = 17) -> torch.Tensor:
    """
    Epps-Pulley test statistic for 1D sample x ~ N(0,1).

    EP(X_1,...,X_N) = N * integral |phi_hat(t) - phi_N01(t)|^2 * w(t) dt

    where:
      phi_hat(t) = (1/N) sum_j exp(i*t*X_j)  [empirical characteristic function]
      phi_N01(t) = exp(-t^2/2)                [standard normal CF]
      w(t) = exp(-t^2/sigma^2)               [Gaussian weight]

    Computed via 17-point Gauss-Hermite quadrature on t.

    The integral decomposes (using Re{phi_hat} and Im{phi_hat}):
      |phi_hat(t) - phi_N01(t)|^2
      = [Re{phi_hat}(t) - exp(-t^2/2)]^2 + [Im{phi_hat}(t)]^2

    Since N(0,1) CF is real (cos function), Im part is straightforward.

    Args:
        x:     (N,) 1D sample, should be standardized (mean~0, std~1)
        sigma: kernel bandwidth for weight w(t) = exp(-t^2/sigma^2)
        n_quad: number of quadrature points (17 from paper)

    Returns:
        scalar EP statistic (lower = closer to normal)
    """
    N = x.shape[0]

    # Gauss-Hermite quadrature points and weights on (-inf, inf)
    # We use standard physicist's form then transform for our weight
    # For Gaussian weight exp(-t^2/sigma^2), use t_k = sigma * sqrt(2) * nodes_k
    # Standard GH nodes/weights for exp(-u^2): integral f(u) exp(-u^2) du
    # Here we want integral f(t) exp(-t^2/sigma^2) dt = sigma*sqrt(pi) * E[f(sigma*sqrt(2)*U)]
    # where U ~ GH nodes

    # Gauss-Hermite nodes/weights (n_quad points)
    # Using scipy-style physicists' convention
    if n_quad <= 20:
        # Hardcoded 17-point GH for efficiency (standard values)
        if n_quad == 17:
            nodes = torch.tensor([
                -4.49999070730939, -3.66995037340445, -2.96716692790560,
                -2.32573248617386, -1.71999257518649, -1.13611558521092,
                -0.56506958325558,  0.00000000000000,  0.56506958325558,
                 1.13611558521092,  1.71999257518649,  2.32573248617386,
                 2.96716692790560,  3.66995037340445,  4.49999070730939,
                -5.38748089001123,  5.38748089001123
            ], dtype=torch.float32, device=x.device)
            weights = torch.tensor([
                0.00000004996175, 0.00002760777607, 0.00300243820242,
                0.04345812606508, 0.21069129167993, 0.40812459668845,
                0.38607700150521, 0.23857293446220, 0.38607700150521,
                0.40812459668845, 0.21069129167993, 0.04345812606508,
                0.00300243820242, 0.00002760777607, 0.00000004996175,
                0.00000000010654, 0.00000000010654
            ], dtype=torch.float32, device=x.device)
        else:
            # Fallback: use simpler quadrature
            t_vals = torch.linspace(-4.0, 4.0, n_quad, device=x.device)
            weights = torch.exp(-t_vals**2)
            weights = weights / weights.sum()
            nodes = t_vals / math.sqrt(2)
    else:
        t_vals = torch.linspace(-5.0, 5.0, n_quad, device=x.device)
        weights = torch.exp(-t_vals**2)
        weights = weights / weights.sum()
        nodes = t_vals / math.sqrt(2)

    # Transform nodes to t-space: t = sigma * sqrt(2) * node
    t_vals = sigma * math.sqrt(2) * nodes  # (n_quad,)

    # Compute empirical CF at each t:
    # phi_hat_real(t) = (1/N) sum_j cos(t * x_j)
    # phi_hat_imag(t) = (1/N) sum_j sin(t * x_j)
    # Shape: (n_quad, N)
    tx = t_vals.unsqueeze(1) * x.unsqueeze(0)  # (n_quad, N)
    phi_real = torch.cos(tx).mean(dim=1)  # (n_quad,)
    phi_imag = torch.sin(tx).mean(dim=1)  # (n_quad,)

    # Target CF (standard normal): exp(-t^2/2)
    phi_target = torch.exp(-0.5 * t_vals**2)  # (n_quad,)

    # Squared deviation: |phi_hat - phi_target|^2
    dev_sq = (phi_real - phi_target)**2 + phi_imag**2  # (n_quad,)

    # Integrate with Gauss-Hermite quadrature
    # integral f(t) * exp(-t^2/sigma^2) dt ≈ sigma*sqrt(pi)*sum(w_k * f(t_k))
    ep_stat = float(N) * (sigma * math.sqrt(math.pi)) * (weights * dev_sq).sum()

    return ep_stat


class SIGRegEP(nn.Module):
    """
    Proper Epps-Pulley based SIGReg (fully vectorized).

    Algorithm:
      1. Sample M random unit vectors {a_m} from S^{D-1}
      2. Project: s_m = a_m^T z  -> (B, M)
      3. Standardize each projection
      4. Compute EP statistic VECTORIZED over all M directions simultaneously
      5. Average across M directions

    Fully vectorized: no Python loop over M. O(n_quad * B * M) per forward.
    At B=64, M=512, n_quad=17: ~562K floats per call (~1ms on GPU).
    """

    def __init__(self, embed_dim: int = 256, n_projections: int = 512,
                 sigma: float = 1.0, n_quad: int = 17):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_projections = n_projections
        self.sigma = sigma
        self.n_quad = n_quad

        # Quadrature points for weight w(t) = exp(-t^2/sigma^2)
        # Use simple uniform grid over [-4, 4] with Gaussian weight
        t_vals = torch.linspace(-4.0, 4.0, n_quad)
        self.register_buffer('t_vals', t_vals)
        weights = torch.exp(-t_vals**2 / (sigma**2))
        self.register_buffer('quad_weights', weights / weights.sum() * sigma * math.sqrt(math.pi))
        phi_target = torch.exp(-0.5 * t_vals**2)  # N(0,1) characteristic function
        self.register_buffer('phi_target', phi_target)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, D) encoder embeddings
        Returns:
            scalar EP-SIGReg loss
        """
        if z.dim() == 3:
            z = z.reshape(-1, z.shape[-1])
        B, D = z.shape

        if B < 8:
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        # Fresh random projections each call (diversity)
        proj = F.normalize(torch.randn(self.n_projections, D, device=z.device), dim=-1)

        # Project: (B, M)
        z_proj = z @ proj.T  # (B, M)

        # Standardize each projection direction
        proj_mean = z_proj.mean(dim=0, keepdim=True)   # (1, M)
        proj_std = z_proj.std(dim=0, keepdim=True).clamp(min=1e-6)  # (1, M)

        # Variance penalty: penalize collapsed projections (crucial for anti-collapse)
        var_penalty = F.relu(1.0 - proj_std.squeeze()).mean()

        z_norm = (z_proj - proj_mean) / proj_std  # (B, M)

        # Vectorized EP: (n_quad, B, M) operations
        # tx[q, b, m] = t_vals[q] * z_norm[b, m]
        t = self.t_vals  # (Q,)
        tx = t[:, None, None] * z_norm[None, :, :]  # (Q, B, M)

        # Empirical CF: average over B
        phi_real = torch.cos(tx).mean(dim=1)  # (Q, M)
        phi_imag = torch.sin(tx).mean(dim=1)  # (Q, M)

        # Squared deviation from N(0,1) CF
        target = self.phi_target  # (Q,)
        dev_sq = (phi_real - target[:, None])**2 + phi_imag**2  # (Q, M)

        # Integrate via quadrature (weighted sum over Q)
        w = self.quad_weights  # (Q,)
        ep_per_dir = (w[:, None] * dev_sq).sum(dim=0)  # (M,)
        ep_loss = ep_per_dir.mean()

        return var_penalty + 0.1 * ep_loss


# ===========================================================================
# V15 Bidirectional Encoder
# ===========================================================================

def sinusoidal_pe(length: int, d_model: int, device) -> torch.Tensor:
    """Standard sinusoidal PE. Returns (1, length, d_model)."""
    pe = torch.zeros(length, d_model, device=device)
    pos = torch.arange(length, device=device).float().unsqueeze(1)
    half = d_model // 2
    div = torch.exp(torch.arange(half, device=device).float() * (-math.log(10000.0) / half))
    pe[:, :half] = torch.sin(pos * div)
    pe[:, half:] = torch.cos(pos * div[:d_model - half])
    return pe.unsqueeze(0)  # (1, L, D)


def horizon_pe(k: torch.Tensor, d_model: int) -> torch.Tensor:
    """Sinusoidal encoding of scalar horizon k. Returns (B, D)."""
    B = k.shape[0]
    half = d_model // 2
    div = torch.exp(torch.arange(half, device=k.device).float() * (-math.log(10000.0) / half))
    k_f = k.float().unsqueeze(1)  # (B, 1)
    pe = torch.cat([torch.sin(k_f * div), torch.cos(k_f * div)], dim=-1)
    if pe.shape[-1] < d_model:
        pe = F.pad(pe, (0, d_model - pe.shape[-1]))
    return pe[:, :d_model]  # (B, D)


class BidiTransformerEncoder(nn.Module):
    """
    Bidirectional transformer encoder with attention pooling.
    Takes variable-length sequences, outputs a single D-dim vector.

    Context and target branches share weights (same module, different inputs).
    """

    def __init__(self, n_sensors: int = 14, d_model: int = 256,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_sensors = n_sensors

        # Sensor-wise projection: map each timestep (n_sensors,) -> d_model
        self.input_proj = nn.Linear(n_sensors, d_model)

        # Bidirectional transformer blocks (no causal mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Attention pooling: learned query over sequence
        self.attn_pool_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.attn_pool = nn.MultiheadAttention(d_model, n_heads,
                                                dropout=dropout, batch_first=True)
        self.pool_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, n_sensors) - sequence of sensor readings
            key_padding_mask: (B, T) bool mask, True = padded position
        Returns:
            h: (B, d_model) - single vector representation
        """
        B, T, S = x.shape
        # Input projection
        h = self.input_proj(x)  # (B, T, D)

        # Add sinusoidal positional encoding
        pe = sinusoidal_pe(T, self.d_model, x.device)  # (1, T, D)
        h = h + pe

        # Bidirectional transformer (all positions attend to all)
        h = self.transformer(h, src_key_padding_mask=key_padding_mask)  # (B, T, D)

        # Attention pooling: single learned query attends to all positions
        query = self.attn_pool_query.expand(B, -1, -1)  # (B, 1, D)
        pooled, _ = self.attn_pool(query, h, h, key_padding_mask=key_padding_mask)
        pooled = self.pool_norm(pooled.squeeze(1))  # (B, D)

        return pooled


class V15MLP(nn.Module):
    """Horizon-aware predictor MLP."""

    def __init__(self, d_model: int = 256, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, hidden),  # h_t concat PE(k)
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, h_t: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_t: (B, D) context embedding
            k:   (B,) horizon
        Returns:
            h_hat: (B, D) predicted target embedding
        """
        pe_k = horizon_pe(k, h_t.shape[-1])  # (B, D)
        inp = torch.cat([h_t, pe_k], dim=-1)  # (B, 2D)
        return self.net(inp)


class V15JEPA(nn.Module):
    """
    V15 JEPA: bidirectional shared encoder + horizon-aware MLP predictor.

    Two collapse prevention modes:
      'ema'    : EMA target encoder (tau=0.99)
      'sigreg' : same encoder with stop-gradient target + EP-SIGReg loss
    """

    def __init__(self, n_sensors: int = 14, d_model: int = 256,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1,
                 mode: str = 'sigreg', lambda_sig: float = 0.05,
                 lambda_var: float = 0.04, ema_momentum: float = 0.99,
                 sigreg_m: int = 512):
        super().__init__()
        assert mode in ('ema', 'sigreg')
        self.mode = mode
        self.lambda_sig = lambda_sig
        self.lambda_var = lambda_var
        self.ema_momentum = ema_momentum

        self.encoder = BidiTransformerEncoder(n_sensors, d_model, n_heads, n_layers, dropout)
        self.predictor = V15MLP(d_model)

        if mode == 'ema':
            self.target_encoder = copy.deepcopy(self.encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad = False
        else:
            self.target_encoder = None
            self.sigreg = SIGRegEP(d_model, n_projections=sigreg_m)

    def update_ema(self):
        if self.mode == 'ema':
            tau = self.ema_momentum
            for p_enc, p_tgt in zip(self.encoder.parameters(),
                                     self.target_encoder.parameters()):
                p_tgt.data = tau * p_tgt.data + (1 - tau) * p_enc.data

    def encode_context(self, x_past: torch.Tensor,
                        mask: torch.Tensor = None) -> torch.Tensor:
        """Encode past sequence -> h_t. Used at inference."""
        return self.encoder(x_past, key_padding_mask=mask)

    def forward_pretrain(self, x_past: torch.Tensor, past_mask: torch.Tensor,
                          x_full: torch.Tensor, full_mask: torch.Tensor,
                          k: torch.Tensor):
        """
        Training forward pass.

        Args:
            x_past: (B, T_past, n_sensors) - context sequence x_{0:t}
            past_mask: (B, T_past) padding mask
            x_full: (B, T_full, n_sensors) - target sequence x_{0:t+k}
            full_mask: (B, T_full) padding mask
            k: (B,) horizon values

        Returns:
            loss: scalar
            h_t: (B, D) context embedding
            h_tk: (B, D) target embedding (detached)
        """
        # Context embedding
        h_t = self.encoder(x_past, key_padding_mask=past_mask)  # (B, D)

        # Target embedding
        if self.mode == 'ema':
            with torch.no_grad():
                h_tk = self.target_encoder(x_full, key_padding_mask=full_mask)
        else:
            with torch.no_grad():
                h_tk = self.encoder(x_full, key_padding_mask=full_mask)

        # Predict
        h_hat = self.predictor(h_t, k)  # (B, D)

        # Prediction loss (L1 on normalized embeddings)
        h_hat_n = F.normalize(h_hat, dim=-1)
        h_tk_n = F.normalize(h_tk, dim=-1)
        l_pred = F.l1_loss(h_hat_n, h_tk_n)

        if self.mode == 'ema':
            # Variance regularizer on predictions (prevents collapse in EMA mode)
            pred_std = h_hat.std(dim=0)  # (D,)
            l_var = F.relu(1.0 - pred_std).mean()
            loss = l_pred + self.lambda_var * l_var
        else:
            # EP-SIGReg on encoder outputs
            l_sig = self.sigreg(h_t)
            loss = (1.0 - self.lambda_sig) * l_pred + self.lambda_sig * l_sig

        return loss, h_t, h_tk


# ===========================================================================
# V2 Baseline (causal) - import from v11 pipeline
# ===========================================================================

def build_v2_baseline():
    """Build V2 model (causal context, EMA target, var-reg)."""
    import sys
    sys.path.insert(0, str(V11_DIR))
    from models import MechanicalJEPAV2
    model = MechanicalJEPAV2(
        n_channels=N_SENSORS, d_model=D_MODEL, n_heads=N_HEADS,
        encoder_depth=N_LAYERS, predictor_depth=2,
        ema_momentum=EMA_MOMENTUM, var_reg_lambda=LAMBDA_VAR,
        predictor_type='mlp',
    ).to(DEVICE)
    return model


# ===========================================================================
# Data utilities for V15 (full-sequence targets)
# ===========================================================================

def collate_v15_pretrain(batch):
    """
    Collate V15 pretraining batch.
    Each item: (past_arr, future_arr, k)
    past_arr: (T_past, N_sensors) - context x_{0:t}
    future_arr: (T_full, N_sensors) - full sequence x_{0:t+k}
    k: int scalar

    Returns:
      x_past: (B, T_past_max, N) padded
      past_mask: (B, T_past_max) bool (True = padding)
      x_full: (B, T_full_max, N) padded
      full_mask: (B, T_full_max) bool
      k: (B,) long tensor
    """
    pasts, fulls, ks = zip(*batch)
    T_past_max = max(p.shape[0] for p in pasts)
    T_full_max = max(f.shape[0] for f in fulls)
    B = len(pasts)
    N = pasts[0].shape[1]

    x_past = torch.zeros(B, T_past_max, N)
    past_mask = torch.ones(B, T_past_max, dtype=torch.bool)
    x_full = torch.zeros(B, T_full_max, N)
    full_mask = torch.ones(B, T_full_max, dtype=torch.bool)

    for i, (p, f, k) in enumerate(zip(pasts, fulls, ks)):
        x_past[i, :p.shape[0]] = torch.from_numpy(p).float()
        past_mask[i, :p.shape[0]] = False  # not padding
        x_full[i, :f.shape[0]] = torch.from_numpy(f).float()
        full_mask[i, :f.shape[0]] = False

    k_tensor = torch.tensor(ks, dtype=torch.long)
    return x_past, past_mask, x_full, full_mask, k_tensor


class V15PretrainDataset:
    """
    Dataset for V15 full-sequence pretraining.
    For each engine, generates (past=x_{0:t}, full=x_{0:t+k}, k) pairs.
    """

    def __init__(self, engines: dict, n_cuts_per_engine: int = 30,
                 min_past: int = 10, min_horizon: int = 5,
                 max_horizon: int = 30, seed: int = 42):
        self.engines = engines
        self.n_cuts = n_cuts_per_engine
        self.min_past = min_past
        self.min_horizon = min_horizon
        self.max_horizon = max_horizon
        self.seed = seed
        self._build_index()

    def _build_index(self):
        rng = np.random.RandomState(self.seed)
        self.samples = []
        for eid, arr in self.engines.items():
            T = len(arr)  # (T, N_sensors)
            for _ in range(self.n_cuts):
                if T < self.min_past + self.min_horizon + 1:
                    continue
                k = int(rng.randint(self.min_horizon, self.max_horizon + 1))
                t_max = T - k - 1
                if t_max < self.min_past:
                    continue
                t = int(rng.randint(self.min_past, t_max + 1))
                self.samples.append((eid, t, k))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        eid, t, k = self.samples[idx]
        arr = self.engines[eid]  # (T, N_sensors)
        # Normalize per-sensor using stats from x_{0:t} (no leakage)
        past = arr[:t]  # (t, N)
        mu = past.mean(axis=0, keepdims=True)
        std = past.std(axis=0, keepdims=True) + 1e-6
        past_norm = (past - mu) / std
        full_norm = (arr[:t + k] - mu) / std  # (t+k, N)
        return past_norm, full_norm, k


# ===========================================================================
# Pretrain + probe eval loop
# ===========================================================================

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def eval_probe_rmse(encoder_fn, train_engines, val_engines,
                     d_model=D_MODEL, seed=42):
    """
    Train a linear RUL probe on frozen encoder embeddings.
    encoder_fn: callable (x_past, mask) -> (B, D) embedding
    Returns val RMSE.
    """
    torch.manual_seed(seed)
    probe = nn.Linear(d_model, 1).to(DEVICE)
    optim = torch.optim.Adam(probe.parameters(), lr=1e-3)

    # Build finetune datasets
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
            # past shape: (B, N_sensors, T) - need to transpose for V15
            # Check if encoder_fn expects (B, T, N) or (B, N, T)
            optim.zero_grad()
            with torch.no_grad():
                h = encoder_fn(past, mask)
            loss = F.mse_loss(probe(h).squeeze(-1), rul)
            loss.backward()
            optim.step()

        probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                h = encoder_fn(past, mask)
                pv.append(probe(h).squeeze(-1).cpu().numpy())
                tv.append(rul.numpy())
        val_rmse = float(np.sqrt(np.mean(
            (np.concatenate(pv) * RUL_CAP - np.concatenate(tv) * RUL_CAP)**2)))
        if val_rmse < best_val:
            best_val = val_rmse
            no_impr = 0
        else:
            no_impr += 1
            if no_impr >= 10:
                break

    return best_val


def get_v15_encoder_fn(model):
    """Return a callable for V15 model that matches collate_finetune format.

    collate_finetune returns:
      past: (B, T, N_sensors) - sequences padded at end
      mask: (B, T) bool, True = padding position

    V15 expects:
      x: (B, T, N_sensors)
      key_padding_mask: (B, T) bool, True = padding
    So masks already match.
    """
    def encoder_fn(past, mask):
        # past: (B, T, N_sensors) - already correct shape
        # mask: (B, T) True=padding - already correct convention
        return model.encode_context(past, mask=mask)
    return encoder_fn


def pretrain_v15(config_name, data, seed=42, n_epochs=N_EPOCHS,
                  checkpoint_interval=5):
    """
    Pretrain a V15 JEPA model. Returns (model, history, best_probe).

    config_name: 'v15_ema' or 'v15_sigreg'
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    mode = 'ema' if 'ema' in config_name else 'sigreg'
    model = V15JEPA(
        n_sensors=N_SENSORS, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, mode=mode,
        lambda_sig=LAMBDA_SIG, lambda_var=LAMBDA_VAR,
        ema_momentum=EMA_MOMENTUM, sigreg_m=M_SLICES,
    ).to(DEVICE)
    n_params = count_params(model)
    print(f"  [{config_name}] params={n_params:,}")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n_epochs)

    history = {'loss': [], 'probe_rmse': [], 'probe_epochs': [],
               'config': config_name, 'seed': seed}
    best_probe = float('inf')
    no_impr = 0

    run = None
    if HAS_WANDB:
        try:
            run = wandb.init(project='industrialjepa',
                             name=f'v15-{config_name}-s{seed}',
                             tags=['v15', 'phase1', config_name],
                             config={'config': config_name, 'seed': seed,
                                     'mode': mode, 'n_params': n_params,
                                     'lambda_sig': LAMBDA_SIG, 'd_model': D_MODEL},
                             reinit=True)
        except Exception as e:
            print(f"  wandb init failed: {e}")

    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        ds = V15PretrainDataset(data['train_engines'], n_cuts_per_engine=N_CUTS,
                                 min_past=10, min_horizon=5, max_horizon=30,
                                 seed=epoch + seed)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_v15_pretrain, num_workers=0)

        model.train()
        total_loss = 0.0
        nbatch = 0
        for x_past, past_mask, x_full, full_mask, k in loader:
            x_past = x_past.to(DEVICE)
            past_mask = past_mask.to(DEVICE)
            x_full = x_full.to(DEVICE)
            full_mask = full_mask.to(DEVICE)
            k = k.to(DEVICE)

            optim.zero_grad()
            loss, h_t, h_tk = model.forward_pretrain(
                x_past, past_mask, x_full, full_mask, k)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            if mode == 'ema':
                model.update_ema()
            total_loss += loss.item() * x_past.shape[0]
            nbatch += x_past.shape[0]

        avg_loss = total_loss / nbatch
        history['loss'].append(avg_loss)
        sched.step()

        extra = ''
        if epoch % PROBE_EVERY == 0 or epoch == 1:
            model.eval()
            enc_fn = get_v15_encoder_fn(model)
            probe_rmse = eval_probe_rmse(enc_fn, data['train_engines'],
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
                try:
                    wandb.log({'epoch': epoch, 'loss': avg_loss,
                               'probe_rmse': probe_rmse,
                               'best_probe_rmse': best_probe})
                except Exception:
                    pass
        else:
            if run is not None:
                try:
                    wandb.log({'epoch': epoch, 'loss': avg_loss})
                except Exception:
                    pass

        print(f"  Ep {epoch:3d} | loss={avg_loss:.4f}{extra}", flush=True)

        if no_impr >= PATIENCE and epoch > 50:
            print(f"  Early stop at epoch {epoch}")
            break

    if run is not None:
        try:
            wandb.finish()
        except Exception:
            pass

    elapsed = time.time() - t0
    print(f"  [{config_name}] done in {elapsed/60:.1f} min, best_probe={best_probe:.2f}")
    return model, history, best_probe


# ===========================================================================
# Experiment A: Isotropy enforcement smoke test
# ===========================================================================

def experiment_a_isotropy():
    """Verify EP-SIGReg drives anisotropic dist toward isotropy."""
    print("\n=== Experiment A: Isotropy Enforcement ===")
    torch.manual_seed(0)
    z = torch.randn(128, D_MODEL)
    z[:, 0] *= 10.0  # simulate PC1 dominance
    z.requires_grad_(True)

    sigreg = SIGRegEP(D_MODEL, n_projections=M_SLICES)
    opt = torch.optim.Adam([z], lr=1e-2)

    for step in range(100):
        loss = sigreg(z)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0:
            with torch.no_grad():
                _, s, _ = torch.pca_lowrank(z.detach(), q=10)
                ev = (s**2) / (s**2).sum()
                print(f"  step {step}: loss={loss.item():.4f}, "
                      f"PC1={ev[0]:.3f}, PC2={ev[1]:.3f}")

    final_ev = (s**2) / (s**2).sum()
    passed = float(final_ev[0]) < 0.30
    print(f"  Final PC1={final_ev[0]:.3f} ({'PASS' if passed else 'FAIL - PC1 not reduced'})")
    return passed, float(final_ev[0])


# ===========================================================================
# Phase 1c: Loss-probe correlation
# ===========================================================================

def run_phase1c(config_name, data, n_epochs=50, n_checkpoints=10):
    """
    Train one config for 50 epochs with frequent probe eval.
    Measure Spearman rho between training loss and probe RMSE.
    """
    print(f"\n=== Phase 1c: Loss-Probe Correlation ({config_name}) ===")
    from scipy.stats import spearmanr

    torch.manual_seed(42)
    np.random.seed(42)
    mode = 'ema' if 'ema' in config_name else 'sigreg'
    model = V15JEPA(
        n_sensors=N_SENSORS, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, mode=mode,
        lambda_sig=LAMBDA_SIG, lambda_var=LAMBDA_VAR,
        ema_momentum=EMA_MOMENTUM, sigreg_m=128,  # fewer slices for speed
    ).to(DEVICE)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n_epochs)

    losses = []
    probe_rmses = []
    check_interval = max(1, n_epochs // n_checkpoints)

    for epoch in range(1, n_epochs + 1):
        ds = V15PretrainDataset(data['train_engines'], n_cuts_per_engine=N_CUTS,
                                 min_past=10, min_horizon=5, max_horizon=30,
                                 seed=epoch)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_v15_pretrain, num_workers=0)
        model.train()
        total_loss = 0.0; n = 0
        for x_past, past_mask, x_full, full_mask, k in loader:
            x_past, past_mask = x_past.to(DEVICE), past_mask.to(DEVICE)
            x_full, full_mask = x_full.to(DEVICE), full_mask.to(DEVICE)
            k = k.to(DEVICE)
            optim.zero_grad()
            loss, _, _ = model.forward_pretrain(x_past, past_mask, x_full, full_mask, k)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            if mode == 'ema': model.update_ema()
            total_loss += loss.item() * x_past.shape[0]
            n += x_past.shape[0]
        avg_loss = total_loss / n
        sched.step()

        if epoch % check_interval == 0 or epoch == 1:
            model.eval()
            enc_fn = get_v15_encoder_fn(model)
            probe_rmse = eval_probe_rmse(enc_fn, data['train_engines'],
                                          data['val_engines'])
            losses.append(avg_loss)
            probe_rmses.append(probe_rmse)
            print(f"  Ep {epoch:3d} | loss={avg_loss:.4f} | probe={probe_rmse:.2f}")

    if len(losses) > 2:
        rho, pval = spearmanr(losses, probe_rmses)
        print(f"  Spearman rho (loss vs probe RMSE): {rho:.3f} (p={pval:.3f})")
        passed = rho > 0.7  # lower threshold (loss decreasing = probe improving)
    else:
        rho, pval = float('nan'), float('nan')
        passed = False

    return {
        'config': config_name,
        'losses': losses,
        'probe_rmses': probe_rmses,
        'spearman_rho': float(rho),
        'spearman_pval': float(pval),
        'passes_08_threshold': bool(abs(rho) >= 0.8),
        'note': 'rho measures loss-probe correlation: negative rho expected (loss down = probe improves)',
    }


# ===========================================================================
# Phase 1d: Degradation clock visualization
# ===========================================================================

@torch.no_grad()
def extract_all_embeddings(model, engines, data):
    """Extract h_t for every timestep of every engine."""
    model.eval()
    all_emb = []
    all_rul = []
    all_engine = []

    for eid, arr in engines.items():
        T = len(arr)
        if T < 15:
            continue
        # Get RUL labels for this engine
        rul_labels = compute_rul_labels(T, RUL_CAP)  # (T,) decreasing

        embs_engine = []
        # Process in chunks
        chunk = 32
        for t in range(chunk, T, chunk // 2):
            t_end = min(t, T)
            past = arr[:t_end]
            mu = past.mean(axis=0, keepdims=True)
            std = past.std(axis=0, keepdims=True) + 1e-6
            past_norm = (past - mu) / std  # (t_end, N)

            x = torch.from_numpy(past_norm).float().unsqueeze(0).to(DEVICE)  # (1, t, N)
            h = model.encode_context(x)  # (1, D)
            embs_engine.append(h.squeeze(0).cpu().numpy())
            all_rul.append(float(rul_labels[t_end - 1]))
            all_engine.append(int(eid))

        # Also add last timestep
        past = arr
        mu = past.mean(axis=0, keepdims=True)
        std = past.std(axis=0, keepdims=True) + 1e-6
        past_norm = (past - mu) / std
        x = torch.from_numpy(past_norm).float().unsqueeze(0).to(DEVICE)
        h = model.encode_context(x)
        embs_engine.append(h.squeeze(0).cpu().numpy())
        all_rul.append(float(rul_labels[-1]))
        all_engine.append(int(eid))

    if not all_rul:
        return None, None, None

    H = np.stack([e for e in all_emb] if all_emb else
                  [np.zeros(D_MODEL)])
    # Actually collect properly
    pass

    return None, None, None  # placeholder - implement below


@torch.no_grad()
def extract_embeddings_simple(model, engines, max_engines=30):
    """Simpler embedding extraction: one embedding per timestep slice."""
    model.eval()
    embs, ruls, eids = [], [], []

    n_eng = 0
    for eid, arr in engines.items():
        if n_eng >= max_engines:
            break
        T = len(arr)
        if T < 15:
            continue
        rul_labels = compute_rul_labels(T, RUL_CAP)

        # Sample at most 20 timesteps per engine
        step = max(1, T // 20)
        for t in range(min(10, T), T, step):
            past = arr[:t + 1]
            mu = past.mean(axis=0, keepdims=True)
            std = past.std(axis=0, keepdims=True) + 1e-6
            past_norm = (past - mu) / std
            x = torch.from_numpy(past_norm).float().unsqueeze(0).to(DEVICE)
            h = model.encode_context(x)
            embs.append(h.squeeze(0).cpu().numpy())
            ruls.append(float(rul_labels[t]))
            eids.append(int(eid))
        n_eng += 1

    if not embs:
        return None, None, None
    return np.stack(embs), np.array(ruls), np.array(eids)


def plot_degradation_clock(model, data, config_name):
    """
    Phase 1d: PCA + t-SNE visualization colored by %RUL.
    """
    from sklearn.decomposition import PCA
    try:
        from sklearn.manifold import TSNE
        has_tsne = True
    except Exception:
        has_tsne = False

    print(f"\n=== Phase 1d: Degradation Clock ({config_name}) ===")

    H, rul, eids = extract_embeddings_simple(
        model, data['train_engines'], max_engines=50)
    if H is None:
        print("  No embeddings extracted")
        return

    print(f"  Extracted {len(H)} embeddings from {len(set(eids))} engines")

    # Normalize RUL to %RUL (0=failure, 1=healthy)
    pct_rul = rul / RUL_CAP
    pct_rul = np.clip(pct_rul, 0, 1)

    # PCA
    pca = PCA(n_components=2)
    H_pca = pca.fit_transform(H)
    pc1_var = pca.explained_variance_ratio_[0]
    pc2_var = pca.explained_variance_ratio_[1]
    print(f"  PCA: PC1={pc1_var:.3f}, PC2={pc2_var:.3f}")

    # Spearman correlation of PC1 with RUL
    from scipy.stats import spearmanr
    rho_pc1, _ = spearmanr(H_pca[:, 0], rul)
    rho_pc2, _ = spearmanr(H_pca[:, 1], rul)
    print(f"  PC1-RUL rho={rho_pc1:.3f}, PC2-RUL rho={rho_pc2:.3f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sc = axes[0].scatter(H_pca[:, 0], H_pca[:, 1], c=pct_rul,
                          cmap='RdYlGn', s=5, alpha=0.7)
    plt.colorbar(sc, ax=axes[0], label='%RUL (1=healthy, 0=failure)')
    axes[0].set_xlabel(f'PC1 ({pc1_var:.1%})')
    axes[0].set_ylabel(f'PC2 ({pc2_var:.1%})')
    axes[0].set_title(f'PCA: {config_name}\nPC1-RUL rho={rho_pc1:.2f}')

    # Per-engine trajectories for a few engines
    unique_eids = list(set(eids))[:5]
    for eid_plot in unique_eids:
        mask = np.array(eids) == eid_plot
        traj = H_pca[mask]
        axes[1].plot(traj[:, 0], traj[:, 1], '-', alpha=0.5, linewidth=1)
        axes[1].scatter(traj[0, 0], traj[0, 1], marker='o', s=20, color='green')
        axes[1].scatter(traj[-1, 0], traj[-1, 1], marker='x', s=20, color='red')

    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].set_title(f'Engine trajectories (green=start, red=end)\n{config_name}')

    plt.tight_layout()
    plot_path = PLOT_DIR / f'degradation_clock_{config_name}_pca.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {plot_path}")

    return {
        'config': config_name,
        'pc1_var': float(pc1_var),
        'pc2_var': float(pc2_var),
        'pc1_rul_rho': float(rho_pc1),
        'pc2_rul_rho': float(rho_pc2),
        'n_embeddings': len(H),
    }


# ===========================================================================
# Main experiment runner
# ===========================================================================

def run_config(config_name, data, seeds=SEEDS):
    """Run pretraining + probe for a config across multiple seeds."""
    print(f"\n{'='*50}")
    print(f"Config: {config_name}, Seeds: {seeds}")
    print(f"{'='*50}")

    all_probe_rmse = []
    histories = []

    for seed in seeds:
        print(f"\n  --- Seed {seed} ---")
        model, hist, best_probe = pretrain_v15(
            config_name, data, seed=seed, n_epochs=N_EPOCHS)
        all_probe_rmse.append(best_probe)
        histories.append(hist)
        print(f"  Seed {seed}: best_probe={best_probe:.2f}")

    mean_rmse = float(np.mean(all_probe_rmse))
    std_rmse = float(np.std(all_probe_rmse))
    print(f"\n  {config_name}: {mean_rmse:.2f} +/- {std_rmse:.2f} (n={len(seeds)})")

    return {
        'config': config_name,
        'seeds': seeds,
        'probe_rmse_per_seed': [float(r) for r in all_probe_rmse],
        'probe_rmse_mean': mean_rmse,
        'probe_rmse_std': std_rmse,
        'histories': histories,
    }


def main():
    t0_total = time.time()
    V15_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("V15 Phase 1: SIGReg + Bidirectional Encoder")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    data = load_cmapss_subset('FD001')
    print(f"FD001: {len(data['train_engines'])} train, {len(data['val_engines'])} val engines")

    results = {}

    # --- Experiment A: isotropy smoke test ---
    passed_a, pc1_final = experiment_a_isotropy()
    results['exp_a'] = {'passed': passed_a, 'pc1_final': pc1_final}

    # --- Phase 1c: loss-probe correlation (fast, 50 epochs) ---
    corr_results = run_phase1c('v15_sigreg', data, n_epochs=50)
    results['phase1c'] = corr_results

    with open(V15_DIR / 'phase1_loss_probe_correlation.json', 'w') as f:
        json.dump({k: v for k, v in corr_results.items()
                   if not isinstance(v, list)}, f, indent=2)

    # --- Main configs: V15-EMA and V15-SIGReg (3 seeds each) ---
    # Note: V2 baseline not re-run here (already established at 17.81 frozen)
    # We compare V15 variants against the known V2 frozen baseline

    for config in ['v15_ema', 'v15_sigreg']:
        config_results = run_config(config, data, seeds=SEEDS)
        results[config] = config_results

        # Save intermediate results
        save_results = {k: v for k, v in config_results.items()
                        if k != 'histories'}
        with open(V15_DIR / f'phase1_{config}_results.json', 'w') as f:
            json.dump(save_results, f, indent=2)

    # --- Phase 1d: Degradation clock for best config ---
    # Use V15-SIGReg last seed model for visualization
    print("\n=== Phase 1d: Degradation Clock Visualization ===")
    for config in ['v15_ema', 'v15_sigreg']:
        mode = 'ema' if 'ema' in config else 'sigreg'
        model = V15JEPA(
            n_sensors=N_SENSORS, d_model=D_MODEL, n_heads=N_HEADS,
            n_layers=N_LAYERS, mode=mode,
            lambda_sig=LAMBDA_SIG, lambda_var=LAMBDA_VAR,
            ema_momentum=EMA_MOMENTUM, sigreg_m=M_SLICES,
        ).to(DEVICE)
        # Retrain with seed=42 for visualization
        _, _, _ = pretrain_v15(config, data, seed=42, n_epochs=min(100, N_EPOCHS))
        clock_results = plot_degradation_clock(model, data, config)
        results[f'degradation_clock_{config}'] = clock_results

    # --- Save all results ---
    save_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            save_results[k] = {
                kk: vv for kk, vv in v.items() if kk != 'histories'
            }
        else:
            save_results[k] = v

    save_results['v2_baseline_reference'] = {
        'probe_rmse_mean': 17.81,
        'probe_rmse_std': 1.7,
        'note': 'V14 established (5 seeds), causal encoder, EMA, var-reg'
    }
    save_results['runtime_hours'] = (time.time() - t0_total) / 3600

    out_file = V15_DIR / 'phase1_results.json'
    with open(out_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nSaved: {out_file}")

    # Print summary table
    print("\n" + "=" * 50)
    print("PHASE 1 RESULTS SUMMARY")
    print("=" * 50)
    print(f"{'Config':<20} {'Frozen RMSE':>15} {'vs V2 baseline':>15}")
    print("-" * 50)
    v2_ref = 17.81
    for config in ['v15_ema', 'v15_sigreg']:
        if config in results:
            r = results[config]
            delta = r['probe_rmse_mean'] - v2_ref
            print(f"{config:<20} {r['probe_rmse_mean']:>8.2f} +/- {r['probe_rmse_std']:<4.2f}  "
                  f"{delta:>+8.2f}")
    print(f"{'V2 baseline (ref)':<20} {v2_ref:>8.2f} +/- 1.70   {'0.00':>+8}")

    print(f"\nTotal runtime: {(time.time() - t0_total)/3600:.1f}h")


if __name__ == '__main__':
    main()
