#!/usr/bin/env python3
"""
KH-JEPA: Koopman-Hierarchical Joint Embedding Predictive Architecture
======================================================================

A world-class time series foundation model architecture combining:
1. JEPA: Predict in latent space, not observation space
2. Koopman: Linear dynamics enable O(1) any-horizon prediction
3. Hierarchy: Multi-resolution captures different time scales
4. VICReg: Prevents representation collapse
5. Cross-Variate: Captures sensor dependencies

Target: Beat TTT (0.358 MSE) on ETTh1 horizon-96

Architecture toggles allow systematic ablation studies.

References:
- I-JEPA (Meta, 2023): Joint embedding predictive architecture
- C-JEPA (NeurIPS 2024): VICReg for JEPA stability
- Koopman (Nature Comms 2018): Linear dynamics in lifted space
- iTransformer (ICLR 2024): Cross-variate attention
"""

import sys
import time
import json
import math
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

from prepare import get_dataloaders

# ============================================================================
# HYPERPARAMETERS - Agent can modify these
# ============================================================================

# Training
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_EPOCHS = 2
GRADIENT_CLIP = 1.0

# Sequence lengths (CRITICAL: SOTA uses 512)
SEQ_LEN = 96            # Input context length (try 512 for SOTA)
PRED_LEN = 96           # Prediction horizon (96, 192, 336, 720)

# Model architecture
D_MODEL = 256           # Model dimension
N_HEADS = 8             # Attention heads
E_LAYERS = 3            # Encoder layers
D_FF = 512              # Feedforward dimension
DROPOUT = 0.1

# Patch configuration
USE_PATCHES = True
PATCH_LEN = 16
STRIDE = 8              # Non-overlapping: stride = patch_len

# =========================
# KH-JEPA ARCHITECTURE FLAGS
# =========================

# Core JEPA
USE_JEPA = True
LATENT_DIM = 128
EMA_MOMENTUM = 0.996    # Higher = more stable target (0.996 standard)
EMA_WARMUP = True       # Gradually increase momentum

# Innovation #1: Koopman Predictor
USE_KOOPMAN = False     # Replace MLP predictor with Koopman operator
KOOPMAN_RANK = 64       # Rank of Koopman approximation (< LATENT_DIM for efficiency)
KOOPMAN_STABLE = True   # Constrain eigenvalues to unit disk

# Innovation #2: VICReg Regularization
USE_VICREG = False      # Variance-Invariance-Covariance regularization
VICREG_VAR_WEIGHT = 0.04
VICREG_COV_WEIGHT = 0.04

# Innovation #3: Cross-Variate Attention
USE_CROSS_VARIATE = False   # Attention across channels (iTransformer style)
CROSS_VARIATE_LAYERS = 2    # Number of cross-variate attention layers

# Innovation #4: Hierarchy
USE_HIERARCHY = False       # Multi-resolution latent spaces
HIERARCHY_LEVELS = 3        # Number of resolution levels
HIERARCHY_FACTORS = [1, 4, 16]  # Downsampling factors per level

# Loss configuration
USE_HUBER_LOSS = False      # Huber loss (robust to outliers)
HUBER_DELTA = 1.0
MULTI_HORIZON_LOSS = False  # Loss at multiple prediction steps


# ============================================================================
# CORE COMPONENTS
# ============================================================================

class PositionalEncoding(nn.Module):
    """Learnable positional encoding with optional sinusoidal initialization."""

    def __init__(self, d_model: int, max_len: int = 5000, learnable: bool = False):
        super().__init__()
        self.learnable = learnable

        if learnable:
            self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        else:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class PatchEmbedding(nn.Module):
    """Patch-based embedding preserving channel structure for cross-variate attention."""

    def __init__(self, num_features: int, patch_len: int, stride: int, d_model: int,
                 channel_independent: bool = False):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_features = num_features
        self.channel_independent = channel_independent

        if channel_independent:
            # Each channel gets separate embedding (PatchTST style)
            self.proj = nn.Linear(patch_len, d_model)
        else:
            # All channels embedded together
            self.proj = nn.Linear(patch_len * num_features, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) input time series
        Returns:
            (B, num_patches, d_model) if not channel_independent
            (B, C, num_patches, d_model) if channel_independent
        """
        B, T, C = x.shape

        if self.channel_independent:
            # (B, T, C) -> (B, C, T)
            x = x.permute(0, 2, 1)
            # Create patches per channel
            patches = x.unfold(2, self.patch_len, self.stride)  # (B, C, num_patches, patch_len)
            return self.proj(patches)  # (B, C, num_patches, d_model)
        else:
            # Create patches across all channels
            patches = x.unfold(1, self.patch_len, self.stride)  # (B, num_patches, C, patch_len)
            patches = patches.permute(0, 1, 3, 2)  # (B, num_patches, patch_len, C)
            patches = patches.reshape(B, -1, self.patch_len * C)  # (B, num_patches, patch_len * C)
            return self.proj(patches)  # (B, num_patches, d_model)


class RevIN(nn.Module):
    """Reversible Instance Normalization for distribution shift handling."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

        # Store statistics for denormalization
        self.mean = None
        self.stdev = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x: torch.Tensor):
        self.mean = x.mean(dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        x = x * self.stdev + self.mean
        return x


# ============================================================================
# CROSS-VARIATE ATTENTION (Innovation #3)
# ============================================================================

class CrossVariateAttention(nn.Module):
    """
    Attention across channels at each time step (iTransformer style).

    Key insight: While temporal patterns are local, channel dependencies are global.
    Standard transformers miss cross-channel structure by treating channels independently.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, d_model) - channel-first representation
        Returns:
            (B, C, T, d_model) - with cross-channel attention applied
        """
        B, C, T, D = x.shape

        # Reshape: attend across channels at each timestep
        x_flat = x.permute(0, 2, 1, 3).reshape(B * T, C, D)  # (B*T, C, D)

        # Self-attention across channels
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        attn_out = self.dropout(attn_out)
        x_flat = self.norm(x_flat + attn_out)

        # Reshape back
        return x_flat.view(B, T, C, D).permute(0, 2, 1, 3)  # (B, C, T, D)


class CrossVariateEncoder(nn.Module):
    """
    Encoder with alternating temporal and cross-variate attention.

    Architecture:
        For each layer:
            1. Temporal attention (within each channel)
            2. Cross-variate attention (across channels at each time)
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_temporal_layers: int,
                 n_cross_variate_layers: int, dropout: float = 0.1):
        super().__init__()

        self.temporal_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                       activation='gelu', batch_first=True)
            for _ in range(n_temporal_layers)
        ])

        self.cross_variate_layers = nn.ModuleList([
            CrossVariateAttention(d_model, n_heads, dropout)
            for _ in range(n_cross_variate_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, d_model) - channel-first patch embeddings
        Returns:
            (B, C, T, d_model) - encoded representation
        """
        B, C, T, D = x.shape

        # Apply temporal attention within each channel
        x_temporal = x.view(B * C, T, D)
        for layer in self.temporal_layers:
            x_temporal = layer(x_temporal)
        x = x_temporal.view(B, C, T, D)

        # Apply cross-variate attention
        for layer in self.cross_variate_layers:
            x = layer(x)

        return x


# ============================================================================
# KOOPMAN PREDICTOR (Innovation #1)
# ============================================================================

class KoopmanPredictor(nn.Module):
    """
    Koopman operator for linear dynamics in latent space.

    Theory: Any nonlinear dynamical system can be represented as linear dynamics
    in a (possibly infinite-dimensional) lifted space. We learn a finite-dimensional
    approximation.

    Key insight: z_{t+k} = K^k @ z_t, where K is the Koopman matrix.
    Matrix powers enable O(1) prediction for any horizon.

    Parameterization: K = V @ diag(λ) @ V^(-1)
    - V: eigenvector matrix (learnable)
    - λ: eigenvalues (constrained to unit disk for stability)

    For complex eigenvalues (oscillatory dynamics):
        λ = r * exp(iθ) where r ∈ (0,1), θ ∈ (-π, π)
    """

    def __init__(self, latent_dim: int, rank: int = None, stable: bool = True,
                 use_complex: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.rank = rank or latent_dim
        self.stable = stable
        self.use_complex = use_complex

        # Low-rank factorization: K = U @ S @ V^T
        # This is more stable than full eigendecomposition
        self.U = nn.Parameter(torch.randn(latent_dim, self.rank) * 0.02)
        self.V = nn.Parameter(torch.randn(latent_dim, self.rank) * 0.02)

        if use_complex:
            # Complex eigenvalues for oscillatory dynamics
            # Parameterize as magnitude (r) and phase (θ)
            self.log_magnitude = nn.Parameter(torch.zeros(self.rank))  # log(r), r ∈ (0, 1)
            self.phase = nn.Parameter(torch.zeros(self.rank))  # θ ∈ (-π, π)
        else:
            # Real eigenvalues only
            self.eigenvalues = nn.Parameter(torch.zeros(self.rank))

        # Residual connection for stability
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def get_koopman_matrix(self) -> torch.Tensor:
        """Construct Koopman matrix K = U @ diag(λ) @ V^T

        For complex eigenvalues, we use 2x2 rotation blocks to preserve
        oscillatory dynamics while keeping the matrix real-valued.

        Complex eigenvalue λ = r*exp(iθ) maps to rotation block:
        [r*cos(θ)  -r*sin(θ)]
        [r*sin(θ)   r*cos(θ)]
        """
        if self.use_complex:
            # Eigenvalues as complex numbers: λ = r * exp(iθ)
            if self.stable:
                r = torch.sigmoid(self.log_magnitude)  # r ∈ (0, 1) for stability
            else:
                r = torch.exp(self.log_magnitude)  # unconstrained

            theta = self.phase

            # For real-valued output, construct block-diagonal with 2x2 rotation matrices
            # Each complex eigenvalue pair becomes a rotation block
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)

            # Build block-diagonal S matrix with 2x2 rotation blocks
            # S is (rank, rank) with 2x2 blocks on diagonal
            # For efficiency, we construct it explicitly
            half_rank = self.rank // 2

            # Each 2x2 block: [[r*cos, -r*sin], [r*sin, r*cos]]
            # We'll construct the diagonal matrix differently:
            # Use the eigenvalue diagonal but with rotation encoding
            S = torch.zeros(self.rank, self.rank, device=r.device, dtype=r.dtype)

            for i in range(half_rank):
                rc = r[i] * cos_theta[i]
                rs = r[i] * sin_theta[i]
                S[2*i, 2*i] = rc
                S[2*i, 2*i+1] = -rs
                S[2*i+1, 2*i] = rs
                S[2*i+1, 2*i+1] = rc

            # Handle odd rank (last eigenvalue is real)
            if self.rank % 2 == 1:
                S[-1, -1] = r[-1]

            # K = U @ S @ V^T
            K = self.U @ S @ self.V.T
        else:
            if self.stable:
                eigenvalues = torch.tanh(self.eigenvalues)  # ∈ (-1, 1)
            else:
                eigenvalues = self.eigenvalues

            # K = U @ diag(λ) @ V^T
            K = self.U @ torch.diag(eigenvalues) @ self.V.T
        return K

    def forward(self, z: torch.Tensor, horizon: int = 1) -> torch.Tensor:
        """
        Predict z_{t+horizon} from z_t using Koopman dynamics.

        Args:
            z: (B, latent_dim) current latent state
            horizon: number of steps to predict ahead
        Returns:
            z_pred: (B, latent_dim) predicted latent state
        """
        K = self.get_koopman_matrix()

        # K^horizon via repeated squaring (efficient for large horizons)
        K_power = torch.matrix_power(K, horizon)

        # z_{t+h} = K^h @ z_t + residual
        z_pred = z @ K_power.T

        # Residual connection for training stability
        if horizon == 1:
            z_pred = z_pred + self.residual_weight * z

        return z_pred

    def predict_sequence(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Predict sequence of future states efficiently.

        Args:
            z: (B, latent_dim) initial latent state
            seq_len: number of steps to predict
        Returns:
            z_seq: (B, seq_len, latent_dim) predicted sequence
        """
        K = self.get_koopman_matrix()

        z_seq = []
        z_current = z
        for t in range(seq_len):
            z_current = z_current @ K.T + self.residual_weight * z_current
            z_seq.append(z_current)

        return torch.stack(z_seq, dim=1)


class MLPPredictor(nn.Module):
    """Standard MLP predictor (baseline for Koopman comparison)."""

    def __init__(self, latent_dim: int, hidden_dim: int, pred_len: int, dropout: float = 0.1):
        super().__init__()
        self.pred_len = pred_len
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim * pred_len),
        )

    def forward(self, z: torch.Tensor, horizon: int = None) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim)
        Returns:
            (B, pred_len, latent_dim)
        """
        out = self.net(z)
        return out.view(-1, self.pred_len, self.latent_dim)

    def predict_sequence(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        return self.forward(z)[:, :seq_len]


# ============================================================================
# VICREG REGULARIZATION (Innovation #2)
# ============================================================================

def vicreg_loss(z: torch.Tensor, var_weight: float = 0.04, cov_weight: float = 0.04,
                eps: float = 1e-4) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    VICReg: Variance-Invariance-Covariance Regularization

    Prevents representation collapse in self-supervised learning.
    From: Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization"

    For JEPA, we apply to predicted latents to ensure diverse predictions.

    Args:
        z: (B, D) or (B, T, D) latent representations
        var_weight: weight for variance loss
        cov_weight: weight for covariance loss

    Returns:
        total_loss: weighted sum of variance and covariance losses
        loss_dict: individual loss components
    """
    if z.dim() == 3:
        # (B, T, D) -> (B*T, D)
        z = z.reshape(-1, z.size(-1))

    B, D = z.shape

    # Variance loss: std of each dimension should be at least 1
    # Prevents collapse to constant
    std = z.std(dim=0)
    var_loss = F.relu(1 - std).mean()

    # Covariance loss: off-diagonal elements of covariance should be 0
    # Prevents collapse to low-dimensional subspace
    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = (z_centered.T @ z_centered) / (B - 1 + eps)

    # Zero out diagonal
    off_diag_mask = ~torch.eye(D, dtype=torch.bool, device=z.device)
    cov_loss = cov[off_diag_mask].pow(2).mean()

    total_loss = var_weight * var_loss + cov_weight * cov_loss

    return total_loss, {'var_loss': var_loss, 'cov_loss': cov_loss}


# ============================================================================
# HIERARCHICAL ENCODER (Innovation #4)
# ============================================================================

class HierarchicalEncoder(nn.Module):
    """
    Multi-resolution encoder for capturing different time scales.

    Key insight: Different phenomena operate at different time scales:
    - Micro (seconds): noise, rapid fluctuations
    - Meso (minutes): local patterns, event responses
    - Macro (hours): trends, seasonality, regime changes

    Architecture:
        Level 1: Full resolution
        Level 2: 4x downsampled
        Level 3: 16x downsampled

    Each level has its own encoder and Koopman/predictor.
    Predictions are combined coarse-to-fine.
    """

    def __init__(self, num_features: int, d_model: int, n_heads: int, d_ff: int,
                 n_layers: int, latent_dim: int, factors: List[int] = [1, 4, 16],
                 dropout: float = 0.1):
        super().__init__()

        self.factors = factors
        self.n_levels = len(factors)

        # Per-level encoders
        self.encoders = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                          activation='gelu', batch_first=True),
                num_layers=n_layers
            )
            for _ in range(self.n_levels)
        ])

        # Per-level projections to latent
        self.to_latents = nn.ModuleList([
            nn.Linear(d_model, latent_dim)
            for _ in range(self.n_levels)
        ])

        # Downsampling (average pooling)
        self.downsample = nn.ModuleList([
            nn.AvgPool1d(kernel_size=f, stride=f) if f > 1 else nn.Identity()
            for f in factors
        ])

        # Per-level input projections
        self.input_projs = nn.ModuleList([
            nn.Linear(num_features, d_model)
            for _ in range(self.n_levels)
        ])

        # Cross-level fusion
        self.fusion = nn.Linear(latent_dim * self.n_levels, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, T, C) input time series
        Returns:
            z_fused: (B, latent_dim) fused multi-scale latent
            z_levels: list of per-level latents
        """
        B, T, C = x.shape
        z_levels = []

        for i, (factor, downsample, proj, encoder, to_latent) in enumerate(
            zip(self.factors, self.downsample, self.input_projs, self.encoders, self.to_latents)
        ):
            # Downsample
            if factor > 1:
                x_level = downsample(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, T//factor, C)
            else:
                x_level = x

            # Encode
            x_level = proj(x_level)  # (B, T//factor, d_model)
            x_level = encoder(x_level)  # (B, T//factor, d_model)

            # Pool to single vector and project to latent
            z_level = to_latent(x_level.mean(dim=1))  # (B, latent_dim)
            z_levels.append(z_level)

        # Fuse all levels
        z_concat = torch.cat(z_levels, dim=-1)  # (B, latent_dim * n_levels)
        z_fused = self.fusion(z_concat)  # (B, latent_dim)

        return z_fused, z_levels


# ============================================================================
# MAIN MODEL: KH-JEPA
# ============================================================================

class KHJEPAForecaster(nn.Module):
    """
    Koopman-Hierarchical JEPA Forecaster.

    Combines:
    1. JEPA: Predict in latent space
    2. Koopman: Linear dynamics for efficient prediction
    3. Hierarchy: Multi-resolution for different time scales
    4. VICReg: Prevent collapse
    5. Cross-Variate: Capture channel dependencies

    All components are toggleable for ablation studies.
    """

    def __init__(
        self,
        num_features: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 256,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        # JEPA
        use_jepa: bool = True,
        latent_dim: int = 128,
        ema_momentum: float = 0.996,
        # Koopman
        use_koopman: bool = False,
        koopman_rank: int = 64,
        koopman_stable: bool = True,
        # VICReg
        use_vicreg: bool = False,
        vicreg_var_weight: float = 0.04,
        vicreg_cov_weight: float = 0.04,
        # Cross-Variate
        use_cross_variate: bool = False,
        cross_variate_layers: int = 2,
        # Hierarchy
        use_hierarchy: bool = False,
        hierarchy_factors: List[int] = [1, 4, 16],
        # Patches
        use_patches: bool = True,
        patch_len: int = 16,
        stride: int = 8,
    ):
        super().__init__()

        # Store config
        self.num_features = num_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_jepa = use_jepa
        self.latent_dim = latent_dim
        self.ema_momentum = ema_momentum
        self.use_koopman = use_koopman
        self.use_vicreg = use_vicreg
        self.vicreg_var_weight = vicreg_var_weight
        self.vicreg_cov_weight = vicreg_cov_weight
        self.use_cross_variate = use_cross_variate
        self.use_hierarchy = use_hierarchy
        self.use_patches = use_patches

        # RevIN
        self.revin = RevIN(num_features)

        # Patch embedding
        if use_patches:
            self.num_patches = (seq_len - patch_len) // stride + 1
            if use_cross_variate:
                # Keep channels separate for cross-variate attention
                self.patch_embed = PatchEmbedding(num_features, patch_len, stride, d_model,
                                                   channel_independent=True)
            else:
                self.patch_embed = PatchEmbedding(num_features, patch_len, stride, d_model,
                                                   channel_independent=False)
        else:
            self.input_proj = nn.Linear(num_features, d_model)

        # Positional encoding
        max_len = max(seq_len, pred_len) + 100
        self.pos_embed = PositionalEncoding(d_model, max_len)

        # Encoder
        if use_hierarchy:
            self.encoder = HierarchicalEncoder(
                num_features, d_model, n_heads, d_ff, e_layers, latent_dim,
                factors=hierarchy_factors, dropout=dropout
            )
        elif use_cross_variate:
            self.encoder = CrossVariateEncoder(
                d_model, n_heads, d_ff, e_layers, cross_variate_layers, dropout
            )
            self.to_latent = nn.Linear(d_model, latent_dim)
        else:
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                          activation='gelu', batch_first=True),
                num_layers=e_layers
            )
            self.to_latent = nn.Linear(d_model, latent_dim)

        # Predictor
        if use_jepa:
            if use_koopman:
                self.predictor = KoopmanPredictor(latent_dim, rank=koopman_rank,
                                                   stable=koopman_stable)
            else:
                self.predictor = MLPPredictor(latent_dim, d_ff, pred_len, dropout)

            # Decoder from latent to observation
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, num_features),
            )

            # Target encoder (EMA)
            self._build_target_encoder(d_model, n_heads, d_ff, e_layers, dropout)
        else:
            # Direct prediction (no latent space)
            self.predictor = nn.Linear(d_model, num_features * pred_len)

    def _build_target_encoder(self, d_model, n_heads, d_ff, e_layers, dropout):
        """Build target encoder as copy of online encoder (for EMA).

        CRITICAL: Target encoder must match online encoder architecture
        to ensure valid JEPA comparisons.
        """
        if self.use_hierarchy:
            # Match HierarchicalEncoder architecture
            # For hierarchy mode, create a simplified target encoder
            # that processes at full resolution only
            self.target_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                          activation='gelu', batch_first=True),
                num_layers=e_layers
            )
            self.target_input_proj = nn.Linear(self.num_features, d_model)
            self.target_to_latent = nn.Linear(d_model, self.latent_dim)

            # Initialize target_input_proj with first level's input_proj
            if hasattr(self.encoder, 'input_projs'):
                for p_online, p_target in zip(
                    self.encoder.input_projs[0].parameters(),
                    self.target_input_proj.parameters()
                ):
                    p_target.data.copy_(p_online.data)
                    p_target.requires_grad = False

            # Initialize encoder with first level's encoder
            if hasattr(self.encoder, 'encoders'):
                for p_online, p_target in zip(
                    self.encoder.encoders[0].parameters(),
                    self.target_encoder.parameters()
                ):
                    p_target.data.copy_(p_online.data)
                    p_target.requires_grad = False

            # Initialize to_latent with first level's to_latent
            if hasattr(self.encoder, 'to_latents'):
                for p_online, p_target in zip(
                    self.encoder.to_latents[0].parameters(),
                    self.target_to_latent.parameters()
                ):
                    p_target.data.copy_(p_online.data)
                    p_target.requires_grad = False

        elif self.use_cross_variate:
            # Match CrossVariateEncoder architecture
            self.target_encoder = CrossVariateEncoder(
                d_model, n_heads, d_ff,
                n_temporal_layers=e_layers,
                n_cross_variate_layers=2,  # Use same as online
                dropout=dropout
            )
            self.target_to_latent = nn.Linear(d_model, self.latent_dim)

            # Initialize with online encoder weights
            for p_online, p_target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                p_target.data.copy_(p_online.data)
                p_target.requires_grad = False

            if hasattr(self, 'to_latent'):
                for p_online, p_target in zip(self.to_latent.parameters(), self.target_to_latent.parameters()):
                    p_target.data.copy_(p_online.data)
                    p_target.requires_grad = False

        else:
            # Standard TransformerEncoder
            self.target_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                          activation='gelu', batch_first=True),
                num_layers=e_layers
            )
            self.target_to_latent = nn.Linear(d_model, self.latent_dim)

            # Initialize with online encoder weights
            if hasattr(self, 'encoder') and isinstance(self.encoder, nn.TransformerEncoder):
                for p_online, p_target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                    p_target.data.copy_(p_online.data)
                    p_target.requires_grad = False

            if hasattr(self, 'to_latent'):
                for p_online, p_target in zip(self.to_latent.parameters(), self.target_to_latent.parameters()):
                    p_target.data.copy_(p_online.data)
                    p_target.requires_grad = False

    @torch.no_grad()
    def update_ema(self, momentum: float = None):
        """Update target encoder with EMA of online encoder.

        Handles all encoder architectures (standard, cross-variate, hierarchical).
        """
        if not self.use_jepa:
            return

        m = momentum if momentum is not None else self.ema_momentum

        if self.use_hierarchy:
            # Update target encoder from first level's encoder
            if hasattr(self.encoder, 'encoders'):
                for p_online, p_target in zip(
                    self.encoder.encoders[0].parameters(),
                    self.target_encoder.parameters()
                ):
                    p_target.data = m * p_target.data + (1 - m) * p_online.data

            if hasattr(self, 'target_input_proj') and hasattr(self.encoder, 'input_projs'):
                for p_online, p_target in zip(
                    self.encoder.input_projs[0].parameters(),
                    self.target_input_proj.parameters()
                ):
                    p_target.data = m * p_target.data + (1 - m) * p_online.data

            if hasattr(self.encoder, 'to_latents'):
                for p_online, p_target in zip(
                    self.encoder.to_latents[0].parameters(),
                    self.target_to_latent.parameters()
                ):
                    p_target.data = m * p_target.data + (1 - m) * p_online.data

        elif self.use_cross_variate:
            # Update cross-variate target encoder
            for p_online, p_target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                p_target.data = m * p_target.data + (1 - m) * p_online.data

            if hasattr(self, 'to_latent'):
                for p_online, p_target in zip(self.to_latent.parameters(), self.target_to_latent.parameters()):
                    p_target.data = m * p_target.data + (1 - m) * p_online.data

        else:
            # Standard TransformerEncoder
            if hasattr(self, 'encoder') and isinstance(self.encoder, nn.TransformerEncoder):
                for p_online, p_target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                    p_target.data = m * p_target.data + (1 - m) * p_online.data

        if hasattr(self, 'to_latent') and not self.use_hierarchy and not self.use_cross_variate:
            for p_online, p_target in zip(self.to_latent.parameters(), self.target_to_latent.parameters()):
                p_target.data = m * p_target.data + (1 - m) * p_online.data

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        if self.use_hierarchy:
            z, _ = self.encoder(x)  # Already returns latent
            return z

        # Patch embedding
        if self.use_patches:
            x = self.patch_embed(x)  # (B, num_patches, d_model) or (B, C, num_patches, d_model)
        else:
            x = self.input_proj(x)

        # Cross-variate or standard encoding
        if self.use_cross_variate:
            # x is (B, C, num_patches, d_model)
            B, C, T, D = x.shape

            # Add positional encoding per channel
            x = x.view(B * C, T, D)
            x = self.pos_embed(x)
            x = x.view(B, C, T, D)

            # Encode with cross-variate attention
            x = self.encoder(x)  # (B, C, T, d_model)

            # Pool across time and channels
            x = x.mean(dim=(1, 2))  # (B, d_model)
            z = self.to_latent(x)  # (B, latent_dim)
        else:
            x = self.pos_embed(x)
            x = self.encoder(x)  # (B, T, d_model)
            x = x.mean(dim=1)  # (B, d_model)
            z = self.to_latent(x)  # (B, latent_dim)

        return z

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: (B, seq_len, num_features) input sequence
            y: (B, pred_len, num_features) target sequence (for training)

        Returns:
            pred: (B, pred_len, num_features) predictions
            loss_dict: dictionary of losses
        """
        B = x.shape[0]

        # RevIN normalization
        x = self.revin(x, 'norm')

        # Encode
        z = self.encode(x)  # (B, latent_dim)

        if self.use_jepa:
            # Predict in latent space
            if self.use_koopman:
                z_pred = self.predictor.predict_sequence(z, self.pred_len)  # (B, pred_len, latent_dim)
            else:
                z_pred = self.predictor(z)  # (B, pred_len, latent_dim)

            # Decode to observation space
            pred = self.decoder(z_pred)  # (B, pred_len, num_features)
        else:
            # Direct prediction
            pred = self.predictor(z)
            pred = pred.view(B, self.pred_len, self.num_features)

        # RevIN denormalization
        pred = self.revin(pred, 'denorm')

        # Compute losses
        loss_dict = {}
        if y is not None:
            loss_dict = self._compute_losses(pred, y, z_pred if self.use_jepa else None, x)

        return pred, loss_dict

    def _compute_losses(self, pred: torch.Tensor, y: torch.Tensor,
                        z_pred: torch.Tensor, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute all losses."""
        loss_dict = {}

        # Main prediction loss
        if USE_HUBER_LOSS:
            mse_loss = F.huber_loss(pred, y, delta=HUBER_DELTA)
        else:
            mse_loss = F.mse_loss(pred, y)
        loss_dict['mse'] = mse_loss

        total_loss = mse_loss

        # JEPA latent loss
        if self.use_jepa and z_pred is not None:
            with torch.no_grad():
                # Encode target with target encoder
                y_norm = self.revin(y, 'norm')

                if self.use_hierarchy:
                    # For hierarchy mode: use target encoder at full resolution
                    y_proj = self.target_input_proj(y_norm)  # (B, pred_len, d_model)
                    y_enc = self.target_encoder(y_proj)  # (B, pred_len, d_model)
                    y_enc = y_enc.mean(dim=1)  # (B, d_model)

                elif self.use_patches:
                    # Pad to match patch structure
                    if self.use_cross_variate:
                        y_padded = F.pad(y_norm, (0, 0, self.seq_len - self.pred_len, 0))
                        y_patches = self.patch_embed(y_padded)
                        B, C, T, D = y_patches.shape

                        # Apply positional encoding per channel
                        y_patches = y_patches.view(B * C, T, D)
                        y_enc = self.pos_embed(y_patches)
                        y_enc = y_enc.view(B, C, T, D)

                        # Encode with cross-variate target encoder
                        y_enc = self.target_encoder(y_enc)  # (B, C, T, d_model)
                        y_enc = y_enc.mean(dim=(1, 2))  # (B, d_model)
                    else:
                        y_padded = F.pad(y_norm, (0, 0, self.seq_len - self.pred_len, 0))
                        y_patches = self.patch_embed(y_padded)
                        y_enc = self.pos_embed(y_patches)
                        y_enc = self.target_encoder(y_enc)
                        y_enc = y_enc.mean(dim=1)
                else:
                    y_enc = self.input_proj(y_norm)
                    y_enc = self.pos_embed(y_enc)
                    y_enc = self.target_encoder(y_enc)
                    y_enc = y_enc.mean(dim=1)

                z_target = self.target_to_latent(y_enc)  # (B, latent_dim)

            # Latent prediction loss
            z_pred_pooled = z_pred.mean(dim=1)  # (B, latent_dim)
            jepa_loss = F.mse_loss(z_pred_pooled, z_target)
            loss_dict['jepa'] = jepa_loss
            total_loss = total_loss + 0.5 * jepa_loss

        # VICReg regularization
        if self.use_vicreg and z_pred is not None:
            vicreg, vicreg_components = vicreg_loss(
                z_pred, self.vicreg_var_weight, self.vicreg_cov_weight
            )
            loss_dict['vicreg'] = vicreg
            loss_dict['var_loss'] = vicreg_components['var_loss']
            loss_dict['cov_loss'] = vicreg_components['cov_loss']
            total_loss = total_loss + vicreg

        loss_dict['total'] = total_loss
        return loss_dict


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model: nn.Module, loader, optimizer, device, epoch: int, total_epochs: int,
                scheduler=None, step_scheduler_per_batch: bool = False):
    """Train for one epoch with optional EMA warmup.

    Args:
        scheduler: Learning rate scheduler (optional)
        step_scheduler_per_batch: If True, step scheduler after each batch (for OneCycleLR)
    """
    model.train()
    total_loss = 0
    total_mse = 0
    n_batches = 0

    # EMA momentum warmup (gradually increase from 0.99 to target)
    if EMA_WARMUP and hasattr(model, 'ema_momentum'):
        base_momentum = 0.99
        target_momentum = model.ema_momentum
        progress = epoch / total_epochs
        current_momentum = base_momentum + (target_momentum - base_momentum) * progress
    else:
        current_momentum = None

    for batch in loader:
        x = batch['x'].to(device)
        y = batch['y'].to(device)

        pred, losses = model(x, y)

        optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()

        # Step scheduler per batch (for OneCycleLR)
        if scheduler is not None and step_scheduler_per_batch:
            scheduler.step()

        if hasattr(model, 'update_ema'):
            model.update_ema(current_momentum)

        total_loss += losses['total'].item()
        total_mse += losses['mse'].item()
        n_batches += 1

    return total_loss / n_batches, total_mse / n_batches


@torch.no_grad()
def evaluate(model: nn.Module, loader, device) -> Tuple[float, float]:
    """Evaluate model on dataset."""
    model.eval()
    total_mse = 0
    total_mae = 0
    n_samples = 0

    for batch in loader:
        x = batch['x'].to(device)
        y = batch['y'].to(device)

        pred, _ = model(x)

        mse = F.mse_loss(pred, y, reduction='sum').item()
        mae = F.l1_loss(pred, y, reduction='sum').item()

        total_mse += mse
        total_mae += mae
        n_samples += y.numel()

    return total_mse / n_samples, total_mae / n_samples


def main():
    start_time = time.time()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Configuration summary
    print("\n" + "=" * 60)
    print("KH-JEPA Configuration")
    print("=" * 60)
    print(f"  SEQ_LEN: {SEQ_LEN}, PRED_LEN: {PRED_LEN}")
    print(f"  USE_JEPA: {USE_JEPA}")
    print(f"  USE_KOOPMAN: {USE_KOOPMAN}")
    print(f"  USE_VICREG: {USE_VICREG}")
    print(f"  USE_CROSS_VARIATE: {USE_CROSS_VARIATE}")
    print(f"  USE_HIERARCHY: {USE_HIERARCHY}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, info = get_dataloaders(
        BATCH_SIZE, SEQ_LEN, PRED_LEN
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Features: {info['num_features']}")

    # Create model
    model = KHJEPAForecaster(
        num_features=info['num_features'],
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        e_layers=E_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        use_jepa=USE_JEPA,
        latent_dim=LATENT_DIM,
        ema_momentum=EMA_MOMENTUM,
        use_koopman=USE_KOOPMAN,
        koopman_rank=KOOPMAN_RANK,
        koopman_stable=KOOPMAN_STABLE,
        use_vicreg=USE_VICREG,
        vicreg_var_weight=VICREG_VAR_WEIGHT,
        vicreg_cov_weight=VICREG_COV_WEIGHT,
        use_cross_variate=USE_CROSS_VARIATE,
        cross_variate_layers=CROSS_VARIATE_LAYERS,
        use_hierarchy=USE_HIERARCHY,
        hierarchy_factors=HIERARCHY_FACTORS,
        use_patches=USE_PATCHES,
        patch_len=PATCH_LEN,
        stride=STRIDE,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    # Optimizer with warmup
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    if WARMUP_EPOCHS > 0:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=LEARNING_RATE,
            epochs=EPOCHS,
            steps_per_epoch=len(train_loader),
            pct_start=WARMUP_EPOCHS / EPOCHS,
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training
    best_val_mse = float('inf')
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # Determine scheduler stepping strategy
    use_onecycle = WARMUP_EPOCHS > 0

    print(f"\nTraining for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_mse = train_epoch(
            model, train_loader, optimizer, device, epoch, EPOCHS,
            scheduler=scheduler if use_onecycle else None,
            step_scheduler_per_batch=use_onecycle
        )
        val_mse, val_mae = evaluate(model, val_loader, device)

        # Step scheduler per epoch (for CosineAnnealingLR)
        if not use_onecycle:
            scheduler.step()

        # Logging
        log_str = f"Epoch {epoch}: train_mse={train_mse:.4f}, val_mse={val_mse:.4f}"
        if USE_VICREG:
            log_str += " [VICReg]"
        if USE_KOOPMAN:
            log_str += " [Koopman]"
        print(log_str)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), results_dir / 'best_model.pt')

    # Final evaluation
    model.load_state_dict(torch.load(results_dir / 'best_model.pt', weights_only=True))
    test_mse, test_mae = evaluate(model, test_loader, device)

    elapsed = time.time() - start_time

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"val_mse: {best_val_mse:.6f}")
    print(f"test_mse: {test_mse:.6f}")
    print(f"test_mae: {test_mae:.6f}")
    print(f"elapsed_seconds: {elapsed:.1f}")
    print(f"parameters: {n_params:,}")
    print("=" * 60)

    # SOTA comparison
    print("\nSOTA Comparison (ETTh1, horizon 96):")
    print(f"  TTT:          0.358 MSE (target)")
    print(f"  iTransformer: 0.386 MSE")
    print(f"  PatchTST:     0.414 MSE")
    print(f"  Ours:         {test_mse:.3f} MSE", end="")
    if test_mse < 0.358:
        print(" ** NEW SOTA! **")
    elif test_mse < 0.386:
        print(" (beats iTransformer)")
    elif test_mse < 0.414:
        print(" (beats PatchTST)")
    else:
        print("")
    print("=" * 60)

    # Architecture summary
    print("\nArchitecture used:")
    print(f"  JEPA: {USE_JEPA}")
    print(f"  Koopman: {USE_KOOPMAN}")
    print(f"  VICReg: {USE_VICREG}")
    print(f"  Cross-Variate: {USE_CROSS_VARIATE}")
    print(f"  Hierarchy: {USE_HIERARCHY}")
    print("=" * 60)

    # Save results
    results = {
        'val_mse': best_val_mse,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'elapsed_seconds': elapsed,
        'parameters': n_params,
        'config': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'seq_len': SEQ_LEN,
            'pred_len': PRED_LEN,
            'd_model': D_MODEL,
            'latent_dim': LATENT_DIM,
            'use_jepa': USE_JEPA,
            'use_koopman': USE_KOOPMAN,
            'use_vicreg': USE_VICREG,
            'use_cross_variate': USE_CROSS_VARIATE,
            'use_hierarchy': USE_HIERARCHY,
        },
        'timestamp': datetime.now().isoformat(),
    }

    with open(results_dir / 'latest_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Output for run.py parsing
    print(f"\nval_loss: {best_val_mse:.6f}")

    return best_val_mse


if __name__ == "__main__":
    main()
