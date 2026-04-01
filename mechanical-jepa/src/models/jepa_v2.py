"""
Mechanical-JEPA v2: Fixed predictor collapse.

Root cause: JEPAPredictor uses learnable positional embeddings initialized to
trunc_normal(std=0.02). These are small relative to the mask token value.
With only 2 transformer layers, the predictor cannot differentiate positions.
It collapses to predicting a context-weighted average, ignoring position entirely.

Fixes implemented (selectable via CLI flags):
  --predictor-pos {learnable,sinusoidal,rope}
    'sinusoidal': Fixed sine/cosine encoding, guarantees position discrimination
    'learnable':  Original (collapsed)
    'rope':       Rotary position embeddings (experimental)

  --predictor-depth N
    Default 2 -> try 4 for more position-processing capacity

  --var-reg LAMBDA
    VICReg-style variance regularization on predictions.
    Penalizes low std across the batch dimension.

  --loss-fn {mse,l1,smooth_l1}
    l1: more robust to outliers, less incentive for "safe" mean prediction
    smooth_l1: L1 far from target, L2 near target

  --pred-separate-tokens
    Per-position learnable mask tokens instead of one shared token.
    Guarantees initial diversity across positions.

  --vicreg-coeff LAMBDA
    Add VICReg variance + covariance loss on encoder outputs.
    Prevents encoder from collapsing too.

Architecture: backward-compatible with v1 checkpoints via config keys.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
import copy

from .jepa import PatchEmbed1D, TransformerBlock, JEPAEncoder


# =============================================================================
# Positional Encoding Helpers
# =============================================================================

def sinusoidal_pos_encoding(n_positions: int, d_model: int) -> torch.Tensor:
    """
    Standard sinusoidal positional encoding from "Attention is All You Need".
    Returns (1, n_positions, d_model) tensor.
    """
    pe = torch.zeros(n_positions, d_model)
    position = torch.arange(0, n_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, N, D)


# =============================================================================
# JEPA Predictor V2: Collapse Prevention
# =============================================================================

class JEPAPredictorV2(nn.Module):
    """
    Improved predictor with multiple collapse-prevention mechanisms.

    Key changes from V1:
    1. pos_enc_type: 'sinusoidal' gives deterministic, diverse positional signals
    2. separate_mask_tokens: per-position tokens instead of one shared token
    3. depth: more layers give capacity to learn position-dependent transforms
    """

    def __init__(
        self,
        n_patches: int = 16,
        embed_dim: int = 256,
        predictor_dim: int = 128,
        depth: int = 4,
        n_heads: int = 4,
        pos_enc_type: str = 'sinusoidal',  # 'sinusoidal' | 'learnable'
        separate_mask_tokens: bool = False,
    ):
        super().__init__()
        self.n_patches = n_patches
        self.embed_dim = embed_dim
        self.predictor_dim = predictor_dim
        self.pos_enc_type = pos_enc_type
        self.separate_mask_tokens = separate_mask_tokens

        # Project encoder dim to predictor dim
        self.input_proj = nn.Linear(embed_dim, predictor_dim)

        # Positional encoding
        if pos_enc_type == 'sinusoidal':
            # Fixed — guaranteed position discrimination, no learning needed
            pe = sinusoidal_pos_encoding(n_patches, predictor_dim)
            self.register_buffer('pos_embed', pe)  # (1, N, predictor_dim)
        elif pos_enc_type == 'learnable':
            # Original learnable — tends to collapse
            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, predictor_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            raise ValueError(f"Unknown pos_enc_type: {pos_enc_type}")

        # Mask tokens: shared (original) or per-position (prevents collapse)
        if separate_mask_tokens:
            # One learnable token per position — guarantees initial diversity
            self.mask_tokens = nn.Parameter(torch.zeros(1, n_patches, predictor_dim))
            nn.init.trunc_normal_(self.mask_tokens, std=0.02)
        else:
            # Original: single shared token expanded to all positions
            self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
            nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_dim, n_heads, mlp_ratio=4.0, dropout=0.1)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(predictor_dim)

        # Project back to encoder dim for loss computation
        self.output_proj = nn.Linear(predictor_dim, embed_dim)

    def forward(
        self,
        context_embeds: torch.Tensor,
        context_indices: torch.Tensor,
        mask_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            context_embeds: (B, n_context, D) embeddings of visible patches
            context_indices: (B, n_context) indices of visible patches
            mask_indices: (B, n_mask) indices of masked patches

        Returns:
            predictions: (B, n_mask, D) predicted embeddings for masked patches
        """
        B = context_embeds.shape[0]
        n_context = context_embeds.shape[1]
        n_mask = mask_indices.shape[1]

        # Project context to predictor dimension
        context = self.input_proj(context_embeds)  # (B, n_context, predictor_dim)

        # Add positional embeddings to context
        pos_embed = self.pos_embed.expand(B, -1, -1)  # (B, N, predictor_dim)
        context_pos = torch.gather(
            pos_embed, 1,
            context_indices.unsqueeze(-1).expand(-1, -1, self.predictor_dim)
        )
        context = context + context_pos

        # Create mask tokens with positional embeddings
        if self.separate_mask_tokens:
            # Gather per-position tokens
            mask_tokens_full = self.mask_tokens.expand(B, -1, -1)  # (B, N, predictor_dim)
            mask_tokens = torch.gather(
                mask_tokens_full, 1,
                mask_indices.unsqueeze(-1).expand(-1, -1, self.predictor_dim)
            )  # (B, n_mask, predictor_dim)
        else:
            # Shared token expanded to all mask positions
            mask_tokens = self.mask_token.expand(B, n_mask, -1)

        # Add positional embedding to mask tokens
        mask_pos = torch.gather(
            pos_embed, 1,
            mask_indices.unsqueeze(-1).expand(-1, -1, self.predictor_dim)
        )
        mask_tokens = mask_tokens + mask_pos

        # Concatenate context and mask tokens
        x = torch.cat([context, mask_tokens], dim=1)  # (B, n_context + n_mask, predictor_dim)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Extract predictions for masked positions (last n_mask tokens)
        predictions = x[:, n_context:]  # (B, n_mask, predictor_dim)

        # Project to encoder dimension
        predictions = self.output_proj(predictions)  # (B, n_mask, D)

        return predictions


# =============================================================================
# VICReg Regularization Loss
# =============================================================================

def vicreg_loss(z: torch.Tensor, sim_coeff: float = 0.0, std_coeff: float = 1.0, cov_coeff: float = 0.04) -> torch.Tensor:
    """
    VICReg regularization (variance + covariance terms only, no invariance).
    Prevents encoder representation collapse.

    Args:
        z: (B, D) or (B, N, D) embeddings
        std_coeff: Weight for variance term (prevents collapse)
        cov_coeff: Weight for covariance term (decorrelates dimensions)

    Returns:
        scalar loss
    """
    if z.dim() == 3:
        z = z.reshape(-1, z.shape[-1])  # flatten batch x patches

    # Center
    z = z - z.mean(dim=0, keepdim=True)
    N, D = z.shape

    # Variance: penalize std below 1
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    var_loss = F.relu(1.0 - std).mean()

    # Covariance: penalize off-diagonal covariance
    if N > 1:
        cov = (z.T @ z) / (N - 1)
        cov_loss = (cov ** 2).sum() - (cov ** 2).diagonal().sum()
        cov_loss = cov_loss / D
    else:
        cov_loss = torch.tensor(0.0, device=z.device)

    return std_coeff * var_loss + cov_coeff * cov_loss


# =============================================================================
# Variance Regularization on Predictions
# =============================================================================

def prediction_var_loss(predictions: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    """
    Penalize low variance across predicted positions.

    predictions: (B, n_mask, D) — predictions for different positions
    Returns: scalar loss, high when all positions get same prediction.
    """
    # Variance across patch positions (dim=1), then mean over batch and dims
    pred_var = predictions.var(dim=1).mean()  # scalar
    # Penalize if var falls below threshold
    loss = F.relu(threshold - pred_var)
    return loss


# =============================================================================
# Mechanical-JEPA V2: Full Model
# =============================================================================

class MechanicalJEPAV2(nn.Module):
    """
    Mechanical-JEPA V2 with predictor collapse fixes.

    New args (vs V1):
      predictor_pos: 'sinusoidal' (recommended) | 'learnable' (original)
      predictor_depth: depth of predictor transformer (was hardcoded to 2)
      separate_mask_tokens: True = per-position mask tokens
      loss_fn: 'mse' | 'l1' | 'smooth_l1'
      var_reg_lambda: variance regularization on predictions (0 = off)
      vicreg_lambda: VICReg regularization on encoder outputs (0 = off)
    """

    def __init__(
        self,
        n_channels: int = 3,
        window_size: int = 4096,
        patch_size: int = 256,
        embed_dim: int = 512,
        encoder_depth: int = 4,
        predictor_depth: int = 4,
        n_heads: int = 4,
        mask_ratio: float = 0.5,
        ema_decay: float = 0.996,
        # New collapse-prevention args
        predictor_pos: str = 'sinusoidal',   # 'sinusoidal' | 'learnable'
        separate_mask_tokens: bool = False,
        loss_fn: str = 'mse',               # 'mse' | 'l1' | 'smooth_l1'
        var_reg_lambda: float = 0.0,        # prediction variance regularization
        vicreg_lambda: float = 0.0,         # encoder VICReg regularization
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.ema_decay = ema_decay
        self.n_patches = window_size // patch_size
        self.loss_fn = loss_fn
        self.var_reg_lambda = var_reg_lambda
        self.vicreg_lambda = vicreg_lambda

        # Context encoder (trainable)
        self.encoder = JEPAEncoder(
            n_channels=n_channels,
            window_size=window_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=encoder_depth,
            n_heads=n_heads,
        )

        # Target encoder (EMA of context encoder)
        self.target_encoder = copy.deepcopy(self.encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Predictor V2 with collapse prevention
        self.predictor = JEPAPredictorV2(
            n_patches=self.n_patches,
            embed_dim=embed_dim,
            predictor_dim=embed_dim // 2,
            depth=predictor_depth,
            n_heads=n_heads,
            pos_enc_type=predictor_pos,
            separate_mask_tokens=separate_mask_tokens,
        )

    @torch.no_grad()
    def _update_target_encoder(self):
        """Update target encoder with EMA."""
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = self.ema_decay * param_k.data + (1 - self.ema_decay) * param_q.data

    def _generate_mask(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random mask indices."""
        n_mask = int(self.n_patches * self.mask_ratio)
        n_context = self.n_patches - n_mask

        indices = torch.stack([
            torch.randperm(self.n_patches, device=device)
            for _ in range(batch_size)
        ])

        mask_indices = indices[:, :n_mask]
        context_indices = indices[:, n_mask:]

        return mask_indices, context_indices

    def _compute_prediction_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute main prediction loss (MSE, L1, or Smooth L1).
        Both inputs normalized along embedding dimension.
        """
        predictions_norm = F.normalize(predictions, dim=-1)
        targets_norm = F.normalize(targets, dim=-1)

        if self.loss_fn == 'mse':
            return F.mse_loss(predictions_norm, targets_norm)
        elif self.loss_fn == 'l1':
            return F.l1_loss(predictions_norm, targets_norm)
        elif self.loss_fn == 'smooth_l1':
            return F.smooth_l1_loss(predictions_norm, targets_norm)
        else:
            raise ValueError(f"Unknown loss_fn: {self.loss_fn}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.

        Returns:
            loss: scalar (total loss, including regularization)
            predictions: (B, n_mask, D)
            targets: (B, n_mask, D)
        """
        B = x.shape[0]
        device = x.device

        # Generate mask
        mask_indices, context_indices = self._generate_mask(B, device)

        # Get target embeddings (all patches, no masking)
        with torch.no_grad():
            target_embeds = self.target_encoder(x, return_all_tokens=True)[:, 1:]  # Remove CLS
            targets = torch.gather(
                target_embeds, 1,
                mask_indices.unsqueeze(-1).expand(-1, -1, target_embeds.shape[-1])
            )

        # Get context embeddings (visible patches only)
        context_embeds = self.encoder(x, mask_indices=mask_indices, return_all_tokens=True)[:, 1:]

        # Predict masked patch embeddings
        predictions = self.predictor(context_embeds, context_indices, mask_indices)

        # Main prediction loss
        loss = self._compute_prediction_loss(predictions, targets)

        # Optional: Prediction variance regularization
        # Penalizes low variance across positions (prevents predictor collapse)
        if self.var_reg_lambda > 0:
            var_loss = prediction_var_loss(predictions, threshold=0.1)
            loss = loss + self.var_reg_lambda * var_loss

        # Optional: VICReg regularization on encoder outputs
        # Prevents encoder representation collapse
        if self.vicreg_lambda > 0:
            all_encoder_out = self.encoder(x, return_all_tokens=True)[:, 1:]  # (B, N, D)
            vic_loss = vicreg_loss(all_encoder_out)
            loss = loss + self.vicreg_lambda * vic_loss

        return loss, predictions, targets

    def get_embeddings(self, x: torch.Tensor, pool: str = 'mean') -> torch.Tensor:
        """
        Get embeddings for downstream tasks.
        Mean-pool over patch tokens (preferred: JEPA trains patch tokens, not CLS).
        """
        if pool == 'cls':
            return self.encoder(x, return_all_tokens=False)
        else:
            all_tokens = self.encoder(x, return_all_tokens=True)
            return all_tokens[:, 1:].mean(dim=1)

    def train_step(self, x: torch.Tensor) -> torch.Tensor:
        loss, _, _ = self.forward(x)
        return loss

    @torch.no_grad()
    def update_ema(self):
        self._update_target_encoder()
