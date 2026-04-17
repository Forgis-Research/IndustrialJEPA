"""
Mechanical-JEPA V3: JEPA with SIGReg (No EMA).

Key change from V2: Remove EMA target encoder, replace with SIGReg on encoder outputs.
This follows the LeJEPA / LeWorldModel approach of using regularization instead of
momentum encoders for collapse prevention.

Architecture change:
- V2: encoder (trainable) + target_encoder (EMA of encoder) + predictor
  Loss = L1(normalize(predictions), normalize(EMA_targets)) + var_reg
- V3: single encoder + predictor
  Loss = L1(normalize(predictions), normalize(sg(targets))) + sigreg_coeff * SIGReg(encoder_out)
  where sg = stop_gradient on targets (detach)

Benefits:
- Simpler: no EMA hyperparameter, no copy of encoder weights
- ~50% memory savings (no target_encoder)
- Faster: no EMA update step
- Principled: SIGReg directly regularizes the embedding distribution

The prediction target is the SAME encoder's output (just detached from the computation graph).
This is similar to BYOL / SimSiam but with the isotropic Gaussian regularization preventing
the trivial solution where all embeddings collapse to a constant.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import copy

from .jepa import PatchEmbed1D, TransformerBlock, JEPAEncoder
from .jepa_v2 import JEPAPredictorV2, sinusoidal_pos_encoding, prediction_var_loss
from .sigreg import SIGReg


class MechanicalJEPAV3(nn.Module):
    """
    Mechanical-JEPA V3: SIGReg replaces EMA.

    Key differences from V2:
    1. Single encoder (no EMA target_encoder)
    2. SIGReg on encoder outputs for collapse prevention
    3. Stop-gradient on prediction targets (not EMA)
    4. No var_reg_lambda needed (SIGReg handles this)

    Args:
        n_channels: Number of input channels
        window_size: Input window size in samples
        patch_size: Patch size in samples
        embed_dim: Encoder embedding dimension
        encoder_depth: Number of transformer blocks in encoder
        predictor_depth: Number of transformer blocks in predictor
        n_heads: Number of attention heads
        mask_ratio: Fraction of patches to mask
        predictor_pos: Positional encoding type ('sinusoidal' | 'learnable')
        loss_fn: Loss function ('mse' | 'l1' | 'smooth_l1')
        sigreg_coeff: SIGReg regularization coefficient
        sigreg_projections: Number of random projections for SIGReg
        var_reg_lambda: Optional prediction variance regularization (additional)
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
        mask_ratio: float = 0.625,
        predictor_pos: str = 'sinusoidal',
        loss_fn: str = 'l1',
        sigreg_coeff: float = 0.1,
        sigreg_projections: int = 64,
        var_reg_lambda: float = 0.0,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.n_patches = window_size // patch_size
        self.loss_fn = loss_fn
        self.sigreg_coeff = sigreg_coeff
        self.var_reg_lambda = var_reg_lambda

        # Single encoder (no EMA copy)
        self.encoder = JEPAEncoder(
            n_channels=n_channels,
            window_size=window_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=encoder_depth,
            n_heads=n_heads,
        )

        # Predictor (same as V2)
        self.predictor = JEPAPredictorV2(
            n_patches=self.n_patches,
            embed_dim=embed_dim,
            predictor_dim=embed_dim // 2,
            depth=predictor_depth,
            n_heads=n_heads,
            pos_enc_type=predictor_pos,
            separate_mask_tokens=False,
        )

        # SIGReg module with persistent random projections
        self.sigreg = SIGReg(
            embed_dim=embed_dim,
            n_projections=sigreg_projections,
            method='moments',
        )

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

    def _compute_prediction_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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

        Key difference from V2: targets are encoder outputs with STOP GRADIENT,
        not a separate EMA encoder.
        """
        B = x.shape[0]
        device = x.device

        # Generate mask
        mask_indices, context_indices = self._generate_mask(B, device)

        # Get ALL patch embeddings from encoder (for both context and targets)
        all_embeds = self.encoder(x, return_all_tokens=True)[:, 1:]  # Remove CLS, (B, N, D)

        # Apply SIGReg on all encoder outputs (before masking)
        # This encourages the ENCODER to produce isotropic Gaussian embeddings
        sigreg_loss_val = self.sigreg(all_embeds)

        # Targets: stop-gradient on all encoder outputs (no EMA!)
        target_embeds = all_embeds.detach()

        # Gather targets for masked positions
        targets = torch.gather(
            target_embeds, 1,
            mask_indices.unsqueeze(-1).expand(-1, -1, all_embeds.shape[-1])
        )

        # Context: gather visible patches
        context_embeds = torch.gather(
            all_embeds, 1,
            context_indices.unsqueeze(-1).expand(-1, -1, all_embeds.shape[-1])
        )

        # Predict masked patches
        predictions = self.predictor(context_embeds, context_indices, mask_indices)

        # Prediction loss
        pred_loss = self._compute_prediction_loss(predictions, targets)

        # Optional: prediction variance regularization
        var_loss = 0.0
        if self.var_reg_lambda > 0:
            var_loss = self.var_reg_lambda * prediction_var_loss(predictions, threshold=0.1)

        # Total loss
        loss = pred_loss + self.sigreg_coeff * sigreg_loss_val + var_loss

        return loss, predictions, targets

    def get_embeddings(self, x: torch.Tensor, pool: str = 'mean') -> torch.Tensor:
        """Get embeddings for downstream tasks (same as V2)."""
        if pool == 'cls':
            return self.encoder(x, return_all_tokens=False)
        else:
            all_tokens = self.encoder(x, return_all_tokens=True)
            return all_tokens[:, 1:].mean(dim=1)

    def train_step(self, x: torch.Tensor) -> torch.Tensor:
        loss, _, _ = self.forward(x)
        return loss

    def update_ema(self):
        """No-op: V3 doesn't use EMA."""
        pass
