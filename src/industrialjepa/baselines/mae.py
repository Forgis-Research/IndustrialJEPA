# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Masked AutoEncoder (MAE) baseline for industrial time series.

Unlike standard MAE that reconstructs masked input, this variant:
1. Masks random patches of setpoint signals
2. Predicts corresponding effort signals (cross-modal prediction)

This tests whether learning to predict effort from masked setpoint
captures the same physics-based relationships as JEPA.

Reference: He et al., "Masked Autoencoders Are Scalable Vision Learners" (2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .base import BaselineConfig, BaselineModel, PatchEmbedding, TransformerEncoder


@dataclass
class MAEConfig(BaselineConfig):
    """Configuration for MAE model."""

    # MAE-specific
    mask_ratio: float = 0.75  # Fraction of patches to mask
    decoder_hidden_dim: int = 128  # Smaller decoder
    decoder_num_layers: int = 2
    decoder_num_heads: int = 4


class MAE(BaselineModel):
    """
    Masked AutoEncoder for Setpoint->Effort prediction.

    Architecture:
    1. Patch embedding of setpoint
    2. Random masking of patches
    3. Encoder processes only visible patches
    4. Decoder predicts effort for ALL patches (including masked)
    5. Loss computed on masked patches only

    For anomaly detection:
    - Use full reconstruction error (no masking)
    - Higher error = more anomalous
    """

    def __init__(self, config: MAEConfig):
        super().__init__(config)
        self.config = config
        self.mask_ratio = config.mask_ratio

        # Decoder patch embedding (effort)
        self.effort_patch_embed = PatchEmbedding(
            input_dim=config.effort_dim,
            hidden_dim=config.hidden_dim,
            patch_size=config.patch_size,
            seq_len=config.seq_len,
        )

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)

        # Decoder
        self.decoder_embed = nn.Linear(config.hidden_dim, config.decoder_hidden_dim)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.decoder_hidden_dim,
            nhead=config.decoder_num_heads,
            dim_feedforward=config.decoder_hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, config.decoder_num_layers)
        self.decoder_norm = nn.LayerNorm(config.decoder_hidden_dim)

        # Decoder position embeddings (for full sequence including masked)
        num_patches = config.seq_len // config.patch_size
        self.decoder_pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, config.decoder_hidden_dim) * 0.02
        )

        # Prediction head: predict effort patches
        patch_effort_dim = config.effort_dim * config.patch_size
        self.effort_head = nn.Linear(config.decoder_hidden_dim, patch_effort_dim)

    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Random masking of patches.

        Args:
            x: [B, L, D] patch embeddings (excluding CLS)
            mask_ratio: Fraction to mask

        Returns:
            x_masked: [B, L_vis, D] visible patches only
            mask: [B, L] boolean mask (True = masked)
            ids_restore: [B, L] indices to restore original order
        """
        B, L, D = x.shape
        num_keep = int(L * (1 - mask_ratio))

        # Random shuffle indices
        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep first num_keep
        ids_keep = ids_shuffle[:, :num_keep]

        # Gather visible patches
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )

        # Create binary mask (True = masked)
        mask = torch.ones(B, L, device=x.device, dtype=torch.bool)
        mask.scatter_(1, ids_keep, False)

        return x_masked, mask, ids_restore

    def forward_encoder(
        self,
        setpoint: torch.Tensor,
        setpoint_mask: Optional[torch.Tensor] = None,
        apply_masking: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode setpoint with random masking.

        Args:
            setpoint: [B, T, 14] setpoint signals
            setpoint_mask: [B, 14] validity mask
            apply_masking: Whether to apply random masking

        Returns:
            latent: [B, 1 + L_vis, D] encoded visible patches (with CLS)
            mask: [B, L] patch mask
            ids_restore: [B, L] restore indices
        """
        # Patch embedding
        x, num_patches = self.patch_embed(setpoint, setpoint_mask)
        # x: [B, 1 + num_patches, D] (with CLS token)

        # Separate CLS and patches
        cls_token = x[:, :1]  # [B, 1, D]
        patches = x[:, 1:]  # [B, num_patches, D]

        if apply_masking:
            # Random masking
            patches_masked, mask, ids_restore = self.random_masking(
                patches, self.mask_ratio
            )
        else:
            # No masking (for inference)
            patches_masked = patches
            B, L, D = patches.shape
            mask = torch.zeros(B, L, device=patches.device, dtype=torch.bool)
            ids_restore = torch.arange(L, device=patches.device).unsqueeze(0).expand(B, -1)

        # Add CLS back
        x_vis = torch.cat([cls_token, patches_masked], dim=1)

        # Encode
        latent = self.encoder(x_vis)

        return latent, mask, ids_restore

    def forward_decoder(
        self,
        latent: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode to predict effort patches.

        Args:
            latent: [B, 1 + L_vis, D] encoded visible patches
            ids_restore: [B, L] indices to restore order

        Returns:
            effort_pred: [B, num_patches, patch_size * effort_dim]
        """
        B = latent.shape[0]
        num_patches = self.config.seq_len // self.config.patch_size

        # Project to decoder dim
        x = self.decoder_embed(latent)  # [B, 1 + L_vis, D_dec]

        # Separate CLS and visible patches
        cls_token = x[:, :1]
        vis_patches = x[:, 1:]
        L_vis = vis_patches.shape[1]
        L_mask = num_patches - L_vis

        # Create mask tokens
        mask_tokens = self.mask_token.expand(B, L_mask, -1)
        mask_tokens = self.decoder_embed(
            self.mask_token.expand(B, L_mask, self.config.hidden_dim)
        )[:, :, :self.config.decoder_hidden_dim]

        # Unshuffle to original order
        # First, concatenate visible and mask tokens
        x_full = torch.cat([vis_patches, mask_tokens], dim=1)  # [B, L, D_dec]

        # Gather to restore order
        x_full = torch.gather(
            x_full,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, self.config.decoder_hidden_dim),
        )

        # Add CLS back and positional embedding
        x_full = torch.cat([cls_token, x_full], dim=1)
        x_full = x_full + self.decoder_pos_embed

        # Decode
        x_full = self.decoder(x_full)
        x_full = self.decoder_norm(x_full)

        # Predict effort (skip CLS)
        effort_pred = self.effort_head(x_full[:, 1:])  # [B, L, patch_size * effort_dim]

        return effort_pred

    def forward(
        self,
        setpoint: torch.Tensor,
        effort: torch.Tensor,
        setpoint_mask: Optional[torch.Tensor] = None,
        effort_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.

        Args:
            setpoint: [B, T, 14] setpoint signals
            effort: [B, T, 7] effort signals (target)
            setpoint_mask: [B, 14] validity mask for setpoint
            effort_mask: [B, 7] validity mask for effort

        Returns:
            Dict with 'loss', 'loss_masked', 'loss_full'
        """
        B, T, _ = setpoint.shape

        # Encode with masking
        latent, mask, ids_restore = self.forward_encoder(
            setpoint, setpoint_mask, apply_masking=True
        )

        # Decode
        effort_pred = self.forward_decoder(latent, ids_restore)
        # effort_pred: [B, num_patches, patch_size * effort_dim]

        # Reshape target effort to patches
        num_patches = T // self.config.patch_size
        effort_target = effort.reshape(
            B, num_patches, self.config.patch_size * self.config.effort_dim
        )

        # Apply effort mask if provided
        if effort_mask is not None:
            # Expand mask to patch dimension
            effort_mask_expanded = effort_mask.unsqueeze(1).unsqueeze(1).expand(
                B, num_patches, self.config.patch_size, -1
            ).reshape(B, num_patches, -1)

            effort_pred = effort_pred * effort_mask_expanded
            effort_target = effort_target * effort_mask_expanded

        # Compute MSE loss on masked patches only
        loss_per_patch = F.mse_loss(effort_pred, effort_target, reduction='none')
        loss_per_patch = loss_per_patch.mean(dim=-1)  # [B, num_patches]

        # Masked loss (only on masked patches)
        if mask.any():
            loss_masked = loss_per_patch[mask].mean()
        else:
            loss_masked = loss_per_patch.mean()

        # Full loss (for reference)
        loss_full = loss_per_patch.mean()

        return {
            "loss": loss_masked,  # Train on masked only (like MAE)
            "loss_masked": loss_masked,
            "loss_full": loss_full,
        }

    def encode(
        self,
        setpoint: torch.Tensor,
        setpoint_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract representations (CLS token).

        Args:
            setpoint: [B, T, 14] setpoint signals
            setpoint_mask: [B, 14] validity mask

        Returns:
            [B, hidden_dim] CLS token representation
        """
        latent, _, _ = self.forward_encoder(
            setpoint, setpoint_mask, apply_masking=False
        )
        return latent[:, 0]  # CLS token

    @torch.no_grad()
    def compute_anomaly_score(
        self,
        setpoint: torch.Tensor,
        effort: torch.Tensor,
        setpoint_mask: Optional[torch.Tensor] = None,
        effort_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute anomaly scores (reconstruction error).

        Args:
            setpoint: [B, T, 14] setpoint signals
            effort: [B, T, 7] effort signals
            setpoint_mask: [B, 14] validity mask
            effort_mask: [B, 7] validity mask

        Returns:
            [B] anomaly scores (higher = more anomalous)
        """
        self.eval()
        B, T, _ = setpoint.shape

        # Encode without masking
        latent, mask, ids_restore = self.forward_encoder(
            setpoint, setpoint_mask, apply_masking=False
        )

        # Decode
        effort_pred = self.forward_decoder(latent, ids_restore)

        # Reshape target
        num_patches = T // self.config.patch_size
        effort_target = effort.reshape(
            B, num_patches, self.config.patch_size * self.config.effort_dim
        )

        # Apply mask if provided
        if effort_mask is not None:
            effort_mask_expanded = effort_mask.unsqueeze(1).unsqueeze(1).expand(
                B, num_patches, self.config.patch_size, -1
            ).reshape(B, num_patches, -1)

            effort_pred = effort_pred * effort_mask_expanded
            effort_target = effort_target * effort_mask_expanded

            # Compute MSE per sample, normalized by valid dimensions
            diff = (effort_pred - effort_target) ** 2
            valid_count = effort_mask_expanded.sum(dim=(1, 2)).clamp(min=1)
            scores = diff.sum(dim=(1, 2)) / valid_count
        else:
            # Simple MSE per sample
            scores = F.mse_loss(
                effort_pred, effort_target, reduction='none'
            ).mean(dim=(1, 2))

        return scores
