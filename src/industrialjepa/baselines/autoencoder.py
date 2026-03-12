# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Autoencoder baselines for industrial time series.

Two variants to test different hypotheses:

1. EffortAutoencoder: Encode/decode EFFORT only (no setpoint)
   - Tests: Can effort patterns alone detect anomalies?
   - If JEPA >> EffortAE, proves setpoint (causal input) is essential

2. SetpointToEffort: Direct supervised prediction
   - Tests: Does latent space prediction (JEPA) beat raw value prediction?
   - If JEPA >> S2E, proves latent space filtering helps

The key insight from the execution plan:
- JEPA predicts in LATENT space (filters noise)
- MAE predicts in RAW space (wastes capacity on noise)
- EffortAE has NO causal input (tests if setpoint matters)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional

from .base import BaselineConfig, BaselineModel, PatchEmbedding, TransformerEncoder


@dataclass
class AutoencoderConfig(BaselineConfig):
    """Configuration for Autoencoder models."""

    # Decoder architecture
    decoder_hidden_dim: int = 256
    decoder_num_layers: int = 4
    decoder_num_heads: int = 8

    # Bottleneck
    latent_dim: int = 64
    use_bottleneck: bool = True


class EffortAutoencoder(BaselineModel):
    """
    Effort-only Autoencoder (NO setpoint input).

    This baseline tests whether the causal structure matters:
    - Encodes ONLY effort signals
    - Reconstructs effort from its own encoding
    - Does NOT see setpoint (the causal input)

    If JEPA (setpoint→effort) beats this, it proves:
    - The setpoint→effort causal structure is essential
    - You can't just learn effort patterns in isolation

    For anomaly detection:
    - Train on healthy effort patterns
    - Anomalies have high reconstruction error
    """

    def __init__(self, config: AutoencoderConfig):
        # Don't call BaselineModel.__init__ since we need different patch embedding
        nn.Module.__init__(self)
        self.config = config

        # Effort patch embedding (NOT setpoint!)
        self.patch_embed = PatchEmbedding(
            input_dim=config.effort_dim,  # 13 (7 joint + 6 Cartesian), not 14
            hidden_dim=config.hidden_dim,
            patch_size=config.patch_size,
            seq_len=config.seq_len,
        )

        # Encoder
        self.encoder = TransformerEncoder(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )

        # Bottleneck (optional)
        if config.use_bottleneck:
            self.bottleneck = nn.Sequential(
                nn.Linear(config.hidden_dim, config.latent_dim),
                nn.GELU(),
                nn.Linear(config.latent_dim, config.hidden_dim),
            )
        else:
            self.bottleneck = nn.Identity()

        # Decoder
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

        # Project encoder→decoder if dims differ
        if config.hidden_dim != config.decoder_hidden_dim:
            self.enc_to_dec = nn.Linear(config.hidden_dim, config.decoder_hidden_dim)
        else:
            self.enc_to_dec = nn.Identity()

        # Reconstruction head (predict effort patches)
        self.recon_head = nn.Sequential(
            nn.Linear(config.decoder_hidden_dim, config.decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(config.decoder_hidden_dim, config.effort_dim * config.patch_size),
        )

    def forward(
        self,
        setpoint: torch.Tensor,  # Ignored! Only here for API compatibility
        effort: torch.Tensor,
        setpoint_mask: Optional[torch.Tensor] = None,
        effort_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.

        NOTE: setpoint is IGNORED - this model only sees effort.
        """
        B, T, _ = effort.shape

        # Encode effort (NOT setpoint!)
        x, _ = self.patch_embed(effort, effort_mask)
        x = self.encoder(x)
        x = self.bottleneck(x)

        # Decode
        x = self.enc_to_dec(x)
        x = self.decoder(x)
        x = self.decoder_norm(x)

        # Reconstruct effort (skip CLS token)
        effort_pred = self.recon_head(x[:, 1:])

        # Reshape to [B, T, effort_dim]
        num_patches = effort_pred.shape[1]
        effort_pred = effort_pred.reshape(B, num_patches * self.config.patch_size, self.config.effort_dim)

        # Reshape target
        effort_target = effort[:, :effort_pred.shape[1], :]

        # Apply mask if provided
        if effort_mask is not None:
            effort_mask_expanded = effort_mask.unsqueeze(1)
            effort_pred = effort_pred * effort_mask_expanded
            effort_target = effort_target * effort_mask_expanded

        # Reconstruction loss
        mse = F.mse_loss(effort_pred, effort_target)

        return {
            "loss": mse,
            "mse": mse,
        }

    def encode(
        self,
        setpoint: torch.Tensor,  # Ignored
        setpoint_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """This model doesn't use setpoint - raises error."""
        raise NotImplementedError(
            "EffortAutoencoder doesn't encode setpoint. Use encode_effort() instead."
        )

    def encode_effort(
        self,
        effort: torch.Tensor,
        effort_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode effort to representation."""
        x, _ = self.patch_embed(effort, effort_mask)
        x = self.encoder(x)
        x = self.bottleneck(x)
        return x[:, 0]  # CLS token

    @torch.no_grad()
    def compute_anomaly_score(
        self,
        setpoint: torch.Tensor,  # Ignored
        effort: torch.Tensor,
        setpoint_mask: Optional[torch.Tensor] = None,
        effort_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute anomaly scores (reconstruction error)."""
        self.eval()
        output = self.forward(setpoint, effort, setpoint_mask, effort_mask)
        # Return per-sample loss
        B, T, _ = effort.shape

        # Re-compute per sample
        x, _ = self.patch_embed(effort, effort_mask)
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.enc_to_dec(x)
        x = self.decoder(x)
        x = self.decoder_norm(x)
        effort_pred = self.recon_head(x[:, 1:])
        num_patches = effort_pred.shape[1]
        effort_pred = effort_pred.reshape(B, num_patches * self.config.patch_size, self.config.effort_dim)
        effort_target = effort[:, :effort_pred.shape[1], :]

        if effort_mask is not None:
            effort_mask_expanded = effort_mask.unsqueeze(1)
            effort_pred = effort_pred * effort_mask_expanded
            effort_target = effort_target * effort_mask_expanded
            diff = (effort_pred - effort_target) ** 2
            valid_count = effort_mask.sum(dim=-1) * effort_pred.shape[1]
            scores = diff.sum(dim=(1, 2)) / valid_count.clamp(min=1)
        else:
            scores = F.mse_loss(effort_pred, effort_target, reduction='none').mean(dim=(1, 2))

        return scores

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SetpointToEffort(BaselineModel):
    """
    Direct Setpoint→Effort prediction (supervised baseline).

    This is the simplest causal model:
    - Encodes setpoint
    - Directly predicts raw effort values
    - No masking, no latent space prediction

    Compared to JEPA:
    - Same causal structure (setpoint→effort)
    - But predicts RAW values, not latent embeddings
    - If JEPA >> S2E, proves latent space helps

    This is essentially what MAE does without masking.
    """

    def __init__(self, config: AutoencoderConfig):
        super().__init__(config)
        self.config = config

        # Bottleneck (optional)
        if config.use_bottleneck:
            self.bottleneck = nn.Sequential(
                nn.Linear(config.hidden_dim, config.latent_dim),
                nn.GELU(),
                nn.Linear(config.latent_dim, config.hidden_dim),
            )
        else:
            self.bottleneck = nn.Identity()

        # Decoder
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

        # Project encoder→decoder if dims differ
        if config.hidden_dim != config.decoder_hidden_dim:
            self.enc_to_dec = nn.Linear(config.hidden_dim, config.decoder_hidden_dim)
        else:
            self.enc_to_dec = nn.Identity()

        # Effort prediction head
        self.effort_head = nn.Sequential(
            nn.Linear(config.decoder_hidden_dim, config.decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(config.decoder_hidden_dim, config.effort_dim * config.patch_size),
        )

    def forward(
        self,
        setpoint: torch.Tensor,
        effort: torch.Tensor,
        setpoint_mask: Optional[torch.Tensor] = None,
        effort_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass: encode setpoint, predict effort."""
        B, T, _ = setpoint.shape

        # Encode setpoint
        x, _ = self.patch_embed(setpoint, setpoint_mask)
        x = self.encoder(x)
        x = self.bottleneck(x)

        # Decode
        x = self.enc_to_dec(x)
        x = self.decoder(x)
        x = self.decoder_norm(x)

        # Predict effort (skip CLS)
        effort_pred = self.effort_head(x[:, 1:])
        num_patches = effort_pred.shape[1]
        effort_pred = effort_pred.reshape(B, num_patches * self.config.patch_size, self.config.effort_dim)

        # Align target length
        effort_target = effort[:, :effort_pred.shape[1], :]

        # Apply mask if provided
        if effort_mask is not None:
            effort_mask_expanded = effort_mask.unsqueeze(1)
            effort_pred = effort_pred * effort_mask_expanded
            effort_target = effort_target * effort_mask_expanded

        mse = F.mse_loss(effort_pred, effort_target)

        return {
            "loss": mse,
            "mse": mse,
        }

    def encode(
        self,
        setpoint: torch.Tensor,
        setpoint_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract representations (CLS token)."""
        x, _ = self.patch_embed(setpoint, setpoint_mask)
        x = self.encoder(x)
        x = self.bottleneck(x)
        return x[:, 0]

    @torch.no_grad()
    def compute_anomaly_score(
        self,
        setpoint: torch.Tensor,
        effort: torch.Tensor,
        setpoint_mask: Optional[torch.Tensor] = None,
        effort_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute anomaly scores (prediction error)."""
        self.eval()
        B, T, _ = setpoint.shape

        # Encode setpoint, predict effort
        x, _ = self.patch_embed(setpoint, setpoint_mask)
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.enc_to_dec(x)
        x = self.decoder(x)
        x = self.decoder_norm(x)
        effort_pred = self.effort_head(x[:, 1:])
        num_patches = effort_pred.shape[1]
        effort_pred = effort_pred.reshape(B, num_patches * self.config.patch_size, self.config.effort_dim)
        effort_target = effort[:, :effort_pred.shape[1], :]

        if effort_mask is not None:
            effort_mask_expanded = effort_mask.unsqueeze(1)
            effort_pred = effort_pred * effort_mask_expanded
            effort_target = effort_target * effort_mask_expanded
            diff = (effort_pred - effort_target) ** 2
            valid_count = effort_mask.sum(dim=-1) * effort_pred.shape[1]
            scores = diff.sum(dim=(1, 2)) / valid_count.clamp(min=1)
        else:
            scores = F.mse_loss(effort_pred, effort_target, reduction='none').mean(dim=(1, 2))

        return scores


# Aliases for backwards compatibility
Autoencoder = EffortAutoencoder  # The "Autoencoder" in execution plan is effort-only
