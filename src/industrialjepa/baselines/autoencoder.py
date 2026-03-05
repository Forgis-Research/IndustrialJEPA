# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Autoencoder baseline for industrial time series.

Simple encoder-decoder architecture:
1. Encode setpoint signals to latent representation
2. Decode to predict effort signals

This is the simplest baseline - direct supervised prediction of
Effort from Setpoint without any self-supervised pretraining.

For anomaly detection:
- Train only on healthy data
- High reconstruction error = anomaly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional

from .base import BaselineConfig, BaselineModel


@dataclass
class AutoencoderConfig(BaselineConfig):
    """Configuration for Autoencoder model."""

    # Decoder architecture
    decoder_hidden_dim: int = 256
    decoder_num_layers: int = 4
    decoder_num_heads: int = 8

    # Bottleneck
    latent_dim: int = 64  # Compressed representation
    use_bottleneck: bool = True


class Autoencoder(BaselineModel):
    """
    Simple Autoencoder for Setpoint->Effort prediction.

    Architecture:
    1. Patch embedding of setpoint
    2. Transformer encoder
    3. Optional bottleneck (compressed representation)
    4. Transformer decoder
    5. Effort prediction head

    Unlike MAE, this processes all patches (no masking) and
    learns a direct mapping from setpoint to effort.
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

        # Project from encoder to decoder if dimensions differ
        if config.hidden_dim != config.decoder_hidden_dim:
            self.enc_to_dec = nn.Linear(config.hidden_dim, config.decoder_hidden_dim)
        else:
            self.enc_to_dec = nn.Identity()

        # Effort prediction head (predicts per-timestep effort)
        self.effort_head = nn.Sequential(
            nn.Linear(config.decoder_hidden_dim, config.decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(config.decoder_hidden_dim, config.effort_dim * config.patch_size),
        )

    def forward_encoder(
        self,
        setpoint: torch.Tensor,
        setpoint_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode setpoint to latent representation.

        Args:
            setpoint: [B, T, 14] setpoint signals
            setpoint_mask: [B, 14] validity mask

        Returns:
            [B, 1 + num_patches, hidden_dim] encoded representation
        """
        # Patch embedding
        x, _ = self.patch_embed(setpoint, setpoint_mask)
        # x: [B, 1 + num_patches, hidden_dim]

        # Encode
        x = self.encoder(x)

        # Bottleneck (applied to all tokens)
        x = self.bottleneck(x)

        return x

    def forward_decoder(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode latent to effort prediction.

        Args:
            latent: [B, 1 + num_patches, hidden_dim] encoded representation

        Returns:
            [B, T, effort_dim] predicted effort
        """
        B = latent.shape[0]

        # Project to decoder dimension
        x = self.enc_to_dec(latent)

        # Decode (skip CLS for prediction)
        x = self.decoder(x)
        x = self.decoder_norm(x)

        # Predict effort (skip CLS token)
        effort_patches = self.effort_head(x[:, 1:])
        # effort_patches: [B, num_patches, patch_size * effort_dim]

        # Reshape to [B, T, effort_dim]
        num_patches = effort_patches.shape[1]
        effort = effort_patches.reshape(
            B, num_patches * self.config.patch_size, self.config.effort_dim
        )

        return effort

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
            Dict with 'loss', 'mse', 'mae'
        """
        # Encode
        latent = self.forward_encoder(setpoint, setpoint_mask)

        # Decode
        effort_pred = self.forward_decoder(latent)

        # Apply effort mask if provided
        if effort_mask is not None:
            effort_mask_expanded = effort_mask.unsqueeze(1)  # [B, 1, 7]
            effort_pred = effort_pred * effort_mask_expanded
            effort = effort * effort_mask_expanded

            # Compute loss only on valid dimensions
            diff = (effort_pred - effort) ** 2
            valid_count = effort_mask.sum(dim=-1, keepdim=True).unsqueeze(1)  # [B, 1, 1]
            valid_count = valid_count.expand_as(diff).clamp(min=1)
            mse = (diff / valid_count).sum() / diff.numel() * self.config.effort_dim
        else:
            mse = F.mse_loss(effort_pred, effort)

        # MAE for reference
        mae = F.l1_loss(effort_pred, effort)

        return {
            "loss": mse,
            "mse": mse,
            "mae": mae,
        }

    def encode(
        self,
        setpoint: torch.Tensor,
        setpoint_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract representations (CLS token after bottleneck).

        Args:
            setpoint: [B, T, 14] setpoint signals
            setpoint_mask: [B, 14] validity mask

        Returns:
            [B, hidden_dim] CLS token representation
        """
        latent = self.forward_encoder(setpoint, setpoint_mask)
        return latent[:, 0]  # CLS token

    def get_latent(
        self,
        setpoint: torch.Tensor,
        setpoint_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get bottleneck representation (if using bottleneck).

        Args:
            setpoint: [B, T, 14] setpoint signals
            setpoint_mask: [B, 14] validity mask

        Returns:
            [B, 1 + num_patches, latent_dim] or [B, 1 + num_patches, hidden_dim]
        """
        # Patch embedding
        x, _ = self.patch_embed(setpoint, setpoint_mask)

        # Encode
        x = self.encoder(x)

        # Get pre-projection representation
        if self.config.use_bottleneck:
            # Apply only the first linear layer of bottleneck
            return self.bottleneck[0](x)  # [B, L, latent_dim]
        else:
            return x

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

        # Encode and decode
        latent = self.forward_encoder(setpoint, setpoint_mask)
        effort_pred = self.forward_decoder(latent)

        # Apply mask if provided
        if effort_mask is not None:
            effort_mask_expanded = effort_mask.unsqueeze(1)
            effort_pred = effort_pred * effort_mask_expanded
            effort = effort * effort_mask_expanded

            # MSE per sample, normalized by valid dimensions
            diff = (effort_pred - effort) ** 2
            valid_count = effort_mask.sum(dim=-1) * effort.shape[1]  # [B]
            scores = diff.sum(dim=(1, 2)) / valid_count.clamp(min=1)
        else:
            # Simple MSE per sample
            scores = F.mse_loss(
                effort_pred, effort, reduction='none'
            ).mean(dim=(1, 2))

        return scores


class VariationalAutoencoder(Autoencoder):
    """
    Variational Autoencoder variant with KL regularization.

    Adds stochastic latent representation with:
    - Gaussian encoder (predicts mean and log-variance)
    - Reparameterization trick
    - KL divergence regularization

    This encourages smoother latent space which may improve
    anomaly detection generalization.
    """

    def __init__(self, config: AutoencoderConfig):
        super().__init__(config)

        # Replace bottleneck with VAE components
        if config.use_bottleneck:
            self.mu_proj = nn.Linear(config.hidden_dim, config.latent_dim)
            self.logvar_proj = nn.Linear(config.hidden_dim, config.latent_dim)
            self.latent_to_hidden = nn.Linear(config.latent_dim, config.hidden_dim)

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  # Use mean at inference

    def forward_encoder(
        self,
        setpoint: torch.Tensor,
        setpoint_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Encode setpoint to latent distribution.

        Returns:
            latent: [B, 1 + num_patches, hidden_dim]
            mu: [B, 1 + num_patches, latent_dim]
            logvar: [B, 1 + num_patches, latent_dim]
        """
        # Patch embedding
        x, _ = self.patch_embed(setpoint, setpoint_mask)

        # Encode
        x = self.encoder(x)

        if self.config.use_bottleneck:
            # Predict distribution parameters
            mu = self.mu_proj(x)
            logvar = self.logvar_proj(x)

            # Sample latent
            z = self.reparameterize(mu, logvar)

            # Project back to hidden dim
            x = self.latent_to_hidden(z)

            return x, mu, logvar
        else:
            return x, None, None

    def forward(
        self,
        setpoint: torch.Tensor,
        effort: torch.Tensor,
        setpoint_mask: Optional[torch.Tensor] = None,
        effort_mask: Optional[torch.Tensor] = None,
        kl_weight: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass with KL regularization.

        Args:
            setpoint: [B, T, 14] setpoint signals
            effort: [B, T, 7] effort signals (target)
            setpoint_mask: [B, 14] validity mask for setpoint
            effort_mask: [B, 7] validity mask for effort
            kl_weight: Weight for KL divergence term

        Returns:
            Dict with 'loss', 'recon_loss', 'kl_loss'
        """
        # Encode
        latent, mu, logvar = self.forward_encoder(setpoint, setpoint_mask)

        # Decode
        effort_pred = self.forward_decoder(latent)

        # Reconstruction loss
        if effort_mask is not None:
            effort_mask_expanded = effort_mask.unsqueeze(1)
            effort_pred = effort_pred * effort_mask_expanded
            effort = effort * effort_mask_expanded

        recon_loss = F.mse_loss(effort_pred, effort)

        # KL divergence
        if mu is not None and logvar is not None:
            # KL(q(z|x) || p(z)) where p(z) = N(0, I)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            kl_loss = torch.tensor(0.0, device=setpoint.device)

        # Total loss (ELBO)
        loss = recon_loss + kl_weight * kl_loss

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }
