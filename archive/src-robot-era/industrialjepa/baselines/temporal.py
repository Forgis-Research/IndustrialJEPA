# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Temporal Self-Prediction model for industrial time series anomaly detection.

Unlike static Setpoint→Effort prediction, this model predicts FUTURE effort
from PAST context (both setpoint and effort). This addresses the fundamental
problem that setpoint alone cannot predict contact forces.

Key insight: Even if absolute force values vary (missing screw = lower force),
temporal dynamics are disrupted. Normal screwdriving has a characteristic
force profile: approach → contact → resistance increase → plateau.
Anomalies disrupt this pattern.

Architecture:
    Input:  [setpoint(t-k:t), effort(t-k:t)]  (context window)
    Output: effort(t+1:t+n)                    (future prediction)

Two modes:
1. Direct: Predict raw future effort values (simpler, good baseline)
2. JEPA:   Predict future effort embeddings using EMA target encoder
           (learns abstract representations, better for transfer)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

from .base import BaselineConfig, BaselineModel, PatchEmbedding, TransformerEncoder


@dataclass
class TemporalConfig(BaselineConfig):
    """Configuration for Temporal Self-Prediction model."""

    # Context/target split
    context_ratio: float = 0.5  # First 50% is context, rest is target

    # Prediction mode
    prediction_mode: Literal["direct", "jepa"] = "jepa"

    # JEPA-specific parameters
    ema_decay: float = 0.996  # EMA decay for target encoder
    predictor_hidden_dim: int = 256  # Predictor MLP hidden dim

    # Direct prediction decoder
    decoder_hidden_dim: int = 128
    decoder_num_layers: int = 2
    decoder_num_heads: int = 4


class TemporalPredictor(BaselineModel):
    """
    Temporal Self-Prediction for anomaly detection.

    Predicts future effort from past context (setpoint + effort).
    Anomalies cause higher prediction error because they disrupt
    the expected temporal dynamics.

    Key differences from static prediction:
    1. Uses effort history (not just setpoint)
    2. Predicts future (temporal structure)
    3. Anomalies disrupt dynamics, not just magnitude

    For anomaly detection:
    - Train on normal data only
    - High prediction error = temporal dynamics disrupted = anomaly
    """

    def __init__(self, config: TemporalConfig):
        # Don't call parent __init__ - we need custom patch embedding
        nn.Module.__init__(self)
        self.config = config

        # Combined input dimension: setpoint + effort
        combined_dim = config.setpoint_dim + config.effort_dim  # 14 + 13 = 27

        # Context length (in timesteps)
        self.context_len = int(config.seq_len * config.context_ratio)
        self.target_len = config.seq_len - self.context_len

        # Context encoder: processes [setpoint, effort] jointly
        self.context_patch_embed = PatchEmbedding(
            input_dim=combined_dim,
            hidden_dim=config.hidden_dim,
            patch_size=config.patch_size,
            seq_len=self.context_len,
        )

        self.context_encoder = TransformerEncoder(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )

        if config.prediction_mode == "jepa":
            # JEPA-style: predict embeddings, not raw values
            self._init_jepa_components(config)
        else:
            # Direct: predict raw effort values
            self._init_direct_components(config)

    def _init_jepa_components(self, config: TemporalConfig):
        """Initialize JEPA-style components with EMA target encoder."""

        # Target encoder (for future effort) - will be EMA updated
        self.target_patch_embed = PatchEmbedding(
            input_dim=config.effort_dim,
            hidden_dim=config.hidden_dim,
            patch_size=config.patch_size,
            seq_len=self.target_len,
        )

        self.target_encoder = TransformerEncoder(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )

        # Predictor MLP: context embedding → target embedding
        # The predictor is asymmetric (smaller) to prevent collapse
        self.predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.predictor_hidden_dim),
            nn.GELU(),
            nn.Linear(config.predictor_hidden_dim, config.predictor_hidden_dim),
            nn.GELU(),
            nn.Linear(config.predictor_hidden_dim, config.hidden_dim),
        )

        # Also need per-patch prediction for temporal alignment
        # Maps context patches to target patch predictions
        num_context_patches = self.context_len // config.patch_size
        num_target_patches = self.target_len // config.patch_size

        self.temporal_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim * num_context_patches, config.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim * num_target_patches),
        )
        self.num_context_patches = num_context_patches
        self.num_target_patches = num_target_patches

        # Initialize EMA parameters (copy of target encoder)
        self.ema_target_patch_embed = copy.deepcopy(self.target_patch_embed)
        self.ema_target_encoder = copy.deepcopy(self.target_encoder)

        # Freeze EMA parameters (updated via EMA, not gradients)
        for param in self.ema_target_patch_embed.parameters():
            param.requires_grad = False
        for param in self.ema_target_encoder.parameters():
            param.requires_grad = False

    def _init_direct_components(self, config: TemporalConfig):
        """Initialize direct prediction components (predict raw effort)."""

        # Decoder for predicting future effort
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

        # Query embeddings for future timesteps
        num_target_patches = self.target_len // config.patch_size
        self.target_queries = nn.Parameter(
            torch.randn(1, num_target_patches, config.decoder_hidden_dim) * 0.02
        )

        # Prediction head: output raw effort patches
        patch_effort_dim = config.effort_dim * config.patch_size
        self.effort_head = nn.Linear(config.decoder_hidden_dim, patch_effort_dim)

    @torch.no_grad()
    def update_ema(self):
        """Update EMA target encoder parameters."""
        if self.config.prediction_mode != "jepa":
            return

        decay = self.config.ema_decay

        # Update patch embedding
        for ema_param, param in zip(
            self.ema_target_patch_embed.parameters(),
            self.target_patch_embed.parameters()
        ):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

        # Update encoder
        for ema_param, param in zip(
            self.ema_target_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

    def _encode_context(
        self,
        setpoint: torch.Tensor,
        effort: torch.Tensor,
        setpoint_mask: Optional[torch.Tensor] = None,
        effort_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode context window (past setpoint + effort).

        Args:
            setpoint: [B, T, 14] full setpoint sequence
            effort: [B, T, 13] full effort sequence
            *_mask: validity masks

        Returns:
            context_repr: [B, hidden_dim] or [B, num_patches, hidden_dim]
        """
        # Split into context portion
        context_setpoint = setpoint[:, :self.context_len]  # [B, context_len, 14]
        context_effort = effort[:, :self.context_len]  # [B, context_len, 13]

        # Concatenate setpoint and effort along feature dimension
        context_combined = torch.cat([context_setpoint, context_effort], dim=-1)
        # Shape: [B, context_len, 27]

        # Apply masks if provided
        if setpoint_mask is not None and effort_mask is not None:
            combined_mask = torch.cat([setpoint_mask, effort_mask], dim=-1)
        else:
            combined_mask = None

        # Patch embedding
        patches, num_patches = self.context_patch_embed(context_combined, combined_mask)
        # patches: [B, 1 + num_patches, hidden_dim]

        # Encode
        encoded = self.context_encoder(patches)
        # [B, 1 + num_patches, hidden_dim]

        return encoded

    def _encode_target_ema(
        self,
        effort: torch.Tensor,
        effort_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode target effort using EMA encoder (no gradients).

        Args:
            effort: [B, T, 13] full effort sequence
            effort_mask: validity mask

        Returns:
            target_repr: [B, num_patches, hidden_dim]
        """
        # Extract target portion
        target_effort = effort[:, self.context_len:]  # [B, target_len, 13]

        # Patch embedding (EMA version)
        patches, _ = self.ema_target_patch_embed(target_effort, effort_mask)

        # Encode (EMA version)
        encoded = self.ema_target_encoder(patches)

        # Return patch representations (skip CLS)
        return encoded[:, 1:]  # [B, num_patches, hidden_dim]

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
            effort: [B, T, 13] effort signals
            setpoint_mask: [B, 14] validity mask for setpoint
            effort_mask: [B, 13] validity mask for effort

        Returns:
            Dict with 'loss' and metrics
        """
        if self.config.prediction_mode == "jepa":
            return self._forward_jepa(setpoint, effort, setpoint_mask, effort_mask)
        else:
            return self._forward_direct(setpoint, effort, setpoint_mask, effort_mask)

    def _forward_jepa(
        self,
        setpoint: torch.Tensor,
        effort: torch.Tensor,
        setpoint_mask: Optional[torch.Tensor] = None,
        effort_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """JEPA-style forward: predict embeddings."""
        B = setpoint.shape[0]

        # Encode context (with gradients)
        context_encoded = self._encode_context(
            setpoint, effort, setpoint_mask, effort_mask
        )
        # [B, 1 + num_context_patches, hidden_dim]

        # Get context patch representations (skip CLS)
        context_patches = context_encoded[:, 1:]  # [B, num_context_patches, hidden_dim]

        # Flatten and predict target patches
        context_flat = context_patches.reshape(B, -1)  # [B, num_context_patches * hidden_dim]
        predicted_flat = self.temporal_predictor(context_flat)
        # [B, num_target_patches * hidden_dim]

        predicted_target = predicted_flat.reshape(
            B, self.num_target_patches, self.config.hidden_dim
        )
        # [B, num_target_patches, hidden_dim]

        # Encode target with EMA (no gradients)
        with torch.no_grad():
            target_encoded = self._encode_target_ema(effort, effort_mask)
            # [B, num_target_patches, hidden_dim]

        # JEPA loss: MSE between predicted and target embeddings
        # (smooth L1 is more robust to outliers)
        loss = F.smooth_l1_loss(predicted_target, target_encoded)

        # Also compute cosine similarity for monitoring
        pred_norm = F.normalize(predicted_target, dim=-1)
        target_norm = F.normalize(target_encoded, dim=-1)
        cosine_sim = (pred_norm * target_norm).sum(dim=-1).mean()

        return {
            "loss": loss,
            "cosine_similarity": cosine_sim,
        }

    def _forward_direct(
        self,
        setpoint: torch.Tensor,
        effort: torch.Tensor,
        setpoint_mask: Optional[torch.Tensor] = None,
        effort_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Direct forward: predict raw effort values."""
        B, T, _ = setpoint.shape

        # Encode context
        context_encoded = self._encode_context(
            setpoint, effort, setpoint_mask, effort_mask
        )
        # [B, 1 + num_context_patches, hidden_dim]

        # Project to decoder dimension
        decoder_input = self.decoder_embed(context_encoded)
        # [B, 1 + num_context_patches, decoder_hidden_dim]

        # Add target queries
        queries = self.target_queries.expand(B, -1, -1)
        decoder_input = torch.cat([decoder_input, queries], dim=1)

        # Decode
        decoded = self.decoder(decoder_input)
        decoded = self.decoder_norm(decoded)

        # Extract target predictions (last num_target_patches)
        num_target_patches = self.target_len // self.config.patch_size
        target_decoded = decoded[:, -num_target_patches:]

        # Predict effort
        effort_pred = self.effort_head(target_decoded)
        # [B, num_target_patches, patch_size * effort_dim]

        # Get target effort
        target_effort = effort[:, self.context_len:]
        target_effort = target_effort.reshape(
            B, num_target_patches, self.config.patch_size * self.config.effort_dim
        )

        # Apply effort mask
        if effort_mask is not None:
            mask_expanded = effort_mask.unsqueeze(1).unsqueeze(1).expand(
                B, num_target_patches, self.config.patch_size, -1
            ).reshape(B, num_target_patches, -1)

            effort_pred = effort_pred * mask_expanded
            target_effort = target_effort * mask_expanded

        # MSE loss
        loss = F.mse_loss(effort_pred, target_effort)

        return {
            "loss": loss,
        }

    def encode(
        self,
        setpoint: torch.Tensor,
        setpoint_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract representations. For temporal model, we need effort too,
        but this interface is for compatibility. Returns context encoding
        assuming effort is zeros (not ideal, but maintains interface).

        For proper encoding, use encode_with_effort().
        """
        B, T, _ = setpoint.shape
        # Create dummy effort
        effort = torch.zeros(B, T, self.config.effort_dim, device=setpoint.device)
        return self.encode_with_effort(setpoint, effort, setpoint_mask, None)

    def encode_with_effort(
        self,
        setpoint: torch.Tensor,
        effort: torch.Tensor,
        setpoint_mask: Optional[torch.Tensor] = None,
        effort_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract representations from context (setpoint + effort).

        Returns CLS token representation.
        """
        context_encoded = self._encode_context(
            setpoint, effort, setpoint_mask, effort_mask
        )
        return context_encoded[:, 0]  # CLS token

    @torch.no_grad()
    def compute_anomaly_score(
        self,
        setpoint: torch.Tensor,
        effort: torch.Tensor,
        setpoint_mask: Optional[torch.Tensor] = None,
        effort_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute anomaly scores based on temporal prediction error.

        Higher score = temporal dynamics disrupted = likely anomaly.

        Args:
            setpoint: [B, T, 14] setpoint signals
            effort: [B, T, 13] effort signals
            setpoint_mask: [B, 14] validity mask
            effort_mask: [B, 13] validity mask

        Returns:
            [B] anomaly scores (higher = more anomalous)
        """
        self.eval()

        if self.config.prediction_mode == "jepa":
            return self._anomaly_score_jepa(setpoint, effort, setpoint_mask, effort_mask)
        else:
            return self._anomaly_score_direct(setpoint, effort, setpoint_mask, effort_mask)

    def _anomaly_score_jepa(
        self,
        setpoint: torch.Tensor,
        effort: torch.Tensor,
        setpoint_mask: Optional[torch.Tensor] = None,
        effort_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Anomaly score based on embedding prediction error."""
        B = setpoint.shape[0]

        # Encode context
        context_encoded = self._encode_context(
            setpoint, effort, setpoint_mask, effort_mask
        )
        context_patches = context_encoded[:, 1:]

        # Predict target embeddings
        context_flat = context_patches.reshape(B, -1)
        predicted_flat = self.temporal_predictor(context_flat)
        predicted_target = predicted_flat.reshape(
            B, self.num_target_patches, self.config.hidden_dim
        )

        # Encode actual target
        target_encoded = self._encode_target_ema(effort, effort_mask)

        # Anomaly score: embedding distance
        # Use both L2 and cosine for robustness
        l2_error = ((predicted_target - target_encoded) ** 2).mean(dim=(1, 2))

        pred_norm = F.normalize(predicted_target, dim=-1)
        target_norm = F.normalize(target_encoded, dim=-1)
        cosine_dist = 1 - (pred_norm * target_norm).sum(dim=-1).mean(dim=1)

        # Combined score (both are important)
        scores = l2_error + cosine_dist

        return scores

    def _anomaly_score_direct(
        self,
        setpoint: torch.Tensor,
        effort: torch.Tensor,
        setpoint_mask: Optional[torch.Tensor] = None,
        effort_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Anomaly score based on raw prediction error."""
        B, T, _ = setpoint.shape

        # Forward pass to get predictions
        context_encoded = self._encode_context(
            setpoint, effort, setpoint_mask, effort_mask
        )

        decoder_input = self.decoder_embed(context_encoded)
        queries = self.target_queries.expand(B, -1, -1)
        decoder_input = torch.cat([decoder_input, queries], dim=1)

        decoded = self.decoder(decoder_input)
        decoded = self.decoder_norm(decoded)

        num_target_patches = self.target_len // self.config.patch_size
        target_decoded = decoded[:, -num_target_patches:]
        effort_pred = self.effort_head(target_decoded)

        # Get target effort
        target_effort = effort[:, self.context_len:]
        target_effort = target_effort.reshape(
            B, num_target_patches, self.config.patch_size * self.config.effort_dim
        )

        # Apply mask
        if effort_mask is not None:
            mask_expanded = effort_mask.unsqueeze(1).unsqueeze(1).expand(
                B, num_target_patches, self.config.patch_size, -1
            ).reshape(B, num_target_patches, -1)
            effort_pred = effort_pred * mask_expanded
            target_effort = target_effort * mask_expanded
            valid_count = mask_expanded.sum(dim=(1, 2)).clamp(min=1)
            scores = ((effort_pred - target_effort) ** 2).sum(dim=(1, 2)) / valid_count
        else:
            scores = F.mse_loss(effort_pred, target_effort, reduction='none').mean(dim=(1, 2))

        return scores

    def get_num_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Convenience function
def create_temporal_predictor(
    prediction_mode: str = "jepa",
    seq_len: int = 256,
    hidden_dim: int = 256,
) -> TemporalPredictor:
    """Create a temporal predictor with sensible defaults."""
    config = TemporalConfig(
        prediction_mode=prediction_mode,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
    )
    return TemporalPredictor(config)
