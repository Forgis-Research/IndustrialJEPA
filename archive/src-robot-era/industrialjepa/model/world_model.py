# SPDX-FileCopyrightText: 2025-2026 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
JEPA World Model for Industrial Robotics.

Learns to predict next state in latent space given current state and command.
Uses EMA target encoder to prevent collapse (standard JEPA approach).

Architecture:
    observation(t) -> Encoder -> z(t) -> Predictor(z(t), cmd(t)) -> z_pred(t+1)
    observation(t+1) -> EMA_Encoder -> z_target(t+1)

    Loss = ||z_pred(t+1) - z_target(t+1)||^2

Key design choices:
1. Predictor is SMALLER than encoder (asymmetry prevents collapse)
2. EMA target provides stable training signal
3. Optional decoder for reconstruction / visualization
4. Sequence model (Transformer) for temporal context
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WorldModelConfig:
    """Configuration for JEPA World Model."""

    # Input dimensions (will be set from data)
    obs_dim: int = 13  # effort signals (7 joint + 6 Cartesian)
    cmd_dim: int = 14  # setpoint signals (7 pos + 7 vel)

    # Sequence parameters
    seq_len: int = 256  # window size
    pred_horizon: int = 1  # predict this many steps ahead

    # Architecture
    latent_dim: int = 256
    hidden_dim: int = 512
    num_encoder_layers: int = 4
    num_predictor_layers: int = 2  # Smaller than encoder!
    num_heads: int = 8
    dropout: float = 0.1

    # EMA
    ema_momentum: float = 0.996
    ema_momentum_end: float = 0.9999  # Anneal to this
    ema_anneal_steps: int = 10000

    # Training
    use_decoder: bool = True  # For reconstruction loss
    reconstruction_weight: float = 0.1 #randomly turned off neurons during training (avoid overreliance)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class StateEncoder(nn.Module):
    """
    Encodes observation sequence to latent representation.

    Uses Transformer encoder for temporal modeling.
    Output is a sequence of latent states z(1), z(2), ..., z(T).
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(config.obs_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )

        # Positional encoding
        self.pos_enc = PositionalEncoding(
            config.hidden_dim,
            max_len=config.seq_len + 100,
            dropout=config.dropout
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers,
        )

        # Output projection to latent space
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, seq_len, obs_dim) - observation sequence

        Returns:
            z: (batch, seq_len, latent_dim) - latent state sequence
        """
        # Project input
        x = self.input_proj(obs)  # (B, T, hidden_dim)

        # Add positional encoding
        x = self.pos_enc(x)

        # Transformer encoding (causal mask for autoregressive)
        # Note: We use causal mask so z(t) only depends on obs(1:t)
        mask = nn.Transformer.generate_square_subsequent_mask(
            x.size(1), device=x.device
        )
        x = self.transformer(x, mask=mask, is_causal=True)

        # Project to latent space
        z = self.output_proj(x)  # (B, T, latent_dim)

        return z


class DynamicsPredictor(nn.Module):
    """
    Predicts next latent state given current latent state and command.

    IMPORTANT: This is intentionally SMALLER than the encoder.
    The asymmetry forces the encoder to do the heavy lifting,
    preventing representation collapse.
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config

        # Command projection
        self.cmd_proj = nn.Sequential(
            nn.Linear(config.cmd_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )

        # Combine latent + command
        self.combine = nn.Sequential(
            nn.Linear(config.latent_dim + config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )

        # Predictor MLP (smaller than encoder)
        layers = []
        for _ in range(config.num_predictor_layers):
            layers.extend([
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
            ])
        self.mlp = nn.Sequential(*layers)

        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.latent_dim)

    def forward(
        self,
        z: torch.Tensor,
        cmd: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z: (batch, seq_len, latent_dim) - current latent states
            cmd: (batch, seq_len, cmd_dim) - commands at each timestep

        Returns:
            z_pred: (batch, seq_len, latent_dim) - predicted next latent states
        """
        # Project command
        cmd_emb = self.cmd_proj(cmd)  # (B, T, hidden_dim)

        # Combine latent + command
        combined = torch.cat([z, cmd_emb], dim=-1)  # (B, T, latent_dim + hidden_dim)
        x = self.combine(combined)  # (B, T, hidden_dim)

        # Predict
        x = self.mlp(x)
        z_pred = self.output_proj(x)  # (B, T, latent_dim)

        return z_pred


class StateDecoder(nn.Module):
    """
    Decodes latent state back to observation space.

    Optional but useful for:
    1. Reconstruction loss (prevents collapse)
    2. Visualization of predictions
    3. Anomaly detection (compare decoded prediction vs actual)
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config

        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.obs_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, seq_len, latent_dim) - latent states

        Returns:
            obs_recon: (batch, seq_len, obs_dim) - reconstructed observations
        """
        return self.decoder(z)


class JEPAWorldModel(nn.Module):
    """
    JEPA-based World Model for Industrial Robotics.

    Learns dynamics in latent space: z(t+1) = f(z(t), cmd(t))

    Training uses EMA target encoder for stable targets.
    Optional decoder enables reconstruction loss and visualization.
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config

        # Online encoder (trained with gradients)
        self.encoder = StateEncoder(config)

        # EMA target encoder (no gradients, updated via momentum)
        self.ema_encoder = copy.deepcopy(self.encoder)
        for param in self.ema_encoder.parameters():
            param.requires_grad = False

        # Dynamics predictor
        self.predictor = DynamicsPredictor(config)

        # Optional decoder
        self.decoder = StateDecoder(config) if config.use_decoder else None

        # EMA step counter for momentum annealing
        self.register_buffer('ema_step', torch.tensor(0))

    @torch.no_grad()
    def update_ema(self):
        """Update EMA encoder parameters with momentum."""
        # Anneal momentum from ema_momentum to ema_momentum_end
        step = self.ema_step.item()
        if step < self.config.ema_anneal_steps:
            momentum = self.config.ema_momentum + (
                self.config.ema_momentum_end - self.config.ema_momentum
            ) * step / self.config.ema_anneal_steps
        else:
            momentum = self.config.ema_momentum_end

        # Update EMA parameters
        for online_param, ema_param in zip(
            self.encoder.parameters(),
            self.ema_encoder.parameters()
        ):
            ema_param.data = momentum * ema_param.data + (1 - momentum) * online_param.data

        self.ema_step += 1

    def forward(
        self,
        obs_t: torch.Tensor,
        cmd_t: torch.Tensor,
        obs_t1: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            obs_t: (batch, seq_len, obs_dim) - observations at time t
            cmd_t: (batch, seq_len, cmd_dim) - commands at time t
            obs_t1: (batch, seq_len, obs_dim) - observations at time t+1 (for target)

        Returns:
            dict with:
                z_t: encoded current states
                z_pred: predicted next states
                z_target: EMA-encoded target states (if obs_t1 provided)
                obs_recon: decoded current states (if decoder enabled)
                obs_pred: decoded predicted states (if decoder enabled)
        """
        # Encode current observation
        z_t = self.encoder(obs_t)  # (B, T, latent_dim)

        # Predict next latent state
        z_pred = self.predictor(z_t, cmd_t)  # (B, T, latent_dim)

        outputs = {
            'z_t': z_t,
            'z_pred': z_pred,
        }

        # Compute target (EMA encoder, no gradients)
        if obs_t1 is not None:
            with torch.no_grad():
                z_target = self.ema_encoder(obs_t1)
            outputs['z_target'] = z_target

        # Decode (optional)
        if self.decoder is not None:
            outputs['obs_recon'] = self.decoder(z_t)
            outputs['obs_pred'] = self.decoder(z_pred)

        return outputs

    @torch.no_grad()
    def rollout(
        self,
        obs_init: torch.Tensor,
        cmd_sequence: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        """
        Multi-step rollout in latent space.

        Args:
            obs_init: (batch, context_len, obs_dim) - initial observations
            cmd_sequence: (batch, steps, cmd_dim) - future commands
            steps: number of steps to predict

        Returns:
            z_rollout: (batch, steps, latent_dim) - predicted latent states
        """
        self.eval()

        # Encode initial context
        z = self.encoder(obs_init)  # (B, context_len, latent_dim)
        z_last = z[:, -1:, :]  # Take last state (B, 1, latent_dim)

        rollout_states = []

        for t in range(steps):
            cmd_t = cmd_sequence[:, t:t+1, :]  # (B, 1, cmd_dim)
            z_next = self.predictor(z_last, cmd_t)  # (B, 1, latent_dim)
            rollout_states.append(z_next)
            z_last = z_next

        return torch.cat(rollout_states, dim=1)  # (B, steps, latent_dim)

    def compute_loss(
        self,
        obs_t: torch.Tensor,
        cmd_t: torch.Tensor,
        obs_t1: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute training losses.

        Args:
            obs_t: current observations
            cmd_t: commands
            obs_t1: next observations (target)
            mask: optional validity mask for heterogeneous data

        Returns:
            dict with loss components
        """
        outputs = self.forward(obs_t, cmd_t, obs_t1)

        # JEPA loss: predict in latent space
        # Use last (seq_len - 1) positions to predict the shifted target
        z_pred = outputs['z_pred'][:, :-1, :]  # (B, T-1, latent)
        z_target = outputs['z_target'][:, 1:, :]  # (B, T-1, latent) - shifted by 1

        L_jepa = F.mse_loss(z_pred, z_target)

        losses = {'L_jepa': L_jepa}

        # Reconstruction loss (optional)
        if self.decoder is not None:
            obs_recon = outputs['obs_recon']

            if mask is not None:
                # Only compute loss on valid dimensions
                L_recon = F.mse_loss(obs_recon * mask, obs_t * mask)
            else:
                L_recon = F.mse_loss(obs_recon, obs_t)

            losses['L_recon'] = L_recon

        # Total loss
        total_loss = L_jepa
        if 'L_recon' in losses:
            total_loss = total_loss + self.config.reconstruction_weight * losses['L_recon']

        losses['total'] = total_loss

        return losses


def create_world_model(
    obs_dim: int,
    cmd_dim: int,
    seq_len: int = 256,
    latent_dim: int = 256,
    **kwargs,
) -> JEPAWorldModel:
    """
    Factory function to create a JEPA World Model.

    Args:
        obs_dim: dimension of observation (effort signals)
        cmd_dim: dimension of command (setpoint signals)
        seq_len: sequence length
        latent_dim: latent space dimension
        **kwargs: additional config options

    Returns:
        JEPAWorldModel instance
    """
    config = WorldModelConfig(
        obs_dim=obs_dim,
        cmd_dim=cmd_dim,
        seq_len=seq_len,
        latent_dim=latent_dim,
        **kwargs,
    )
    return JEPAWorldModel(config)
