# SPDX-FileCopyrightText: 2025 Industrial World Model Authors
# SPDX-License-Identifier: MIT

"""
FiLM (Feature-wise Linear Modulation) for action conditioning.

Allows control actions to modulate the hidden state dynamics,
enabling "what-if" queries and action-conditioned prediction.

Based on "FiLM: Visual Reasoning with a General Conditioning Layer"
(Perez et al., 2018).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FiLMConditioner(nn.Module):
    """
    Feature-wise Linear Modulation for action conditioning.

    Given an action vector, produces scale (gamma) and shift (beta)
    parameters to modulate hidden states:

        h' = gamma(a) * h + beta(a)
    """

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            action_dim: Dimension of action input
            hidden_dim: Dimension of hidden state to modulate
            num_layers: Number of layers in conditioning network
            dropout: Dropout rate
        """
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Action embedding network
        layers = []
        current_dim = action_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim

        self.action_encoder = nn.Sequential(*layers) if layers else nn.Identity()

        # Gamma (scale) and Beta (shift) projections
        self.gamma_proj = nn.Linear(current_dim, hidden_dim)
        self.beta_proj = nn.Linear(current_dim, hidden_dim)

        # Initialize to identity (gamma=1, beta=0)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(
        self,
        h: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply FiLM conditioning.

        Args:
            h: [B, L, D] hidden states
            action: [B, A] action vector OR [B, L, A] per-timestep actions

        Returns:
            [B, L, D] modulated hidden states
        """
        # Encode action
        a_enc = self.action_encoder(action)  # [B, D] or [B, L, D]

        # Compute gamma and beta
        gamma = self.gamma_proj(a_enc)  # [B, D] or [B, L, D]
        beta = self.beta_proj(a_enc)    # [B, D] or [B, L, D]

        # Expand if action is per-sequence (not per-timestep)
        if gamma.dim() == 2:
            gamma = gamma.unsqueeze(1)  # [B, 1, D]
            beta = beta.unsqueeze(1)    # [B, 1, D]

        # Apply modulation
        return gamma * h + beta


class ActionEmbedding(nn.Module):
    """
    Embedding module for different types of industrial actions.

    Handles:
    - Continuous actions (setpoints, parameters)
    - Discrete actions (mode switches, commands)
    - Mixed actions
    """

    def __init__(
        self,
        continuous_dim: int = 0,
        discrete_dims: list = None,
        output_dim: int = 32,
        max_discrete: int = 100,
    ):
        """
        Args:
            continuous_dim: Number of continuous action dimensions
            discrete_dims: List of vocab sizes for discrete actions
            output_dim: Output embedding dimension
            max_discrete: Maximum vocabulary size for discrete embeddings
        """
        super().__init__()
        self.continuous_dim = continuous_dim
        self.discrete_dims = discrete_dims or []
        self.output_dim = output_dim

        # Continuous action projection
        if continuous_dim > 0:
            self.continuous_proj = nn.Sequential(
                nn.Linear(continuous_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
            )
        else:
            self.continuous_proj = None

        # Discrete action embeddings
        self.discrete_embeds = nn.ModuleList([
            nn.Embedding(min(dim, max_discrete), output_dim)
            for dim in self.discrete_dims
        ])

        # Final projection (combines continuous and discrete)
        total_input = 0
        if continuous_dim > 0:
            total_input += output_dim
        total_input += len(self.discrete_dims) * output_dim

        if total_input > 0:
            self.final_proj = nn.Linear(total_input, output_dim)
        else:
            self.final_proj = None

    def forward(
        self,
        continuous_actions: Optional[torch.Tensor] = None,
        discrete_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Embed actions.

        Args:
            continuous_actions: [B, continuous_dim] or [B, L, continuous_dim]
            discrete_actions: [B, num_discrete] or [B, L, num_discrete] (long)

        Returns:
            [B, output_dim] or [B, L, output_dim] action embeddings
        """
        embeddings = []

        # Process continuous actions
        if continuous_actions is not None and self.continuous_proj is not None:
            cont_emb = self.continuous_proj(continuous_actions)
            embeddings.append(cont_emb)

        # Process discrete actions
        if discrete_actions is not None and len(self.discrete_embeds) > 0:
            for i, embed in enumerate(self.discrete_embeds):
                disc_emb = embed(discrete_actions[..., i])
                embeddings.append(disc_emb)

        # Combine
        if len(embeddings) == 0:
            raise ValueError("No actions provided")
        elif len(embeddings) == 1:
            return embeddings[0]
        else:
            combined = torch.cat(embeddings, dim=-1)
            return self.final_proj(combined)


class TimestepEmbedding(nn.Module):
    """
    Timestep embedding for temporal conditioning.

    Uses sinusoidal embeddings similar to diffusion models,
    useful for temporal prediction tasks.
    """

    def __init__(
        self,
        output_dim: int = 256,
        max_timesteps: int = 10000,
    ):
        super().__init__()
        self.output_dim = output_dim

        # MLP projection
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Linear(output_dim * 4, output_dim),
        )

        # Precompute sinusoidal embeddings
        half_dim = output_dim // 2
        emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32)
            * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1))
        )
        self.register_buffer("emb_scale", emb)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [B] or [B, L] timestep indices

        Returns:
            [B, D] or [B, L, D] timestep embeddings
        """
        # Ensure float for computation
        timesteps = timesteps.float()

        # Add dimension if needed
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(-1)  # [B, 1]

        # Sinusoidal embedding
        emb = timesteps * self.emb_scale  # [..., half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [..., D]

        # MLP projection
        return self.mlp(emb)


class MultiConditioner(nn.Module):
    """
    Combines multiple conditioning signals for rich control.

    Supports:
    - Actions (continuous + discrete)
    - Timesteps
    - Domain embeddings
    - Operating condition embeddings
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int = 32,
        num_domains: int = 4,
        num_conditions: int = 10,
        use_timestep: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Action conditioning
        self.action_film = FiLMConditioner(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )

        # Domain embedding
        self.domain_embed = nn.Embedding(num_domains, hidden_dim)

        # Operating condition embedding
        self.condition_embed = nn.Embedding(num_conditions, hidden_dim)

        # Timestep embedding
        self.use_timestep = use_timestep
        if use_timestep:
            self.timestep_embed = TimestepEmbedding(output_dim=hidden_dim)

        # Final combination
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        h: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        domain: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply multiple conditioning signals.

        Args:
            h: [B, L, D] hidden states
            action: [B, action_dim] action vector
            domain: [B] domain indices
            condition: [B] operating condition indices
            timestep: [B] or [B, L] timestep indices

        Returns:
            [B, L, D] conditioned hidden states
        """
        B, L, D = h.shape

        # Start with action conditioning
        if action is not None:
            h = self.action_film(h, action)

        # Add domain embedding
        if domain is not None:
            domain_emb = self.domain_embed(domain)  # [B, D]
            h = h + domain_emb.unsqueeze(1)

        # Add condition embedding
        if condition is not None:
            cond_emb = self.condition_embed(condition)  # [B, D]
            h = h + cond_emb.unsqueeze(1)

        # Add timestep embedding
        if timestep is not None and self.use_timestep:
            time_emb = self.timestep_embed(timestep)  # [B, D] or [B, L, D]
            if time_emb.dim() == 2:
                time_emb = time_emb.unsqueeze(1)
            h = h + time_emb

        return h
