# SPDX-FileCopyrightText: 2025 Industrial World Model Authors
# SPDX-License-Identifier: MIT

"""
Hybrid Mamba-Transformer backbone for industrial dynamics modeling.

Combines:
1. Mamba blocks for efficient O(L) sequential processing
2. Sparse attention layers for critical interactions
3. FiLM conditioning for action input
4. Stochastic latent state for uncertainty

Architecture:
    [Mamba x 4] → [Sparse Attn] → [Mamba x 4] → [Sparse Attn] → ...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

from industrialjepa.model.backbone.mamba import MambaBlock, BidirectionalMamba
from industrialjepa.model.backbone.attention import SparseAttention, LocalAttention
from industrialjepa.model.backbone.film import FiLMConditioner, MultiConditioner
from industrialjepa.model.config import BackboneConfig


@dataclass
class BackboneOutput:
    """Output from the hybrid backbone."""

    hidden_states: torch.Tensor  # [B, L, D] final hidden states
    all_hidden_states: Optional[List[torch.Tensor]] = None  # Per-layer outputs
    attention_weights: Optional[List[torch.Tensor]] = None  # Attention weights
    importance_scores: Optional[torch.Tensor] = None  # [B, L] token importance


class HybridBlock(nn.Module):
    """
    Single hybrid block: Mamba + optional Sparse Attention.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_attention: bool = False,
        attention_type: str = "sparse",  # "sparse", "local", "full"
        window_size: int = 128,
        top_k: int = 64,
    ):
        super().__init__()
        self.use_attention = use_attention

        # Mamba block
        self.mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

        # Optional attention
        if use_attention:
            if attention_type == "sparse":
                self.attention = SparseAttention(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    top_k=top_k,
                    window_size=window_size,
                )
            elif attention_type == "local":
                self.attention = LocalAttention(
                    d_model=d_model,
                    n_heads=n_heads,
                    window_size=window_size,
                    dropout=dropout,
                )
            else:  # full
                self.attention = nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=n_heads,
                    dropout=dropout,
                    batch_first=True,
                )

            self.attn_norm = nn.LayerNorm(d_model)
            self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        importance_mask: Optional[torch.Tensor] = None,
        cache: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Forward pass.

        Args:
            x: [B, L, D] input
            importance_mask: [B, L] for sparse attention
            cache: Optional cache for Mamba

        Returns:
            output: [B, L, D]
            new_cache: Updated cache
        """
        # Mamba
        x, new_cache = self.mamba(x, cache=cache)

        # Optional attention
        if self.use_attention:
            residual = x
            x = self.attn_norm(x)

            if isinstance(self.attention, nn.MultiheadAttention):
                x, _ = self.attention(x, x, x)
            elif isinstance(self.attention, SparseAttention):
                x = self.attention(x, importance_mask=importance_mask)
            else:
                x = self.attention(x)

            x = self.attn_dropout(x)
            x = residual + x

        return x, new_cache


class HybridMambaTransformer(nn.Module):
    """
    Full Hybrid Mamba-Transformer backbone for dynamics modeling.
    """

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim

        # Input projection
        self.input_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.input_norm = nn.LayerNorm(config.hidden_dim)

        # Build blocks
        self.blocks = nn.ModuleList()
        block_idx = 0

        for layer_idx in range(config.num_attention_layers):
            # Add Mamba blocks
            for _ in range(config.num_mamba_blocks):
                use_attn = (block_idx + 1) % config.attention_every_n == 0
                self.blocks.append(HybridBlock(
                    d_model=config.hidden_dim,
                    d_state=config.mamba.d_state,
                    d_conv=config.mamba.d_conv,
                    expand=config.mamba.expand,
                    n_heads=config.transformer.n_heads,
                    dropout=config.mamba.dropout,
                    use_attention=use_attn,
                    attention_type="sparse",
                    window_size=128,
                    top_k=64,
                ))
                block_idx += 1

        # Action conditioning
        if config.use_film_conditioning:
            self.action_conditioner = FiLMConditioner(
                action_dim=config.action_dim,
                hidden_dim=config.hidden_dim,
            )
        else:
            self.action_conditioner = None

        # Output projection
        self.output_norm = nn.LayerNorm(config.hidden_dim)

        # Importance score predictor (for sparse attention)
        self.importance_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        cache: Optional[List[Tuple]] = None,
        return_all_hidden_states: bool = False,
    ) -> BackboneOutput:
        """
        Forward pass through hybrid backbone.

        Args:
            x: [B, L, D] input (quantized token embeddings)
            action: [B, action_dim] optional action for conditioning
            cache: Optional list of caches for each block
            return_all_hidden_states: Whether to return all layer outputs

        Returns:
            BackboneOutput with hidden states and metadata
        """
        B, L, D = x.shape

        # Input processing
        h = self.input_proj(x)
        h = self.input_norm(h)

        # Apply action conditioning at input
        if action is not None and self.action_conditioner is not None:
            h = self.action_conditioner(h, action)

        # Compute importance scores for sparse attention
        importance_scores = self.importance_predictor(h).squeeze(-1)  # [B, L]

        # Process through blocks
        all_hidden_states = [h] if return_all_hidden_states else None
        new_caches = []

        for i, block in enumerate(self.blocks):
            block_cache = cache[i] if cache is not None else None
            h, new_cache = block(h, importance_mask=importance_scores, cache=block_cache)
            new_caches.append(new_cache)

            if return_all_hidden_states:
                all_hidden_states.append(h)

            # Re-apply action conditioning every few blocks
            if action is not None and self.action_conditioner is not None:
                if (i + 1) % 4 == 0:
                    h = self.action_conditioner(h, action)

        # Output normalization
        h = self.output_norm(h)

        return BackboneOutput(
            hidden_states=h,
            all_hidden_states=all_hidden_states,
            importance_scores=importance_scores,
        )

    def predict_next(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        steps: int = 1,
    ) -> torch.Tensor:
        """
        Autoregressive prediction of next states.

        Args:
            x: [B, L, D] input sequence
            action: [B, action_dim] action for conditioning
            steps: Number of steps to predict

        Returns:
            [B, steps, D] predicted states
        """
        predictions = []
        h = x

        for _ in range(steps):
            output = self.forward(h, action=action)
            # Use last hidden state as next prediction
            next_state = output.hidden_states[:, -1:, :]
            predictions.append(next_state)
            # Append to sequence
            h = torch.cat([h, next_state], dim=1)

        return torch.cat(predictions, dim=1)

    def get_num_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


class DynamicsModel(nn.Module):
    """
    Full dynamics model combining backbone with state prediction heads.
    """

    def __init__(
        self,
        config: BackboneConfig,
        output_dim: int = 256,
        num_quantiles: int = 9,
    ):
        super().__init__()
        self.config = config
        self.output_dim = output_dim
        self.num_quantiles = num_quantiles

        # Backbone
        self.backbone = HybridMambaTransformer(config)

        # State prediction head (mean)
        self.state_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, output_dim),
        )

        # Quantile prediction head (for uncertainty)
        self.quantile_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, output_dim * num_quantiles),
        )

        # Quantile levels (e.g., [0.1, 0.2, ..., 0.9])
        quantiles = torch.linspace(0.1, 0.9, num_quantiles)
        self.register_buffer("quantiles", quantiles)

    def forward(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with state and uncertainty prediction.

        Args:
            x: [B, L, D] input sequence
            action: [B, action_dim] optional action

        Returns:
            Dict with 'mean', 'quantiles', 'hidden_states'
        """
        # Backbone
        backbone_output = self.backbone(x, action=action)
        h = backbone_output.hidden_states

        # State prediction (mean)
        state_pred = self.state_head(h)  # [B, L, output_dim]

        # Quantile prediction
        quantile_pred = self.quantile_head(h)  # [B, L, output_dim * num_quantiles]
        quantile_pred = quantile_pred.view(
            *h.shape[:-1], self.output_dim, self.num_quantiles
        )  # [B, L, output_dim, num_quantiles]

        return {
            "mean": state_pred,
            "quantiles": quantile_pred,
            "hidden_states": h,
            "importance_scores": backbone_output.importance_scores,
        }

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute prediction loss including quantile loss.

        Args:
            predictions: Output from forward()
            targets: [B, L, output_dim] ground truth

        Returns:
            Dict with 'total', 'mse', 'quantile' losses
        """
        # MSE loss on mean prediction
        mse_loss = F.mse_loss(predictions["mean"], targets)

        # Quantile loss
        quantile_pred = predictions["quantiles"]  # [B, L, D, Q]
        targets_expanded = targets.unsqueeze(-1)  # [B, L, D, 1]

        # Pinball loss
        errors = targets_expanded - quantile_pred  # [B, L, D, Q]
        quantile_loss = torch.max(
            self.quantiles * errors,
            (self.quantiles - 1) * errors,
        ).mean()

        total_loss = mse_loss + 0.5 * quantile_loss

        return {
            "total": total_loss,
            "mse": mse_loss,
            "quantile": quantile_loss,
        }
