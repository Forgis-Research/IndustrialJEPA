# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Base class for all baseline models.

All baselines share:
1. Common encoder architecture (Transformer or Mamba)
2. Input: Setpoint signals [B, T, 14]
3. Validity masks for variable DOF robots
4. Anomaly scoring interface
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Literal


@dataclass
class BaselineConfig:
    """Base configuration shared by all baseline models."""

    # Input dimensions (from FactoryNet unified schema)
    setpoint_dim: int = 14  # 7 DOF x 2 (pos + vel)
    effort_dim: int = 7  # 7 DOF max

    # Sequence
    seq_len: int = 256
    patch_size: int = 16  # Patch embedding

    # Encoder architecture
    encoder_type: Literal["transformer", "mamba"] = "transformer"
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01


class PatchEmbedding(nn.Module):
    """
    Embed time series into patches.

    Converts [B, T, C] -> [B, num_patches, hidden_dim]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        patch_size: int,
        seq_len: int,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Linear projection of flattened patches
        self.proj = nn.Linear(input_dim * patch_size, hidden_dim)

        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, hidden_dim) * 0.02
        )

        # CLS token for sequence-level representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Args:
            x: [B, T, C] input time series
            mask: [B, C] validity mask (1 = valid, 0 = padded)

        Returns:
            patches: [B, 1 + num_patches, hidden_dim] (with CLS token)
            num_patches: Number of patches (excluding CLS)
        """
        B, T, C = x.shape

        # Apply validity mask if provided (zero out padded dimensions)
        if mask is not None:
            x = x * mask.unsqueeze(1)  # [B, T, C]

        # Reshape into patches: [B, num_patches, patch_size * C]
        x = x.reshape(B, self.num_patches, self.patch_size * C)

        # Project to hidden dim
        x = self.proj(x)  # [B, num_patches, hidden_dim]

        # Add positional embeddings
        x = x + self.pos_embed

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        return x, self.num_patches


class TransformerEncoder(nn.Module):
    """Standard Transformer encoder."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] input embeddings

        Returns:
            [B, L, D] encoded representations
        """
        x = self.encoder(x)
        return self.norm(x)


class BaselineModel(nn.Module, ABC):
    """
    Abstract base class for all baseline models.

    Subclasses must implement:
    - forward(): Training forward pass returning loss
    - encode(): Extract representations
    - compute_anomaly_score(): Compute per-window anomaly scores
    """

    def __init__(self, config: BaselineConfig):
        super().__init__()
        self.config = config

        # Patch embedding for setpoint input
        self.patch_embed = PatchEmbedding(
            input_dim=config.setpoint_dim,
            hidden_dim=config.hidden_dim,
            patch_size=config.patch_size,
            seq_len=config.seq_len,
        )

        # Shared encoder
        if config.encoder_type == "transformer":
            self.encoder = TransformerEncoder(
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                dropout=config.dropout,
            )
        else:
            raise NotImplementedError("Mamba encoder not yet implemented for baselines")

    @abstractmethod
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
            Dict with 'loss' and other metrics
        """
        pass

    @abstractmethod
    def encode(
        self,
        setpoint: torch.Tensor,
        setpoint_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract representations from setpoint.

        Args:
            setpoint: [B, T, 14] setpoint signals
            setpoint_mask: [B, 14] validity mask

        Returns:
            [B, hidden_dim] or [B, T, hidden_dim] representations
        """
        pass

    @abstractmethod
    def compute_anomaly_score(
        self,
        setpoint: torch.Tensor,
        effort: torch.Tensor,
        setpoint_mask: Optional[torch.Tensor] = None,
        effort_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute anomaly scores for each window.

        Higher score = more anomalous.

        Args:
            setpoint: [B, T, 14] setpoint signals
            effort: [B, T, 7] effort signals
            setpoint_mask: [B, 14] validity mask
            effort_mask: [B, 7] validity mask

        Returns:
            [B] anomaly scores
        """
        pass

    def get_num_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_pretrained(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "config": self.config,
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu"):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(device)
