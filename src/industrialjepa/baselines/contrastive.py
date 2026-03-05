# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Contrastive learning baseline for industrial time series.

Implements SimCLR-style contrastive learning:
1. Create two augmented views of each setpoint window
2. Learn representations where same-window views are similar
3. Different-window views are dissimilar

Augmentations for time series:
- Temporal jittering (slight shifts)
- Gaussian noise injection
- Amplitude scaling
- Feature masking

For anomaly detection:
- Compute distance to mean healthy representation
- Or use one-class contrastive learning (healthy vs augmented)

Reference: Chen et al., "A Simple Framework for Contrastive Learning" (2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .base import BaselineConfig, BaselineModel


@dataclass
class ContrastiveConfig(BaselineConfig):
    """Configuration for Contrastive model."""

    # Contrastive learning
    temperature: float = 0.07
    projection_dim: int = 128

    # Augmentations
    noise_std: float = 0.1
    jitter_ratio: float = 0.1  # Max temporal shift as fraction of seq_len
    scale_range: tuple = (0.8, 1.2)  # Amplitude scaling range
    feature_mask_ratio: float = 0.1  # Fraction of features to mask

    # Training mode
    use_effort_pairs: bool = True  # Use (setpoint, effort) as positive pairs
    augmentation_types: List[str] = field(
        default_factory=lambda: ["noise", "scale", "jitter"]
    )


class TimeSeriesAugmenter(nn.Module):
    """
    Augmentation module for time series data.
    """

    def __init__(self, config: ContrastiveConfig):
        super().__init__()
        self.config = config

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise."""
        noise = torch.randn_like(x) * self.config.noise_std
        return x + noise

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        """Random amplitude scaling."""
        scale_min, scale_max = self.config.scale_range
        scale = torch.empty(x.shape[0], 1, 1, device=x.device).uniform_(
            scale_min, scale_max
        )
        return x * scale

    def jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Temporal jittering (circular shift)."""
        B, T, C = x.shape
        max_shift = int(T * self.config.jitter_ratio)
        if max_shift == 0:
            return x

        shifts = torch.randint(-max_shift, max_shift + 1, (B,), device=x.device)
        result = torch.zeros_like(x)
        for i in range(B):
            result[i] = torch.roll(x[i], shifts[i].item(), dims=0)
        return result

    def feature_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly mask some features."""
        B, T, C = x.shape
        num_mask = int(C * self.config.feature_mask_ratio)
        if num_mask == 0:
            return x

        mask = torch.ones(B, 1, C, device=x.device)
        for i in range(B):
            indices = torch.randperm(C, device=x.device)[:num_mask]
            mask[i, 0, indices] = 0
        return x * mask

    def forward(
        self,
        x: torch.Tensor,
        augmentation_types: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Apply random augmentations.

        Args:
            x: [B, T, C] input time series
            augmentation_types: List of augmentation names to apply

        Returns:
            [B, T, C] augmented time series
        """
        if augmentation_types is None:
            augmentation_types = self.config.augmentation_types

        # Apply augmentations in sequence
        for aug_type in augmentation_types:
            if aug_type == "noise":
                x = self.add_noise(x)
            elif aug_type == "scale":
                x = self.scale(x)
            elif aug_type == "jitter":
                x = self.jitter(x)
            elif aug_type == "feature_mask":
                x = self.feature_mask(x)

        return x


class ContrastiveModel(BaselineModel):
    """
    Contrastive learning model for Setpoint->Effort.

    Two training modes:
    1. SimCLR: Two augmented views of setpoint as positive pairs
    2. Cross-modal: (Setpoint, Effort) as positive pairs

    The cross-modal mode is more aligned with the JEPA objective,
    learning that setpoint and corresponding effort should have
    similar representations.
    """

    def __init__(self, config: ContrastiveConfig):
        super().__init__(config)
        self.config = config
        self.temperature = config.temperature

        # Augmenter
        self.augmenter = TimeSeriesAugmenter(config)

        # Separate encoder for effort (if using cross-modal)
        if config.use_effort_pairs:
            from .base import PatchEmbedding, TransformerEncoder

            self.effort_patch_embed = PatchEmbedding(
                input_dim=config.effort_dim,
                hidden_dim=config.hidden_dim,
                patch_size=config.patch_size,
                seq_len=config.seq_len,
            )
            self.effort_encoder = TransformerEncoder(
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                dropout=config.dropout,
            )

        # Projection head (MLP)
        self.projector = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_dim, config.projection_dim),
        )

    def forward_setpoint(
        self,
        setpoint: torch.Tensor,
        setpoint_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode setpoint to representation.

        Args:
            setpoint: [B, T, 14] setpoint signals
            setpoint_mask: [B, 14] validity mask

        Returns:
            [B, hidden_dim] representation
        """
        x, _ = self.patch_embed(setpoint, setpoint_mask)
        x = self.encoder(x)
        return x[:, 0]  # CLS token

    def forward_effort(
        self,
        effort: torch.Tensor,
        effort_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode effort to representation.

        Args:
            effort: [B, T, 7] effort signals
            effort_mask: [B, 7] validity mask

        Returns:
            [B, hidden_dim] representation
        """
        x, _ = self.effort_patch_embed(effort, effort_mask)
        x = self.effort_encoder(x)
        return x[:, 0]  # CLS token

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """
        Project representation to contrastive space.

        Args:
            h: [B, hidden_dim] representation

        Returns:
            [B, projection_dim] normalized projection
        """
        z = self.projector(h)
        return F.normalize(z, dim=-1)

    def nt_xent_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """
        NT-Xent loss (normalized temperature-scaled cross entropy).

        Args:
            z1: [B, D] first view projections
            z2: [B, D] second view projections

        Returns:
            Scalar loss
        """
        B = z1.shape[0]
        device = z1.device

        # Concatenate both views: [2B, D]
        z = torch.cat([z1, z2], dim=0)

        # Compute similarity matrix: [2B, 2B]
        sim = torch.mm(z, z.t()) / self.temperature

        # Mask out self-similarities
        mask = torch.eye(2 * B, device=device, dtype=torch.bool)
        sim.masked_fill_(mask, float("-inf"))

        # Labels: positive pairs are (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=device),
            torch.arange(B, device=device),
        ])

        # Cross entropy loss
        loss = F.cross_entropy(sim, labels)

        return loss

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
            effort: [B, T, 7] effort signals
            setpoint_mask: [B, 14] validity mask for setpoint
            effort_mask: [B, 7] validity mask for effort

        Returns:
            Dict with 'loss' and metrics
        """
        if self.config.use_effort_pairs:
            # Cross-modal contrastive: (setpoint, effort) pairs
            return self._forward_cross_modal(
                setpoint, effort, setpoint_mask, effort_mask
            )
        else:
            # SimCLR: two augmented views of setpoint
            return self._forward_simclr(setpoint, setpoint_mask)

    def _forward_simclr(
        self,
        setpoint: torch.Tensor,
        setpoint_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """SimCLR-style contrastive learning."""
        # Create two augmented views
        view1 = self.augmenter(setpoint)
        view2 = self.augmenter(setpoint)

        # Encode
        h1 = self.forward_setpoint(view1, setpoint_mask)
        h2 = self.forward_setpoint(view2, setpoint_mask)

        # Project
        z1 = self.project(h1)
        z2 = self.project(h2)

        # Contrastive loss
        loss = self.nt_xent_loss(z1, z2)

        # Compute alignment metrics
        with torch.no_grad():
            # Cosine similarity between positive pairs
            pos_sim = F.cosine_similarity(z1, z2).mean()
            # Average pairwise similarity (should be low for negatives)
            all_sim = torch.mm(z1, z2.t()).mean()

        return {
            "loss": loss,
            "pos_similarity": pos_sim,
            "avg_similarity": all_sim,
        }

    def _forward_cross_modal(
        self,
        setpoint: torch.Tensor,
        effort: torch.Tensor,
        setpoint_mask: Optional[torch.Tensor] = None,
        effort_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Cross-modal contrastive: setpoint-effort pairs."""
        # Encode both modalities
        h_setpoint = self.forward_setpoint(setpoint, setpoint_mask)
        h_effort = self.forward_effort(effort, effort_mask)

        # Project to shared space
        z_setpoint = self.project(h_setpoint)
        z_effort = self.project(h_effort)

        # Contrastive loss (setpoint and effort from same window are positive pairs)
        loss = self.nt_xent_loss(z_setpoint, z_effort)

        # Metrics
        with torch.no_grad():
            pos_sim = F.cosine_similarity(z_setpoint, z_effort).mean()
            all_sim = torch.mm(z_setpoint, z_effort.t()).mean()

        return {
            "loss": loss,
            "pos_similarity": pos_sim,
            "avg_similarity": all_sim,
        }

    def encode(
        self,
        setpoint: torch.Tensor,
        setpoint_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract representations.

        Args:
            setpoint: [B, T, 14] setpoint signals
            setpoint_mask: [B, 14] validity mask

        Returns:
            [B, hidden_dim] representation
        """
        return self.forward_setpoint(setpoint, setpoint_mask)

    def encode_both(
        self,
        setpoint: torch.Tensor,
        effort: torch.Tensor,
        setpoint_mask: Optional[torch.Tensor] = None,
        effort_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Encode both setpoint and effort.

        Returns:
            (setpoint_repr, effort_repr) both [B, hidden_dim]
        """
        h_setpoint = self.forward_setpoint(setpoint, setpoint_mask)
        h_effort = self.forward_effort(effort, effort_mask)
        return h_setpoint, h_effort

    @torch.no_grad()
    def compute_anomaly_score(
        self,
        setpoint: torch.Tensor,
        effort: torch.Tensor,
        setpoint_mask: Optional[torch.Tensor] = None,
        effort_mask: Optional[torch.Tensor] = None,
        healthy_center: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute anomaly scores.

        For cross-modal mode:
        - Anomaly = low similarity between setpoint and effort representations
        - (Faults break the setpoint->effort relationship)

        For SimCLR mode:
        - Distance to healthy cluster center

        Args:
            setpoint: [B, T, 14] setpoint signals
            effort: [B, T, 7] effort signals
            setpoint_mask: [B, 14] validity mask
            effort_mask: [B, 7] validity mask
            healthy_center: [hidden_dim] mean healthy representation

        Returns:
            [B] anomaly scores (higher = more anomalous)
        """
        self.eval()

        if self.config.use_effort_pairs:
            # Cross-modal: use dissimilarity as anomaly score
            h_setpoint = self.forward_setpoint(setpoint, setpoint_mask)
            h_effort = self.forward_effort(effort, effort_mask)

            z_setpoint = self.project(h_setpoint)
            z_effort = self.project(h_effort)

            # Negative cosine similarity (higher = more anomalous)
            scores = 1 - F.cosine_similarity(z_setpoint, z_effort)

        else:
            # SimCLR: distance to healthy center
            h = self.forward_setpoint(setpoint, setpoint_mask)

            if healthy_center is not None:
                # Euclidean distance to healthy center
                scores = torch.norm(h - healthy_center.unsqueeze(0), dim=-1)
            else:
                # Use norm as proxy (assumes healthy samples cluster near origin)
                scores = torch.norm(h, dim=-1)

        return scores

    def compute_healthy_center(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Compute mean representation of healthy samples.

        Args:
            dataloader: DataLoader with healthy samples
            device: Device to use

        Returns:
            [hidden_dim] mean representation
        """
        self.eval()
        all_repr = []

        with torch.no_grad():
            for batch in dataloader:
                setpoint = batch["setpoint"].to(device)
                setpoint_mask = batch.get("setpoint_mask")
                if setpoint_mask is not None:
                    setpoint_mask = setpoint_mask.to(device)

                h = self.forward_setpoint(setpoint, setpoint_mask)
                all_repr.append(h)

        all_repr = torch.cat(all_repr, dim=0)
        return all_repr.mean(dim=0)
