"""
DCSSL Architecture and Baseline Models.

Implements:
1. TCN Encoder (Temporal Convolutional Network) — from Bai et al. 2018 + Franceschi et al. 2019
2. DCSSL (Dual-Dimensional Contrastive SSL) — Shen et al. 2026
3. SimCLR baseline
4. SupCon baseline
5. RUL Prediction Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


# =====================================================================
# TCN Encoder (core encoder used by DCSSL and SimCLR)
# =====================================================================

class Chomp1d(nn.Module):
    """Remove future-leaking padding for causal convolution."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TCNResidualBlock(nn.Module):
    """
    TCN residual block with dilated causal convolution.

    Based on Bai et al. 2018 "An Empirical Evaluation of Generic Convolutional
    and Recurrent Networks for Sequence Modeling".

    Uses BatchNorm instead of WeightNorm for stability (weight_norm can produce NaN).
    """

    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int,
                 stride: int, dilation: int, padding: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity='relu')

    def forward(self, x):
        # First conv block
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        # Second conv block
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Skip connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNEncoder(nn.Module):
    """
    Temporal Convolutional Network encoder.

    Input:  (batch, channels, time) e.g. (B, 2, 1024)
    Output: (batch, hidden_dim)  — uses global average pooling over time

    Based on Franceschi et al. 2019 "Unsupervised Scalable Representation Learning
    for Multivariate Time Series" and Bai et al. 2018.
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 64,
        out_channels: int = 128,
        kernel_size: int = 3,
        n_blocks: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        num_levels = n_blocks
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = hidden_channels if i < num_levels - 1 else out_channels
            padding = (kernel_size - 1) * dilation_size
            layers.append(TCNResidualBlock(
                in_ch, out_ch, kernel_size,
                stride=1, dilation=dilation_size,
                padding=padding, dropout=dropout
            ))
        self.network = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, time)
        Returns:
            (batch, out_channels) — global average pooled representation
        """
        out = self.network(x)  # (batch, out_channels, time)
        return out.mean(dim=-1)  # global average pool


# =====================================================================
# Projection Head
# =====================================================================

class ProjectionHead(nn.Module):
    """Standard 2-layer MLP projection head for contrastive learning."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# =====================================================================
# Contrastive Losses
# =====================================================================

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

    Args:
        z1, z2: (batch, dim) — two views of the same samples
        temperature: temperature scaling factor

    Returns:
        scalar loss
    """
    batch_size = z1.shape[0]
    device = z1.device

    if batch_size < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Normalize — add small epsilon to avoid NaN when norm is 0
    z1 = F.normalize(z1 + 1e-8, dim=1)
    z2 = F.normalize(z2 + 1e-8, dim=1)

    # Concatenate all representations
    z = torch.cat([z1, z2], dim=0)  # (2B, dim)

    # Similarity matrix
    sim = torch.mm(z, z.T) / temperature  # (2B, 2B)

    # Clamp to avoid overflow in exp
    sim = torch.clamp(sim, min=-100.0, max=100.0)

    # Mask out self-similarities
    mask = torch.eye(2 * batch_size, device=device).bool()
    sim.masked_fill_(mask, -1e9)

    # Labels: positive pairs are (i, i+B) and (i+B, i)
    labels = torch.arange(batch_size, device=device)
    labels = torch.cat([labels + batch_size, labels], dim=0)  # (2B,)

    loss = F.cross_entropy(sim, labels)

    # Check for NaN
    if torch.isnan(loss):
        return torch.tensor(0.0, device=device, requires_grad=True)

    return loss


def temporal_contrastive_loss(
    z: torch.Tensor,
    bearing_indices: torch.Tensor,
    time_indices: torch.Tensor,
    n_snapshots: torch.Tensor,
    temperature: float = 0.1,
    temporal_window: float = 0.1,
) -> torch.Tensor:
    """
    Temporal-level contrastive loss (DCSSL Eq. 1 / "within-bearing" loss).

    Nearby timesteps (within temporal_window fraction of bearing lifetime)
    are positive pairs. Distant timesteps are negatives.

    Args:
        z: (batch, dim) — projected representations
        bearing_indices: (batch,) — which bearing each sample is from
        time_indices: (batch,) — snapshot index within bearing
        n_snapshots: (batch,) — total snapshots in that bearing
        temperature: NT-Xent temperature
        temporal_window: fraction of bearing lifetime to consider "nearby"

    Returns:
        scalar loss
    """
    batch_size = z.shape[0]
    device = z.device

    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.T) / temperature  # (B, B)

    # Mask out same-sample (diagonal)
    eye_mask = torch.eye(batch_size, device=device).bool()

    # Compute normalized time positions: t / n_snapshots
    t_norm = time_indices.float() / n_snapshots.float()  # (batch,)

    # Build positive mask: same bearing AND nearby in time
    same_bearing = bearing_indices.unsqueeze(1) == bearing_indices.unsqueeze(0)  # (B, B)
    time_diff = (t_norm.unsqueeze(1) - t_norm.unsqueeze(0)).abs()  # (B, B)
    nearby = time_diff < temporal_window

    pos_mask = same_bearing & nearby & ~eye_mask

    # If no positives in batch, return zero loss
    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # For each sample, compute loss against its positives
    # Using InfoNCE formulation
    sim_masked = sim.clone()
    sim_masked.masked_fill_(eye_mask, -1e9)

    # Log-sum-exp over all non-self pairs (denominators)
    log_sum_exp = torch.logsumexp(sim_masked, dim=1)  # (B,)

    # Average log-prob of positive pairs
    pos_sim = sim * pos_mask.float()  # zero out non-positives
    pos_count = pos_mask.float().sum(dim=1).clamp(min=1)
    avg_pos_sim = pos_sim.sum(dim=1) / pos_count

    loss = -(avg_pos_sim - log_sum_exp)

    # Only include samples that have at least one positive
    has_pos = pos_mask.any(dim=1)
    if has_pos.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return loss[has_pos].mean()


def instance_contrastive_loss(
    z: torch.Tensor,
    bearing_indices: torch.Tensor,
    time_indices: torch.Tensor,
    n_snapshots: torch.Tensor,
    temperature: float = 0.1,
    rul_window: float = 0.1,
    rul: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Instance-level contrastive loss (DCSSL Eq. 2 / "cross-bearing" loss).

    Samples from different bearings at similar degradation stages
    (similar normalized RUL position) form positive pairs.

    Args:
        z: (batch, dim)
        bearing_indices: (batch,)
        time_indices: (batch,)
        n_snapshots: (batch,)
        temperature: NT-Xent temperature
        rul_window: RUL proximity threshold for positive pairs
        rul: (batch,) optional — if provided, use actual RUL instead of normalized time
             for degradation stage proximity (fixes FPT distribution shift problem)

    Returns:
        scalar loss
    """
    batch_size = z.shape[0]
    device = z.device

    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.T) / temperature  # (B, B)

    eye_mask = torch.eye(batch_size, device=device).bool()

    if rul is not None:
        # Use actual RUL values for degradation stage — fixes FPT distribution shift
        # Samples with similar RUL from different bearings are positive pairs
        rul_flat = rul.squeeze().float()
        stage_diff = (rul_flat.unsqueeze(1) - rul_flat.unsqueeze(0)).abs()
        similar_stage = stage_diff < rul_window
    else:
        # Fall back to normalized time position (original, FPT-dependent)
        t_norm = time_indices.float() / n_snapshots.float()
        time_diff = (t_norm.unsqueeze(1) - t_norm.unsqueeze(0)).abs()
        similar_stage = time_diff < rul_window

    # Positive mask: different bearings + similar degradation stage
    diff_bearing = bearing_indices.unsqueeze(1) != bearing_indices.unsqueeze(0)

    pos_mask = diff_bearing & similar_stage & ~eye_mask

    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    sim_masked = sim.clone()
    sim_masked.masked_fill_(eye_mask, -1e9)
    log_sum_exp = torch.logsumexp(sim_masked, dim=1)

    pos_sim = sim * pos_mask.float()
    pos_count = pos_mask.float().sum(dim=1).clamp(min=1)
    avg_pos_sim = pos_sim.sum(dim=1) / pos_count

    loss = -(avg_pos_sim - log_sum_exp)

    has_pos = pos_mask.any(dim=1)
    if has_pos.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return loss[has_pos].mean()


def supcon_loss(
    z: torch.Tensor,
    rul: torch.Tensor,
    temperature: float = 0.1,
    rul_window: float = 0.1,
) -> torch.Tensor:
    """
    Supervised contrastive loss where samples with similar RUL are positives.

    From Khosla et al. NeurIPS 2020 "Supervised Contrastive Learning",
    adapted for RUL regression with a proximity criterion.
    """
    batch_size = z.shape[0]
    device = z.device

    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.T) / temperature

    eye_mask = torch.eye(batch_size, device=device).bool()

    # Similar RUL = positive pairs
    rul_diff = (rul.unsqueeze(1) - rul.unsqueeze(0)).abs()
    pos_mask = (rul_diff < rul_window) & ~eye_mask

    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    sim_masked = sim.clone()
    sim_masked.masked_fill_(eye_mask, -1e9)
    log_sum_exp = torch.logsumexp(sim_masked, dim=1)

    pos_sim = sim * pos_mask.float()
    pos_count = pos_mask.float().sum(dim=1).clamp(min=1)
    avg_pos_sim = pos_sim.sum(dim=1) / pos_count

    loss = -(avg_pos_sim - log_sum_exp)
    has_pos = pos_mask.any(dim=1)
    if has_pos.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return loss[has_pos].mean()


# =====================================================================
# RUL Prediction Head
# =====================================================================

class RULHead(nn.Module):
    """
    MLP prediction head for RUL regression.

    Optionally takes elapsed time as additional input (strongly improves
    performance on bearings with long healthy phases — the model learns
    that early in life = high RUL).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, use_elapsed_time: bool = True):
        super().__init__()
        self.use_elapsed_time = use_elapsed_time
        actual_in = in_dim + 1 if use_elapsed_time else in_dim
        self.net = nn.Sequential(
            nn.Linear(actual_in, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # RUL in [0, 1]
        )

    def forward(self, x: torch.Tensor,
                elapsed_time: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_elapsed_time and elapsed_time is not None:
            x = torch.cat([x, elapsed_time.view(-1, 1)], dim=-1)
        return self.net(x).squeeze(-1)


# =====================================================================
# DCSSL Full Model
# =====================================================================

class DCSSSLModel(nn.Module):
    """
    DCSSL: Dual-Dimensional Contrastive Self-Supervised Learning framework.

    Stage 1: Encoder + ProjectionHead for SSL pretraining
    Stage 2: Encoder + RULHead for RUL regression
    """

    def __init__(
        self,
        in_channels: int = 2,
        encoder_hidden: int = 64,
        encoder_out: int = 128,
        n_tcn_blocks: int = 8,
        kernel_size: int = 3,
        proj_hidden: int = 128,
        proj_out: int = 64,
        rul_hidden: int = 64,
        dropout: float = 0.1,
        temperature: float = 0.1,
        lambda_temporal: float = 1.0,
        lambda_instance: float = 1.0,
        temporal_window: float = 0.1,
        rul_window: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature
        self.lambda_temporal = lambda_temporal
        self.lambda_instance = lambda_instance
        self.temporal_window = temporal_window
        self.rul_window = rul_window

        self.encoder = TCNEncoder(
            in_channels=in_channels,
            hidden_channels=encoder_hidden,
            out_channels=encoder_out,
            kernel_size=kernel_size,
            n_blocks=n_tcn_blocks,
            dropout=dropout,
        )
        self.projector = ProjectionHead(encoder_out, proj_hidden, proj_out)
        self.rul_head = RULHead(encoder_out, rul_hidden, use_elapsed_time=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to representation space."""
        return self.encoder(x)

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """Project to contrastive space."""
        return self.projector(h)

    def predict_rul(self, x: torch.Tensor,
                    elapsed_time: Optional[torch.Tensor] = None) -> torch.Tensor:
        """End-to-end RUL prediction."""
        h = self.encoder(x)
        return self.rul_head(h, elapsed_time)

    def contrastive_loss(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
        bearing_indices: torch.Tensor,
        time_indices: torch.Tensor,
        n_snapshots: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute dual-dimensional contrastive loss.

        Args:
            view1, view2: (batch, channels, time) — two augmented views
            bearing_indices: (batch,) — which bearing
            time_indices: (batch,) — snapshot position in bearing
            n_snapshots: (batch,) — total snapshots in that bearing
            rul: (batch,) optional — actual RUL labels for degradation-stage proximity
        """
        rul = kwargs.get("rul", None)

        # Encode both views
        h1 = self.encoder(view1)  # (batch, encoder_out)
        h2 = self.encoder(view2)

        # Instance-level NT-Xent loss (standard SimCLR style)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        loss_ntxent = nt_xent_loss(z1, z2, self.temperature)

        # Temporal-level contrastive loss (within-bearing)
        # Use mean of both views' representations
        z_mean = F.normalize((F.normalize(z1, dim=1) + F.normalize(z2, dim=1)) / 2, dim=1)
        loss_temporal = temporal_contrastive_loss(
            z_mean, bearing_indices, time_indices, n_snapshots,
            self.temperature, self.temporal_window
        )

        # Instance-level cross-bearing loss
        # Use actual RUL for degradation-stage proximity (fixes FPT distribution shift)
        loss_instance = instance_contrastive_loss(
            z_mean, bearing_indices, time_indices, n_snapshots,
            self.temperature, self.rul_window, rul=rul
        )

        total_loss = (loss_ntxent +
                      self.lambda_temporal * loss_temporal +
                      self.lambda_instance * loss_instance)

        return total_loss, {
            "loss_ntxent": loss_ntxent.item(),
            "loss_temporal": loss_temporal.item(),
            "loss_instance": loss_instance.item(),
            "total": total_loss.item(),
        }


# =====================================================================
# SimCLR Baseline
# =====================================================================

class SimCLRModel(nn.Module):
    """
    SimCLR adapted for time series bearing data.
    Simplest contrastive baseline.
    """

    def __init__(
        self,
        in_channels: int = 2,
        encoder_hidden: int = 64,
        encoder_out: int = 128,
        n_tcn_blocks: int = 8,
        kernel_size: int = 3,
        proj_hidden: int = 128,
        proj_out: int = 64,
        rul_hidden: int = 64,
        dropout: float = 0.1,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature
        self.encoder = TCNEncoder(
            in_channels=in_channels,
            hidden_channels=encoder_hidden,
            out_channels=encoder_out,
            kernel_size=kernel_size,
            n_blocks=n_tcn_blocks,
            dropout=dropout,
        )
        self.projector = ProjectionHead(encoder_out, proj_hidden, proj_out)
        self.rul_head = RULHead(encoder_out, rul_hidden, use_elapsed_time=False)

    def encode(self, x):
        return self.encoder(x)

    def predict_rul(self, x, elapsed_time=None):
        h = self.encoder(x)
        return self.rul_head(h, elapsed_time)

    def contrastive_loss(self, view1, view2, **kwargs):
        h1 = self.encoder(view1)
        h2 = self.encoder(view2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        loss = nt_xent_loss(z1, z2, self.temperature)
        return loss, {"loss_ntxent": loss.item(), "total": loss.item()}


# =====================================================================
# SupCon Baseline
# =====================================================================

class SupConModel(nn.Module):
    """
    Supervised Contrastive Learning baseline.
    Uses RUL labels to define positive pairs during pretraining.
    """

    def __init__(
        self,
        in_channels: int = 2,
        encoder_hidden: int = 64,
        encoder_out: int = 128,
        n_tcn_blocks: int = 8,
        kernel_size: int = 3,
        proj_hidden: int = 128,
        proj_out: int = 64,
        rul_hidden: int = 64,
        dropout: float = 0.1,
        temperature: float = 0.1,
        rul_window: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature
        self.rul_window = rul_window
        self.encoder = TCNEncoder(
            in_channels=in_channels,
            hidden_channels=encoder_hidden,
            out_channels=encoder_out,
            kernel_size=kernel_size,
            n_blocks=n_tcn_blocks,
            dropout=dropout,
        )
        self.projector = ProjectionHead(encoder_out, proj_hidden, proj_out)
        self.rul_head = RULHead(encoder_out, rul_hidden, use_elapsed_time=False)

    def encode(self, x):
        return self.encoder(x)

    def predict_rul(self, x, elapsed_time=None):
        h = self.encoder(x)
        return self.rul_head(h, elapsed_time)

    def contrastive_loss(self, view1, view2, rul=None, **kwargs):
        h1 = self.encoder(view1)
        h2 = self.encoder(view2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)

        # Standard NT-Xent within views
        loss_ntxent = nt_xent_loss(z1, z2, self.temperature)

        # Supervised contrastive using RUL labels
        if rul is not None:
            z_all = torch.cat([z1, z2], dim=0)
            rul_all = torch.cat([rul.squeeze(), rul.squeeze()], dim=0)
            loss_supcon = supcon_loss(z_all, rul_all, self.temperature, self.rul_window)
            total = loss_ntxent + loss_supcon
            return total, {"loss_ntxent": loss_ntxent.item(),
                           "loss_supcon": loss_supcon.item(), "total": total.item()}

        return loss_ntxent, {"loss_ntxent": loss_ntxent.item(), "total": loss_ntxent.item()}


# =====================================================================
# Trivial Baselines (for sanity check)
# =====================================================================

class TrivialMeanPredictor:
    """Always predict the mean RUL from training data."""

    def __init__(self):
        self.mean_rul = 0.5

    def fit(self, rul_labels):
        self.mean_rul = float(np.mean(rul_labels))

    def predict(self, n_samples):
        return np.full(n_samples, self.mean_rul)


class LinearDecayPredictor:
    """Predict linearly decaying RUL from 1 to 0 over bearing lifetime."""

    def predict_bearing(self, n_snapshots):
        return np.linspace(1.0, 0.0, n_snapshots)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
