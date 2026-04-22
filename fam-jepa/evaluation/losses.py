"""
Canonical loss functions for FAM training.

Positive-weighted BCE for imbalanced probability surfaces p(t, Δt).
"""

import torch
import torch.nn.functional as F
from typing import Optional


def weighted_bce_loss(
    p_surface: torch.Tensor,
    y_surface: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Positive-weighted BCE for probability surface p(t, Δt).

    Handles class imbalance where events are rare: most cells are y=0.
    Without pos_weight, a trivial p=0 model gets low loss.
    With pos_weight = N_neg / N_pos, positive cells contribute equally.

    Args:
        p_surface: (B, T, K) predicted probabilities (after sigmoid)
                   or logits if using logit version
        y_surface: (B, T, K) binary targets, 1 = event within Δt_k of time t
        pos_weight: scalar or (K,) tensor. If None, auto-computed from y_surface.
        mask: (B, T) bool tensor, True = padding (exclude from loss)

    Returns:
        Scalar loss (mean over valid cells).
    """
    if pos_weight is None:
        pos_weight = compute_pos_weight(y_surface, mask)

    # Use BCEWithLogitsLoss-style if p_surface is logits
    # For pre-sigmoid logits, use F.binary_cross_entropy_with_logits
    # For post-sigmoid probs, use F.binary_cross_entropy with manual weighting
    loss = F.binary_cross_entropy_with_logits(
        p_surface, y_surface.float(),
        pos_weight=pos_weight,
        reduction='none',
    )  # (B, T, K)

    if mask is not None:
        # mask: (B, T) True=padding -> expand to (B, T, K)
        mask_3d = mask.unsqueeze(-1).expand_as(loss)
        loss = loss.masked_fill(mask_3d, 0.0)
        n_valid = (~mask_3d).sum().clamp(min=1)
        return loss.sum() / n_valid
    else:
        return loss.mean()


def compute_pos_weight(
    y_surface: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Auto-compute pos_weight = N_neg / N_pos from labels.

    Returns scalar tensor. Clamped to [1, 1000] to prevent explosion
    when positive rate is extremely low.

    Args:
        y_surface: (B, T, K) binary targets
        mask: (B, T) bool, True = padding
    """
    if mask is not None:
        mask_3d = mask.unsqueeze(-1).expand_as(y_surface)
        valid = ~mask_3d
        n_pos = (y_surface * valid.float()).sum()
        n_total = valid.sum().float()
    else:
        n_pos = y_surface.sum()
        n_total = torch.tensor(y_surface.numel(), dtype=torch.float32,
                               device=y_surface.device)

    n_neg = n_total - n_pos
    # Clamp to avoid div-by-zero and extreme weights
    pw = (n_neg / n_pos.clamp(min=1.0)).clamp(min=1.0, max=1000.0)
    return pw


def build_label_surface(
    time_to_event: torch.Tensor,
    horizons: torch.Tensor,
) -> torch.Tensor:
    """
    Build binary label surface y(t, Δt) from time-to-event values.

    y(t, Δt) = 1 if event occurs within Δt steps of time t, else 0.

    Args:
        time_to_event: (B, T) time-to-event at each observation time.
                       inf = no event in future.
        horizons: (K,) tensor of horizon values [Δt_1, ..., Δt_K]

    Returns:
        y_surface: (B, T, K) binary labels
    """
    # time_to_event: (B, T) -> (B, T, 1)
    # horizons: (K,) -> (1, 1, K)
    tte = time_to_event.unsqueeze(-1)  # (B, T, 1)
    h = horizons.unsqueeze(0).unsqueeze(0)  # (1, 1, K)

    # Event within horizon: tte <= Δt and tte is finite (not inf)
    y_surface = (tte <= h) & (tte.isfinite())
    return y_surface.float()
