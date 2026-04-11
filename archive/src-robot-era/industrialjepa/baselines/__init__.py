# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Baseline models for comparison with IndustrialJEPA.

Implements several self-supervised learning approaches:
- MAE: Masked AutoEncoder (reconstruct masked portions)
- Autoencoder: Simple encoder-decoder (reconstruct Effort from Setpoint)
- Contrastive: SimCLR-style contrastive learning
- Temporal: Temporal self-prediction (predict future effort from past context)
"""

from .base import BaselineModel, BaselineConfig
from .mae import MAE, MAEConfig
from .autoencoder import (
    EffortAutoencoder,
    SetpointToEffort,
    AutoencoderConfig,
    Autoencoder,  # Alias for EffortAutoencoder
)
from .contrastive import ContrastiveModel, ContrastiveConfig
from .temporal import TemporalPredictor, TemporalConfig

__all__ = [
    "BaselineModel",
    "BaselineConfig",
    "MAE",
    "MAEConfig",
    "EffortAutoencoder",
    "SetpointToEffort",
    "Autoencoder",  # Alias for EffortAutoencoder
    "AutoencoderConfig",
    "ContrastiveModel",
    "ContrastiveConfig",
    "TemporalPredictor",
    "TemporalConfig",
]
