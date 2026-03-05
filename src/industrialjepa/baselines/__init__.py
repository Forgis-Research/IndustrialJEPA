# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Baseline models for comparison with IndustrialJEPA.

Implements several self-supervised learning approaches:
- MAE: Masked AutoEncoder (reconstruct masked portions)
- Autoencoder: Simple encoder-decoder (reconstruct Effort from Setpoint)
- Contrastive: SimCLR-style contrastive learning
"""

from .base import BaselineModel, BaselineConfig
from .mae import MAE, MAEConfig
from .autoencoder import Autoencoder, AutoencoderConfig
from .contrastive import ContrastiveModel, ContrastiveConfig

__all__ = [
    "BaselineModel",
    "BaselineConfig",
    "MAE",
    "MAEConfig",
    "Autoencoder",
    "AutoencoderConfig",
    "ContrastiveModel",
    "ContrastiveConfig",
]
