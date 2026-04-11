# SPDX-FileCopyrightText: 2025 Industrial World Model Authors
# SPDX-License-Identifier: MIT

"""Hybrid Mamba-Transformer backbone for dynamics modeling."""

from industrialjepa.model.backbone.mamba import MambaBlock, SelectiveSSM
from industrialjepa.model.backbone.attention import SparseAttention
from industrialjepa.model.backbone.film import FiLMConditioner
from industrialjepa.model.backbone.hybrid import HybridMambaTransformer

__all__ = [
    "MambaBlock",
    "SelectiveSSM",
    "SparseAttention",
    "FiLMConditioner",
    "HybridMambaTransformer",
]
