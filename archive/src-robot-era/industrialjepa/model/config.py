# SPDX-FileCopyrightText: 2025 Industrial World Model Authors
# SPDX-License-Identifier: MIT

"""Configuration for IndustrialWorldLM models."""

from dataclasses import dataclass, field
from typing import List, Optional, Literal


@dataclass
class VQVAEConfig:
    """Configuration for the Hierarchical VQ-VAE Tokenizer."""

    # Multi-scale patching
    scales: List[int] = field(default_factory=lambda: [16, 64, 256])
    hidden_dims: List[int] = field(default_factory=lambda: [64, 128, 256])

    # Codebook settings
    codebook_size: int = 8192
    codebook_dim: int = 256
    num_domain_codebooks: int = 4
    domain_codebook_size: int = 1024
    anomaly_codebook_size: int = 256

    # Quantization settings
    use_gm_vq: bool = True  # Gaussian Mixture VQ
    commitment_weight: float = 0.25
    entropy_weight: float = 0.1

    # Encoder settings
    encoder_layers: int = 4
    encoder_heads: int = 8
    encoder_dropout: float = 0.1


@dataclass
class MambaConfig:
    """Configuration for Mamba blocks."""

    d_model: int = 512
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.1


@dataclass
class TransformerConfig:
    """Configuration for sparse attention layers."""

    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 8192


@dataclass
class BackboneConfig:
    """Configuration for the Hybrid Mamba-Transformer backbone."""

    hidden_dim: int = 512
    num_mamba_blocks: int = 4
    num_attention_layers: int = 6  # Sparse attention every N mamba blocks
    attention_every_n: int = 4

    mamba: MambaConfig = field(default_factory=MambaConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)

    # Action conditioning
    action_dim: int = 32
    use_film_conditioning: bool = True


@dataclass
class LatentConfig:
    """Configuration for stochastic latent state."""

    hidden_dim: int = 512
    latent_dim: int = 1024  # num_categories * category_size
    num_categories: int = 32
    category_size: int = 32

    # KL settings
    free_bits: float = 1.0
    kl_weight: float = 0.1


@dataclass
class LLMConfig:
    """Configuration for LLM reasoning head."""

    llm_id: str = "meta-llama/Llama-3.2-3B"
    projector_hidden_dim: int = 1024
    num_soft_tokens: int = 8
    freeze_llm: bool = True
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32


@dataclass
class IndustrialWorldLMConfig:
    """Full configuration for IndustrialWorldLM."""

    # Model size preset
    model_size: Literal["base", "large", "xl"] = "large"

    # Component configs
    vqvae: VQVAEConfig = field(default_factory=VQVAEConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    latent: LatentConfig = field(default_factory=LatentConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Input settings
    input_channels: int = 1  # Will be set dynamically
    max_seq_len: int = 8192

    # Training settings
    dropout: float = 0.1

    # Domain settings
    domains: List[str] = field(default_factory=lambda: [
        "manufacturing", "energy", "aerospace", "chemical"
    ])

    @classmethod
    def from_preset(cls, size: Literal["base", "large", "xl"]) -> "IndustrialWorldLMConfig":
        """Create config from a preset size."""
        presets = {
            "base": {
                "vqvae": VQVAEConfig(
                    codebook_size=4096,
                    encoder_layers=4,
                ),
                "backbone": BackboneConfig(
                    hidden_dim=512,
                    num_mamba_blocks=3,
                    num_attention_layers=3,
                ),
                "latent": LatentConfig(hidden_dim=512),
                "llm": LLMConfig(llm_id="google/gemma-2-2b"),
            },
            "large": {
                "vqvae": VQVAEConfig(
                    codebook_size=8192,
                    encoder_layers=6,
                ),
                "backbone": BackboneConfig(
                    hidden_dim=768,
                    num_mamba_blocks=4,
                    num_attention_layers=6,
                ),
                "latent": LatentConfig(hidden_dim=768),
                "llm": LLMConfig(llm_id="meta-llama/Llama-3.2-3B"),
            },
            "xl": {
                "vqvae": VQVAEConfig(
                    codebook_size=16384,
                    encoder_layers=8,
                ),
                "backbone": BackboneConfig(
                    hidden_dim=1024,
                    num_mamba_blocks=6,
                    num_attention_layers=12,
                ),
                "latent": LatentConfig(hidden_dim=1024),
                "llm": LLMConfig(llm_id="meta-llama/Llama-3.1-8B"),
            },
        }

        preset = presets[size]
        return cls(
            model_size=size,
            vqvae=preset["vqvae"],
            backbone=preset["backbone"],
            latent=preset["latent"],
            llm=preset["llm"],
        )

    def __post_init__(self):
        """Ensure consistency across configs."""
        # Sync hidden dimensions
        self.backbone.mamba.d_model = self.backbone.hidden_dim
        self.backbone.transformer.d_model = self.backbone.hidden_dim
        self.latent.hidden_dim = self.backbone.hidden_dim
