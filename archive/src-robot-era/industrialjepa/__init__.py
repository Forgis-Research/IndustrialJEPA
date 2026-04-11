"""
IndustrialJEPA: JEPA-based fault detection and cross-machine transfer for industrial robotics.

This package implements Joint Embedding Predictive Architecture (JEPA) for learning
machine physics from industrial time series data. The key insight is that by predicting
Effort from Setpoint in latent space, the model learns transferable physics rather than
hardware-specific statistics.

Main components:
- model: JEPA architecture (encoder, predictor, backbone)
- data: FactoryNet dataloader and transforms
- training: JEPA training loop and losses
- evaluation: Fault detection and Q&A benchmarks
"""

__version__ = "0.1.0"
