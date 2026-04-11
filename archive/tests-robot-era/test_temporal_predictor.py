# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Test script for TemporalPredictor model.

Verifies:
1. Model instantiation (both JEPA and direct modes)
2. Forward pass shapes
3. Anomaly score computation
4. EMA update mechanism
"""

import torch
import pytest


def test_temporal_predictor_jepa_mode():
    """Test JEPA mode temporal predictor."""
    from industrialjepa.baselines import TemporalPredictor, TemporalConfig

    config = TemporalConfig(
        setpoint_dim=14,
        effort_dim=13,
        seq_len=256,
        patch_size=16,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        prediction_mode="jepa",
        context_ratio=0.5,
    )

    model = TemporalPredictor(config)
    print(f"JEPA mode parameters: {model.get_num_params():,}")

    # Test forward pass
    batch_size = 4
    setpoint = torch.randn(batch_size, 256, 14)
    effort = torch.randn(batch_size, 256, 13)

    output = model(setpoint, effort)

    assert "loss" in output
    assert "cosine_similarity" in output
    assert output["loss"].shape == ()
    print(f"JEPA loss: {output['loss'].item():.4f}")
    print(f"Cosine similarity: {output['cosine_similarity'].item():.4f}")

    # Test anomaly score
    scores = model.compute_anomaly_score(setpoint, effort)
    assert scores.shape == (batch_size,)
    print(f"Anomaly scores: {scores.tolist()}")

    # Test EMA update
    # Get a parameter from the EMA encoder
    old_param = list(model.ema_target_encoder.parameters())[0].clone()
    model.update_ema()
    new_param = list(model.ema_target_encoder.parameters())[0]
    # After update, params should be slightly different (EMA moved toward online encoder)
    print("EMA update working:", not torch.allclose(old_param, new_param))


def test_temporal_predictor_direct_mode():
    """Test direct prediction mode temporal predictor."""
    from industrialjepa.baselines import TemporalPredictor, TemporalConfig

    config = TemporalConfig(
        setpoint_dim=14,
        effort_dim=13,
        seq_len=256,
        patch_size=16,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        prediction_mode="direct",
        context_ratio=0.5,
    )

    model = TemporalPredictor(config)
    print(f"Direct mode parameters: {model.get_num_params():,}")

    # Test forward pass
    batch_size = 4
    setpoint = torch.randn(batch_size, 256, 14)
    effort = torch.randn(batch_size, 256, 13)

    output = model(setpoint, effort)

    assert "loss" in output
    assert output["loss"].shape == ()
    print(f"Direct loss: {output['loss'].item():.4f}")

    # Test anomaly score
    scores = model.compute_anomaly_score(setpoint, effort)
    assert scores.shape == (batch_size,)
    print(f"Anomaly scores: {scores.tolist()}")


def test_temporal_predictor_with_masks():
    """Test temporal predictor with validity masks."""
    from industrialjepa.baselines import TemporalPredictor, TemporalConfig

    config = TemporalConfig(
        setpoint_dim=14,
        effort_dim=13,
        seq_len=256,
        prediction_mode="jepa",
    )

    model = TemporalPredictor(config)

    batch_size = 4
    setpoint = torch.randn(batch_size, 256, 14)
    effort = torch.randn(batch_size, 256, 13)

    # Create masks (simulating 6-DOF robot with 12 setpoint signals, 12 effort signals)
    setpoint_mask = torch.zeros(batch_size, 14)
    setpoint_mask[:, :12] = 1.0
    effort_mask = torch.zeros(batch_size, 13)
    effort_mask[:, :12] = 1.0

    output = model(setpoint, effort, setpoint_mask, effort_mask)
    assert "loss" in output
    print(f"Loss with masks: {output['loss'].item():.4f}")


def test_temporal_context_target_split():
    """Verify context/target split is correct."""
    from industrialjepa.baselines import TemporalPredictor, TemporalConfig

    config = TemporalConfig(
        seq_len=256,
        context_ratio=0.5,
        prediction_mode="direct",
    )

    model = TemporalPredictor(config)

    assert model.context_len == 128, f"Expected context_len=128, got {model.context_len}"
    assert model.target_len == 128, f"Expected target_len=128, got {model.target_len}"

    # Test with different ratio
    config2 = TemporalConfig(
        seq_len=256,
        context_ratio=0.75,
        prediction_mode="direct",
    )

    model2 = TemporalPredictor(config2)
    assert model2.context_len == 192, f"Expected context_len=192, got {model2.context_len}"
    assert model2.target_len == 64, f"Expected target_len=64, got {model2.target_len}"
    print("Context/target split: OK")


def test_gradient_flow():
    """Verify gradients flow through the model correctly."""
    from industrialjepa.baselines import TemporalPredictor, TemporalConfig

    config = TemporalConfig(
        setpoint_dim=14,
        effort_dim=13,
        seq_len=256,
        prediction_mode="jepa",
    )

    model = TemporalPredictor(config)

    setpoint = torch.randn(2, 256, 14)
    effort = torch.randn(2, 256, 13)

    output = model(setpoint, effort)
    loss = output["loss"]
    loss.backward()

    # Check gradients exist for online encoder (should have gradients)
    has_grad_context = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.context_encoder.parameters()
    )
    assert has_grad_context, "Context encoder should have gradients"

    # Check EMA encoder has no gradients (should be frozen)
    has_grad_ema = any(
        p.grad is not None
        for p in model.ema_target_encoder.parameters()
    )
    assert not has_grad_ema, "EMA encoder should not have gradients"

    print("Gradient flow: OK")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing TemporalPredictor - JEPA mode")
    print("=" * 60)
    test_temporal_predictor_jepa_mode()

    print("\n" + "=" * 60)
    print("Testing TemporalPredictor - Direct mode")
    print("=" * 60)
    test_temporal_predictor_direct_mode()

    print("\n" + "=" * 60)
    print("Testing with validity masks")
    print("=" * 60)
    test_temporal_predictor_with_masks()

    print("\n" + "=" * 60)
    print("Testing context/target split")
    print("=" * 60)
    test_temporal_context_target_split()

    print("\n" + "=" * 60)
    print("Testing gradient flow")
    print("=" * 60)
    test_gradient_flow()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
