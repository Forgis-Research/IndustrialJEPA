#!/usr/bin/env python3
"""
Mechanical-JEPA Sanity Check Script

Run this BEFORE any overnight training to validate the implementation.
All 8 checks must pass before proceeding.

Usage:
    python sanity_check.py
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Minimal Model Implementation (for sanity checking)
# ============================================================================

class MiniEncoder(nn.Module):
    """Tiny encoder for sanity checks."""
    def __init__(self, input_dim=7, d_model=32, n_heads=2, n_layers=1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.norm(x)


class MiniPredictor(nn.Module):
    """Tiny predictor for sanity checks."""
    def __init__(self, d_model=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, z_context, mask_positions):
        # Predict masked positions from context
        return self.net(z_context)


class MiniMechanicalJEPA(nn.Module):
    """Minimal JEPA for sanity checking."""

    def __init__(self, input_dim=7, d_model=32, n_heads=2, n_layers=1):
        super().__init__()
        self.encoder = MiniEncoder(input_dim, d_model, n_heads, n_layers)
        self.target_encoder = MiniEncoder(input_dim, d_model, n_heads, n_layers)
        self.predictor = MiniPredictor(d_model)

        # Initialize target as copy of encoder
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.d_model = d_model
        self.mask_ratio = 0.3

    def create_mask(self, seq_len, mask_ratio=None):
        """Create random temporal mask."""
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        n_mask = int(seq_len * mask_ratio)
        mask = torch.zeros(seq_len, dtype=torch.bool)
        mask_idx = torch.randperm(seq_len)[:n_mask]
        mask[mask_idx] = True
        return mask

    def encode(self, x):
        """Encode input sequence."""
        return self.encoder(x)

    def compute_loss(self, x):
        """Compute JEPA loss."""
        batch_size, seq_len, input_dim = x.shape

        # Create mask
        mask = self.create_mask(seq_len)
        context_mask = ~mask

        # Encode context (unmasked positions)
        z_context = self.encoder(x)

        # Get target embeddings for masked positions
        with torch.no_grad():
            z_target = self.target_encoder(x)

        # Predict masked positions
        z_pred = self.predictor(z_context, mask)

        # Loss: L2 between predicted and target at masked positions
        loss = F.mse_loss(z_pred[:, mask], z_target[:, mask])

        return loss

    def ema_update(self, decay=0.996):
        """Update target encoder with EMA."""
        with torch.no_grad():
            for p_enc, p_tgt in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                p_tgt.data = decay * p_tgt.data + (1 - decay) * p_enc.data


# ============================================================================
# Synthetic Data for Sanity Checks
# ============================================================================

def generate_synthetic_robot_data(n_episodes=100, seq_len=32, n_joints=7):
    """Generate synthetic robot joint trajectories for testing."""
    data = []
    for _ in range(n_episodes):
        # Random initial position
        q0 = np.random.uniform(-np.pi, np.pi, n_joints)
        # Random smooth trajectory (random walk with momentum)
        trajectory = [q0]
        velocity = np.zeros(n_joints)
        for t in range(seq_len - 1):
            acceleration = np.random.randn(n_joints) * 0.1
            velocity = 0.9 * velocity + acceleration
            q_next = trajectory[-1] + velocity * 0.1
            # Clip to joint limits
            q_next = np.clip(q_next, -np.pi, np.pi)
            trajectory.append(q_next)
        data.append(np.array(trajectory, dtype=np.float32))
    return torch.tensor(np.stack(data))


# ============================================================================
# Sanity Checks
# ============================================================================

def run_sanity_checks():
    """Run all 8 sanity checks."""
    print("=" * 60)
    print("MECHANICAL-JEPA SANITY CHECKS")
    print("=" * 60)
    print()

    all_passed = True

    # ========================================
    # Check 1: Data loads
    # ========================================
    print("[1/8] Data loading...")
    try:
        data = generate_synthetic_robot_data(n_episodes=100, seq_len=32, n_joints=7)
        assert not torch.isnan(data).any(), "NaN in data!"
        assert not torch.isinf(data).any(), "Inf in data!"
        print(f"      Shape: {data.shape}")
        print(f"      Range: [{data.min():.2f}, {data.max():.2f}]")
        print("      ✓ PASSED")
    except Exception as e:
        print(f"      ✗ FAILED: {e}")
        all_passed = False
    print()

    # ========================================
    # Check 2: Forward pass
    # ========================================
    print("[2/8] Forward pass...")
    try:
        model = MiniMechanicalJEPA(input_dim=7, d_model=32, n_heads=2, n_layers=1)
        batch = data[:8]
        z = model.encode(batch)
        assert z.shape == (8, 32, 32), f"Wrong output shape: {z.shape}"
        print(f"      Input: {batch.shape} → Output: {z.shape}")
        print("      ✓ PASSED")
    except Exception as e:
        print(f"      ✗ FAILED: {e}")
        all_passed = False
    print()

    # ========================================
    # Check 3: Loss computes
    # ========================================
    print("[3/8] Loss computation...")
    try:
        loss = model.compute_loss(batch)
        assert torch.isfinite(loss), f"Loss is {loss}!"
        assert loss > 0, f"Loss should be positive: {loss}"
        print(f"      Loss: {loss.item():.4f}")
        print("      ✓ PASSED")
    except Exception as e:
        print(f"      ✗ FAILED: {e}")
        all_passed = False
    print()

    # ========================================
    # Check 4: Loss decreases
    # ========================================
    print("[4/8] Loss decreases over 10 steps...")
    try:
        model = MiniMechanicalJEPA(input_dim=7, d_model=32, n_heads=2, n_layers=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()
            optimizer.step()
            model.ema_update()
            losses.append(loss.item())

        decrease_pct = (losses[0] - losses[-1]) / losses[0] * 100
        print(f"      Loss: {losses[0]:.4f} → {losses[-1]:.4f} ({decrease_pct:.1f}% decrease)")

        if losses[-1] < losses[0]:
            print("      ✓ PASSED")
        else:
            print("      ✗ FAILED: Loss did not decrease")
            all_passed = False
    except Exception as e:
        print(f"      ✗ FAILED: {e}")
        all_passed = False
    print()

    # ========================================
    # Check 5: Gradients flow
    # ========================================
    print("[5/8] Gradients flow...")
    try:
        model = MiniMechanicalJEPA(input_dim=7, d_model=32, n_heads=2, n_layers=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss = model.compute_loss(batch)
        loss.backward()

        all_grads_ok = True
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"      NaN gradient in {name}")
                    all_grads_ok = False
                if torch.isinf(param.grad).any():
                    print(f"      Inf gradient in {name}")
                    all_grads_ok = False
                if param.grad.abs().max() > 100:
                    print(f"      Large gradient in {name}: {param.grad.abs().max():.2f}")

        if all_grads_ok:
            print("      ✓ PASSED")
        else:
            all_passed = False
    except Exception as e:
        print(f"      ✗ FAILED: {e}")
        all_passed = False
    print()

    # ========================================
    # Check 6: EMA updates
    # ========================================
    print("[6/8] EMA updates target encoder...")
    try:
        model = MiniMechanicalJEPA(input_dim=7, d_model=32, n_heads=2, n_layers=1)

        # Get initial target params
        tgt_before = sum(p.sum().item() for p in model.target_encoder.parameters())

        # Train a step (encoder changes)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        optimizer.zero_grad()
        loss = model.compute_loss(batch)
        loss.backward()
        optimizer.step()

        # EMA update
        model.ema_update(decay=0.9)

        tgt_after = sum(p.sum().item() for p in model.target_encoder.parameters())

        if abs(tgt_after - tgt_before) > 1e-6:
            print(f"      Target changed: {tgt_before:.4f} → {tgt_after:.4f}")
            print("      ✓ PASSED")
        else:
            print("      ✗ FAILED: Target did not change")
            all_passed = False
    except Exception as e:
        print(f"      ✗ FAILED: {e}")
        all_passed = False
    print()

    # ========================================
    # Check 7: Overfits single batch
    # ========================================
    print("[7/8] Overfits single batch (100 steps)...")
    try:
        model = MiniMechanicalJEPA(input_dim=7, d_model=32, n_heads=2, n_layers=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        single_batch = data[:4]

        initial_loss = None
        for i in range(100):
            optimizer.zero_grad()
            loss = model.compute_loss(single_batch)
            loss.backward()
            optimizer.step()
            model.ema_update()
            if initial_loss is None:
                initial_loss = loss.item()

        final_loss = loss.item()
        print(f"      Loss: {initial_loss:.4f} → {final_loss:.6f}")

        if final_loss < 0.1:
            print("      ✓ PASSED")
        else:
            print("      ✗ FAILED: Could not overfit (loss still high)")
            all_passed = False
    except Exception as e:
        print(f"      ✗ FAILED: {e}")
        all_passed = False
    print()

    # ========================================
    # Check 8: Masking works
    # ========================================
    print("[8/8] Masking works...")
    try:
        model = MiniMechanicalJEPA(input_dim=7, d_model=32, n_heads=2, n_layers=1)
        mask = model.create_mask(seq_len=32, mask_ratio=0.3)

        masked_count = mask.sum().item()
        expected = int(32 * 0.3)

        print(f"      Masked: {masked_count}/{32} positions (expected ~{expected})")

        if 5 <= masked_count <= 15:  # Reasonable range
            print("      ✓ PASSED")
        else:
            print("      ✗ FAILED: Unexpected mask count")
            all_passed = False
    except Exception as e:
        print(f"      ✗ FAILED: {e}")
        all_passed = False
    print()

    # ========================================
    # Summary
    # ========================================
    print("=" * 60)
    if all_passed:
        print("ALL SANITY CHECKS PASSED ✓")
        print("Safe to proceed to viability test.")
    else:
        print("SOME CHECKS FAILED ✗")
        print("Fix issues before proceeding.")
    print("=" * 60)

    return all_passed


# ============================================================================
# Embedding Collapse Check
# ============================================================================

def check_embedding_collapse(model, data, n_samples=100):
    """Check if embeddings have collapsed to a constant."""
    print("\n[COLLAPSE CHECK]")

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(data[:n_samples])

        # Flatten to (n_samples, d_model) by taking mean over time
        emb_flat = embeddings.mean(dim=1)

        # Check variance
        var = emb_flat.var(dim=0).mean().item()
        print(f"  Embedding variance: {var:.6f}")

        # Check pairwise distances
        dists = torch.cdist(emb_flat, emb_flat)
        # Exclude diagonal
        mask = ~torch.eye(n_samples, dtype=torch.bool)
        mean_dist = dists[mask].mean().item()
        print(f"  Mean pairwise distance: {mean_dist:.4f}")

        if var < 0.001:
            print("  ⚠️  WARNING: Very low variance — possible collapse!")
            return False
        if mean_dist < 0.01:
            print("  ⚠️  WARNING: Embeddings too similar — possible collapse!")
            return False

        print("  ✓ No collapse detected")
        return True


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    passed = run_sanity_checks()

    if passed:
        print("\nRunning collapse check...")
        data = generate_synthetic_robot_data(n_episodes=100, seq_len=32, n_joints=7)
        model = MiniMechanicalJEPA(input_dim=7, d_model=32, n_heads=2, n_layers=1)

        # Quick training
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(50):
            optimizer.zero_grad()
            loss = model.compute_loss(data[:32])
            loss.backward()
            optimizer.step()
            model.ema_update()

        check_embedding_collapse(model, data)

    sys.exit(0 if passed else 1)
