#!/usr/bin/env python3
"""
Mechanical-JEPA Viability Test

Run this AFTER sanity_check.py passes.
Tests on 1k episodes for 10 epochs (~30 min).

Usage:
    python viability_test.py
"""

import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from sanity check (reuse model definition)
from sanity_check import MiniMechanicalJEPA, generate_synthetic_robot_data, check_embedding_collapse


def run_viability_test():
    """Run 30-minute viability test."""
    print("=" * 60)
    print("MECHANICAL-JEPA VIABILITY TEST")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    # ========================================
    # Config
    # ========================================
    config = {
        'n_episodes': 1000,
        'seq_len': 64,
        'n_joints': 7,
        'd_model': 64,      # Slightly larger than sanity check
        'n_heads': 4,
        'n_layers': 2,
        'batch_size': 32,
        'epochs': 10,
        'lr': 1e-4,
    }
    print("Config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    # ========================================
    # Data
    # ========================================
    print("[1/5] Generating data...")
    start = time.time()

    # TODO: Replace with real OXE data loading
    # For now, use synthetic data
    data = generate_synthetic_robot_data(
        n_episodes=config['n_episodes'],
        seq_len=config['seq_len'],
        n_joints=config['n_joints']
    )
    print(f"      Shape: {data.shape}")
    print(f"      Time: {time.time() - start:.1f}s")
    print()

    # Train/val split
    n_train = int(0.9 * len(data))
    train_data = data[:n_train]
    val_data = data[n_train:]
    print(f"      Train: {len(train_data)}, Val: {len(val_data)}")
    print()

    # ========================================
    # Model
    # ========================================
    print("[2/5] Creating model...")
    model = MiniMechanicalJEPA(
        input_dim=config['n_joints'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers']
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Parameters: {n_params:,}")
    print()

    # ========================================
    # Training
    # ========================================
    print("[3/5] Training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    train_losses = []
    val_losses = []

    for epoch in range(config['epochs']):
        epoch_start = time.time()

        # Train
        model.train()
        epoch_loss = 0
        n_batches = 0
        for i in range(0, len(train_data), config['batch_size']):
            batch = train_data[i:i+config['batch_size']]
            if len(batch) < config['batch_size']:
                continue

            optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.ema_update()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(train_loss)

        # Validate
        model.eval()
        with torch.no_grad():
            val_loss = model.compute_loss(val_data).item()
        val_losses.append(val_loss)

        scheduler.step()

        epoch_time = time.time() - epoch_start
        print(f"      Epoch {epoch+1:2d}/{config['epochs']}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"time={epoch_time:.1f}s")

    print()

    # ========================================
    # Evaluation
    # ========================================
    print("[4/5] Evaluating...")

    # Check loss trend
    loss_decreased = train_losses[-1] < train_losses[0]
    loss_decrease_pct = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
    print(f"      Train loss: {train_losses[0]:.4f} → {train_losses[-1]:.4f} ({loss_decrease_pct:.1f}% decrease)")

    # Check for overfitting
    val_ratio = val_losses[-1] / train_losses[-1] if train_losses[-1] > 0 else float('inf')
    print(f"      Val/Train ratio: {val_ratio:.2f} (should be <2.0)")

    # Check for collapse
    print()
    collapse_ok = check_embedding_collapse(model, val_data, n_samples=50)

    # Simple "classification" test: can we distinguish different trajectories?
    print("\n[5/5] Embedding quality check...")
    model.eval()
    with torch.no_grad():
        emb = model.encode(val_data[:50]).mean(dim=1)  # (50, d_model)

        # Check if embeddings for same trajectory are more similar
        # than for different trajectories (basic sanity)
        self_sim = F.cosine_similarity(emb[:25], emb[:25]).mean()
        cross_sim = F.cosine_similarity(emb[:25], emb[25:]).mean()
        print(f"      Self-similarity: {self_sim:.4f}")
        print(f"      Cross-similarity: {cross_sim:.4f}")

    # ========================================
    # Verdict
    # ========================================
    print()
    print("=" * 60)

    passed = True
    if not loss_decreased:
        print("✗ FAIL: Loss did not decrease")
        passed = False
    if val_ratio > 3.0:
        print("✗ FAIL: Severe overfitting (val/train > 3)")
        passed = False
    if not collapse_ok:
        print("✗ FAIL: Embedding collapse detected")
        passed = False

    if passed:
        print("VIABILITY TEST PASSED ✓")
        print("Safe to proceed to full training.")
    else:
        print("VIABILITY TEST FAILED ✗")
        print("Debug issues before scaling up.")

    print("=" * 60)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return passed


if __name__ == "__main__":
    passed = run_viability_test()
    sys.exit(0 if passed else 1)
