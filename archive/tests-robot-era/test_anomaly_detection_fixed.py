# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Test anomaly detection with FIXED normalization.

Previous experiment failed (AUC-ROC = 0.47) because:
1. Per-window z-score normalization destroyed magnitude information
2. Phase mixing (loosening + tightening) confused the model

This test applies the fixes:
1. Global normalization (preserves absolute magnitude)
2. Tightening-only phase handling (unified physics)
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score

from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig
from industrialjepa.baselines import SetpointToEffort, AutoencoderConfig


def run_experiment():
    print("=" * 60)
    print("EXPERIMENT: Anomaly Detection with Fixed Normalization")
    print("=" * 60)

    # Fixed config: global normalization + tightening only
    config = FactoryNetConfig(
        dataset_name="Forgis/factorynet-hackathon",
        subset="aursad",
        window_size=256,
        stride=128,
        normalize=True,
        norm_mode="global",  # FIX 1: Use global normalization
        train_healthy_only=True,
        aursad_phase_handling="tightening_only",  # FIX 2: Only tightening phase
    )

    print("\nConfig:")
    print(f"  norm_mode: {config.norm_mode}")
    print(f"  aursad_phase_handling: {config.aursad_phase_handling}")
    print(f"  train_healthy_only: {config.train_healthy_only}")

    # Load datasets
    print("\nLoading datasets...")
    train_ds = FactoryNetDataset(config, split="train")
    test_ds = FactoryNetDataset(config, split="test")

    print(f"Train windows: {len(train_ds)}")
    print(f"Test windows: {len(test_ds)}")

    # Count anomalies in test set
    test_labels = []
    test_faults = []
    for i in range(len(test_ds)):
        _, _, meta = test_ds[i]
        test_labels.append(1 if meta["is_anomaly"] else 0)
        test_faults.append(meta["fault_type"])

    n_anomaly = sum(test_labels)
    n_healthy = len(test_labels) - n_anomaly
    print(f"Test set: {n_healthy} healthy, {n_anomaly} anomalies")

    # Create model
    model_config = AutoencoderConfig(
        setpoint_dim=14,
        effort_dim=7,
        seq_len=256,
        patch_size=16,
        hidden_dim=128,  # Smaller for faster training
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        decoder_hidden_dim=128,
        decoder_num_layers=2,
        use_bottleneck=False,
        learning_rate=1e-3,
        weight_decay=0.01,
    )

    model = SetpointToEffort(model_config)
    print(f"\nModel parameters: {model.get_num_params():,}")

    # Training
    print("\nTraining on healthy data...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # Simple training loop (mini epochs for speed)
    batch_size = 64
    n_epochs = 5

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        n_batches = 0

        # Shuffle indices
        indices = torch.randperm(len(train_ds))

        for i in range(0, len(train_ds), batch_size):
            batch_idx = indices[i:i+batch_size]
            if len(batch_idx) < batch_size:
                continue

            # Load batch
            setpoints = []
            efforts = []
            for idx in batch_idx:
                s, e, _ = train_ds[idx.item()]
                setpoints.append(s)
                efforts.append(e)

            setpoint = torch.stack(setpoints)
            effort = torch.stack(efforts)

            # Forward
            optimizer.zero_grad()
            output = model(setpoint=setpoint, effort=effort)
            loss = output["loss"]

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}/{n_epochs}: loss = {avg_loss:.4f}")

    # Evaluation
    print("\nEvaluating on test set...")
    model.eval()
    all_scores = []

    with torch.no_grad():
        for i in range(0, len(test_ds), batch_size):
            setpoints = []
            efforts = []
            for j in range(i, min(i + batch_size, len(test_ds))):
                s, e, _ = test_ds[j]
                setpoints.append(s)
                efforts.append(e)

            setpoint = torch.stack(setpoints)
            effort = torch.stack(efforts)

            scores = model.compute_anomaly_score(setpoint=setpoint, effort=effort)
            all_scores.extend(scores.tolist())

    # Compute metrics
    all_scores = np.array(all_scores)
    test_labels = np.array(test_labels)

    healthy_scores = all_scores[test_labels == 0]
    faulty_scores = all_scores[test_labels == 1]

    print("\nResults:")
    print(f"  Healthy scores: mean={healthy_scores.mean():.4f}, std={healthy_scores.std():.4f}")
    print(f"  Faulty scores:  mean={faulty_scores.mean():.4f}, std={faulty_scores.std():.4f}")

    # Separation in standard deviations
    separation = (faulty_scores.mean() - healthy_scores.mean()) / healthy_scores.std()
    print(f"  Separation: {separation:.2f} std")

    # AUC-ROC
    if n_anomaly > 0:
        auc = roc_auc_score(test_labels, all_scores)
        print(f"  AUC-ROC: {auc:.4f}")

        if auc > 0.6:
            print("\n[PASS] Anomaly detection shows improvement!")
        else:
            print("\n[FAIL] Still cannot distinguish healthy from faulty.")

    # Per-fault analysis
    print("\nPer-fault analysis:")
    fault_types = set(test_faults)
    for fault in sorted(fault_types):
        fault_mask = np.array([f == fault for f in test_faults])
        fault_scores = all_scores[fault_mask]
        fault_labels = test_labels[fault_mask]

        if len(fault_scores) == 0:
            continue

        print(f"  {fault}: n={len(fault_scores)}, "
              f"mean_score={fault_scores.mean():.4f}, "
              f"anomaly_rate={fault_labels.mean():.2f}")

    return auc if n_anomaly > 0 else None


if __name__ == "__main__":
    run_experiment()
