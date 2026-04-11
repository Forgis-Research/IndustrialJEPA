# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Train SetpointToEffort model and test anomaly detection on AURSAD.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig
from industrialjepa.baselines import SetpointToEffort, AutoencoderConfig


def main():
    print("=" * 60)
    print("SETPOINT->EFFORT ANOMALY DETECTION (AURSAD)")
    print("=" * 60)

    # Config - use global normalization
    data_config = FactoryNetConfig(
        dataset_name="Forgis/factorynet-hackathon",
        subset="aursad",
        window_size=256,
        stride=128,
        normalize=True,
        norm_mode="global",  # Preserve magnitude
        train_healthy_only=True,
        aursad_phase_handling="tightening_only",
    )

    print("\nLoading data...")
    train_ds = FactoryNetDataset(data_config, split="train")
    test_ds = FactoryNetDataset(data_config, split="test")

    print(f"Train: {len(train_ds)} windows")
    print(f"Test: {len(test_ds)} windows")

    # Model
    model_config = AutoencoderConfig(
        setpoint_dim=14,
        effort_dim=7,
        seq_len=256,
        patch_size=16,
        hidden_dim=128,
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    print(f"Model params: {model.get_num_params():,}")

    # Training
    print("\nTraining...")
    batch_size = 32
    n_epochs = 10

    model.train()
    for epoch in range(n_epochs):
        indices = torch.randperm(len(train_ds))
        total_loss = 0
        n_batches = 0

        for i in range(0, min(len(train_ds), 2000), batch_size):  # Cap for speed
            batch_idx = indices[i:i+batch_size]
            if len(batch_idx) < batch_size:
                continue

            setpoints = []
            efforts = []
            for idx in batch_idx:
                s, e, _ = train_ds[idx.item()]
                setpoints.append(s)
                efforts.append(e)

            setpoint = torch.stack(setpoints)
            effort = torch.stack(efforts)

            optimizer.zero_grad()
            output = model(setpoint=setpoint, effort=effort)
            loss = output["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        print(f"  Epoch {epoch+1}/{n_epochs}: loss = {total_loss/n_batches:.4f}")

    # Evaluation
    print("\nEvaluating...")
    model.eval()

    scores = []
    labels = []
    fault_types = []

    with torch.no_grad():
        for i in tqdm(range(len(test_ds)), desc="Testing"):
            setpoint, effort, meta = test_ds[i]
            setpoint = setpoint.unsqueeze(0)
            effort = effort.unsqueeze(0)

            score = model.compute_anomaly_score(setpoint=setpoint, effort=effort)
            scores.append(score.item())
            labels.append(1 if meta["is_anomaly"] else 0)
            fault_types.append(meta["fault_type"])

    scores = np.array(scores)
    labels = np.array(labels)

    # Results
    healthy_scores = scores[labels == 0]
    faulty_scores = scores[labels == 1]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Healthy: n={len(healthy_scores)}, mean={healthy_scores.mean():.4f}, std={healthy_scores.std():.4f}")
    print(f"Faulty:  n={len(faulty_scores)}, mean={faulty_scores.mean():.4f}, std={faulty_scores.std():.4f}")

    effect = (faulty_scores.mean() - healthy_scores.mean()) / healthy_scores.std()
    print(f"Effect size: {effect:+.2f} std")

    auc = roc_auc_score(labels, scores)
    print(f"AUC-ROC: {auc:.4f}")

    # Per-fault
    print("\nPer-fault AUC:")
    for fault in set(fault_types):
        if fault == "normal":
            continue
        mask = np.array([f == fault or l == 0 for f, l in zip(fault_types, labels)])
        sub_labels = np.array([1 if f == fault else 0 for f in fault_types])[mask]
        sub_scores = scores[mask]
        if sub_labels.sum() > 0:
            sub_auc = roc_auc_score(sub_labels, sub_scores)
            print(f"  {fault}: AUC = {sub_auc:.4f}")

    if auc > 0.7:
        print("\n[SUCCESS] Prediction error can detect anomalies!")
    elif auc > 0.6:
        print("\n[MODERATE] Some detection capability, but weak signal.")
    else:
        print("\n[FAIL] Prediction error cannot reliably detect anomalies.")


if __name__ == "__main__":
    main()
