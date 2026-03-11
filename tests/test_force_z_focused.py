# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Test anomaly detection focused on force_z channel only.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig
from industrialjepa.baselines import SetpointToEffort, AutoencoderConfig


def main():
    print("=" * 60)
    print("FORCE_Z FOCUSED ANOMALY DETECTION")
    print("=" * 60)

    data_config = FactoryNetConfig(
        dataset_name="Forgis/factorynet-hackathon",
        subset="aursad",
        window_size=256,
        stride=128,
        normalize=True,
        norm_mode="global",
        train_healthy_only=True,
        aursad_phase_handling="tightening_only",
    )

    print("\nLoading data...")
    train_ds = FactoryNetDataset(data_config, split="train")
    test_ds = FactoryNetDataset(data_config, split="test")

    # Find force_z index in effort columns
    force_z_idx = train_ds.effort_cols.index('effort_force_z')
    print(f"Effort columns: {train_ds.effort_cols}")
    print(f"force_z index: {force_z_idx}")

    # Model
    model_config = AutoencoderConfig(
        setpoint_dim=data_config.unified_setpoint_dim,
        effort_dim=data_config.unified_effort_dim,
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

    # Training
    print("\nTraining...")
    batch_size = 32
    n_epochs = 10

    model.train()
    for epoch in range(n_epochs):
        indices = torch.randperm(len(train_ds))
        total_loss = 0
        n_batches = 0

        for i in range(0, min(len(train_ds), 2000), batch_size):
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

    # Evaluation - compute FORCE_Z ONLY prediction error
    print("\nEvaluating with FORCE_Z focused scoring...")
    model.eval()

    all_scores = []
    force_z_scores = []
    labels = []
    fault_types = []

    with torch.no_grad():
        for i in tqdm(range(len(test_ds)), desc="Testing"):
            setpoint, effort, meta = test_ds[i]
            setpoint = setpoint.unsqueeze(0)
            effort = effort.unsqueeze(0)

            # Get full prediction
            B, T, _ = setpoint.shape
            x, _ = model.patch_embed(setpoint, None)
            x = model.encoder(x)
            x = model.bottleneck(x)
            x = model.enc_to_dec(x)
            x = model.decoder(x)
            x = model.decoder_norm(x)
            effort_pred = model.effort_head(x[:, 1:])
            num_patches = effort_pred.shape[1]
            effort_pred = effort_pred.reshape(B, num_patches * model.config.patch_size, model.config.effort_dim)

            # Align lengths
            effort_target = effort[:, :effort_pred.shape[1], :]

            # Full prediction error (all channels)
            full_error = F.mse_loss(effort_pred, effort_target, reduction='none').mean(dim=(1, 2))

            # Force_z only error
            force_z_pred = effort_pred[:, :, force_z_idx]
            force_z_target = effort_target[:, :, force_z_idx]
            force_z_error = F.mse_loss(force_z_pred, force_z_target, reduction='none').mean(dim=1)

            all_scores.append(full_error.item())
            force_z_scores.append(force_z_error.item())
            labels.append(1 if meta["is_anomaly"] else 0)
            fault_types.append(meta["fault_type"])

    all_scores = np.array(all_scores)
    force_z_scores = np.array(force_z_scores)
    labels = np.array(labels)

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Full model scores
    auc_full = roc_auc_score(labels, all_scores)
    print(f"Full model AUC: {auc_full:.4f}")

    # Force_z only scores
    auc_force_z = roc_auc_score(labels, force_z_scores)
    print(f"Force_z only AUC: {auc_force_z:.4f}")

    # Per-fault with force_z
    print("\nPer-fault (force_z only):")
    for fault in ['missing_screw', 'damaged_screw', 'extra_component']:
        mask = np.array([f == fault or l == 0 for f, l in zip(fault_types, labels)])
        sub_labels = np.array([1 if f == fault else 0 for f in fault_types])[mask]
        sub_scores = force_z_scores[mask]
        if sub_labels.sum() > 0:
            sub_auc = roc_auc_score(sub_labels, sub_scores)
            print(f"  {fault}: AUC = {sub_auc:.4f}")


if __name__ == "__main__":
    main()
