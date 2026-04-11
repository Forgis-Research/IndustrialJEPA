# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Direct analysis: Does the model predict higher force_z than actual for missing_screw?
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig
from industrialjepa.baselines import SetpointToEffort, AutoencoderConfig


def main():
    print("="*60)
    print("DIRECT FORCE_Z PREDICTION ANALYSIS")
    print("="*60)

    config = FactoryNetConfig(
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
    train_ds = FactoryNetDataset(config, split="train")
    test_ds = FactoryNetDataset(config, split="test")

    force_z_idx = train_ds.effort_cols.index('effort_force_z')
    print(f"force_z index: {force_z_idx}")

    # Model
    model_config = AutoencoderConfig(
        setpoint_dim=config.unified_setpoint_dim,
        effort_dim=config.unified_effort_dim,
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

    # Analysis
    print("\nAnalyzing predictions...")
    model.eval()

    healthy_pred = []
    healthy_actual = []
    missing_pred = []
    missing_actual = []

    with torch.no_grad():
        for i in range(len(test_ds)):
            setpoint, effort, meta = test_ds[i]
            fault = meta["fault_type"]

            setpoint = setpoint.unsqueeze(0)
            effort = effort.unsqueeze(0)

            # Get prediction
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

            # Get force_z values
            pred_force_z = effort_pred[0, :, force_z_idx].mean().item()
            actual_force_z = effort[0, :effort_pred.shape[1], force_z_idx].mean().item()

            if fault == "normal":
                healthy_pred.append(pred_force_z)
                healthy_actual.append(actual_force_z)
            elif fault == "missing_screw":
                missing_pred.append(pred_force_z)
                missing_actual.append(actual_force_z)

    healthy_pred = np.array(healthy_pred)
    healthy_actual = np.array(healthy_actual)
    missing_pred = np.array(missing_pred)
    missing_actual = np.array(missing_actual)

    print("\n" + "="*60)
    print("FORCE_Z PREDICTIONS")
    print("="*60)

    print("\nHealthy samples:")
    print(f"  Predicted: {healthy_pred.mean():.4f} +/- {healthy_pred.std():.4f}")
    print(f"  Actual:    {healthy_actual.mean():.4f} +/- {healthy_actual.std():.4f}")
    healthy_error = healthy_pred - healthy_actual
    print(f"  Error:     {healthy_error.mean():.4f} +/- {healthy_error.std():.4f}")

    print("\nMissing_screw samples:")
    print(f"  Predicted: {missing_pred.mean():.4f} +/- {missing_pred.std():.4f}")
    print(f"  Actual:    {missing_actual.mean():.4f} +/- {missing_actual.std():.4f}")
    missing_error = missing_pred - missing_actual
    print(f"  Error:     {missing_error.mean():.4f} +/- {missing_error.std():.4f}")

    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    print(f"\nPredicted force_z difference (missing - healthy): {missing_pred.mean() - healthy_pred.mean():.4f}")
    print(f"Actual force_z difference (missing - healthy):    {missing_actual.mean() - healthy_actual.mean():.4f}")

    # The key question: Does the model predict HIGHER force_z for missing_screw
    # even though actual is LOWER?
    if missing_pred.mean() > healthy_pred.mean():
        print("\nModel predicts HIGHER force_z for missing_screw!")
        print("But actual is LOWER. This SHOULD create detectable prediction error.")
    elif missing_pred.mean() < healthy_pred.mean():
        print("\nModel predicts LOWER force_z for missing_screw.")
        print("This is WRONG - model is learning to predict the anomaly, not the healthy baseline!")

    # Error-based detection
    healthy_abs_error = np.abs(healthy_error)
    missing_abs_error = np.abs(missing_error)
    error_auc = roc_auc_score(
        [0]*len(healthy_abs_error) + [1]*len(missing_abs_error),
        list(healthy_abs_error) + list(missing_abs_error)
    )
    print(f"\nForce_z abs error AUC: {error_auc:.4f}")


if __name__ == "__main__":
    main()
