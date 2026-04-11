# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Check how normalization affects force_z signal for missing_screw.
"""

import numpy as np
from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig


def main():
    # Without normalization
    config_raw = FactoryNetConfig(
        dataset_name="Forgis/factorynet-hackathon",
        subset="aursad",
        window_size=256,
        stride=256,
        normalize=False,
        train_healthy_only=False,
        aursad_phase_handling="tightening_only",
    )

    # With global normalization
    config_global = FactoryNetConfig(
        dataset_name="Forgis/factorynet-hackathon",
        subset="aursad",
        window_size=256,
        stride=256,
        normalize=True,
        norm_mode="global",
        train_healthy_only=False,
        aursad_phase_handling="tightening_only",
    )

    print("Loading datasets...")
    ds_raw = FactoryNetDataset(config_raw, split="test")
    ds_global = FactoryNetDataset(config_global, split="test")

    # Find force_z index
    force_z_idx = ds_raw.effort_cols.index('effort_force_z')
    print(f"force_z index: {force_z_idx}")

    # Check global normalization stats
    print(f"\nGlobal stats computed from:")
    print(f"  effort_mean[force_z]: {ds_global.effort_mean[force_z_idx]:.4f}")
    print(f"  effort_std[force_z]: {ds_global.effort_std[force_z_idx]:.4f}")

    # Compare raw vs normalized for healthy and missing_screw
    healthy_raw = []
    healthy_norm = []
    missing_raw = []
    missing_norm = []

    for i in range(len(ds_raw)):
        _, effort_raw, meta_raw = ds_raw[i]
        _, effort_norm, meta_norm = ds_global[i]

        fault = meta_raw["fault_type"]
        force_z_raw_val = effort_raw[:, force_z_idx].mean().item()
        force_z_norm_val = effort_norm[:, force_z_idx].mean().item()

        if fault == "normal":
            healthy_raw.append(force_z_raw_val)
            healthy_norm.append(force_z_norm_val)
        elif fault == "missing_screw":
            missing_raw.append(force_z_raw_val)
            missing_norm.append(force_z_norm_val)

    healthy_raw = np.array(healthy_raw)
    healthy_norm = np.array(healthy_norm)
    missing_raw = np.array(missing_raw)
    missing_norm = np.array(missing_norm)

    print("\n" + "="*60)
    print("FORCE_Z COMPARISON")
    print("="*60)

    print("\nRAW values:")
    print(f"  Healthy: mean={healthy_raw.mean():.2f}, std={healthy_raw.std():.2f}")
    print(f"  Missing: mean={missing_raw.mean():.2f}, std={missing_raw.std():.2f}")
    raw_effect = (missing_raw.mean() - healthy_raw.mean()) / healthy_raw.std()
    print(f"  Effect size: {raw_effect:.2f} std")

    print("\nGLOBAL NORMALIZED values:")
    print(f"  Healthy: mean={healthy_norm.mean():.4f}, std={healthy_norm.std():.4f}")
    print(f"  Missing: mean={missing_norm.mean():.4f}, std={missing_norm.std():.4f}")
    norm_effect = (missing_norm.mean() - healthy_norm.mean()) / healthy_norm.std()
    print(f"  Effect size: {norm_effect:.2f} std")

    print("\n" + "="*60)
    if abs(norm_effect) < abs(raw_effect) * 0.5:
        print("WARNING: Global normalization is REDUCING the signal!")
        print("This explains why prediction error doesn't detect missing_screw.")
    else:
        print("Signal preserved after normalization.")


if __name__ == "__main__":
    main()
