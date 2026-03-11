# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""Debug why variance analysis showed +0.99 effect but AUC is 0.47."""

import numpy as np
from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig


def debug_variance():
    config = FactoryNetConfig(
        dataset_name="Forgis/factorynet-hackathon",
        subset="aursad",
        window_size=256,
        stride=256,
        normalize=False,
        train_healthy_only=False,
        aursad_phase_handling="tightening_only",
    )

    ds = FactoryNetDataset(config, split="test")

    healthy_var = []
    missing_var = []
    damaged_var = []
    extra_var = []

    for i in range(len(ds)):
        _, effort, meta = ds[i]
        fault_type = meta["fault_type"]
        is_anomaly = meta["is_anomaly"]

        var = np.var(effort.numpy())

        if not is_anomaly:
            healthy_var.append(var)
        elif fault_type == "missing_screw":
            missing_var.append(var)
        elif fault_type == "damaged_screw":
            damaged_var.append(var)
        elif fault_type == "extra_component":
            extra_var.append(var)

    healthy_var = np.array(healthy_var)
    missing_var = np.array(missing_var)
    damaged_var = np.array(damaged_var)
    extra_var = np.array(extra_var)

    print("Variance distributions:")
    print(f"Healthy:         n={len(healthy_var)}, mean={healthy_var.mean():.4f}, std={healthy_var.std():.4f}")
    print(f"Missing screw:   n={len(missing_var)}, mean={missing_var.mean():.4f}, std={missing_var.std():.4f}")
    print(f"Damaged screw:   n={len(damaged_var)}, mean={damaged_var.mean():.4f}, std={damaged_var.std():.4f}")
    print(f"Extra component: n={len(extra_var)}, mean={extra_var.mean():.4f}, std={extra_var.std():.4f}")

    # Effect sizes
    print("\nEffect sizes (vs healthy):")
    print(f"  missing_screw:   {(missing_var.mean() - healthy_var.mean()) / healthy_var.std():.2f} std")
    print(f"  damaged_screw:   {(damaged_var.mean() - healthy_var.mean()) / healthy_var.std():.2f} std")
    print(f"  extra_component: {(extra_var.mean() - healthy_var.mean()) / healthy_var.std():.2f} std")

    # Percentile overlap
    print("\nOverlap analysis:")
    healthy_p90 = np.percentile(healthy_var, 90)
    print(f"  Healthy 90th percentile: {healthy_p90:.4f}")
    print(f"  Missing screw > healthy p90: {(missing_var > healthy_p90).mean()*100:.1f}%")
    print(f"  Damaged screw > healthy p90: {(damaged_var > healthy_p90).mean()*100:.1f}%")
    print(f"  Extra component > healthy p90: {(extra_var > healthy_p90).mean()*100:.1f}%")


if __name__ == "__main__":
    debug_variance()
