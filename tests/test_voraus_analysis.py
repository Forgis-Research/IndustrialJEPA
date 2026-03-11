# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Analyze voraus-AD dataset for setpoint→effort anomaly detection viability.

Questions:
1. What fault types exist?
2. Do faults affect effort magnitude/variance?
3. Can setpoint predict effort?
4. Can prediction error detect anomalies?
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig


def analyze_voraus():
    print("=" * 70)
    print("VORAUS-AD DATASET ANALYSIS")
    print("=" * 70)

    # Load dataset
    config = FactoryNetConfig(
        dataset_name="Forgis/factorynet-hackathon",
        subset="voraus-ad",
        window_size=256,
        stride=256,
        normalize=False,  # Raw values first
        train_healthy_only=False,
    )

    print("\nLoading voraus-AD...")
    ds = FactoryNetDataset(config, split="test")
    print(f"Test windows: {len(ds)}")

    # Analyze structure
    fault_counts = {}
    healthy_efforts = []
    faulty_efforts = {}
    healthy_vars = []
    faulty_vars = {}

    for i in range(len(ds)):
        setpoint, effort, meta = ds[i]
        fault_type = meta["fault_type"]
        is_anomaly = meta["is_anomaly"]

        effort_np = effort.numpy()
        effort_mag = np.abs(effort_np).mean()
        effort_var = np.var(effort_np)

        # Count faults
        fault_counts[fault_type] = fault_counts.get(fault_type, 0) + 1

        if is_anomaly:
            if fault_type not in faulty_efforts:
                faulty_efforts[fault_type] = []
                faulty_vars[fault_type] = []
            faulty_efforts[fault_type].append(effort_mag)
            faulty_vars[fault_type].append(effort_var)
        else:
            healthy_efforts.append(effort_mag)
            healthy_vars.append(effort_var)

    # Print fault distribution
    print("\n--- FAULT DISTRIBUTION ---")
    for fault, count in sorted(fault_counts.items(), key=lambda x: -x[1]):
        print(f"  {fault}: {count}")

    # Compute statistics
    healthy_efforts = np.array(healthy_efforts)
    healthy_vars = np.array(healthy_vars)

    print(f"\n--- EFFORT STATISTICS ---")
    print(f"Healthy: n={len(healthy_efforts)}, "
          f"mean_mag={healthy_efforts.mean():.4f}, std={healthy_efforts.std():.4f}, "
          f"mean_var={healthy_vars.mean():.4f}")

    print("\nPer-fault analysis:")
    all_faulty_mags = []
    all_faulty_vars = []
    fault_labels = []

    for fault, mags in sorted(faulty_efforts.items()):
        mags = np.array(mags)
        vars_ = np.array(faulty_vars[fault])
        all_faulty_mags.extend(mags)
        all_faulty_vars.extend(vars_)
        fault_labels.extend([fault] * len(mags))

        mag_effect = (mags.mean() - healthy_efforts.mean()) / healthy_efforts.std()
        var_effect = (vars_.mean() - healthy_vars.mean()) / healthy_vars.std()

        print(f"  {fault}: n={len(mags)}, "
              f"mag_effect={mag_effect:+.2f} std, "
              f"var_effect={var_effect:+.2f} std")

    # Overall effect
    all_faulty_mags = np.array(all_faulty_mags)
    all_faulty_vars = np.array(all_faulty_vars)

    if len(all_faulty_mags) > 0:
        overall_mag_effect = (all_faulty_mags.mean() - healthy_efforts.mean()) / healthy_efforts.std()
        overall_var_effect = (all_faulty_vars.mean() - healthy_vars.mean()) / healthy_vars.std()
        print(f"\nOverall faulty: mag_effect={overall_mag_effect:+.2f} std, var_effect={overall_var_effect:+.2f} std")

    # Simple anomaly detection test
    print("\n--- SIMPLE ANOMALY DETECTION (magnitude as score) ---")
    all_scores = list(healthy_efforts) + list(all_faulty_mags)
    all_labels = [0] * len(healthy_efforts) + [1] * len(all_faulty_mags)

    if sum(all_labels) > 0:
        auc_mag = roc_auc_score(all_labels, all_scores)
        print(f"AUC-ROC (effort magnitude): {auc_mag:.4f}")

        all_var_scores = list(healthy_vars) + list(all_faulty_vars)
        auc_var = roc_auc_score(all_labels, all_var_scores)
        print(f"AUC-ROC (effort variance): {auc_var:.4f}")

        if auc_mag > 0.7 or auc_var > 0.7:
            print("\n[PROMISING] voraus-AD shows detectable anomaly signal!")
        elif auc_mag > 0.6 or auc_var > 0.6:
            print("\n[MODERATE] Some signal present, worth investigating further.")
        else:
            print("\n[WEAK] Limited anomaly signal in raw effort statistics.")


if __name__ == "__main__":
    analyze_voraus()
