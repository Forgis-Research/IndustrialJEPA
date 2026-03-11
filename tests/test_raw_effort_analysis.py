# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Analyze RAW (unnormalized) effort values for healthy vs faulty.
"""

import numpy as np
from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig


def analyze_raw_effort():
    print("=" * 60)
    print("RAW EFFORT ANALYSIS (No normalization)")
    print("=" * 60)

    config = FactoryNetConfig(
        dataset_name="Forgis/factorynet-hackathon",
        subset="aursad",
        window_size=256,
        stride=256,
        normalize=False,  # RAW values
        train_healthy_only=False,
        aursad_phase_handling="tightening_only",
    )

    ds = FactoryNetDataset(config, split="test")
    print(f"Windows: {len(ds)}")

    # Collect raw effort values
    healthy_max_efforts = []
    faulty_max_efforts = []
    healthy_mean_efforts = []
    faulty_mean_efforts = []
    fault_efforts = {}

    for i in range(min(len(ds), 500)):
        setpoint, effort, meta = ds[i]
        is_anomaly = meta["is_anomaly"]
        fault_type = meta["fault_type"]

        effort_np = effort.numpy()

        # Various effort metrics
        max_effort = np.abs(effort_np).max()
        mean_effort = np.abs(effort_np).mean()

        if is_anomaly:
            faulty_max_efforts.append(max_effort)
            faulty_mean_efforts.append(mean_effort)
            if fault_type not in fault_efforts:
                fault_efforts[fault_type] = {"max": [], "mean": []}
            fault_efforts[fault_type]["max"].append(max_effort)
            fault_efforts[fault_type]["mean"].append(mean_effort)
        else:
            healthy_max_efforts.append(max_effort)
            healthy_mean_efforts.append(mean_effort)

    # Stats
    healthy_max = np.array(healthy_max_efforts)
    faulty_max = np.array(faulty_max_efforts)
    healthy_mean = np.array(healthy_mean_efforts)
    faulty_mean = np.array(faulty_mean_efforts)

    print(f"\n--- MAX EFFORT ---")
    print(f"Healthy: {healthy_max.mean():.2f} +/- {healthy_max.std():.2f}")
    print(f"Faulty:  {faulty_max.mean():.2f} +/- {faulty_max.std():.2f}")
    effect_max = (faulty_max.mean() - healthy_max.mean()) / healthy_max.std()
    print(f"Effect size: {effect_max:.2f} std")

    print(f"\n--- MEAN EFFORT ---")
    print(f"Healthy: {healthy_mean.mean():.2f} +/- {healthy_mean.std():.2f}")
    print(f"Faulty:  {faulty_mean.mean():.2f} +/- {faulty_mean.std():.2f}")
    effect_mean = (faulty_mean.mean() - healthy_mean.mean()) / healthy_mean.std()
    print(f"Effect size: {effect_mean:.2f} std")

    print(f"\n--- PER-FAULT BREAKDOWN (Raw Values) ---")
    for fault, data in sorted(fault_efforts.items()):
        max_arr = np.array(data["max"])
        mean_arr = np.array(data["mean"])
        print(f"  {fault}: n={len(max_arr)}, "
              f"max_effort={max_arr.mean():.2f}, "
              f"mean_effort={mean_arr.mean():.2f}")

    # Compare to healthy
    print(f"\n--- EFFECT SIZE VS HEALTHY (per fault) ---")
    for fault, data in sorted(fault_efforts.items()):
        max_arr = np.array(data["max"])
        effect = (max_arr.mean() - healthy_max.mean()) / healthy_max.std()
        print(f"  {fault}: {effect:+.2f} std from healthy")


if __name__ == "__main__":
    analyze_raw_effort()
