# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Quick test: Compare per-episode vs global normalization.

This demonstrates why per-episode normalization fails for anomaly detection.
"""

import numpy as np
from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig


def analyze_normalization_mode(norm_mode: str, phase_handling: str):
    """Analyze effort statistics under different normalization."""
    print(f"\n{'='*60}")
    print(f"Normalization: {norm_mode}, Phase: {phase_handling}")
    print("=" * 60)

    config = FactoryNetConfig(
        dataset_name="Forgis/factorynet-hackathon",
        subset="aursad",
        window_size=256,
        stride=256,  # Non-overlapping for faster loading
        normalize=True,
        norm_mode=norm_mode,
        train_healthy_only=False,  # Get both for analysis
        aursad_phase_handling=phase_handling,
    )

    ds = FactoryNetDataset(config, split="test")
    print(f"Windows: {len(ds)}")

    # Collect statistics
    healthy_efforts = []
    faulty_efforts = []
    fault_stats = {}

    for i in range(min(len(ds), 500)):  # Sample for speed
        setpoint, effort, meta = ds[i]
        is_anomaly = meta["is_anomaly"]
        fault_type = meta["fault_type"]

        # Get mean absolute effort per window
        effort_mag = np.abs(effort.numpy()).mean()

        if is_anomaly:
            faulty_efforts.append(effort_mag)
            if fault_type not in fault_stats:
                fault_stats[fault_type] = []
            fault_stats[fault_type].append(effort_mag)
        else:
            healthy_efforts.append(effort_mag)

    healthy_efforts = np.array(healthy_efforts)
    faulty_efforts = np.array(faulty_efforts) if faulty_efforts else np.array([0])

    print(f"\nHealthy effort magnitude: mean={healthy_efforts.mean():.4f}, std={healthy_efforts.std():.4f}")
    if len(faulty_efforts) > 1:
        print(f"Faulty effort magnitude:  mean={faulty_efforts.mean():.4f}, std={faulty_efforts.std():.4f}")

        # Effect size
        effect = (healthy_efforts.mean() - faulty_efforts.mean()) / healthy_efforts.std()
        print(f"Effect size (healthy - faulty): {effect:.2f} std")

        if abs(effect) > 0.5:
            print("-> Significant difference! Magnitude preserves anomaly signal.")
        else:
            print("-> Small difference. Normalization may be destroying signal.")

    print("\nPer-fault breakdown:")
    for fault, mags in sorted(fault_stats.items()):
        mags = np.array(mags)
        print(f"  {fault}: n={len(mags)}, mean_effort={mags.mean():.4f}")


if __name__ == "__main__":
    # Compare the two modes
    print("\n" + "#"*60)
    print("# HYPOTHESIS: Per-episode normalization destroys anomaly signal")
    print("# because missing_screw = LOW effort gets normalized to look normal")
    print("#"*60)

    # This should show NO difference (normalization destroys signal)
    analyze_normalization_mode("episode", "tightening_only")

    # This should show a difference (magnitude preserved)
    analyze_normalization_mode("global", "tightening_only")
