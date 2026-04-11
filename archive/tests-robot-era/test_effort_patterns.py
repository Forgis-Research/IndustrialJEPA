# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Analyze effort PATTERNS (variance, rate of change) for healthy vs faulty.

The magnitude analysis showed faults don't change overall effort level.
Maybe faults show up in the dynamics?
"""

import numpy as np
from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig


def compute_pattern_features(effort):
    """Extract pattern features from effort tensor."""
    effort_np = effort.numpy()

    # Variance within window
    variance = np.var(effort_np, axis=0).mean()

    # Rate of change (diff)
    diff = np.diff(effort_np, axis=0)
    mean_rate = np.abs(diff).mean()
    max_rate = np.abs(diff).max()

    # Frequency content (simple: zero crossings)
    centered = effort_np - effort_np.mean(axis=0)
    zero_crossings = np.sum(np.diff(np.sign(centered), axis=0) != 0)

    # Peak-to-peak range
    ptp = np.ptp(effort_np, axis=0).mean()

    return {
        "variance": variance,
        "mean_rate": mean_rate,
        "max_rate": max_rate,
        "zero_crossings": zero_crossings,
        "peak_to_peak": ptp,
    }


def analyze_patterns():
    print("=" * 60)
    print("EFFORT PATTERN ANALYSIS (Variance, Rate of Change)")
    print("=" * 60)

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

    healthy_features = []
    faulty_features = []
    fault_features = {}

    for i in range(min(len(ds), 500)):
        setpoint, effort, meta = ds[i]
        is_anomaly = meta["is_anomaly"]
        fault_type = meta["fault_type"]

        features = compute_pattern_features(effort)

        if is_anomaly:
            faulty_features.append(features)
            if fault_type not in fault_features:
                fault_features[fault_type] = []
            fault_features[fault_type].append(features)
        else:
            healthy_features.append(features)

    # Aggregate
    def aggregate(feature_list):
        if not feature_list:
            return {}
        return {
            k: (np.mean([f[k] for f in feature_list]),
                np.std([f[k] for f in feature_list]))
            for k in feature_list[0].keys()
        }

    healthy_agg = aggregate(healthy_features)
    faulty_agg = aggregate(faulty_features)

    print("\nFeature comparison (healthy vs faulty):")
    print(f"{'Feature':<20} {'Healthy':<20} {'Faulty':<20} {'Effect Size':<15}")
    print("-" * 75)

    for feature in healthy_agg.keys():
        h_mean, h_std = healthy_agg[feature]
        f_mean, f_std = faulty_agg[feature]
        effect = (f_mean - h_mean) / h_std if h_std > 0 else 0
        print(f"{feature:<20} {h_mean:.4f} +/- {h_std:.4f}  {f_mean:.4f} +/- {f_std:.4f}  {effect:+.2f} std")

    print("\nPer-fault pattern analysis:")
    for fault, features in sorted(fault_features.items()):
        agg = aggregate(features)
        var_effect = (agg["variance"][0] - healthy_agg["variance"][0]) / healthy_agg["variance"][1]
        rate_effect = (agg["mean_rate"][0] - healthy_agg["mean_rate"][0]) / healthy_agg["mean_rate"][1]
        print(f"  {fault}: variance effect={var_effect:+.2f}, rate effect={rate_effect:+.2f}")


if __name__ == "__main__":
    analyze_patterns()
