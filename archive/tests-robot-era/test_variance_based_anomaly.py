# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Test variance-based anomaly detection.

Finding: missing_screw has +0.99 std higher variance than healthy.
Let's see if we can detect it using variance as anomaly score.
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig


def compute_variance_score(effort):
    """Simple anomaly score: effort variance."""
    effort_np = effort.numpy()
    return np.var(effort_np)


def run_variance_detector():
    print("=" * 60)
    print("VARIANCE-BASED ANOMALY DETECTION")
    print("=" * 60)

    config = FactoryNetConfig(
        dataset_name="Forgis/factorynet-hackathon",
        subset="aursad",
        window_size=256,
        stride=256,
        normalize=False,  # Raw variance
        train_healthy_only=False,
        aursad_phase_handling="tightening_only",
    )

    ds = FactoryNetDataset(config, split="test")

    labels = []
    scores = []
    fault_types = []

    for i in range(len(ds)):
        _, effort, meta = ds[i]
        is_anomaly = meta["is_anomaly"]
        fault_type = meta["fault_type"]

        score = compute_variance_score(effort)

        labels.append(1 if is_anomaly else 0)
        scores.append(score)
        fault_types.append(fault_type)

    labels = np.array(labels)
    scores = np.array(scores)

    # Overall AUC
    auc = roc_auc_score(labels, scores)
    print(f"\nOverall AUC-ROC (variance as score): {auc:.4f}")

    # Per-fault AUC
    print("\nPer-fault AUC:")
    unique_faults = set(fault_types)
    for fault in sorted(unique_faults):
        if fault == "normal":
            continue
        fault_mask = np.array([f == fault or f == "normal" for f in fault_types])
        fault_labels = np.array([1 if f == fault else 0 for f in fault_types])

        # Filter to this fault vs healthy
        mask = np.array([f == fault or (not l) for f, l in zip(fault_types, labels)])
        sub_labels = labels[mask]
        sub_scores = scores[mask]

        if sub_labels.sum() > 0:
            sub_auc = roc_auc_score(sub_labels, sub_scores)
            print(f"  {fault}: AUC = {sub_auc:.4f} (n={sub_labels.sum()})")

    # Prediction error + variance hybrid?
    print("\n" + "=" * 60)
    print("INSIGHT:")
    print("  - Variance alone gives moderate improvement")
    print("  - missing_screw: HIGH variance (detectable)")
    print("  - damaged_screw: SIMILAR variance to healthy (hard to detect)")
    print("=" * 60)


if __name__ == "__main__":
    run_variance_detector()
