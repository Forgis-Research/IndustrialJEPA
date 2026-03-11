# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Test anomaly detection using effort_force_z (the end-effector signal).
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import json

print("Loading AURSAD...")
ds = load_dataset("Forgis/factorynet-hackathon", data_dir="aursad", split="train")
df = ds.to_pandas()

# Load metadata
meta_path = hf_hub_download(
    repo_id="Forgis/factorynet-hackathon",
    filename="metadata/aursad_metadata.json",
    repo_type="dataset"
)
with open(meta_path) as f:
    metadata_list = json.load(f)

metadata = {m['episode_id']: m for m in metadata_list}

# Compute force_z stats per episode
print("\nComputing per-episode force_z statistics...")

scores = []
labels = []
fault_types_list = []

for ep_id, ep_data in df.groupby('episode_id'):
    meta = metadata.get(ep_id, {})
    fault_label = meta.get('fault_label', 'unknown')

    # Skip loosening (different operation)
    if fault_label == 'loosening':
        continue

    # End-effector force (the screwdriver push force)
    force_z = ep_data['effort_force_z'].values

    # Anomaly score: max force during operation
    # Higher force = more resistance = healthy screw engagement
    score = np.abs(force_z).max()

    # Label: normal=0, fault=1
    is_anomaly = fault_label not in ['normal', 'loosening']

    scores.append(score)
    labels.append(1 if is_anomaly else 0)
    fault_types_list.append(fault_label)

scores = np.array(scores)
labels = np.array(labels)

# Since higher force = healthy, we need to flip the score
# (or use 1-AUC which is equivalent)
# Actually, let's use -score so that lower force = higher anomaly score
anomaly_scores = -scores

print(f"\nTotal tightening episodes: {len(scores)}")
print(f"Healthy: {(labels==0).sum()}, Faulty: {(labels==1).sum()}")

# Results
healthy_force = scores[labels == 0]
faulty_force = scores[labels == 1]

print(f"\n" + "="*60)
print("RESULTS: Using effort_force_z (end-effector push force)")
print("="*60)
print(f"Healthy force_z: mean={healthy_force.mean():.2f}, std={healthy_force.std():.2f}")
print(f"Faulty force_z:  mean={faulty_force.mean():.2f}, std={faulty_force.std():.2f}")

effect = (faulty_force.mean() - healthy_force.mean()) / healthy_force.std()
print(f"Effect size: {effect:+.2f} std")

# AUC (using -score so lower force = higher anomaly score)
auc = roc_auc_score(labels, anomaly_scores)
print(f"AUC-ROC: {auc:.4f}")

# Per-fault
print("\nPer-fault analysis:")
for fault in ['missing_screw', 'damaged_screw', 'extra_component', 'damaged_thread']:
    mask = np.array([f == fault for f in fault_types_list])
    if mask.sum() == 0:
        continue

    fault_scores = scores[mask]
    fault_effect = (fault_scores.mean() - healthy_force.mean()) / healthy_force.std()

    # AUC for this fault vs healthy
    sub_mask = np.array([f == fault or f == 'normal' for f in fault_types_list])
    sub_labels = np.array([1 if f == fault else 0 for f in fault_types_list])[sub_mask]
    sub_scores = anomaly_scores[sub_mask]
    sub_auc = roc_auc_score(sub_labels, sub_scores)

    print(f"  {fault}: n={mask.sum()}, effect={fault_effect:+.2f} std, AUC={sub_auc:.4f}")

if auc > 0.7:
    print("\n[SUCCESS] End-effector force detects anomalies!")
elif auc > 0.6:
    print("\n[MODERATE] Some detection capability with force_z.")
else:
    print("\n[WEAK] Force_z alone is not sufficient.")
