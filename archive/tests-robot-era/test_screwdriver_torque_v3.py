# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Check if end-effector torque differs between healthy and faulty.
"""

from datasets import load_dataset
from huggingface_hub import hf_hub_download
import numpy as np
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

# fault_label contains the actual fault type!
# loosening = loosening phase (not tightening)
# normal = healthy tightening
# missing_screw, damaged_screw, etc. = fault types during tightening

print("\nAnalyzing by fault_label (the actual fault)...")

fault_stats = {}

for ep_id, ep_data in df.groupby('episode_id'):
    meta = metadata.get(ep_id, {})
    fault_label = meta.get('fault_label', 'unknown')

    # Get end-effector signals
    torque_z = ep_data['effort_torque_z'].values
    force_z = ep_data['effort_force_z'].values

    stats = {
        'torque_z_max': np.abs(torque_z).max(),
        'torque_z_mean': np.abs(torque_z).mean(),
        'force_z_max': np.abs(force_z).max(),
        'force_z_mean': np.abs(force_z).mean(),
    }

    if fault_label not in fault_stats:
        fault_stats[fault_label] = []
    fault_stats[fault_label].append(stats)

# Report
print("\n" + "="*70)
print("END-EFFECTOR TORQUE ANALYSIS BY FAULT TYPE")
print("="*70)

print(f"\n{'Fault Label':<20} {'N':>6} {'torque_z_max':>14} {'force_z_max':>14} {'torque effect':>14} {'force effect':>14}")
print("-"*90)

# Get healthy (normal tightening) as baseline
healthy = fault_stats.get('normal', [])
if healthy:
    healthy_torque = np.array([s['torque_z_max'] for s in healthy])
    healthy_force = np.array([s['force_z_max'] for s in healthy])

    print(f"{'normal (baseline)':<20} {len(healthy):>6} {healthy_torque.mean():>14.4f} {healthy_force.mean():>14.4f} {'--':>14} {'--':>14}")

    # Compare fault types
    for fault_label in ['missing_screw', 'damaged_screw', 'extra_component', 'damaged_thread', 'loosening']:
        if fault_label not in fault_stats:
            continue

        stats_list = fault_stats[fault_label]
        torque = np.array([s['torque_z_max'] for s in stats_list])
        force = np.array([s['force_z_max'] for s in stats_list])

        torque_effect = (torque.mean() - healthy_torque.mean()) / healthy_torque.std()
        force_effect = (force.mean() - healthy_force.mean()) / healthy_force.std()

        print(f"{fault_label:<20} {len(stats_list):>6} {torque.mean():>14.4f} {force.mean():>14.4f} {torque_effect:>+14.2f} {force_effect:>+14.2f}")

print("\n" + "="*70)
print("KEY INSIGHT:")
print("  - missing_screw should have LOWER torque (no thread engagement)")
print("  - damaged_screw might have LOWER torque (slipping)")
print("  - extra_component might have HIGHER force (obstruction)")
print("  - loosening is a different operation (unscrewing)")
print("="*70)
