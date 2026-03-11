# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Check if end-effector torque (the actual screwdriver torque) differs
between healthy and faulty operations.

The user's intuition: missing screw = less resistance = less torque
"""

from datasets import load_dataset
import numpy as np

print("Loading AURSAD with metadata...")

# Load dataset
ds = load_dataset("Forgis/factorynet-hackathon", data_dir="aursad", split="train")
df = ds.to_pandas()

# Load metadata to get fault labels
from huggingface_hub import hf_hub_download
import json

meta_path = hf_hub_download(
    repo_id="Forgis/factorynet-hackathon",
    filename="metadata/aursad_metadata.json",
    repo_type="dataset"
)
with open(meta_path) as f:
    metadata_list = json.load(f)

# Convert to dict by episode_id
metadata = {m['episode_id']: m for m in metadata_list}

# Group by episode and compute end-effector torque statistics
print("\nAnalyzing end-effector signals by fault type...")

fault_stats = {}

for ep_id, ep_data in df.groupby('episode_id'):
    meta = metadata.get(ep_id, {})
    fault = meta.get('fault_type', 'unknown')
    phase = meta.get('phase', 'unknown')

    # Only look at tightening phase (where faults matter)
    if phase != 'tightening':
        continue

    # Get end-effector signals
    torque_z = ep_data['effort_torque_z'].values
    force_z = ep_data['effort_force_z'].values

    # Also check joint 5 (closest to tool) torque
    joint5_torque = ep_data['effort_torque_5'].values

    # Compute stats
    stats = {
        'torque_z_max': np.abs(torque_z).max(),
        'torque_z_mean': np.abs(torque_z).mean(),
        'force_z_max': np.abs(force_z).max(),
        'force_z_mean': np.abs(force_z).mean(),
        'joint5_torque_max': np.abs(joint5_torque).max(),
    }

    if fault not in fault_stats:
        fault_stats[fault] = []
    fault_stats[fault].append(stats)

# Aggregate and compare
print("\n" + "="*70)
print("END-EFFECTOR TORQUE ANALYSIS (Tightening Phase Only)")
print("="*70)

healthy_stats = fault_stats.get('normal', [])

if healthy_stats:
    healthy_torque_z = np.array([s['torque_z_max'] for s in healthy_stats])
    healthy_force_z = np.array([s['force_z_max'] for s in healthy_stats])

    print(f"\nHealthy (n={len(healthy_stats)}):")
    print(f"  effort_torque_z (max): {healthy_torque_z.mean():.4f} +/- {healthy_torque_z.std():.4f}")
    print(f"  effort_force_z (max):  {healthy_force_z.mean():.4f} +/- {healthy_force_z.std():.4f}")

    print("\nPer-fault comparison:")
    print(f"{'Fault':<20} {'N':>5} {'torque_z':>12} {'effect':>10} {'force_z':>12} {'effect':>10}")
    print("-"*70)

    for fault, stats_list in sorted(fault_stats.items()):
        if fault == 'normal':
            continue

        fault_torque_z = np.array([s['torque_z_max'] for s in stats_list])
        fault_force_z = np.array([s['force_z_max'] for s in stats_list])

        torque_effect = (fault_torque_z.mean() - healthy_torque_z.mean()) / healthy_torque_z.std()
        force_effect = (fault_force_z.mean() - healthy_force_z.mean()) / healthy_force_z.std()

        print(f"{fault:<20} {len(stats_list):>5} {fault_torque_z.mean():>12.4f} {torque_effect:>+10.2f} {fault_force_z.mean():>12.4f} {force_effect:>+10.2f}")

print("\n" + "="*70)
print("INTERPRETATION:")
print("- Negative effect = LOWER torque/force than healthy")
print("- missing_screw should show LOWER torque (no thread resistance)")
print("- damaged_screw might show LOWER torque (slipping threads)")
print("="*70)
