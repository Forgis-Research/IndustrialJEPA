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

# Check fault distribution
print("\nFault type distribution:")
fault_counts = {}
phase_counts = {}
for m in metadata_list:
    ft = m.get('fault_type', 'unknown')
    fl = m.get('fault_label', 'unknown')
    fault_counts[ft] = fault_counts.get(ft, 0) + 1
    phase_counts[fl] = phase_counts.get(fl, 0) + 1

for ft, count in sorted(fault_counts.items()):
    print(f"  {ft}: {count}")

print("\nPhase (fault_label) distribution:")
for fl, count in sorted(phase_counts.items()):
    print(f"  {fl}: {count}")

# Analyze end-effector torque by fault type
print("\n" + "="*70)
print("END-EFFECTOR TORQUE ANALYSIS")
print("="*70)

fault_stats = {}

for ep_id, ep_data in df.groupby('episode_id'):
    meta = metadata.get(ep_id, {})
    fault = meta.get('fault_type', 'unknown')
    phase = meta.get('fault_label', 'unknown')  # 'loosening' or 'tightening' etc

    # Get end-effector signals - these are the screwdriver signals!
    torque_z = ep_data['effort_torque_z'].values
    force_z = ep_data['effort_force_z'].values

    # Key for grouping: fault + phase
    key = f"{fault}_{phase}"

    stats = {
        'torque_z_max': np.abs(torque_z).max(),
        'torque_z_mean': np.abs(torque_z).mean(),
        'force_z_max': np.abs(force_z).max(),
        'force_z_mean': np.abs(force_z).mean(),
        'fault_type': fault,
        'phase': phase,
    }

    if key not in fault_stats:
        fault_stats[key] = []
    fault_stats[key].append(stats)

# Report
print(f"\n{'Group':<30} {'N':>5} {'torque_z_max':>14} {'force_z_max':>14}")
print("-"*70)

# Get healthy tightening as baseline
healthy_key = 'normal_tightening'
if healthy_key in fault_stats:
    healthy = fault_stats[healthy_key]
    healthy_torque = np.array([s['torque_z_max'] for s in healthy])
    healthy_force = np.array([s['force_z_max'] for s in healthy])
    print(f"{'normal_tightening (baseline)':<30} {len(healthy):>5} {healthy_torque.mean():>14.4f} {healthy_force.mean():>14.4f}")
    print()

    # Compare other tightening groups
    for key, stats_list in sorted(fault_stats.items()):
        if 'tightening' not in key or key == healthy_key:
            continue

        torque = np.array([s['torque_z_max'] for s in stats_list])
        force = np.array([s['force_z_max'] for s in stats_list])

        torque_effect = (torque.mean() - healthy_torque.mean()) / healthy_torque.std()
        force_effect = (force.mean() - healthy_force.mean()) / healthy_force.std()

        print(f"{key:<30} {len(stats_list):>5} {torque.mean():>14.4f} {force.mean():>14.4f}  "
              f"(torque: {torque_effect:+.2f} std, force: {force_effect:+.2f} std)")

print("\n" + "="*70)
print("INSIGHT: If torque_z is the screwdriver torque, missing_screw should")
print("have LOWER torque_z because there's no screw thread to grip.")
print("="*70)
