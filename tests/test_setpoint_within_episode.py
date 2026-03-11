# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Check how setpoint varies WITHIN an episode (timestep to timestep).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import load_dataset

print("Loading AURSAD...")
ds = load_dataset("Forgis/factorynet-hackathon", data_dir="aursad", split="train")
df = ds.to_pandas()

# Get one healthy episode
ep_ids = df['episode_id'].unique()
ep1 = df[df['episode_id'] == ep_ids[0]]

print(f"\nEpisode: {ep_ids[0]}")
print(f"Timesteps: {len(ep1)}")

# Setpoint columns
setpoint_cols = [c for c in df.columns if 'setpoint' in c]
effort_cols = [c for c in df.columns if 'effort_torque' in c or 'effort_force' in c]

print(f"\nSetpoint columns: {setpoint_cols}")
print(f"Effort columns: {effort_cols[:6]}...")

# Analyze variation within episode
print("\n" + "="*60)
print("SETPOINT VARIATION WITHIN EPISODE")
print("="*60)

for col in setpoint_cols[:6]:
    values = ep1[col].values
    print(f"{col}:")
    print(f"  Range: [{values.min():.3f}, {values.max():.3f}]")
    print(f"  Std:   {values.std():.4f}")
    print(f"  This is the robot arm MOVING through different positions")

# Plot setpoint trajectory
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot setpoint_pos_0 over time
ax1 = axes[0]
for i, col in enumerate(['setpoint_pos_0', 'setpoint_pos_1', 'setpoint_pos_2']):
    ax1.plot(ep1[col].values, label=col, alpha=0.7)
ax1.set_xlabel('Timestep')
ax1.set_ylabel('Position (rad)')
ax1.set_title('Setpoint Position Over Time (One Episode)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot effort over time
ax2 = axes[1]
for col in ['effort_torque_0', 'effort_force_z']:
    if col in ep1.columns:
        ax2.plot(ep1[col].values, label=col, alpha=0.7)
ax2.set_xlabel('Timestep')
ax2.set_ylabel('Effort')
ax2.set_title('Effort Over Time (One Episode)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('C:/Users/Jonaspetersen/dev/IndustrialJEPA/tests/setpoint_trajectory.png', dpi=100)
print("\nSaved plot to tests/setpoint_trajectory.png")

# Compare two episodes (healthy vs missing_screw)
from huggingface_hub import hf_hub_download
import json

meta_path = hf_hub_download(
    repo_id="Forgis/factorynet-hackathon",
    filename="metadata/aursad_metadata.json",
    repo_type="dataset"
)
with open(meta_path) as f:
    metadata = {m['episode_id']: m for m in json.load(f)}

# Find one healthy and one missing_screw tightening episode
healthy_ep = None
missing_ep = None

for ep_id in ep_ids:
    meta = metadata.get(ep_id, {})
    if meta.get('fault_label') == 'normal' and healthy_ep is None:
        healthy_ep = ep_id
    elif meta.get('fault_label') == 'missing_screw' and missing_ep is None:
        missing_ep = ep_id
    if healthy_ep and missing_ep:
        break

print("\n" + "="*60)
print("COMPARING HEALTHY vs MISSING_SCREW EPISODE")
print("="*60)

if healthy_ep and missing_ep:
    h_data = df[df['episode_id'] == healthy_ep]
    m_data = df[df['episode_id'] == missing_ep]

    print(f"\nHealthy episode: {healthy_ep} ({len(h_data)} timesteps)")
    print(f"Missing episode: {missing_ep} ({len(m_data)} timesteps)")

    # Compare setpoint trajectories
    print("\nSetpoint comparison (mean over episode):")
    for col in ['setpoint_pos_0', 'setpoint_pos_1', 'setpoint_pos_2']:
        h_mean = h_data[col].mean()
        m_mean = m_data[col].mean()
        print(f"  {col}: healthy={h_mean:.3f}, missing={m_mean:.3f}, diff={m_mean-h_mean:.3f}")

    # Compare effort
    print("\nEffort comparison (mean over episode):")
    for col in ['effort_force_z', 'effort_torque_5']:
        if col in h_data.columns:
            h_mean = h_data[col].mean()
            m_mean = m_data[col].mean()
            print(f"  {col}: healthy={h_mean:.3f}, missing={m_mean:.3f}, diff={m_mean-h_mean:.3f}")

print("\n" + "="*60)
print("KEY INSIGHT:")
print("Setpoint is a TIME-VARYING trajectory (robot arm moving).")
print("Each episode has a different trajectory (different screw position).")
print("The model should predict effort at EACH TIMESTEP from the")
print("corresponding setpoint at that timestep.")
print("="*60)
