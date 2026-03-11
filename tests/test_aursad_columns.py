# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""Check what setpoint and effort actually are in AURSAD."""

from datasets import load_dataset
import pandas as pd

print("Loading AURSAD...")
ds = load_dataset("Forgis/factorynet-hackathon", data_dir="aursad", split="train")

# Get column names
df = ds.to_pandas()
print(f"\nAll columns ({len(df.columns)}):")
for col in sorted(df.columns):
    print(f"  {col}")

# Check setpoint columns
setpoint_cols = [c for c in df.columns if 'setpoint' in c.lower()]
print(f"\nSetpoint columns: {setpoint_cols}")

# Check effort columns
effort_cols = [c for c in df.columns if 'effort' in c.lower() or 'torque' in c.lower() or 'current' in c.lower()]
print(f"\nEffort-related columns: {effort_cols}")

# Sample some data
print("\nSample values (first episode):")
ep1 = df[df['episode_id'] == df['episode_id'].iloc[0]]
print(f"Episode: {ep1['episode_id'].iloc[0]}")
print(f"Rows: {len(ep1)}")

if setpoint_cols:
    print(f"\nSetpoint stats:")
    for col in setpoint_cols[:6]:
        print(f"  {col}: min={ep1[col].min():.3f}, max={ep1[col].max():.3f}, mean={ep1[col].mean():.3f}")

if effort_cols:
    print(f"\nEffort stats:")
    for col in effort_cols[:6]:
        print(f"  {col}: min={ep1[col].min():.3f}, max={ep1[col].max():.3f}, mean={ep1[col].mean():.3f}")
