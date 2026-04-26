"""
Post-process Phase 1 FD002 17-channel ablation results.
Run this once fd002_condition_input_results.json is available.
"""

import json
import sys
from pathlib import Path

V12_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v12')

p1_path = V12_DIR / 'fd002_condition_input_results.json'
if not p1_path.exists():
    print("fd002_condition_input_results.json not found. Waiting...")
    sys.exit(1)

with open(p1_path) as f:
    p1 = json.load(f)

print("=" * 60)
print("Phase 1.3: 17-Channel FD002 Ablation Results")
print("=" * 60)
print(f"Pretrain best probe RMSE: {p1['pretrain_best_probe_rmse']:.2f}")
print(f"\n17ch frozen: {p1['frozen_mean_rmse']:.2f} +/- {p1['frozen_std_rmse']:.2f}")
print(f"17ch e2e:    {p1['e2e_mean_rmse']:.2f} +/- {p1['e2e_std_rmse']:.2f}")
print(f"\nBaseline (14ch) frozen: {p1['baseline_frozen_rmse']:.2f}")
print(f"Baseline (14ch) e2e:    {p1['baseline_e2e_rmse']:.2f}")
print(f"\nImprovement frozen: {p1['improvement_frozen']:+.2f}")
print(f"Improvement e2e:    {p1['improvement_e2e']:+.2f}")
print(f"\nVerdict: {p1['verdict']}")

print("\nKill criterion check:")
if p1['frozen_mean_rmse'] < 20:
    print(f"  CONFIRMED: 17ch frozen RMSE={p1['frozen_mean_rmse']:.2f} < 20")
    print("  Condition-awareness hypothesis validated. V13: condition-as-input-token.")
elif p1['frozen_mean_rmse'] < 24:
    print(f"  PARTIAL: improvement={p1['improvement_frozen']:+.2f} but not <20")
    print("  Partial support for condition-awareness hypothesis.")
else:
    print(f"  NOT TRIGGERED: 17ch RMSE={p1['frozen_mean_rmse']:.2f} not better than baseline {p1['baseline_frozen_rmse']:.2f}")
    print("  Kill criterion triggered: condition-as-input-channels doesn't help FD002.")

print("\nPer-seed results:")
for seed_r in p1['frozen_per_seed']:
    print(f"  Frozen seed {p1['seeds'][p1['frozen_per_seed'].index(seed_r)]}: {seed_r:.2f}")

print("\nDone.")
