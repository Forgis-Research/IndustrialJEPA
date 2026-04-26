"""
Post-process STAR label sweep results.
Run this once star_label_efficiency.json is available.
"""

import json
import sys
import numpy as np
from pathlib import Path

V12_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v12')
PLOTS_V12 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/analysis/plots/v12')

star_path = V12_DIR / 'star_label_efficiency.json'
if not star_path.exists():
    print("star_label_efficiency.json not found. Run phase2_star_label_sweep.py first.")
    sys.exit(1)

with open(star_path) as f:
    star_data = json.load(f)

# V11 numbers
V11_E2E = {5: 25.33, 10: 18.66, 20: 16.54, 50: 14.93, 100: 14.23}  # 100% = V12 5-seed mean
V11_E2E_STD = {5: 5.13, 10: 0.84, 20: 0.80, 50: 0.41, 100: 0.39}  # 100% = V12 5-seed std
STAR_100_PAPER = 10.61
STAR_100_REPLICATION = 12.19

print("=" * 60)
print("STAR Label Efficiency Results")
print("=" * 60)

star_rmse = {}
star_std = {}
for budget_str, result in star_data['results'].items():
    budget_pct = int(budget_str)
    star_rmse[budget_pct] = result['mean_rmse']
    star_std[budget_pct] = result['std_rmse']
    print(f"  STAR {budget_pct}%: {result['mean_rmse']:.3f} +/- {result['std_rmse']:.3f}")

print(f"\n  STAR 100% (paper): {STAR_100_PAPER}")
print(f"  STAR 100% (replication): {STAR_100_REPLICATION}")

print("\n--- Kill Criterion Check ---")
if 20 in star_rmse:
    star_20 = star_rmse[20]
    jepa_20 = V11_E2E[20]
    if star_20 <= 14:
        print(f"KILL CRITERION TRIGGERED: STAR@20%={star_20:.2f} <= 14")
        print("Label-efficiency pitch is DEAD.")
        print("Paper must pivot to H.I. recovery as headline.")
        print("-> JEPA's SSL contribution: H.I. recovery R2=0.926 (no labels)")
    elif star_20 < jepa_20 - 0.5:
        print(f"KILL CRITERION TRIGGERED: STAR@20%={star_20:.2f} beats JEPA@20%={jepa_20:.2f} by {jepa_20-star_20:.2f}")
        print("Label-efficiency pitch dead for frozen, but E2E still valuable.")
    else:
        print(f"Kill criterion NOT triggered: STAR@20%={star_20:.2f}, JEPA@20%={jepa_20:.2f}")
        print("Label-efficiency pitch SURVIVES. JEPA frozen > STAR at 20% labels.")

print("\n--- Full Comparison Table ---")
print(f"{'Budget':>8} | {'JEPA E2E':>10} +/- {'std':>6} | {'STAR':>8} +/- {'std':>6} | {'Delta':>8}")
print("-" * 60)
budgets = sorted(star_rmse.keys(), reverse=True)
for b in budgets:
    j = V11_E2E.get(b, None)
    j_std = V11_E2E_STD.get(b, None)
    s = star_rmse[b]
    s_std = star_std[b]
    if j:
        delta = j - s  # positive = JEPA worse than STAR
        print(f"{b:>7}% | {j:>10.2f} +/- {j_std:>4.2f} | {s:>8.2f} +/- {s_std:>4.2f} | {delta:>+8.2f}")
    else:
        print(f"{b:>7}% | {'N/A':>10}      {'N/A':>6} | {s:>8.2f} +/- {s_std:>4.2f} |      N/A")

# Now run collect_and_plot_star.py to generate the money plot
import subprocess
result = subprocess.run(
    ['python', str(V12_DIR / 'collect_and_plot_star.py')],
    capture_output=True, text=True
)
print("\n--- Money Plot Generation ---")
if result.returncode == 0:
    print(result.stdout)
    print(f"Money plot saved: {PLOTS_V12 / 'label_efficiency_with_star.png'}")
else:
    print("ERROR running collect_and_plot_star.py:")
    print(result.stderr)

print("\nDone.")
