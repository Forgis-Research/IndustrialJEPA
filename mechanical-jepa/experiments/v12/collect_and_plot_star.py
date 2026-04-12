"""
Collect Phase 2 STAR results and generate the money plot:
Label efficiency comparison: JEPA E2E V2, JEPA frozen V2, Supervised LSTM, STAR

Run this script after phase2_star_label_sweep.py completes.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

V12_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v12')
PLOTS_V12 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/analysis/plots/v12')

# V11 numbers
BUDGETS_PCT = [5, 10, 20, 50, 100]
V11_E2E = {5: 25.33, 10: 18.66, 20: 16.54, 50: 14.93, 100: 13.80}
V11_FROZEN = {5: 21.53, 10: 19.93, 20: 19.83, 50: 18.71, 100: 17.81}
V11_E2E_STD = {5: 5.13, 10: 0.84, 20: 0.80, 50: 0.41, 100: 0.75}
V11_FROZEN_STD = {5: 1.96, 10: 0.86, 20: 0.34, 50: 1.13, 100: 1.67}
LSTM = {5: 33.08, 10: 31.22, 20: 18.55, 50: 18.30, 100: 17.36}
LSTM_STD = {5: 9.64, 10: 10.93, 20: 0.81, 50: 0.75, 100: 1.24}

STAR_100_PAPER = 10.61
STAR_100_REPLICATION = 12.19

# Load STAR label efficiency results
star_results_path = V12_DIR / 'star_label_efficiency.json'
if not star_results_path.exists():
    print(f"ERROR: {star_results_path} not found. Run phase2_star_label_sweep.py first.")
    exit(1)

with open(star_results_path) as f:
    star_data = json.load(f)

print("STAR label efficiency results:")
star_rmse = {}
star_std = {}

for budget_str, result in star_data['results'].items():
    budget_pct = int(budget_str)
    star_rmse[budget_pct] = result['mean_rmse']
    star_std[budget_pct] = result['std_rmse']
    print(f"  STAR {budget_pct}%: {result['mean_rmse']:.3f} +/- {result['std_rmse']:.3f}")

# Kill criterion check
print("\nKill criterion check: STAR@20% vs JEPA@20%")
if 20 in star_rmse:
    star_20 = star_rmse[20]
    jepa_20 = V11_E2E[20]
    print(f"  STAR@20%: {star_20:.3f}")
    print(f"  JEPA@20%: {jepa_20:.3f}")
    if star_20 <= 14:
        print(f"  KILL CRITERION TRIGGERED: STAR@20%={star_20:.2f} <= 14. Label-efficiency pitch dead.")
        print("  Paper narrative must pivot to H.I. recovery as headline.")
    elif star_20 < jepa_20 - 0.5:
        print(f"  KILL CRITERION TRIGGERED: STAR beats JEPA by {jepa_20-star_20:.2f} > 0.5 RMSE")
    else:
        print(f"  Kill criterion NOT triggered: STAR@20%={star_20:.2f} vs JEPA@20%={jepa_20:.2f}")

# Generate money plot
fig, ax = plt.subplots(figsize=(10, 6))

x = np.array(BUDGETS_PCT)
e2e_y = np.array([V11_E2E[b] for b in BUDGETS_PCT])
e2e_err = np.array([V11_E2E_STD[b] for b in BUDGETS_PCT])
frozen_y = np.array([V11_FROZEN[b] for b in BUDGETS_PCT])
frozen_err = np.array([V11_FROZEN_STD[b] for b in BUDGETS_PCT])
lstm_y = np.array([LSTM[b] for b in BUDGETS_PCT])
lstm_err = np.array([LSTM_STD[b] for b in BUDGETS_PCT])

star_x = sorted(star_rmse.keys())
star_y = np.array([star_rmse[b] for b in star_x])
star_err = np.array([star_std[b] for b in star_x])

ax.errorbar(x, e2e_y, yerr=e2e_err, marker='o', linewidth=2, markersize=7,
            color='steelblue', label='JEPA E2E V2 (ours)', capsize=4, zorder=4)
ax.errorbar(x, frozen_y, yerr=frozen_err, marker='s', linewidth=2, markersize=7,
            color='darkorange', label='JEPA frozen V2 (ours)', capsize=4, zorder=4)
ax.errorbar(x, lstm_y, yerr=lstm_err, marker='^', linewidth=2, markersize=7,
            color='gray', label='Supervised LSTM (baseline)', capsize=4, linestyle='--', zorder=3)
ax.errorbar(star_x, star_y, yerr=star_err, marker='D', linewidth=2, markersize=7,
            color='darkred', label='STAR (Fan et al. 2024)', capsize=4, linestyle='-.', zorder=4)

ax.axhline(STAR_100_PAPER, color='green', linestyle=':', linewidth=2,
           label=f'STAR@100% (paper) = {STAR_100_PAPER}')

ax.set_xscale('log')
ax.set_xticks(BUDGETS_PCT)
ax.set_xticklabels([f'{b}%' for b in BUDGETS_PCT])
ax.set_xlabel('Label Budget (% of FD001 training engines)', fontsize=12)
ax.set_ylabel('Test RMSE (cycles)', fontsize=12)
ax.set_title('C-MAPSS FD001: Label Efficiency (V12 Money Plot)\nJEPA vs Supervised LSTM vs STAR',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.set_ylim(8, 40)
ax.grid(True, alpha=0.3, which='both')
ax.invert_xaxis()

plt.tight_layout()
out_path = PLOTS_V12 / 'label_efficiency_with_star.png'
plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nMoney plot saved to {out_path}")

# Summary table
print("\nFull comparison table:")
print(f"{'Budget':>8} | {'JEPA E2E':>10} | {'JEPA frozen':>12} | {'LSTM':>8} | {'STAR':>8}")
print("-" * 55)
for b in BUDGETS_PCT:
    star_val = f"{star_rmse.get(b, float('nan')):.2f}" if b in star_rmse else "   -"
    print(f"{b:>7}% | {V11_E2E[b]:>10.2f} | {V11_FROZEN[b]:>12.2f} | {LSTM[b]:>8.2f} | {star_val:>8}")

# Save summary JSON
summary = {
    "budgets_pct": BUDGETS_PCT,
    "jepa_e2e": {str(b): {"mean": V11_E2E[b], "std": V11_E2E_STD[b]} for b in BUDGETS_PCT},
    "jepa_frozen": {str(b): {"mean": V11_FROZEN[b], "std": V11_FROZEN_STD[b]} for b in BUDGETS_PCT},
    "lstm": {str(b): {"mean": LSTM[b], "std": LSTM_STD[b]} for b in BUDGETS_PCT},
    "star": {str(b): {"mean": star_rmse[b], "std": star_std[b]} for b in star_x},
    "star_100_paper": STAR_100_PAPER,
    "star_100_replication": STAR_100_REPLICATION,
    "kill_criterion_star_20_le_14": star_rmse.get(20, 99) <= 14,
    "kill_criterion_star_beats_jepa_20_by_05": star_rmse.get(20, 99) < V11_E2E[20] - 0.5,
}
with open(V12_DIR / 'label_efficiency_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved to {V12_DIR / 'label_efficiency_summary.json'}")
