"""
Generate label efficiency figure for C-MAPSS FD001.
Shows JEPA E2E, JEPA Frozen, and Supervised LSTM across 5 label budgets.
Optional: add STAR results if available.

Output: analysis/plots/v12/label_efficiency_figure.pdf (and .png)
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Paths
V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V12_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v12')
PLOTS_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/analysis/plots/v12')
PAPER_FIGURES_DIR = Path('/home/sagemaker-user/IndustrialJEPA/paper-neurips/figures')

# Load data - use paper-verified numbers from Table 8
# JEPA E2E and Frozen from finetune_results_v2_full.json
# but 100% E2E replaced with V12 5-seed verified result (14.23 +/- 0.39)
with open(V11_DIR / 'finetune_results_v2_full.json') as f:
    v2_data = json.load(f)

with open(V11_DIR / 'finetune_results.json') as f:
    lstm_data = json.load(f)

# Check for STAR label efficiency
star_data = None
star_path = V12_DIR / 'star_label_efficiency.json'
if star_path.exists():
    with open(star_path) as f:
        star_data = json.load(f)
    print("STAR label efficiency data found - including in figure")
else:
    print("STAR label efficiency not available - plotting without STAR")

# Budgets (as percentages for x-axis)
budgets = [5, 10, 20, 50, 100]
budget_keys = ['0.05', '0.1', '0.2', '0.5', '1.0']
n_engines = [4, 8, 17, 42, 85]  # Approximate for FD001

# Use Table 8 values directly (verified paper results)
jepa_e2e_mean = [25.33, 18.66, 16.54, 14.93, 14.23]  # 100% = V12 5-seed verified
jepa_e2e_std  = [5.13, 0.84, 0.80, 0.41, 0.39]        # 100% = V12 5-seed verified
jepa_frozen_mean = [21.53, 19.93, 19.83, 18.71, 17.81]
jepa_frozen_std  = [2.0, 0.86, 0.34, 1.13, 1.67]
lstm_mean = [33.08, 31.22, 18.55, 18.30, 17.36]
lstm_std  = [9.64, 10.93, 0.81, 0.75, 1.24]

print("Data loaded (Table 8 values):")
print(f"JEPA E2E: {jepa_e2e_mean}")
print(f"JEPA Frozen: {jepa_frozen_mean}")
print(f"LSTM: {lstm_mean}")

# Reference lines
star_paper_100 = 12.19  # Our 5-seed replication
aelstm_100 = 13.99  # AE-LSTM SSL (single number, 100% only)

# Create figure
fig, ax = plt.subplots(figsize=(7, 5))

x = np.array(budgets)

# Plot JEPA E2E
ax.plot(x, jepa_e2e_mean, 'o-', color='#1f77b4', linewidth=2.5, markersize=8,
        label='Traj JEPA E2E (ours)', zorder=5)
ax.fill_between(x,
    np.array(jepa_e2e_mean) - np.array(jepa_e2e_std),
    np.array(jepa_e2e_mean) + np.array(jepa_e2e_std),
    alpha=0.15, color='#1f77b4')

# Plot JEPA Frozen
ax.plot(x, jepa_frozen_mean, 's--', color='#aec7e8', linewidth=2, markersize=7,
        label='Traj JEPA Frozen (ours)', zorder=4)
ax.fill_between(x,
    np.array(jepa_frozen_mean) - np.array(jepa_frozen_std),
    np.array(jepa_frozen_mean) + np.array(jepa_frozen_std),
    alpha=0.15, color='#aec7e8')

# Plot Supervised LSTM
ax.plot(x, lstm_mean, '^:', color='#d62728', linewidth=2, markersize=7,
        label='Supervised LSTM (baseline)', zorder=3)
ax.fill_between(x,
    np.array(lstm_mean) - np.array(lstm_std),
    np.array(lstm_mean) + np.array(lstm_std),
    alpha=0.15, color='#d62728')

# Add STAR if available
if star_data is not None:
    star_budgets = [int(k) for k in star_data['results'].keys()]
    star_means = [star_data['results'][k]['mean_rmse'] for k in star_data['results']]
    star_stds = [star_data['results'][k]['std_rmse'] for k in star_data['results']]
    # Sort by budget
    sorted_star = sorted(zip(star_budgets, star_means, star_stds))
    star_budgets = [s[0] for s in sorted_star]
    star_means = [s[1] for s in sorted_star]
    star_stds = [s[2] for s in sorted_star]
    ax.plot(star_budgets, star_means, 'D-', color='#2ca02c', linewidth=2, markersize=7,
            label='STAR Supervised (replic.)', zorder=6)
    ax.fill_between(star_budgets,
        np.array(star_means) - np.array(star_stds),
        np.array(star_means) + np.array(star_stds),
        alpha=0.15, color='#2ca02c')

# Reference lines
ax.axhline(y=star_paper_100, color='#2ca02c', linestyle=':', alpha=0.7, linewidth=1.5,
           label=f'STAR supervised @ 100% (replic. 12.19)')
ax.axhline(y=aelstm_100, color='gray', linestyle=':', alpha=0.7, linewidth=1.5,
           label=f'AE-LSTM SSL @ 100% (13.99)')

# Formatting
ax.set_xlabel('Label Budget (% of training engines)', fontsize=12)
ax.set_ylabel('Test RMSE (cycles)', fontsize=12)
ax.set_title('C-MAPSS FD001 Label Efficiency', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.set_xticks(budgets)
ax.set_xticklabels([f'{b}%\n({n} eng)' for b, n in zip(budgets, n_engines)], fontsize=9)
ax.set_xlim(3, 120)
ax.set_ylim(10, 40)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc='upper left')

# Highlight crossover region
ax.axvspan(8, 12, alpha=0.1, color='yellow', label='_Frozen beats LSTM')
ax.text(10, 38, 'Frozen > LSTM\n(< 10% labels)', ha='center', fontsize=8,
        color='#888800', style='italic')

plt.tight_layout()

# Save
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(PLOTS_DIR / 'label_efficiency_figure.png', dpi=150, bbox_inches='tight')
fig.savefig(PLOTS_DIR / 'label_efficiency_figure.pdf', bbox_inches='tight')
PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(PAPER_FIGURES_DIR / 'v12_label_efficiency.png', dpi=150, bbox_inches='tight')
fig.savefig(PAPER_FIGURES_DIR / 'v12_label_efficiency.pdf', bbox_inches='tight')

print(f"Saved to {PLOTS_DIR / 'label_efficiency_figure.pdf'}")
print(f"Saved to {PAPER_FIGURES_DIR / 'v12_label_efficiency.pdf'}")

# Print statistics
print("\nKey comparison:")
print(f"JEPA E2E @ 10% (8 engines): {jepa_e2e_mean[1]:.2f} vs LSTM: {lstm_mean[1]:.2f} "
      f"(+{(lstm_mean[1]-jepa_e2e_mean[1])/lstm_mean[1]*100:.1f}% JEPA wins)")
print(f"JEPA Frozen @ 5% (4 engines): {jepa_frozen_mean[0]:.2f} vs LSTM: {lstm_mean[0]:.2f} "
      f"(+{(lstm_mean[0]-jepa_frozen_mean[0])/lstm_mean[0]*100:.1f}% JEPA wins)")
