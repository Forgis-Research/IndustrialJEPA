"""
Generate all V12 plots:
1. Label efficiency comparison (V11 numbers now, STAR when available)
2. Phase 0 comprehensive summary
3. H.I. recovery + per-engine tracking scatter
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

V12_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v12')
PLOTS_V12 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/analysis/plots/v12')

# V11 numbers (from RESULTS_FINAL.md)
BUDGETS_PCT = [5, 10, 20, 50, 100]

V11_E2E = {5: 25.33, 10: 18.66, 20: 16.54, 50: 14.93, 100: 14.23}  # 100% = V12 5-seed mean
V11_FROZEN = {5: 21.53, 10: 19.93, 20: 19.83, 50: 18.71, 100: 17.81}
V11_E2E_STD = {5: 5.13, 10: 0.84, 20: 0.80, 50: 0.41, 100: 0.39}  # 100% = V12 5-seed std
V11_FROZEN_STD = {5: 1.96, 10: 0.86, 20: 0.34, 50: 1.13, 100: 1.67}
LSTM = {5: 33.08, 10: 31.22, 20: 18.55, 50: 18.30, 100: 17.36}
LSTM_STD = {5: 9.64, 10: 10.93, 20: 0.81, 50: 0.75, 100: 1.24}

# STAR reference
STAR_100_PAPER = 10.61
STAR_100_REPLICATION = 12.19

# ============================================================
# Plot 1: Label efficiency (V11 only, STAR pending)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

x = np.array(BUDGETS_PCT)
e2e_y = np.array([V11_E2E[b] for b in BUDGETS_PCT])
e2e_err = np.array([V11_E2E_STD[b] for b in BUDGETS_PCT])
frozen_y = np.array([V11_FROZEN[b] for b in BUDGETS_PCT])
frozen_err = np.array([V11_FROZEN_STD[b] for b in BUDGETS_PCT])
lstm_y = np.array([LSTM[b] for b in BUDGETS_PCT])
lstm_err = np.array([LSTM_STD[b] for b in BUDGETS_PCT])

ax.errorbar(x, e2e_y, yerr=e2e_err, marker='o', linewidth=2, markersize=7,
            color='steelblue', label='JEPA E2E V2', capsize=4)
ax.errorbar(x, frozen_y, yerr=frozen_err, marker='s', linewidth=2, markersize=7,
            color='darkorange', label='JEPA frozen V2', capsize=4)
ax.errorbar(x, lstm_y, yerr=lstm_err, marker='^', linewidth=2, markersize=7,
            color='gray', label='Supervised LSTM', capsize=4, linestyle='--')

ax.axhline(STAR_100_PAPER, color='green', linestyle=':', linewidth=2, label=f'STAR@100% (paper) = {STAR_100_PAPER}')
ax.axhline(STAR_100_REPLICATION, color='darkgreen', linestyle='--', linewidth=1.5,
           label=f'STAR@100% (replication) = {STAR_100_REPLICATION:.2f}')

# Annotate: "STAR label sweep PENDING"
ax.text(10, STAR_100_PAPER + 0.8, 'STAR reduced-label sweep: PENDING', fontsize=9,
        color='green', style='italic')

ax.set_xscale('log')
ax.set_xticks(BUDGETS_PCT)
ax.set_xticklabels([f'{b}%' for b in BUDGETS_PCT])
ax.set_xlabel('Label Budget (% of FD001 training engines)', fontsize=12)
ax.set_ylabel('Test RMSE (cycles)', fontsize=12)
ax.set_title('C-MAPSS FD001: Label Efficiency\nJEPA vs LSTM (STAR reduced-label sweep pending)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.set_ylim(8, 40)
ax.grid(True, alpha=0.3, which='both')
ax.invert_xaxis()

plt.tight_layout()
plt.savefig(str(PLOTS_V12 / 'label_efficiency_v11_only.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved label_efficiency_v11_only.png")

# ============================================================
# Plot 2: Phase 0 comprehensive summary
# ============================================================
# Load results
with open(V12_DIR / 'phase0_diagnostics.json') as f:
    p0 = json.load(f)
with open(V12_DIR / 'engine_summary_regressor.json') as f:
    reg = json.load(f)
with open(V12_DIR / 'shuffle_test.json') as f:
    shuffle = json.load(f)
with open(V12_DIR / 'sliding_eval.json') as f:
    sliding = json.load(f)

pred_stds = p0['per_engine_pred_std_all']
rhos = p0['within_engine_rho_all']

fig = plt.figure(figsize=(18, 10))
fig.suptitle('V12 Phase 0: Is V11 Real? Evidence Summary', fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# 1. RMSE comparison bar chart
ax1 = fig.add_subplot(gs[0, 0])
models = ['V11 E2E\n(13.80)', 'V11 frozen\n(17.81)', 'Regressor\n(19.21)', 'Constant\n(43.29)']
rmses = [13.80, 17.81, reg['mean_rmse'], p0['constant_predictor_rmse']]
colors = ['steelblue', 'darkorange', 'red', 'darkred']
bars = ax1.bar(models, rmses, color=colors, alpha=0.8, edgecolor='black')
ax1.set_ylabel('Test RMSE (cycles)')
ax1.set_title('Baseline Comparison\n(lower = better)')
for bar, val in zip(bars, rmses):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f}', ha='center', va='bottom', fontsize=9)
ax1.set_ylim(0, 50)

# 2. Per-engine prediction std histogram
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(pred_stds, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
ax2.axvline(3, color='red', linestyle='--', linewidth=2, label='Constant threshold=3')
ax2.axvline(10, color='green', linestyle='--', linewidth=2, label='Real tracker=10')
ax2.axvline(np.median(pred_stds), color='orange', linewidth=2, label=f'Median={np.median(pred_stds):.1f}')
ax2.set_xlabel('Per-engine prediction std (cycles)')
ax2.set_ylabel('Count')
ax2.set_title('Within-Engine Prediction Variability\n(constant predictor std ~ 0)')
ax2.legend(fontsize=8)

# 3. Per-engine rho histogram
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(rhos, bins=25, color='darkorange', edgecolor='white', alpha=0.8)
ax3.axvline(0.3, color='red', linestyle='--', linewidth=2, label='Low threshold=0.3')
ax3.axvline(0.5, color='green', linestyle='--', linewidth=2, label='Real tracker=0.5')
ax3.axvline(np.median(rhos), color='blue', linewidth=2, label=f'Median={np.median(rhos):.2f}')
ax3.set_xlabel('Within-engine Spearman rho')
ax3.set_ylabel('Count')
ax3.set_title('Within-Engine Tracking Quality\n(constant predictor rho ~ 0)')
ax3.legend(fontsize=8)

# 4. Shuffle test
ax4 = fig.add_subplot(gs[1, 0])
categories = ['Normal\nh_past', 'Shuffled\nh_past']
values = [shuffle['normal_rmse'], shuffle['shuffled_rmse_mean']]
bar_colors = ['steelblue', 'red']
bars4 = ax4.bar(categories, values, color=bar_colors, alpha=0.8, edgecolor='black')
ax4.errorbar([1], [shuffle['shuffled_rmse_mean']], yerr=[shuffle['shuffled_rmse_std']],
             fmt='none', color='black', capsize=5)
ax4.set_ylabel('Test RMSE (cycles)')
ax4.set_title(f'Shuffle Test\nGain from h_past: +{shuffle["rmse_gain_from_h_past"]:.1f}')
for bar, val in zip(bars4, values):
    ax4.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}', ha='center', va='bottom', fontsize=10)
ax4.set_ylim(0, 65)

# 5. Sliding eval vs last-window
ax5 = fig.add_subplot(gs[1, 1])
per_eng_rmses = [r['per_engine_rmse'] for r in sliding['per_engine_results']]
per_eng_stds = [r['pred_std'] for r in sliding['per_engine_results']]
ax5.scatter(per_eng_stds, per_eng_rmses, alpha=0.5, s=30, color='purple')
ax5.axhline(sliding['last_window_rmse'], color='red', linestyle='--',
            label=f'Last-window RMSE={sliding["last_window_rmse"]:.1f}')
ax5.axhline(sliding['sliding_rmse_overall'], color='green', linestyle='--',
            label=f'Sliding RMSE={sliding["sliding_rmse_overall"]:.1f}')
ax5.set_xlabel('Per-engine prediction std')
ax5.set_ylabel('Per-engine sliding RMSE')
ax5.set_title('Sliding Eval: RMSE vs Tracking Variability')
ax5.legend(fontsize=8)

# 6. Per-engine rho vs oracle RUL scatter (from sliding_eval)
ax6 = fig.add_subplot(gs[1, 2])
oracle_ruls = [r['oracle_rul'] for r in sliding['per_engine_results']]
per_eng_rhos = [r['rho'] for r in sliding['per_engine_results']]
sc = ax6.scatter(oracle_ruls, per_eng_rhos, c=[r['T'] for r in sliding['per_engine_results']],
                  cmap='viridis', alpha=0.7, s=30)
ax6.axhline(0.5, color='green', linestyle='--', label='Threshold=0.5')
ax6.axhline(0.3, color='red', linestyle='--', label='Threshold=0.3')
ax6.set_xlabel('Oracle RUL (cycles)')
ax6.set_ylabel('Within-engine Spearman rho')
ax6.set_title('Tracking Quality vs True RUL\n(color = engine length T)')
ax6.legend(fontsize=8)
plt.colorbar(sc, ax=ax6, label='Engine length T')

plt.savefig(str(PLOTS_V12 / 'phase0_comprehensive_summary.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved phase0_comprehensive_summary.png")


# ============================================================
# Plot 3: FD001 vs FD002 val/test gap
# ============================================================
with open(V12_DIR / 'val_test_gap.json') as f:
    vtg = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Phase 1.1: FD001 vs FD002 Val/Test Gap', fontsize=12, fontweight='bold')

for i, (subset, ax) in enumerate(zip(['FD001', 'FD002'], axes)):
    r = vtg[subset]
    categories = ['Val Probe\nRMSE', 'Test\nRMSE']
    values = [r['val_probe_rmse'], r['test_rmse_mean']]
    errors = [0, r['test_rmse_std']]
    bar_colors = ['steelblue', 'darkorange']
    bars = ax.bar(categories, values, color=bar_colors, alpha=0.8, edgecolor='black')
    ax.errorbar([1], [r['test_rmse_mean']], yerr=[r['test_rmse_std']],
                fmt='none', color='black', capsize=5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f}', ha='center', va='bottom')
    gap = r['val_test_gap']
    color = 'red' if gap > 5 else 'green'
    ax.set_title(f'{subset}\nVal/Test Gap = {gap:+.1f}', color=color, fontsize=11)
    ax.set_ylabel('RMSE (cycles)')
    ax.set_ylim(0, 35)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(str(PLOTS_V12 / 'val_test_gap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved val_test_gap.png")

print("\nAll plots generated.")
