"""
Paper-quality figures for V12 findings.
Generates the key plots needed for the NeurIPS submission.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

V12_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v12')
PLOTS_V12 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/analysis/plots/v12')

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

# Load results
with open(V12_DIR / 'phase0_diagnostics.json') as f: p0 = json.load(f)
with open(V12_DIR / 'engine_summary_regressor.json') as f: reg = json.load(f)
with open(V12_DIR / 'shuffle_test.json') as f: shuf = json.load(f)
with open(V12_DIR / 'health_index_recovery.json') as f: hi = json.load(f)
with open(V12_DIR / 'sliding_eval.json') as f: sliding = json.load(f)
with open(V12_DIR / 'val_test_gap.json') as f: vtg = json.load(f)
with open(V12_DIR / 'pca_analysis.json') as f: pca = json.load(f)
with open(V12_DIR / 'multiseed_phase0_diagnostics.json') as f: ms = json.load(f)

# ============================================================
# Figure 1: Main results overview (3-panel)
# For NeurIPS "SSL for PHM" paper
# ============================================================

fig = plt.figure(figsize=(16, 5))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

# Panel A: RMSE comparison + baseline hierarchy
ax_a = fig.add_subplot(gs[0])
methods = ['Constant\nPredictor', 'Ridge\nRegressor\n(60 feats)', 'JEPA\nFrozen V2', 'JEPA\nE2E V2\n(5 seeds)']
rmses = [p0['constant_predictor_rmse'], reg['mean_rmse'],
         17.81, ms['test_rmse_mean']]
stds = [0, reg['std_rmse'], 1.67, ms['test_rmse_std']]
colors = ['#d73027', '#fc8d59', '#fee090', '#4575b4']

bars = ax_a.bar(methods, rmses, color=colors, edgecolor='black', linewidth=0.8, alpha=0.9)
ax_a.errorbar(range(4), rmses, yerr=stds, fmt='none', color='black', capsize=5, linewidth=1.5)
ax_a.set_ylabel('Test RMSE (cycles)')
ax_a.set_title('(a) Baseline Hierarchy\nC-MAPSS FD001', fontweight='bold')
ax_a.set_ylim(0, 50)
for bar, val in zip(bars, rmses):
    ax_a.text(bar.get_x() + bar.get_width()/2, val + 0.8, f'{val:.1f}',
              ha='center', va='bottom', fontweight='bold', fontsize=9)

# Arrow showing "benchmark requires structure"
ax_a.annotate('', xy=(2.5, 20), xytext=(1.5, 20),
               arrowprops=dict(arrowstyle='->', color='darkblue', lw=2))
ax_a.text(2.0, 20.8, '5.4 RMSE\nfrom tracking', ha='center', fontsize=8, color='darkblue')

# Panel B: H.I. recovery evidence
ax_b = fig.add_subplot(gs[1])
categories = ['Piecewise\nLinear', 'Sigmoid', 'Raw RUL\n(normalized)']
with open(V12_DIR / 'hi_alternative_params.json') as f:
    hi_alt = json.load(f)
r2_vals = [hi['r2_val'],
           hi_alt['sigmoid']['r2_val'],
           hi_alt['raw_rul_norm']['r2_val']]
bars_b = ax_b.bar(categories, r2_vals, color='steelblue', edgecolor='black', alpha=0.85)
ax_b.axhline(0.7, color='red', linestyle='--', linewidth=2, label='Target R²=0.7')
ax_b.axhline(1.0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
ax_b.set_ylabel('Validation R²')
ax_b.set_title('(b) Health Index Recovery\n(Frozen encoder, no labels)', fontweight='bold')
ax_b.set_ylim(0, 1.1)
ax_b.legend()
for bar, val in zip(bars_b, r2_vals):
    ax_b.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
              ha='center', va='bottom', fontweight='bold', fontsize=10)
ax_b.text(1, 0.05, 'All > 0.7: robust\nto H.I. parameterization',
          ha='center', fontsize=9, color='darkgreen',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5))

# Panel C: Sliding vs last-window RMSE
ax_c = fig.add_subplot(gs[2])
cut_labels = ['Last\nwindow\n(standard)', 'Sliding\n(all cuts)']
cut_rmses = [sliding['last_window_rmse'], sliding['sliding_rmse_overall']]
bars_c = ax_c.bar(cut_labels, cut_rmses, color=['#d73027', '#4575b4'],
                   edgecolor='black', alpha=0.85)
ax_c.set_ylabel('RMSE (cycles)')
ax_c.set_title('(c) Evaluation Protocol Comparison\nSliding vs Last-Window', fontweight='bold')
ax_c.set_ylim(0, 18)
for bar, val in zip(bars_c, cut_rmses):
    ax_c.text(bar.get_x() + bar.get_width()/2, val + 0.3, f'{val:.1f}',
              ha='center', va='bottom', fontweight='bold', fontsize=12)
ax_c.annotate('', xy=(0.65, 12.5), xytext=(0.65, 14.5),
               arrowprops=dict(arrowstyle='->', color='darkblue', lw=2))
ax_c.text(0.65, 13.5, '15%\nbetter', ha='center', fontsize=9, color='darkblue')

plt.suptitle('V12 Key Results: V11 Validation Summary', fontsize=13, fontweight='bold', y=1.02)
plt.savefig(str(PLOTS_V12 / 'paper_figure1_main_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved paper_figure1_main_results.png")


# ============================================================
# Figure 2: FD002 diagnosis
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('V12 Phase 1: FD002 Diagnosis', fontsize=12, fontweight='bold')

# Left: val/test gap
ax = axes[0]
subsets = ['FD001', 'FD002']
val_rmses = [vtg['FD001']['val_probe_rmse'], vtg['FD002']['val_probe_rmse']]
test_rmses = [vtg['FD001']['test_rmse_mean'], vtg['FD002']['test_rmse_mean']]
test_stds = [vtg['FD001']['test_rmse_std'], vtg['FD002']['test_rmse_std']]
x = np.arange(2)
width = 0.35
b1 = ax.bar(x - width/2, val_rmses, width, label='Val probe RMSE', color='steelblue', alpha=0.8)
b2 = ax.bar(x + width/2, test_rmses, width, label='Test RMSE (frozen)', color='darkorange', alpha=0.8)
ax.errorbar(x + width/2, test_rmses, yerr=test_stds, fmt='none', color='black', capsize=5)
ax.set_xticks(x); ax.set_xticklabels(subsets)
ax.set_ylabel('RMSE (cycles)')
ax.set_title('Val/Test Gap: FD001 vs FD002')
ax.legend()
for xi, (v, t) in enumerate(zip(val_rmses, test_rmses)):
    gap = t - v
    color = 'red' if gap > 5 else 'green'
    ax.text(xi, max(v, t) + 0.7, f'gap={gap:+.1f}', ha='center', color=color, fontweight='bold')

# Right: FD002 per-condition breakdown
with open(V12_DIR / 'fd002_per_condition_breakdown.json') as f:
    cond_data = json.load(f)
ax = axes[1]
cond_rmses = [cond_data['per_condition'].get(str(c), {}).get('rmse', 0) for c in range(6)]
cond_ns = [cond_data['per_condition'].get(str(c), {}).get('n_engines', 0) for c in range(6)]
train_counts = cond_data['train_condition_counts']
test_counts = cond_data['test_condition_counts']

# Color by overrepresentation
train_frac = np.array(train_counts) / sum(train_counts)
test_frac = np.array(test_counts) / max(1, sum(test_counts))
overrep = test_frac / (train_frac + 1e-6)
bar_cols = ['#d73027' if r > 1.5 else '#4575b4' for r in overrep]

x = np.arange(6)
bars = ax.bar(x, cond_rmses, color=bar_cols, alpha=0.85, edgecolor='black')
ax.axhline(cond_data['total_test_rmse'], color='black', linestyle='--',
           label=f'Total RMSE={cond_data["total_test_rmse"]:.1f}')
for xi, (r, n) in enumerate(zip(cond_rmses, cond_ns)):
    if n > 0:
        ax.text(xi, r + 0.3, f'n={n}', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([f'C{i}' for i in range(6)])
ax.set_xlabel('Operating condition')
ax.set_ylabel('RMSE (cycles)')
ax.set_title('Per-condition RMSE\n(red = >1.5x overrepresented vs training)')
ax.legend()

plt.tight_layout()
plt.savefig(str(PLOTS_V12 / 'paper_figure2_fd002.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved paper_figure2_fd002.png")


# ============================================================
# Figure 3: Within-engine tracking evidence
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('V12: Within-Engine Tracking Evidence', fontsize=12, fontweight='bold')

pred_stds = p0['per_engine_pred_std_all']
rhos = p0['within_engine_rho_all']

# 1. Pred std
ax = axes[0]
ax.hist(pred_stds, bins=25, color='steelblue', edgecolor='white', alpha=0.85)
ax.axvline(3, color='red', linestyle='--', linewidth=2, label='Const. threshold=3')
ax.axvline(10, color='green', linestyle='--', linewidth=2, label='Real tracker=10')
ax.axvline(np.median(pred_stds), color='orange', linewidth=2.5,
           label=f'Median={np.median(pred_stds):.1f}')
ax.set_xlabel('Per-engine prediction std (cycles)')
ax.set_ylabel('Count')
ax.set_title('(a) Prediction Variability\n(constant predictor std ~ 0)')
ax.legend(fontsize=9)

# 2. Rho
ax = axes[1]
ax.hist(rhos, bins=25, color='darkorange', edgecolor='white', alpha=0.85)
ax.axvline(0.3, color='red', linestyle='--', linewidth=2, label='Low threshold=0.3')
ax.axvline(0.7, color='green', linestyle='--', linewidth=2, label='Good tracker=0.7')
ax.axvline(np.median(rhos), color='blue', linewidth=2.5,
           label=f'Median={np.median(rhos):.2f}')
ax.set_xlabel('Within-engine Spearman rho')
ax.set_ylabel('Count')
ax.set_title('(b) Tracking Correlation\n(constant predictor rho ~ 0)')
ax.legend(fontsize=9)

# 3. Shuffle test evidence
ax = axes[2]
categories = ['With correct\nh_past', 'With shuffled\nh_past (5 seeds)']
vals = [shuf['normal_rmse'], shuf['shuffled_rmse_mean']]
errs = [0, shuf['shuffled_rmse_std']]
bar_cols2 = ['#4575b4', '#d73027']
b = ax.bar(categories, vals, color=bar_cols2, edgecolor='black', alpha=0.85)
ax.errorbar([1], [shuf['shuffled_rmse_mean']], yerr=[shuf['shuffled_rmse_std']],
            fmt='none', color='black', capsize=5, linewidth=2)
ax.set_ylabel('Test RMSE (cycles)')
ax.set_title(f'(c) Shuffle Test\nh_past carries {shuf["rmse_gain_from_h_past"]:.0f} cycles of information')
for bar, val in zip(b, vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1.5, f'{val:.1f}',
            ha='center', fontweight='bold', fontsize=11)
ax.set_ylim(0, 65)
ax.annotate(f'+{shuf["rmse_gain_from_h_past"]:.0f}',
            xy=(0.5, 35), fontsize=20, ha='center', color='darkred', fontweight='bold')

plt.tight_layout()
plt.savefig(str(PLOTS_V12 / 'paper_figure3_tracking.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved paper_figure3_tracking.png")

print("\nAll paper figures generated.")
