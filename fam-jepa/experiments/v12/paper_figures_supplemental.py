"""
Supplemental figures for the V12 NeurIPS submission.
Generates additional diagnostic plots not in the main paper.
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

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

# Load data
with open(V12_DIR / 'phase0_diagnostics.json') as f: p0 = json.load(f)
with open(V12_DIR / 'frozen_vs_e2e_tracking.json') as f: fte = json.load(f)
with open(V12_DIR / 'val_test_gap.json') as f: vtg = json.load(f)
with open(V12_DIR / 'engine_summary_regressor.json') as f: reg = json.load(f)
with open(V12_DIR / 'extra_fd003_fd004_diagnostics.json') as f: fd34 = json.load(f)


# ============================================================
# Supplemental Figure S1: Multi-subset tracking comparison
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('V12 Supplemental: Tracking Verification Across All C-MAPSS Subsets',
             fontsize=13, fontweight='bold')

# Panel A: RMSE vs. regressor across subsets
ax = axes[0]
subsets = ['FD001', 'FD003', 'FD004']
jepa_rmses = [p0['v11_reported_rmse'], fd34['FD003']['v11_reported_e2e'], fd34['FD004']['v11_reported_e2e']]
reg_rmses = [reg['mean_rmse'], fd34['FD003']['engine_summary_reg_rmse'], fd34['FD004']['engine_summary_reg_rmse']]
x = np.arange(3)
w = 0.35
b1 = ax.bar(x - w/2, jepa_rmses, w, label='JEPA E2E V2', color='steelblue', alpha=0.85)
b2 = ax.bar(x + w/2, reg_rmses, w, label='Engine-summary regressor', color='coral', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(subsets)
ax.set_ylabel('Test RMSE (cycles)')
ax.set_title('(a) JEPA vs. Regressor\n(Regressor cannot track within-engine)')
ax.legend()
for xi, (j, r) in enumerate(zip(jepa_rmses, reg_rmses)):
    delta = r - j
    ax.text(xi, max(j, r) + 0.5, f'+{delta:.1f}', ha='center', fontsize=9, color='darkblue', fontweight='bold')

# Panel B: rho median across subsets
ax = axes[1]
rho_meds = [
    p0['within_engine_rho_median'],
    fd34['FD003']['rho_median'],
    fd34['FD004']['rho_median'],
]
colors = ['steelblue', 'darkorange', 'green']
bars = ax.bar(subsets, rho_meds, color=colors, alpha=0.85, edgecolor='black')
ax.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Minimum threshold=0.5')
ax.axhline(0.7, color='green', linestyle='--', linewidth=1.5, label='Good tracker=0.7', alpha=0.7)
ax.set_ylabel('Within-engine Spearman rho (median)')
ax.set_title('(b) Tracking Quality Across Subsets\n(E2E fine-tuned, seed=0)')
ax.set_ylim(0, 1.0)
ax.legend(fontsize=9)
for bar, val in zip(bars, rho_meds):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
            ha='center', fontweight='bold', fontsize=10)

# Panel C: pred_std median across subsets (confirms non-constant)
ax = axes[2]
std_meds = [
    p0['per_engine_pred_std_median'],
    fd34['FD003']['pred_std_median'],
    fd34['FD004']['pred_std_median'],
]
bars = ax.bar(subsets, std_meds, color=colors, alpha=0.85, edgecolor='black')
ax.axhline(10, color='red', linestyle='--', linewidth=2, label='Min threshold=10')
ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
ax.set_ylabel('Per-engine prediction std (cycles) - median')
ax.set_title('(c) Prediction Variability\n(constant predictor: std~0)')
ax.legend(fontsize=9)
for bar, val in zip(bars, std_meds):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.3, f'{val:.1f}',
            ha='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(str(PLOTS_V12 / 'suppl_figure_S1_multisubset.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved suppl_figure_S1_multisubset.png")


# ============================================================
# Supplemental Figure S2: FD002 diagnosis - comprehensive
# ============================================================

with open(V12_DIR / 'fd002_per_condition_breakdown.json') as f:
    cond_data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('V12 Supplemental: FD002 Distribution Shift Diagnosis',
             fontsize=13, fontweight='bold')

# Panel A: Val probe vs test gap bar chart
ax = axes[0]
fd1 = vtg['FD001']
fd2 = vtg['FD002']
x = np.arange(2)
w = 0.35
b1 = ax.bar(x - w/2, [fd1['val_probe_rmse'], fd2['val_probe_rmse']], w,
            label='Val probe RMSE\n(training dist.)', color='steelblue', alpha=0.85)
b2 = ax.bar(x + w/2, [fd1['test_rmse_mean'], fd2['test_rmse_mean']], w,
            label='Test RMSE\n(canonical test)', color='darkorange', alpha=0.85)
ax.errorbar(x + w/2, [fd1['test_rmse_mean'], fd2['test_rmse_mean']],
            yerr=[fd1['test_rmse_std'], fd2['test_rmse_std']],
            fmt='none', color='black', capsize=5)
ax.set_xticks(x); ax.set_xticklabels(['FD001\n(1 condition)', 'FD002\n(6 conditions)'])
ax.set_ylabel('RMSE (cycles)')
ax.set_title('(a) Val vs. Test Gap\n(SSL learns well; test set has distribution shift)')
ax.legend(fontsize=9)
for xi, (v, t) in enumerate(zip([fd1['val_probe_rmse'], fd2['val_probe_rmse']],
                                  [fd1['test_rmse_mean'], fd2['test_rmse_mean']])):
    color = 'red' if abs(t-v) > 5 else 'darkgreen'
    ax.text(xi, max(v, t) + 0.8, f'gap={t-v:+.1f}', ha='center', color=color,
            fontweight='bold', fontsize=10)

# Panel B: Per-condition RMSE at test time
ax = axes[1]
cond_rmses = [cond_data['per_condition'].get(str(c), {}).get('rmse', 0) for c in range(6)]
cond_ns = [cond_data['per_condition'].get(str(c), {}).get('n_engines', 0) for c in range(6)]
train_counts = cond_data['train_condition_counts']
test_counts = cond_data['test_condition_counts']
train_frac = np.array(train_counts) / sum(train_counts)
test_frac = np.array(test_counts) / max(1, sum(test_counts))
overrep = test_frac / (train_frac + 1e-6)
bar_cols = ['#d73027' if r > 1.5 else '#4575b4' for r in overrep]
x = np.arange(6)
bars = ax.bar(x, cond_rmses, color=bar_cols, alpha=0.85, edgecolor='black')
ax.axhline(cond_data['total_test_rmse'], color='black', linestyle='--', linewidth=1.5,
           label=f'Overall RMSE={cond_data["total_test_rmse"]:.1f}')
for xi, (r, n) in enumerate(zip(cond_rmses, cond_ns)):
    if n > 0:
        ax.text(xi, r + 0.3, f'n={n}', ha='center', va='bottom', fontsize=8)
        ax.text(xi, 0.5, f'{overrep[xi]:.1f}x', ha='center', fontsize=7,
                color='white' if overrep[xi] > 1.5 else 'black', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels([f'C{i}' for i in range(6)])
ax.set_xlabel('Operating condition')
ax.set_ylabel('RMSE (cycles)')
ax.set_title('(b) Per-Condition Test RMSE\n(red = overrepresented at test vs. training)')
ax.legend(fontsize=9)

# Panel C: Train vs test condition fractions
ax = axes[2]
x = np.arange(6)
w = 0.35
b1 = ax.bar(x - w/2, train_frac, w, label='Training (all cycles)', color='steelblue', alpha=0.8)
b2 = ax.bar(x + w/2, test_frac, w, label='Test last-window cycles', color='darkorange', alpha=0.8)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels([f'C{i}' for i in range(6)])
ax.set_xlabel('Operating condition (KMeans)')
ax.set_ylabel('Fraction of samples')
ax.set_title('(c) Condition Distribution Mismatch\n(training vs. test last-windows)')
ax.legend(fontsize=9)
for xi, (tr, te, r) in enumerate(zip(train_frac, test_frac, overrep)):
    if r > 1.5:
        ax.annotate(f'{r:.1f}x\nover',
                    xy=(xi + w/2, te),
                    xytext=(xi + w/2, te + 0.06),
                    ha='center', fontsize=7.5, color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.0))

plt.tight_layout()
plt.savefig(str(PLOTS_V12 / 'suppl_figure_S2_fd002.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved suppl_figure_S2_fd002.png")


# ============================================================
# Supplemental Figure S3: Frozen vs E2E analysis
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('V12 Supplemental: E2E vs. Frozen Fine-tuning Analysis',
             fontsize=12, fontweight='bold')

# Panel A: RMSE vs rho scatter (frozen=orange, e2e=blue)
ax = axes[0]
modes = ['Frozen', 'E2E']
rmses = [fte['frozen']['test_rmse'], fte['e2e']['test_rmse']]
rhos = [fte['frozen']['rho_median'], fte['e2e']['rho_median']]
colors_ab = ['darkorange', 'steelblue']
for mode, rmse, rho, c in zip(modes, rmses, rhos, colors_ab):
    ax.scatter(rho, rmse, s=200, color=c, label=f'{mode}\n(RMSE={rmse:.1f}, rho={rho:.3f})',
               zorder=5, edgecolors='black', linewidths=1.5)
ax.set_xlabel('Median within-engine Spearman rho')
ax.set_ylabel('Test RMSE (cycles)')
ax.set_title('(a) RMSE vs. Tracking Quality\nFrozen tracks better, E2E predicts better')
ax.legend(fontsize=10)
ax.annotate('E2E advantage:\ncalibration, not tracking',
            xy=(fte['e2e']['rho_median'], fte['e2e']['test_rmse']),
            xytext=(0.82, 14.5),
            arrowprops=dict(arrowstyle='->', color='darkblue', lw=1.5),
            fontsize=9, color='darkblue')

# Panel B: What each mode gains/loses
ax = axes[1]
metrics = ['Test RMSE\n(lower=better)', 'Pred std median', 'rho median\n(x10)']
frozen_vals = [fte['frozen']['test_rmse'], fte['frozen']['pred_std_median'], fte['frozen']['rho_median']*10]
e2e_vals = [fte['e2e']['test_rmse'], fte['e2e']['pred_std_median'], fte['e2e']['rho_median']*10]

x = np.arange(3)
w = 0.35
b1 = ax.bar(x - w/2, frozen_vals, w, label='Frozen', color='darkorange', alpha=0.85)
b2 = ax.bar(x + w/2, e2e_vals, w, label='E2E', color='steelblue', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_title('(b) Mode Comparison Summary\n(rho scaled x10 for display)')
ax.legend()
for xi, (f, e) in enumerate(zip(frozen_vals, e2e_vals)):
    color = 'darkgreen' if e < f else 'darkred'  # green if e2e is better
    label = f'{e-f:+.1f}'
    ax.text(xi + w/2, max(f, e) + 0.3, label, ha='center', fontsize=8, color=color)

plt.tight_layout()
plt.savefig(str(PLOTS_V12 / 'suppl_figure_S3_frozen_vs_e2e.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved suppl_figure_S3_frozen_vs_e2e.png")

print("\nAll supplemental figures generated.")
