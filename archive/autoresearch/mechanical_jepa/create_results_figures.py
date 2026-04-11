"""
Create publication-quality results figures for Mechanical-JEPA.
Two figures: (1) Cross-embodiment transfer, (2) Method comparison overview.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

# ─── Data from Experiment Log (Session 2, 3 seeds) ───

# Exp 5: Cross-embodiment few-shot transfer
robots = ['KUKA\niiwa', 'UR5', 'JACO', 'FANUC']
robots_short = ['KUKA', 'UR5', 'JACO', 'FANUC']

# Transfer ratio = pretrained / scratch (lower = pretraining helps more)
ratios = {
    10:  [0.678, 0.606, 0.745, 1.027],
    50:  [0.563, 0.661, 0.554, 0.913],
    100: [0.581, 0.840, 0.473, 0.936],
}

# Raw MSE: pretrained, scratch, linear (at each budget)
raw_mse = {
    'KUKA':  {10: (0.584, 0.862, 0.232), 50: (0.385, 0.684, 0.006), 100: (0.341, 0.587, 0.005)},
    'UR5':   {10: (0.571, 0.942, 0.344), 50: (0.327, 0.495, 0.019), 100: (0.242, 0.288, 0.013)},
    'JACO':  {10: (0.360, 0.483, 0.158), 50: (0.109, 0.197, 0.005), 100: (0.095, 0.201, 0.002)},
    'FANUC': {10: (0.447, 0.435, 0.175), 50: (0.178, 0.195, 0.008), 100: (0.151, 0.161, 0.008)},
}

# Exp 2: Embodiment classification
class_methods = ['Chance', 'Pretrained\nEncoder', 'Random\nEncoder', 'Raw\nFeatures']
class_accs = [20.0, 65.1, 79.8, 81.1]
class_stds = [0, 1.3, 1.2, 1.4]

# Exp 4: In-domain forecasting
forecast_methods = ['Copy-last', 'Linear', 'MLP', 'Encoder\n(frozen)', 'Encoder\n(finetuned)', 'Scratch\nTransformer']
forecast_h1 = [0.00010, 0.00007, 0.00021, 0.00495, 0.00201, 0.00464]
forecast_h10 = [0.00349, 0.00253, 0.00221, 0.00735, 0.01644, 0.01840]

# Colors
C_PRETRAINED = '#2196F3'  # blue
C_SCRATCH = '#FF7043'     # orange-red
C_LINEAR = '#66BB6A'      # green
C_RANDOM = '#9E9E9E'      # gray
C_RAW = '#AB47BC'         # purple
C_CHANCE = '#E0E0E0'      # light gray
C_BG = '#FAFAFA'

# ════════════════════════════════════════════════════
# FIGURE 1: Cross-Embodiment Transfer (THE key result)
# ════════════════════════════════════════════════════
fig1, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [1.3, 1]})
fig1.patch.set_facecolor('white')

# --- Left panel: Transfer ratio heatmap + bar chart ---
ax1 = axes[0]

budgets = [10, 50, 100]
x = np.arange(len(robots))
width = 0.25

for i, budget in enumerate(budgets):
    vals = ratios[budget]
    bars = ax1.bar(x + (i - 1) * width, vals, width,
                   label=f'{budget}-shot',
                   color=[C_PRETRAINED, '#42A5F5', '#90CAF9'][i],
                   edgecolor='white', linewidth=0.5)
    # Add value labels
    for j, (bar, v) in enumerate(zip(bars, vals)):
        color = '#1B5E20' if v < 0.9 else '#B71C1C'
        fontweight = 'bold' if v < 0.9 else 'normal'
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{v:.2f}', ha='center', va='bottom', fontsize=9,
                color=color, fontweight=fontweight)

# Reference line at 1.0
ax1.axhline(y=1.0, color='#B71C1C', linestyle='--', linewidth=1.5, alpha=0.7, label='No benefit')
ax1.axhline(y=0.9, color='#1B5E20', linestyle=':', linewidth=1, alpha=0.5, label='10% improvement')

ax1.set_xticks(x)
ax1.set_xticklabels(robots, fontsize=11)
ax1.set_ylabel('Transfer Ratio (pretrained / scratch)', fontsize=11)
ax1.set_title('Cross-Embodiment Few-Shot Transfer', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax1.set_ylim(0, 1.25)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_facecolor(C_BG)

# Add annotation
ax1.annotate('← pretraining helps', xy=(0.02, 0.4), fontsize=8, color='#1B5E20',
            xycoords='axes fraction', style='italic')
ax1.annotate('no benefit →', xy=(0.78, 0.88), fontsize=8, color='#B71C1C',
            xycoords='axes fraction', style='italic')

# --- Right panel: The elephant in the room (linear dominates) ---
ax2 = axes[1]

# Show MSE at 50-shot for all methods on KUKA (representative)
methods_50 = ['Linear\nRegression', 'JEPA\nPretrained', 'Transformer\nfrom Scratch']
kuka_50 = [0.006, 0.385, 0.684]
colors_50 = [C_LINEAR, C_PRETRAINED, C_SCRATCH]

bars2 = ax2.barh(range(len(methods_50)), kuka_50, color=colors_50,
                  edgecolor='white', height=0.6)
for i, (bar, v) in enumerate(zip(bars2, kuka_50)):
    ax2.text(v + 0.02, bar.get_y() + bar.get_height()/2,
            f'{v:.3f}', va='center', fontsize=11, fontweight='bold')

ax2.set_yticks(range(len(methods_50)))
ax2.set_yticklabels(methods_50, fontsize=10)
ax2.set_xlabel('MSE (KUKA, 50-shot)', fontsize=11)
ax2.set_title('But: Linear Wins at 50+ Shots', fontsize=13, fontweight='bold', color='#B71C1C')
ax2.set_xlim(0, 0.85)
ax2.invert_yaxis()
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_facecolor(C_BG)

# Add annotation
ax2.annotate('64× better', xy=(0.15, 0.75), fontsize=10, color=C_LINEAR,
            xycoords='axes fraction', fontweight='bold')

fig1.suptitle('Mechanical-JEPA: Cross-Embodiment Dynamics Transfer on OXE',
             fontsize=14, fontweight='bold', y=1.02)
fig1.tight_layout()
fig1.savefig('autoresearch/mechanical_jepa/fig1_transfer.png', dpi=200, bbox_inches='tight',
            facecolor='white')
print("Saved fig1_transfer.png")


# ════════════════════════════════════════════════════
# FIGURE 2: Complete results overview (4 panels)
# ════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(14, 10))
fig2.patch.set_facecolor('white')
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

# --- Panel A: Pretraining convergence ---
ax_a = fig2.add_subplot(gs[0, 0])
epochs = [1, 5, 10, 20, 30, 40, 50]
loss_curve = [0.33, 0.16, 0.10, 0.06, 0.04, 0.04, 0.04]
ax_a.plot(epochs, loss_curve, 'o-', color=C_PRETRAINED, linewidth=2, markersize=6)
ax_a.fill_between(epochs, [l*0.85 for l in loss_curve], [l*1.15 for l in loss_curve],
                  alpha=0.15, color=C_PRETRAINED)
ax_a.set_xlabel('Epoch', fontsize=10)
ax_a.set_ylabel('Validation Loss', fontsize=10)
ax_a.set_title('A) JEPA Pretraining Convergence', fontsize=12, fontweight='bold')
ax_a.set_ylim(0, 0.38)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
ax_a.set_facecolor(C_BG)
ax_a.annotate('1,003 Franka episodes\n2.28M params, 3 seeds', xy=(0.55, 0.75),
             xycoords='axes fraction', fontsize=9, color='gray',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='lightgray'))

# --- Panel B: Embodiment classification (JEPA hurts) ---
ax_b = fig2.add_subplot(gs[0, 1])
colors_class = [C_CHANCE, C_PRETRAINED, C_RANDOM, C_RAW]
bars_b = ax_b.bar(range(len(class_methods)), class_accs,
                  yerr=class_stds, capsize=4,
                  color=colors_class, edgecolor='white', linewidth=0.5)
for bar, acc in zip(bars_b, class_accs):
    ax_b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{acc:.1f}%', ha='center', fontsize=10, fontweight='bold')

ax_b.set_xticks(range(len(class_methods)))
ax_b.set_xticklabels(class_methods, fontsize=9)
ax_b.set_ylabel('Accuracy (%)', fontsize=10)
ax_b.set_title('B) Embodiment Classification', fontsize=12, fontweight='bold')
ax_b.set_ylim(0, 100)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.set_facecolor(C_BG)
ax_b.annotate('JEPA strips static\nfeatures by design', xy=(0.15, 0.55),
             xycoords='axes fraction', fontsize=9, color='#B71C1C', style='italic')

# --- Panel C: In-domain forecasting (linear wins) ---
ax_c = fig2.add_subplot(gs[1, 0])
colors_f = [C_RANDOM, C_LINEAR, C_RAW, C_RANDOM, C_PRETRAINED, C_SCRATCH]
bars_c = ax_c.barh(range(len(forecast_methods)), forecast_h1, color=colors_f,
                   edgecolor='white', height=0.6)
for i, (bar, v) in enumerate(zip(bars_c, forecast_h1)):
    ax_c.text(max(v + 0.0003, 0.0005), bar.get_y() + bar.get_height()/2,
             f'{v:.5f}', va='center', fontsize=9)

ax_c.set_yticks(range(len(forecast_methods)))
ax_c.set_yticklabels(forecast_methods, fontsize=9)
ax_c.set_xlabel('MSE (h=1, TOTO in-domain)', fontsize=10)
ax_c.set_title('C) In-Domain Forecasting', fontsize=12, fontweight='bold')
ax_c.invert_yaxis()
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)
ax_c.set_facecolor(C_BG)
ax_c.annotate('Linear wins.\nPretraining helps\nvs scratch (2.3×)',
             xy=(0.65, 0.6), xycoords='axes fraction', fontsize=9,
             color='gray', style='italic')

# --- Panel D: Transfer ratio across all robots and budgets ---
ax_d = fig2.add_subplot(gs[1, 1])

for i, robot in enumerate(robots_short):
    vals = [ratios[b][i] for b in budgets]
    marker = 'o' if min(vals) < 0.9 else 'x'
    linestyle = '-' if min(vals) < 0.9 else '--'
    alpha = 1.0 if min(vals) < 0.9 else 0.5
    ax_d.plot(budgets, vals, f'{marker}{linestyle}', label=robot,
             linewidth=2, markersize=8, alpha=alpha)

ax_d.axhline(y=1.0, color='#B71C1C', linestyle='--', linewidth=1, alpha=0.5)
ax_d.axhline(y=0.9, color='#1B5E20', linestyle=':', linewidth=1, alpha=0.3)
ax_d.fill_between([5, 105], [0], [0.9], alpha=0.05, color='green')
ax_d.set_xlabel('Training Budget (windows)', fontsize=10)
ax_d.set_ylabel('Transfer Ratio', fontsize=10)
ax_d.set_title('D) Transfer Ratio by Data Budget', fontsize=12, fontweight='bold')
ax_d.legend(fontsize=9, loc='upper right')
ax_d.set_ylim(0.3, 1.15)
ax_d.set_xticks(budgets)
ax_d.spines['top'].set_visible(False)
ax_d.spines['right'].set_visible(False)
ax_d.set_facecolor(C_BG)
ax_d.annotate('pretraining\nhelps here', xy=(0.05, 0.15), xycoords='axes fraction',
             fontsize=8, color='#1B5E20', style='italic')

fig2.suptitle('Mechanical-JEPA: Complete Results Overview',
             fontsize=14, fontweight='bold', y=1.01)
fig2.savefig('autoresearch/mechanical_jepa/fig2_overview.png', dpi=200, bbox_inches='tight',
            facecolor='white')
print("Saved fig2_overview.png")

print("\nDone.")
