#!/usr/bin/env python3
"""
FIGURE DESIGN
Message: "Architecture ablation: causal V2 has the best balance of frozen and E2E quality;
         cross-sensor excels at frozen 100% but is brittle; bidirectional V16b collapses."
Type: Horizontal bar chart comparing frozen vs E2E RMSE across variants
Layout: Two panels side by side (Frozen, E2E)
Colors: Forgis palette
Size: NeurIPS single column (5.5in wide, ~2.0in tall)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C = {
    "blue":     "#2B6CB0",
    "teal":     "#2D8A6E",
    "orange":   "#FF5A00",
    "red":      "#DC4B07",
    "gray":     "#878F92",
    "gray_lt":  "#D1D5D8",
    "dark":     "#122128",
    "blue_bg":  "#E8F0FA",
    "teal_bg":  "#E6F5F0",
    "orange_bg":"#FFF0E6",
    "red_bg":   "#FDEDE6",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.labelsize": 9,
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "figure.dpi": 150,
    "figure.constrained_layout.use": True,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# Data from Table 5 in paper
variants = [
    "V2 (causal + EMA)\n\\textit{ours default}",
    "Full-seq target",
    "Cross-sensor",
    "V16b (bidi + VICReg)",
]
# Use simple labels (no LaTeX in matplotlib unless usetex)
labels = [
    "V2 (causal + EMA)\nours default",
    "Full-seq target",
    "Cross-sensor\n(sensor-as-token)",
    "V16b (bidi + VICReg)",
]

frozen_mean = np.array([17.81, 15.70, 14.98, 25.72])
frozen_std  = np.array([1.7, 0.21, 0.22, 1.6])
e2e_mean    = np.array([14.23, 14.32, 14.35, 15.06])
e2e_std     = np.array([0.4, 0.6, 0.9, 1.2])

colors = [C["blue"], C["teal"], C["orange"], C["red"]]
bg_colors = [C["blue_bg"], C["teal_bg"], C["orange_bg"], C["red_bg"]]

y = np.arange(len(labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 1.8),
                                gridspec_kw={"wspace": 0.12})

# Panel (a): Frozen probe RMSE
for i in range(len(labels)):
    ax1.barh(y[i], frozen_mean[i], xerr=frozen_std[i],
             height=0.6, color=bg_colors[i], edgecolor=colors[i],
             linewidth=0.7, capsize=2, ecolor=C["dark"],
             error_kw={"elinewidth": 0.5, "capthick": 0.5})
    ax1.text(frozen_mean[i] + frozen_std[i] + 0.3, y[i],
             f"{frozen_mean[i]:.1f}", fontsize=6.5, color=C["dark"],
             va="center")

ax1.set_yticks(y)
ax1.set_yticklabels(labels, fontsize=7)
ax1.set_xlabel("Test RMSE (frozen probe) $\\downarrow$")
ax1.set_xlim(12, 30)
ax1.invert_yaxis()
ax1.xaxis.grid(True, linewidth=0.2, color=C["gray_lt"], zorder=0)
ax1.set_axisbelow(True)
ax1.text(-0.02, 1.08, "(a) Frozen", transform=ax1.transAxes,
         fontsize=9, fontweight="bold", va="top")

# Panel (b): E2E RMSE
for i in range(len(labels)):
    ax2.barh(y[i], e2e_mean[i], xerr=e2e_std[i],
             height=0.6, color=bg_colors[i], edgecolor=colors[i],
             linewidth=0.7, capsize=2, ecolor=C["dark"],
             error_kw={"elinewidth": 0.5, "capthick": 0.5})
    ax2.text(e2e_mean[i] + e2e_std[i] + 0.15, y[i],
             f"{e2e_mean[i]:.1f}", fontsize=6.5, color=C["dark"],
             va="center")

ax2.set_yticks(y)
ax2.set_yticklabels([])
ax2.set_xlabel("Test RMSE (E2E fine-tuned) $\\downarrow$")
ax2.set_xlim(12, 18)
ax2.invert_yaxis()
ax2.xaxis.grid(True, linewidth=0.2, color=C["gray_lt"], zorder=0)
ax2.set_axisbelow(True)
ax2.text(-0.02, 1.08, "(b) E2E", transform=ax2.transAxes,
         fontsize=9, fontweight="bold", va="top")

out = "C:/Users/Jonaspetersen/dev/IndustrialJEPA/paper-neurips/figures/fig_ablation_summary.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
