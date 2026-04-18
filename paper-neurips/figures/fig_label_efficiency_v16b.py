#!/usr/bin/env python3
"""
FIGURE DESIGN
Message: "Causal V2 degrades gracefully under label scarcity while bidirectional V16b
         collapses catastrophically, proving the causal inductive bias is essential."
Type: Line plot with error ribbons, two series
Colors: blue=V2(ours), orange=V16b
Size: NeurIPS single column (5.5in wide, ~2.2in tall)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C = {
    "blue":     "#2B6CB0",
    "orange":   "#FF5A00",
    "dark":     "#122128",
    "gray":     "#878F92",
    "gray_lt":  "#D1D5D8",
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

# Data
budget_labels = ["100%", "50%", "20%", "10%", "5%"]
x = np.arange(len(budget_labels))

v2_mean  = np.array([14.23, 14.93, 16.54, 18.66, 25.33])
v2_std   = np.array([0.4, 0.4, 0.8, 0.8, 5.1])
v16b_mean = np.array([15.36, 22.43, 27.01, 29.27, 40.65])
v16b_std  = np.array([1.07, 6.28, 5.28, 1.06, 10.18])

fig, ax = plt.subplots(figsize=(5.5, 2.2))

# Ribbons
ax.fill_between(x, v2_mean - v2_std, v2_mean + v2_std,
                color=C["blue"], alpha=0.12, zorder=1)
ax.fill_between(x, v16b_mean - v16b_std, v16b_mean + v16b_std,
                color=C["orange"], alpha=0.10, zorder=1)

# Lines
ax.plot(x, v2_mean, color=C["blue"], linestyle="-", marker="o",
        markersize=4.5, linewidth=1.2, zorder=4, label="V2 (causal) -- ours")
ax.plot(x, v16b_mean, color=C["orange"], linestyle="--", marker="s",
        markersize=4, linewidth=1.0, zorder=3, label="V16b (bidi + VICReg)")

# Direct labels
for i, (vm, vs) in enumerate(zip(v2_mean, v2_std)):
    ax.text(i, vm - max(vs, 1.5) - 0.8, f"{vm:.1f}", fontsize=6, color=C["blue"],
            ha="center", va="top")

for i, (vm, vs) in enumerate(zip(v16b_mean, v16b_std)):
    ypos = vm + max(vs, 1.5) + 0.6
    if i == 4:  # 5% label -- ribbon is huge, put label higher
        ypos = vm + vs + 1.0
    ax.text(i, ypos, f"{vm:.1f}", fontsize=6, color=C["orange"],
            ha="center", va="bottom")

# Gap annotation at 10%
ymid = (v2_mean[3] + v16b_mean[3]) / 2
ax.text(3.0, ymid, f"+{v16b_mean[3]-v2_mean[3]:.0f}", fontsize=7, color=C["gray"],
        fontweight="bold", va="center", ha="center")

# Gap annotation at 5%
ymid5 = (v2_mean[4] + v16b_mean[4]) / 2
ax.text(3.85, ymid5, f"+{v16b_mean[4]-v2_mean[4]:.0f}", fontsize=7, color=C["gray"],
        fontweight="bold", va="center", ha="center")

ax.set_xticks(x)
ax.set_xticklabels(budget_labels)
ax.set_xlabel("Label budget (fraction of FD001 training engines)")
ax.set_ylabel("Test RMSE (cycles) $\\downarrow$")
ax.set_ylim(8, 55)
ax.set_xlim(-0.3, 4.5)

ax.legend(loc="upper left", fontsize=7, handlelength=2.0)

# Light gridlines
ax.yaxis.grid(True, linewidth=0.2, color=C["gray_lt"], zorder=0)
ax.set_axisbelow(True)

out = "C:/Users/Jonaspetersen/dev/IndustrialJEPA/paper-neurips/figures/fig_label_efficiency_v16b.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
