#!/usr/bin/env python3
"""
FIGURE DESIGN
Message: "Causal V2 wins cross-machine transfer on every C-MAPSS target subset,
         beating bidirectional variants by +6 to +11 RMSE."
Type: Grouped bar chart with error bars
Layout: 3 groups (FD002, FD003, FD004), 3 bars each (V2, V16a, V16b)
Colors: Forgis palette - blue=V2(ours), teal=V16a, orange=V16b
Size: NeurIPS single column (5.5in wide, ~2.2in tall)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C = {
    "blue":     "#2B6CB0",
    "blue_bg":  "#E8F0FA",
    "teal":     "#2D8A6E",
    "teal_bg":  "#E6F5F0",
    "orange":   "#FF5A00",
    "orange_bg":"#FFF0E6",
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

# Data from paper
groups = ["FD002", "FD003", "FD004"]
v2_mean   = np.array([27.68, 31.45, 38.32])
v2_std    = np.array([1.00, 3.26, 1.00])
v16a_mean = np.array([32.62, 40.02, 43.60])
v16a_std  = np.array([1.54, 1.22, 1.46])
v16b_mean = np.array([38.04, 37.76, 49.66])
v16b_std  = np.array([2.01, 2.36, 1.70])

x = np.arange(len(groups))
width = 0.22
gap = 0.02

fig, ax = plt.subplots(figsize=(5.5, 2.2))

# Bars with hatching for secondary channel
bars_v2 = ax.bar(x - width - gap, v2_mean, width, yerr=v2_std,
                 color=C["blue_bg"], edgecolor=C["blue"], linewidth=0.7,
                 capsize=2, ecolor=C["dark"], error_kw={"elinewidth": 0.5, "capthick": 0.5},
                 label="V2 (causal) -- ours", zorder=3)

bars_v16a = ax.bar(x, v16a_mean, width, yerr=v16a_std,
                   color=C["teal_bg"], edgecolor=C["teal"], linewidth=0.7,
                   capsize=2, ecolor=C["dark"], error_kw={"elinewidth": 0.5, "capthick": 0.5},
                   hatch="//", label="V16a (bidirectional)", zorder=3)

bars_v16b = ax.bar(x + width + gap, v16b_mean, width, yerr=v16b_std,
                   color=C["orange_bg"], edgecolor=C["orange"], linewidth=0.7,
                   capsize=2, ecolor=C["dark"], error_kw={"elinewidth": 0.5, "capthick": 0.5},
                   hatch="\\\\", label="V16b (bidi + VICReg)", zorder=3)

# Value labels above bars
for bars, means, stds in [(bars_v2, v2_mean, v2_std),
                           (bars_v16a, v16a_mean, v16a_std),
                           (bars_v16b, v16b_mean, v16b_std)]:
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.6,
                f"{m:.1f}", ha="center", va="bottom", fontsize=6.5,
                color=C["dark"])

# Delta annotations between V2 and worst bidirectional
for i in range(3):
    worst_bidi = max(v16a_mean[i], v16b_mean[i])
    delta = worst_bidi - v2_mean[i]
    # Place delta inside the group, centered above V2 bar
    xpos = x[i]  # center of group
    ymid = v2_mean[i] - 2.0
    ax.text(xpos, ymid, f"$\\Delta$+{delta:.0f}", fontsize=6.5, color=C["dark"],
            va="center", ha="center", fontweight="bold")

# Truncate y-axis to focus on differences
ax.set_ylim(22, 55)
ax.set_yticks([25, 30, 35, 40, 45, 50])
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_xlabel("Target subset (FD001-pretrained encoder, frozen probe)")
ax.set_ylabel("Test RMSE (cycles) $\\downarrow$")

ax.legend(loc="upper left", fontsize=7, handlelength=1.8,
          borderpad=0.3, labelspacing=0.3)

# Light horizontal gridlines
ax.yaxis.grid(True, linewidth=0.2, color=C["gray_lt"], zorder=0)
ax.set_axisbelow(True)

out = "C:/Users/Jonaspetersen/dev/IndustrialJEPA/paper-neurips/figures/fig_cross_machine.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
