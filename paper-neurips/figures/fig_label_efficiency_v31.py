#!/usr/bin/env python3
"""
FIGURE: Label efficiency on C-MAPSS FD001 and FD003 (v31 surface-based h-AUROC).

Caption: "Label efficiency on C-MAPSS lifecycle datasets. FD001 retains 92% of
full-label h-AUROC with just 2 training engines (2% of 85). FD003 degrades faster
due to its 4 distinct fault modes requiring more representative engines.
Shaded: +/-1 std over 3 seeds."

v31 data: 3 seeds per cell, h-AUROC from probability surface evaluation.
"""

# FIGURE DESIGN
# Message: "FAM retains 92% of full-label performance with only 2% of labels"
# Type: Line plot with error bands, two series
# Layout: Single panel, log-scale x-axis, 5 tick positions
# Colors: FD001=blue (#0072B2), FD003=orange (#E69F00) -- Okabe-Ito
# Annotations: Arrow at FD001 2% point ("92% retention, 2 engines")
#              90% retention threshold line for FD001
#              Top axis showing engine counts for FD001
# Size: NeurIPS single column (5.5in wide, ~2.5in tall)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

# --- Okabe-Ito colorblind-safe palette ---
C = {
    "blue":     "#0072B2",
    "orange":   "#E69F00",
    "gray":     "#878F92",
    "gray_lt":  "#C0C0C0",
    "dark":     "#122128",
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
    "lines.linewidth": 1.2,
    "lines.markersize": 5,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "legend.handlelength": 2.0,
    "figure.dpi": 150,
    "figure.constrained_layout.use": True,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.format": "pdf",
})

# --- Data (v31, 3 seeds, h-AUROC) ---
budgets_pct = np.array([1, 2, 5, 10, 100])

# FD001 (85 training engines)
fd001_mean = np.array([0.670, 0.724, 0.730, 0.772, 0.786])
fd001_std  = np.array([0.110, 0.013, 0.018, 0.059, 0.033])
fd001_engines = [1, 2, 4, 9, 85]  # actual engine counts

# FD003 (100 training engines)
fd003_mean = np.array([0.513, 0.635, 0.709, 0.830, 0.853])
fd003_std  = np.array([0.220, 0.065, 0.131, 0.018, 0.004])

# 90% retention threshold for FD001
fd001_90pct = 0.786 * 0.90  # = 0.7074

# --- Figure ---
fig, ax = plt.subplots(figsize=(5.5, 2.5))

# Error bands
ax.fill_between(budgets_pct, fd001_mean - fd001_std, fd001_mean + fd001_std,
                color=C["blue"], alpha=0.12, zorder=1, linewidth=0)
ax.fill_between(budgets_pct, fd003_mean - fd003_std, fd003_mean + fd003_std,
                color=C["orange"], alpha=0.12, zorder=1, linewidth=0)

# Lines
ax.plot(budgets_pct, fd001_mean, color=C["blue"], linestyle="-", marker="o",
        markersize=5, linewidth=1.2, zorder=4, label="FD001 (1 fault mode)")
ax.plot(budgets_pct, fd003_mean, color=C["orange"], linestyle="--", marker="s",
        markersize=4.5, linewidth=1.2, zorder=4, label="FD003 (4 fault modes)")

# 90% retention threshold
ax.axhline(fd001_90pct, color=C["gray_lt"], linestyle=":", linewidth=0.7, zorder=2)
ax.text(130, fd001_90pct + 0.006, "90% of FD001",
        fontsize=6.5, color=C["gray"], va="bottom", ha="right")

# Annotation: FD001 at 2%
ax.annotate(
    "92% retention\n(2 engines)",
    xy=(2, 0.724), xytext=(1.4, 0.88),
    fontsize=7.5, color=C["blue"], ha="center", va="bottom",
    arrowprops=dict(arrowstyle="->, head_width=0.12, head_length=0.08",
                    lw=0.6, color=C["blue"],
                    connectionstyle="arc3,rad=-0.2"),
)

# X-axis: log scale with specific ticks
ax.set_xscale("log")
ax.set_xticks(budgets_pct)
ax.set_xticklabels(["1%", "2%", "5%", "10%", "100%"])
ax.xaxis.set_minor_locator(mticker.NullLocator())
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.xaxis.set_major_formatter(mticker.FixedFormatter(["1%", "2%", "5%", "10%", "100%"]))

ax.set_xlabel("Label budget (fraction of training engines)")
ax.set_ylabel("h-AUROC")
ax.set_ylim(0.38, 0.93)
ax.set_xlim(0.8, 130)

# Light horizontal grid
ax.yaxis.grid(True, alpha=0.25, linewidth=0.4, color=C["gray_lt"])
ax.set_axisbelow(True)

# Legend
ax.legend(loc="lower right", borderpad=0.4, labelspacing=0.4)

# Secondary x-axis (top) for engine counts
ax2 = ax.twiny()
ax2.set_xscale("log")
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(budgets_pct)
ax2.set_xticklabels([f"{e}" for e in fd001_engines], fontsize=7, color=C["gray"])
ax2.tick_params(axis="x", which="both", length=0, pad=1)  # no tick marks
ax2.xaxis.set_minor_locator(mticker.NullLocator())  # kill minor ticks
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.spines["bottom"].set_visible(False)
ax2.set_xlabel("FD001 engines", fontsize=7.5, color=C["gray"], labelpad=3)

# --- Save ---
out = Path(__file__).resolve().parent / "fig_label_efficiency_v31.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
