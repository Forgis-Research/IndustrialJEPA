#!/usr/bin/env python3
"""
FIGURE: Evaluation framework for time series event prediction.
Two-panel: (a) p(t,k) probability surface as heatmap, (b) AUROC(k) curve with iAUC shading.

FIGURE DESIGN
Message: "Our evaluation framework captures the full 2D difficulty landscape of event prediction,
         unlike existing metrics that evaluate at single points."
Type: Two-panel -- heatmap + line plot
Layout: (a) left = heatmap of AUROC(t,k), (b) right = AUROC vs k curve
Colors: red-yellow-green heatmap; blue for FAM, gray for baseline
Size: NeurIPS single column (5.5in wide)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

# --- Project palette ---
C = {
    "blue":     "#2B6CB0",
    "orange":   "#FF5A00",
    "teal":     "#2D8A6E",
    "red":      "#DC4B07",
    "gray":     "#878F92",
    "gray_lt":  "#B0B7BA",
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
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "figure.dpi": 150,
    "figure.constrained_layout.use": False,  # manual layout for precise control
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.format": "pdf",
})

# == Synthetic data ===========================================================

# Panel (a): AUROC surface p(t,k)
# t = observation time (columns), k = prediction horizon (rows)
n_t, n_k = 20, 16
t_vals = np.linspace(0, 1, n_t)   # 0 = little history, 1 = full
k_vals = np.arange(1, n_k + 1)    # 1..16

# Natural difficulty: high AUROC at bottom-right (large t, small k),
# low AUROC at top-left (small t, large k)
T, K = np.meshgrid(t_vals, k_vals)
# Combined difficulty score: more history helps, longer horizon hurts
difficulty = 1.5 * T - 0.08 * K  # positive = easier
auroc_surface = 0.60 + 0.35 / (1.0 + np.exp(-3.5 * difficulty))
# Add slight noise for realism
rng = np.random.RandomState(42)
auroc_surface += rng.normal(0, 0.006, auroc_surface.shape)
auroc_surface = np.clip(auroc_surface, 0.52, 0.97)

# Panel (b): AUROC(k) curves (averaged over t)
k_plot = np.arange(1, 17)
# FAM: graceful degradation from 0.95 to ~0.82
fam_auroc = 0.95 - 0.13 * (1 - np.exp(-0.15 * (k_plot - 1)))
fam_auroc += 0.005 * np.array([0, -0.3, -0.1, 0.2, -0.2, 0.1, -0.1, 0.0,
                                0.15, -0.1, 0.05, -0.15, 0.1, -0.05, 0.0, 0.05])
# Baseline: steep drop from 0.90 to ~0.62
base_auroc = 0.90 - 0.30 * (1 - np.exp(-0.20 * (k_plot - 1)))
base_auroc += 0.005 * np.array([0, 0.2, -0.1, -0.3, 0.1, -0.2, 0.15, -0.1,
                                 0.0, 0.1, -0.15, 0.05, -0.1, 0.2, -0.05, 0.0])

# == Figure ====================================================================

fig = plt.figure(figsize=(5.5, 2.2))

# Manual subplot positioning: [left, bottom, width, height]
ax_heat = fig.add_axes([0.07, 0.17, 0.36, 0.74])
ax_cbar = fig.add_axes([0.44, 0.17, 0.012, 0.74])
ax_line = fig.add_axes([0.60, 0.17, 0.37, 0.74])

# == Panel (a): Heatmap ========================================================

# Custom colormap: red (hard) -> yellow -> green (easy)
cmap = LinearSegmentedColormap.from_list(
    "auroc_cmap",
    ["#C44E52", "#E8A838", "#5BAA5B"],
    N=256,
)

im = ax_heat.imshow(
    auroc_surface,
    aspect="auto",
    origin="lower",
    cmap=cmap,
    vmin=0.55, vmax=0.95,
    extent=[-0.5, n_t - 0.5, 0.5, n_k + 0.5],
    interpolation="bilinear",
)

# Axes
ax_heat.set_xlabel("Observation time $t$", fontsize=9)
ax_heat.set_ylabel("Prediction horizon $k$", fontsize=9)
ax_heat.set_xticks([0, n_t - 1])
ax_heat.set_xticklabels(["early", "full"])
ax_heat.set_yticks([1, 4, 8, 12, 16])

# Restore all spines for the heatmap
for spine in ax_heat.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.5)

# Colorbar
cbar = fig.colorbar(im, cax=ax_cbar)
cbar.set_label("AUROC", fontsize=8, labelpad=4)
cbar.ax.tick_params(labelsize=7)

# --- Annotation 1: Prior work at bottom-right (star) ---
x_star = n_t - 2.5
y_star = 1.5
ax_heat.plot(x_star, y_star, marker="*", color=C["dark"], markersize=10,
             markeredgewidth=0.3, zorder=5)
ax_heat.annotate(
    "Prior work:\n$\\mathcal{A}(t{=}\\mathrm{full},\\, k{=}1)$",
    xy=(x_star, y_star),
    xytext=(n_t - 5, 4.5),
    fontsize=6.5, color=C["dark"], ha="center",
    arrowprops=dict(arrowstyle="->,head_width=0.15,head_length=0.1",
                    color=C["dark"], lw=0.6),
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C["dark"], lw=0.5, alpha=0.95),
    zorder=6,
)

# --- Annotation 2: PA-F1 at bottom-left (X) ---
x_pa, y_pa = 1.5, 1.5
ax_heat.plot(x_pa, y_pa, marker="X", color=C["red"], markersize=7, zorder=5)
ax_heat.annotate(
    "PA-F1:\n$k{=}0$, inflated",
    xy=(x_pa, y_pa),
    xytext=(3.5, 6.0),
    fontsize=6.5, color=C["red"], ha="center",
    arrowprops=dict(arrowstyle="->,head_width=0.15,head_length=0.1",
                    color=C["red"], lw=0.6),
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C["red"], lw=0.5, alpha=0.95),
    zorder=6,
)

# --- Annotation 3: iAUC dashed border + label inside heatmap ---
rect = mpatches.FancyBboxPatch(
    (0.2, 1.0), n_t - 1.4, n_k - 0.8,
    boxstyle="round,pad=0.15",
    linewidth=0.9, edgecolor=C["blue"], facecolor="none",
    linestyle=(0, (4, 3)), zorder=4,
)
ax_heat.add_patch(rect)
ax_heat.text(
    (n_t - 1) / 2, 15.0,
    "iAUC = mean over surface",
    fontsize=6.5, color=C["blue"], ha="center", va="top",
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9),
    zorder=7,
)

# Panel label
ax_heat.text(
    -0.15, 1.10, "(a)", transform=ax_heat.transAxes,
    fontsize=10, fontweight="bold", va="top",
)

# == Panel (b): AUROC(k) curve ================================================

# Shaded area under FAM curve only (iAUC visualization)
ax_line.fill_between(
    k_plot, 0.5, fam_auroc,
    color=C["blue"], alpha=0.08, zorder=1,
)

# FAM curve
ax_line.plot(
    k_plot, fam_auroc,
    color=C["blue"], linestyle="-", marker="o", markersize=3.5,
    label="FAM (ours)", zorder=3, linewidth=1.2,
)
# Baseline curve
ax_line.plot(
    k_plot, base_auroc,
    color=C["gray"], linestyle="--", marker="s", markersize=3,
    label="Baseline", zorder=3, linewidth=1.0,
)

# iAUC label inside the shaded area
ax_line.text(
    10, 0.68, "iAUC",
    fontsize=8, color=C["blue"], ha="center", va="center",
    fontstyle="italic", fontweight="bold",
)

# Axes
ax_line.set_xlabel("Prediction horizon $k$", fontsize=9)
ax_line.set_ylabel("AUROC", fontsize=9)
ax_line.set_xlim(0.5, 16.5)
ax_line.set_ylim(0.48, 1.01)
ax_line.set_xticks([1, 4, 8, 12, 16])
ax_line.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# Light horizontal gridlines
ax_line.yaxis.grid(True, linewidth=0.3, color="#D0D0D0", zorder=0)
ax_line.set_axisbelow(True)

# Chance line
ax_line.axhline(0.5, color=C["gray_lt"], linewidth=0.5, linestyle=":", zorder=1)
ax_line.text(15.5, 0.505, "chance", fontsize=6, color=C["gray"], ha="right", va="bottom")

# Legend
ax_line.legend(loc="upper right", fontsize=7)

# Panel label
ax_line.text(
    -0.13, 1.10, "(b)", transform=ax_line.transAxes,
    fontsize=10, fontweight="bold", va="top",
)

# == Save ======================================================================

out = Path(__file__).parent / "fig_evaluation_framework.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved -> {out}")
