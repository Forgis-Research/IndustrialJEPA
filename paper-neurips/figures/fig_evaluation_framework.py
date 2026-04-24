#!/usr/bin/env python3
"""
FIGURE: Evaluation framework for time series event prediction.
Two-panel: (a) p(t,Δt) probability surface as heatmap, (b) AUPRC(Δt) per-horizon curve.

FIGURE DESIGN
Message: "We store a probability surface p(t, Δt). Every existing metric is a lossy
         projection of this surface. AUPRC pools over the entire surface."
Type: Two-panel -- heatmap + line plot
Layout: (a) left = heatmap of p(t,Δt), (b) right = AUPRC vs Δt curve
Colors: viridis heatmap; blue for FAM, gray for baseline
Size: NeurIPS single column (5.5in wide)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

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

# Panel (a): Probability surface p(t, Δt)
# t = observation time (columns), Δt = prediction horizon (rows)
# Use log-spaced Δt values: 1, 5, 10, 20, 50, 100, 150
dt_vals = np.array([1, 2, 5, 10, 20, 50, 100, 150])
n_dt = len(dt_vals)
n_t = 20
t_vals = np.linspace(0, 1, n_t)  # 0 = early, 1 = full observation

# Build probability surface: high p at bottom-right (large t, small Δt),
# low p at top-left (small t, large Δt)
T, DT = np.meshgrid(t_vals, dt_vals)
# Normalize Δt to [0,1] for smooth gradient (log scale)
dt_norm = np.log(DT) / np.log(150)
difficulty = 2.0 * T - 1.8 * dt_norm
prob_surface = 0.85 / (1.0 + np.exp(-3.0 * difficulty))
# Add slight noise for realism
rng = np.random.RandomState(42)
prob_surface += rng.normal(0, 0.01, prob_surface.shape)
prob_surface = np.clip(prob_surface, 0.02, 0.92)

# Panel (b): AUPRC(Δt) curves
dt_plot = np.array([1, 2, 5, 10, 20, 50, 100, 150])
dt_norm_plot = np.log(dt_plot) / np.log(150)

# FAM: graceful degradation from ~0.82 to ~0.35
fam_auprc = 0.82 - 0.50 * (dt_norm_plot ** 1.3)
fam_auprc += 0.01 * np.array([0, -0.3, 0.2, -0.1, 0.15, -0.2, 0.1, -0.05])
fam_auprc = np.clip(fam_auprc, 0.10, 0.90)

# Baseline: steeper drop from ~0.70 to ~0.10
base_auprc = 0.70 - 0.65 * (dt_norm_plot ** 1.0)
base_auprc += 0.01 * np.array([0, 0.2, -0.1, -0.3, 0.1, -0.15, 0.05, 0.0])
base_auprc = np.clip(base_auprc, 0.04, 0.75)

prevalence = 0.05  # chance baseline

# == Figure ====================================================================

fig = plt.figure(figsize=(5.5, 2.4))

# Manual subplot positioning: [left, bottom, width, height]
ax_heat = fig.add_axes([0.07, 0.18, 0.36, 0.72])
ax_cbar = fig.add_axes([0.44, 0.18, 0.012, 0.72])
ax_line = fig.add_axes([0.60, 0.18, 0.37, 0.72])

# == Panel (a): Heatmap of p(t, Δt) ===========================================

im = ax_heat.imshow(
    prob_surface,
    aspect="auto",
    origin="lower",
    cmap="viridis",
    vmin=0.0, vmax=0.90,
    extent=[-0.5, n_t - 0.5, -0.5, n_dt - 0.5],
    interpolation="bilinear",
)

# Axes
ax_heat.set_xlabel("Observation time $t$", fontsize=9)
ax_heat.set_ylabel(r"Prediction horizon $\Delta t$", fontsize=9)
ax_heat.set_xticks([0, n_t - 1])
ax_heat.set_xticklabels(["early", "full"])
ax_heat.set_yticks(range(n_dt))
ax_heat.set_yticklabels([str(v) for v in dt_vals], fontsize=7)

# Restore all spines for the heatmap
for spine in ax_heat.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.5)

# Colorbar
cbar = fig.colorbar(im, cax=ax_cbar)
cbar.set_label(r"$p(t, \Delta t)$", fontsize=8, labelpad=4)
cbar.ax.tick_params(labelsize=7)

# --- Annotation: RMSE → rightmost column (full t, all Δt) ---
# Small label inside heatmap near right edge
ax_heat.text(
    n_t - 1.3, n_dt * 0.55,
    "RMSE",
    fontsize=6.5, color="white", ha="center", va="center",
    fontweight="bold", rotation=90,
    bbox=dict(boxstyle="round,pad=0.12", fc=C["red"], ec="none", alpha=0.85),
    zorder=7,
)
# Vertical line marking the rightmost column
ax_heat.axvline(n_t - 1, color=C["red"], linewidth=0.8, linestyle="-", alpha=0.6, zorder=4)

# --- Annotation: PA-F1 → bottom row (Δt≈0) ---
# Label inside heatmap along bottom
ax_heat.text(
    n_t * 0.25, 0.3,
    r"PA-F1 ($\Delta t{=}0$)",
    fontsize=6.5, color="white", ha="center", va="center",
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.12", fc=C["orange"], ec="none", alpha=0.85),
    zorder=7,
)
# Horizontal line along bottom row
ax_heat.axhline(0, color=C["orange"], linewidth=0.8, linestyle="-", alpha=0.6,
                xmin=0.0, xmax=1.0, zorder=4)

# --- Annotation: F1(Δt=1) → single cell at (full, Δt=1) ---
x_f1, y_f1 = n_t - 2, 0  # bottom-right corner
ax_heat.plot(x_f1, y_f1, marker="*", color="white", markersize=9,
             markeredgewidth=0.4, markeredgecolor=C["dark"], zorder=5)
ax_heat.annotate(
    r"$\mathrm{F1}(\Delta t{=}1)$",
    xy=(x_f1, y_f1),
    xytext=(n_t * 0.45, 2.2),
    fontsize=6.5, color=C["dark"], ha="center", va="bottom",
    arrowprops=dict(arrowstyle="->,head_width=0.12,head_length=0.08",
                    color=C["dark"], lw=0.6),
    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=C["dark"], lw=0.4, alpha=0.9),
    zorder=6,
)

# --- Annotation: AUPRC pools entire surface (dashed border) ---
rect = mpatches.FancyBboxPatch(
    (-0.3, -0.3), n_t - 0.4, n_dt - 0.4,
    boxstyle="round,pad=0.15",
    linewidth=1.0, edgecolor=C["blue"], facecolor="none",
    linestyle=(0, (4, 3)), zorder=4,
)
ax_heat.add_patch(rect)
ax_heat.text(
    (n_t - 1) / 2, n_dt - 0.3,
    "AUPRC: pools entire surface",
    fontsize=6.5, color=C["blue"], ha="center", va="bottom",
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9),
    zorder=7,
)

# Panel label
ax_heat.text(
    -0.15, 1.12, "(a)", transform=ax_heat.transAxes,
    fontsize=10, fontweight="bold", va="top",
)

# == Panel (b): AUPRC(Δt) curve ===============================================

# Shaded area under FAM curve (pooled AUPRC visualization)
ax_line.fill_between(
    dt_plot, prevalence, fam_auprc,
    color=C["blue"], alpha=0.10, zorder=1,
)

# FAM curve
ax_line.plot(
    dt_plot, fam_auprc,
    color=C["blue"], linestyle="-", marker="o", markersize=3.5,
    label="FAM (ours)", zorder=3, linewidth=1.2,
)
# Baseline curve
ax_line.plot(
    dt_plot, base_auprc,
    color=C["gray"], linestyle="--", marker="s", markersize=3,
    label="Baseline", zorder=3, linewidth=1.0,
)

# Pooled AUPRC label inside shaded area
ax_line.text(
    25, 0.30, "pooled\nAUPRC",
    fontsize=7, color=C["blue"], ha="center", va="center",
    fontstyle="italic", fontweight="bold",
)

# Axes (log scale for Δt)
ax_line.set_xscale("log")
ax_line.set_xlabel(r"Prediction horizon $\Delta t$", fontsize=9)
ax_line.set_ylabel("AUPRC", fontsize=9)
ax_line.set_xlim(0.8, 200)
ax_line.set_ylim(-0.02, 1.0)
ax_line.set_xticks(dt_plot)
ax_line.set_xticklabels([str(v) for v in dt_plot], fontsize=7)
ax_line.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax_line.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# Light horizontal gridlines
ax_line.yaxis.grid(True, linewidth=0.3, color="#D0D0D0", zorder=0)
ax_line.set_axisbelow(True)

# Chance / prevalence line
ax_line.axhline(prevalence, color=C["gray_lt"], linewidth=0.6, linestyle=":", zorder=1)
ax_line.text(180, prevalence + 0.015, "prevalence", fontsize=6, color=C["gray"],
             ha="right", va="bottom")

# Legend
ax_line.legend(loc="upper right", fontsize=7)

# Panel label
ax_line.text(
    -0.13, 1.12, "(b)", transform=ax_line.transAxes,
    fontsize=10, fontweight="bold", va="top",
)

# == Save ======================================================================

out_dir = Path(__file__).parent
out_pdf = out_dir / "fig_evaluation_framework.pdf"
out_png = out_dir / "fig_evaluation_framework.png"
fig.savefig(out_pdf)
fig.savefig(out_png, dpi=300, format="png")
plt.close(fig)
print(f"Saved -> {out_pdf}")
print(f"Saved -> {out_png}")
