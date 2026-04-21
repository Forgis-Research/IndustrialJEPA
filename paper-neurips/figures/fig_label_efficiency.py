#!/usr/bin/env python3
"""
FIGURE: label efficiency on C-MAPSS FD001 (v18 honest protocol).
Two-panel: (a) FAM vs STAR across label budgets, (b) From-scratch ablation.

v18 numbers from Phase 1b (100/20/10/5%) + Phase 1c (50%). Honest probe
protocol (AdamW WD=1e-2, val n_cuts_per_engine=10). 5 probe seeds per cell.

STAR numbers are the v11-era in-repo replication (mean +/- std, 5 seeds).
From-scratch numbers are v11-era (only 100/20/10/5%; 50% interpolated on
visual for figure continuity - the original data was not re-run).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

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
    "figure.constrained_layout.use": True,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.format": "pdf",
})

budgets = [100, 50, 20, 10, 5]
budget_labels = ["100%", "50%", "20%", "10%", "5%"]
x = np.arange(len(budgets))

# --- STAR (supervised, v11 replication, 5 seeds) ---
star_mean = np.array([12.19, 13.26, 17.74, 18.72, 24.55])
star_std  = np.array([0.6, 0.7, 3.6, 2.8, 6.4])

# --- FAM E2E (v18 honest, 5 seeds) ---
# Phase 1b (100/20/10/5) + Phase 1c (50):
fam_e2e_mean = np.array([15.08, 15.85, 17.85, 19.62, 21.55])
fam_e2e_std  = np.array([0.10, 0.55, 0.63, 1.36, 1.52])

# --- FAM Frozen (v18 honest, 5 seeds) ---
fam_frz_mean = np.array([17.01, 17.58, 19.53, 20.71, 21.47])
fam_frz_std  = np.array([1.21, 0.47, 0.69, 0.87, 0.87])

# --- From-scratch E2E (v11-era, for panel b only; deltas are the story) ---
# Values at 100, 20, 10, 5 only; 50% not rerun.
scratch_x = np.array([0, 2, 3, 4])  # indices into x for 100, 20, 10, 5
scratch_m = np.array([22.99, 28.87, 35.59, 37.59])
scratch_s = np.array([2.3, 1.5, 2.7, 2.0])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.4),
                                gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.05})

# ========== Panel (a): FAM vs STAR ==========
ax1.fill_between(x, star_mean - star_std, star_mean + star_std,
                 color=C["red"], alpha=0.08, zorder=1)
ax1.fill_between(x, fam_e2e_mean - fam_e2e_std, fam_e2e_mean + fam_e2e_std,
                 color=C["blue"], alpha=0.10, zorder=1)
ax1.fill_between(x, fam_frz_mean - fam_frz_std, fam_frz_mean + fam_frz_std,
                 color=C["teal"], alpha=0.08, zorder=1)

l_star, = ax1.plot(x, star_mean, color=C["red"], linestyle="--", marker="s",
                   markersize=4, linewidth=1.0, zorder=3, label="STAR (supervised)")
l_e2e, = ax1.plot(x, fam_e2e_mean, color=C["blue"], linestyle="-", marker="o",
                  markersize=4.5, linewidth=1.2, zorder=4, label="FAM E2E (ours)")
l_frz, = ax1.plot(x, fam_frz_mean, color=C["teal"], linestyle="-.", marker="D",
                  markersize=3.5, linewidth=1.0, zorder=3, label="FAM frozen (ours)")

# Variance-reduction annotation at 5% (sigma 0.9 vs 6.4, 7x lower)
ax1.annotate("$7\\times$ lower\nseed variance\nat 5\\%",
             xy=(4, fam_frz_mean[-1]), xytext=(3.0, 11),
             fontsize=5.8, color=C["dark"],
             ha="left", va="center", style="italic",
             arrowprops=dict(arrowstyle="->", lw=0.4, color=C["dark"],
                              connectionstyle="arc3,rad=-0.15"))

ax1.set_xticks(x)
ax1.set_xticklabels(budget_labels)
ax1.set_xlabel("Label budget")
ax1.set_ylabel("Test RMSE (cycles) $\\downarrow$")
ax1.set_ylim(9, 42)
ax1.set_xlim(-0.55, 4.8)

ax1.legend(handles=[l_star, l_e2e, l_frz],
           loc="upper left", fontsize=6.5, handlelength=2.0,
           borderpad=0.3, labelspacing=0.35)

ax1.text(-0.12, 1.07, "(a)", transform=ax1.transAxes,
         fontsize=10, fontweight="bold", va="top")

# ========== Panel (b): From-scratch ablation ==========
e2e_x_fs = scratch_x
e2e_m_fs = fam_e2e_mean[e2e_x_fs]
e2e_s_fs = fam_e2e_std[e2e_x_fs]

ax2.fill_between(scratch_x, e2e_m_fs, scratch_m,
                 color=C["gray_lt"], alpha=0.22, zorder=1)

l_pre, = ax2.plot(e2e_x_fs, e2e_m_fs, color=C["blue"], linestyle="-", marker="o",
                  markersize=4.5, linewidth=1.2, zorder=4, label="Pretrained E2E")
ax2.errorbar(e2e_x_fs, e2e_m_fs, yerr=e2e_s_fs, color=C["blue"],
             fmt="none", capsize=2, elinewidth=0.6, capthick=0.5, zorder=3)

l_scr, = ax2.plot(scratch_x, scratch_m, color=C["orange"], linestyle="-", marker="s",
                  markersize=4, linewidth=1.0, zorder=4, label="From scratch (random init)")
ax2.errorbar(scratch_x, scratch_m, yerr=scratch_s, color=C["orange"],
             fmt="none", capsize=2, elinewidth=0.6, capthick=0.5, zorder=3)

p_contrib = mpatches.Patch(color=C["gray_lt"], alpha=0.35, label="Pretraining contribution")

deltas = scratch_m - e2e_m_fs
for i, (xi, di) in enumerate(zip(scratch_x, deltas)):
    ymid = (e2e_m_fs[i] + scratch_m[i]) / 2
    ax2.text(xi + 0.15, ymid, f"+{di:.1f}", fontsize=6.5, color=C["dark"],
             va="center", fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels(budget_labels)
ax2.set_xlabel("Label budget")
ax2.set_ylim(9, 42)
ax2.set_xlim(-0.3, 4.6)
ax2.tick_params(axis="y", labelleft=False)

ax2.legend(handles=[l_pre, l_scr, p_contrib],
           loc="upper left", fontsize=6.5, handlelength=2.0,
           borderpad=0.3, labelspacing=0.35)

ax2.text(-0.02, 1.07, "(b)", transform=ax2.transAxes,
         fontsize=10, fontweight="bold", va="top")

out = Path(__file__).resolve().parent / "fig_label_efficiency.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
