#!/usr/bin/env python3
"""
FIGURE DESIGN
Message: "SSL pretraining yields a clear label-efficiency advantage: at <=20% labels JEPA
         matches or beats supervised STAR, and at 5% labels the frozen encoder wins outright.
         The from-scratch ablation isolates the growing pretraining contribution."
Type: Two-panel line plot with error ribbons
Layout: (a) Label efficiency comparison, (b) From-scratch ablation with shaded gap
Colors: Forgis-derived palette consistent with rest of paper
Size: NeurIPS single column (5.5in wide, ~2.4in tall)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np

# --- Forgis-derived palette ---
C = {
    "blue":     "#2B6CB0",   # csecondary - our method E2E
    "orange":   "#FF5A00",   # cprimary/tiger
    "teal":     "#2D8A6E",   # caccent - frozen
    "red":      "#DC4B07",   # chighlight - STAR
    "gray":     "#878F92",   # csteel
    "gray_lt":  "#B0B7BA",
    "dark":     "#122128",   # cgunmetal
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

# --- Data from paper Table 3 ---
budgets = [100, 50, 20, 10, 5]
budget_labels = ["100%", "50%", "20%", "10%", "5%"]
x = np.arange(len(budgets))

# STAR (supervised)
star_mean = np.array([12.19, 13.26, 17.74, 18.72, 24.55])
star_std  = np.array([0.6, 0.7, 3.6, 2.8, 6.4])

# JEPA E2E
jepa_e2e_mean = np.array([14.23, 14.93, 16.54, 18.66, 25.33])
jepa_e2e_std  = np.array([0.4, 0.4, 0.8, 0.8, 5.1])

# JEPA Frozen
jepa_frz_mean = np.array([17.81, 18.71, 19.83, 19.93, 21.53])
jepa_frz_std  = np.array([1.7, 1.1, 0.3, 0.9, 2.0])

# From-scratch E2E
scratch_x = np.array([0, 2, 3, 4])
scratch_m = np.array([22.99, 32.50, 35.59, 37.59])
scratch_s = np.array([2.3, 1.5, 2.7, 2.0])

# --- Create figure ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.4),
                                gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.05})

# ================================================================
# Panel (a): Label efficiency comparison
# ================================================================

# Plot ribbons first (no label)
ax1.fill_between(x, star_mean - star_std, star_mean + star_std,
                 color=C["red"], alpha=0.08, zorder=1)
ax1.fill_between(x, jepa_e2e_mean - jepa_e2e_std, jepa_e2e_mean + jepa_e2e_std,
                 color=C["blue"], alpha=0.10, zorder=1)
ax1.fill_between(x, jepa_frz_mean - jepa_frz_std, jepa_frz_mean + jepa_frz_std,
                 color=C["teal"], alpha=0.08, zorder=1)

# Plot lines with labels for legend
l_star, = ax1.plot(x, star_mean, color=C["red"], linestyle="--", marker="s",
                   markersize=4, linewidth=1.0, zorder=3, label="STAR (supervised)")
l_e2e, = ax1.plot(x, jepa_e2e_mean, color=C["blue"], linestyle="-", marker="o",
                  markersize=4.5, linewidth=1.2, zorder=4, label="Traj JEPA E2E (ours)")
l_frz, = ax1.plot(x, jepa_frz_mean, color=C["teal"], linestyle="-.", marker="D",
                  markersize=3.5, linewidth=1.0, zorder=3, label="Traj JEPA frozen (ours)")

# Crossover annotation: vertical bracket at 5% showing frozen beats STAR
# Draw a thin vertical line segment
ax1.plot([4.08, 4.08], [21.53+0.4, 24.55-0.4], color=C["dark"], lw=0.5, zorder=5,
         solid_capstyle="butt")
# Horizontal ticks at endpoints
ax1.plot([4.03, 4.13], [21.53+0.4, 21.53+0.4], color=C["dark"], lw=0.5, zorder=5)
ax1.plot([4.03, 4.13], [24.55-0.4, 24.55-0.4], color=C["dark"], lw=0.5, zorder=5)
ax1.text(4.2, 23.0, "frozen\nbeats\nSTAR", fontsize=5.5, color=C["dark"],
         ha="left", va="center", linespacing=0.95, style="italic")

ax1.set_xticks(x)
ax1.set_xticklabels(budget_labels)
ax1.set_xlabel("Label budget")
ax1.set_ylabel("Test RMSE (cycles) $\\downarrow$")
ax1.set_ylim(9, 42)
ax1.set_xlim(-0.55, 4.8)

# Legend (use explicit handles to avoid fill_between entries)
ax1.legend(handles=[l_star, l_e2e, l_frz],
           loc="upper left", fontsize=6.5, handlelength=2.0,
           borderpad=0.3, labelspacing=0.35)

ax1.text(-0.12, 1.07, "(a)", transform=ax1.transAxes,
         fontsize=10, fontweight="bold", va="top")

# ================================================================
# Panel (b): From-scratch ablation
# ================================================================
e2e_x_fs = np.array([0, 2, 3, 4])
e2e_m_fs = jepa_e2e_mean[e2e_x_fs]
e2e_s_fs = jepa_e2e_std[e2e_x_fs]

# Shaded pretraining contribution (draw first so it's behind)
ax2.fill_between(scratch_x, e2e_m_fs, scratch_m,
                 color=C["gray_lt"], alpha=0.22, zorder=1)

# Lines
l_pre, = ax2.plot(e2e_x_fs, e2e_m_fs, color=C["blue"], linestyle="-", marker="o",
                  markersize=4.5, linewidth=1.2, zorder=4, label="Pretrained E2E")
ax2.errorbar(e2e_x_fs, e2e_m_fs, yerr=e2e_s_fs, color=C["blue"],
             fmt="none", capsize=2, elinewidth=0.6, capthick=0.5, zorder=3)

l_scr, = ax2.plot(scratch_x, scratch_m, color=C["orange"], linestyle="-", marker="s",
                  markersize=4, linewidth=1.0, zorder=4, label="From scratch (random init)")
ax2.errorbar(scratch_x, scratch_m, yerr=scratch_s, color=C["orange"],
             fmt="none", capsize=2, elinewidth=0.6, capthick=0.5, zorder=3)

# Patch for legend
p_contrib = mpatches.Patch(color=C["gray_lt"], alpha=0.35, label="Pretraining contribution")

# Delta annotations inside the shaded region
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

# Keep y-tick marks but no labels (shared axis)
ax2.tick_params(axis="y", labelleft=False)

# Legend
ax2.legend(handles=[l_pre, l_scr, p_contrib],
           loc="upper left", fontsize=6.5, handlelength=2.0,
           borderpad=0.3, labelspacing=0.35)

ax2.text(-0.02, 1.07, "(b)", transform=ax2.transAxes,
         fontsize=10, fontweight="bold", va="top")

# Save
out = "C:/Users/Jonaspetersen/dev/IndustrialJEPA/paper-neurips/figures/fig_label_efficiency.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
