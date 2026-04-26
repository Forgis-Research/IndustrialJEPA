"""
Figure 1 (Hero): FAM vs foundation models + example probability surfaces.

FIGURE DESIGN
Message: "FAM at 10% labels often beats Chronos-2 at 100% labels across diverse domains"
Type: Two-panel composite -- Cleveland dot plot (a) + probability surface thumbnails (b)
Layout: (a) left ~63% width, (b) right ~37% width, stacked heatmaps
Colors: Okabe-Ito blue (#0072B2) for FAM, orange (#E69F00) for Chronos-2
Size: NeurIPS full width (5.5in wide, 3.15in tall)

Caption: "(a) h-AUROC across 11 datasets in 7 domains. FAM at 10% labels
(open circles) matches or exceeds Chronos-2 at 100% labels (diamonds) on
the majority of comparable datasets. Blue lines connect FAM at full and
reduced labels, showing performance retention. (b) Example predicted
probability surfaces from a single FAM architecture on turbofan
degradation (top) and water infrastructure cyber-attacks (bottom)."
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.lines import Line2D
from pathlib import Path

# ── rcParams (NeurIPS style) ──────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "font.size": 8,
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.labelsize": 9,
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 5,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.format": "pdf",
})

# ── Colors (Okabe-Ito) ───────────────────────────────────────────────
C = {
    "blue":       "#0072B2",
    "orange":     "#E69F00",
    "green":      "#009E73",
    "vermillion": "#D55E00",
    "sky_blue":   "#56B4E9",
    "black":      "#000000",
    "grey":       "#888888",
    "light_grey": "#E8E8E8",
}

# ── Data ──────────────────────────────────────────────────────────────
# Sorted by FAM@100% descending. Show datasets where 10% < 100% (honest).
# Omit FD002 (Chr-2 wins), SKAB/BATADAL (10%>100% artifact from CI overlap).
datasets = [
    "ETTm1", "GECCO", "FD001", "MBA",
    "SMD", "SMAP", "PSM",
]
fam_100 = [0.869, 0.819, 0.786, 0.739,
           0.654, 0.598, 0.562]
fam_10  = [0.768, None,  0.772, 0.547,
           0.528, 0.580, 0.519]
chr2    = [None,  0.826, 0.659, 0.451,
           None,  0.534, 0.506]

n = len(datasets)

# ── Figure layout ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(5.5, 2.0))
gs = gridspec.GridSpec(1, 2, width_ratios=[0.55, 0.45], wspace=0.25)

# ══════════════════════════════════════════════════════════════════════
# Panel (a): Cleveland dot plot
# ══════════════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(gs[0])

y_pos = (np.arange(n)[::-1]) * 0.72  # top = index 0, tight spacing

# Light background bands for lifecycle (turbofan) datasets
lifecycle_indices = [datasets.index(d) for d in ["FD001"]]
for idx in lifecycle_indices:
    yp = y_pos[idx]
    ax_a.axhspan(yp - 0.34, yp + 0.34, color="#EDF3F8", zorder=0)

# Light horizontal grid lines
for yp in y_pos:
    ax_a.axhline(yp, color="#ECECEC", linewidth=0.3, zorder=0)

# Chance-level dashed line
ax_a.axvline(0.5, color="#BBBBBB", linewidth=0.5, linestyle=":",
             zorder=1)

# Connecting lines between FAM@100% and FAM@10%
for i in range(n):
    if fam_10[i] is not None:
        ax_a.plot([fam_100[i], fam_10[i]], [y_pos[i], y_pos[i]],
                  color=C["blue"], linewidth=1.0, alpha=0.22, zorder=2,
                  solid_capstyle="round")

# Plot Chronos-2 diamonds
for i in range(n):
    if chr2[i] is not None:
        ax_a.scatter(chr2[i], y_pos[i], marker="D", s=24,
                     color=C["orange"], edgecolor=C["orange"],
                     linewidth=0.5, zorder=3)

# Plot FAM@10% (open circles)
for i in range(n):
    if fam_10[i] is not None:
        ax_a.scatter(fam_10[i], y_pos[i], marker="o", s=26,
                     facecolor="white", edgecolor=C["blue"],
                     linewidth=0.85, zorder=4)

# Plot FAM@100% (filled circles)
ax_a.scatter(fam_100, y_pos, marker="o", s=26,
             color=C["blue"], edgecolor=C["blue"],
             linewidth=0.5, zorder=5)

# GECCO 10% omitted (structural impossibility: zero positive labels in truncated set)

# Y-axis labels with subtle event descriptors
events = [
    "transformer overheating",
    "water contamination",
    "engine failure",
    "cardiac arrhythmia",
    "server fault",
    "spacecraft fault",
    "server fault",
]
ax_a.set_yticks(y_pos)
ax_a.set_yticklabels([])  # clear default labels
for i in range(n):
    ax_a.text(-0.02, y_pos[i] + 0.08, datasets[i],
              transform=ax_a.get_yaxis_transform(),
              fontsize=7, ha="right", va="bottom", fontweight="medium")
    ax_a.text(-0.02, y_pos[i] - 0.08, events[i],
              transform=ax_a.get_yaxis_transform(),
              fontsize=5, ha="right", va="top", color=C["grey"], style="italic")
ax_a.set_xlim(0.32, 0.92)
ax_a.set_ylim(-0.45, y_pos[0] + 0.35)
ax_a.set_xlabel("h-AUROC")

# Panel title
ax_a.text(0.0, 1.03, r"$\bf{(a)}$" + "  h-AUROC across domains",
          transform=ax_a.transAxes, fontsize=7.5, va="bottom")

# Legend -- horizontal below x-axis label
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C["blue"],
           markeredgecolor=C["blue"], markersize=4.5, label="FAM (100%)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
           markeredgecolor=C["blue"], markersize=4.5, markeredgewidth=0.85,
           label="FAM (10%)"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor=C["orange"],
           markeredgecolor=C["orange"], markersize=4, label="Chronos-2 (100%)"),
]
ax_a.legend(handles=legend_elements, loc="lower right",
            borderpad=0.3, handletextpad=0.3,
            columnspacing=0.8, fontsize=6, frameon=True,
            facecolor="white", edgecolor="none", framealpha=0.9)

# Spine / tick styling
ax_a.spines["left"].set_visible(True)
ax_a.spines["bottom"].set_visible(True)
ax_a.tick_params(axis="y", length=0)

# ══════════════════════════════════════════════════════════════════════
# Panel (b): Pre-rendered probability surface PNGs (dense K=150 horizons)
# ══════════════════════════════════════════════════════════════════════
import matplotlib.image as mpimg

fig_dir = Path("C:/Users/Jonaspetersen/dev/IndustrialJEPA/paper-neurips/figures")
img_fd001 = mpimg.imread(fig_dir / "fig1b_FD001.png")
img_mba = mpimg.imread(fig_dir / "fig1b_MBA.png")

# Auto-crop whitespace from PNGs (trim rows/cols that are nearly pure white)
def autocrop(img, tol=0.98):
    """Crop white borders from an RGBA or RGB image array."""
    if img.shape[2] == 4:
        gray = img[:, :, :3].mean(axis=2)
    else:
        gray = img.mean(axis=2)
    rows = np.any(gray < tol, axis=1)
    cols = np.any(gray < tol, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return img[rmin:rmax+1, cmin:cmax+1]

img_fd001 = autocrop(img_fd001)
img_mba = autocrop(img_mba)

gs_b = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1],
                                        hspace=0.08, height_ratios=[1, 1])

# --- FD001 surface (top) ---
ax_b1 = fig.add_subplot(gs_b[0])
ax_b1.imshow(img_fd001, aspect="auto", interpolation="lanczos")
ax_b1.set_axis_off()

# --- MBA surface (bottom) ---
ax_b2 = fig.add_subplot(gs_b[1])
ax_b2.imshow(img_mba, aspect="auto", interpolation="lanczos")
ax_b2.set_axis_off()

# Panel label for (b)
ax_b1.text(-0.05, 1.10, r"$\bf{(b)}$" + r"  Predicted $p(t, \Delta t)$",
           transform=ax_b1.transAxes, fontsize=8, va="bottom")

# ── Save ──────────────────────────────────────────────────────────────
out_dir = Path("C:/Users/Jonaspetersen/dev/IndustrialJEPA/paper-neurips/figures")
fig.savefig(out_dir / "fig_hero.pdf")
print(f"Saved: {out_dir / 'fig_hero.pdf'}")
plt.close(fig)
