#!/usr/bin/env python3
"""
FIGURE DESIGN
Message: "Six verification diagnostics confirm the encoder tracks within-engine degradation:
         high Spearman rho, large engine-shuffle degradation, strong HI recovery."
Type: Horizontal bar chart of diagnostic metrics
Layout: Single panel with 6 bars showing diagnostic results
Colors: Forgis palette, highlight for key metrics
Size: NeurIPS single column (5.5in wide, ~1.8in tall)
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

# Diagnostics data from Section 6.1
diagnostics = [
    ("Spearman $\\rho$ (pred vs true RUL)", 0.830, 0.023, 1.0, "correlation"),
    ("Health index $R^2$", 0.926, 0.0, 1.0, "fit"),
    ("PC1 $|\\rho|$ vs health index", 0.797, 0.0, 1.0, "correlation"),
    ("Prediction std (median)", 12.1, 0.7, 20.0, "spread"),
    ("Engine shuffle $\\Delta$RMSE", 41.5, 0.0, 50.0, "degradation"),
    ("Temporal shuffle $\\Delta$RMSE", 20.8, 0.0, 30.0, "degradation"),
]

labels = [d[0] for d in diagnostics]
values = [d[1] for d in diagnostics]
stds = [d[2] for d in diagnostics]

# Normalize to [0, 1] for visual comparison
maxvals = [d[3] for d in diagnostics]
norm_values = [v / m for v, m in zip(values, maxvals)]

y = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(5.5, 1.9))

# Color by category
cat_colors = {
    "correlation": C["blue"],
    "fit": C["teal"],
    "spread": C["orange"],
    "degradation": C["blue"],
}
cat_bg = {
    "correlation": C["blue_bg"],
    "fit": C["teal_bg"],
    "spread": C["orange_bg"],
    "degradation": C["blue_bg"],
}

for i, (label, val, std, maxv, cat) in enumerate(diagnostics):
    color = cat_colors[cat]
    bg = cat_bg[cat]
    nv = val / maxv

    ax.barh(y[i], nv, height=0.55, color=bg, edgecolor=color,
            linewidth=0.7, zorder=3)

    # Value label
    val_str = f"{val:.3f}" if val < 1 else f"{val:.1f}"
    if std > 0:
        val_str += f" $\\pm$ {std}"
    ax.text(nv + 0.02, y[i], val_str, fontsize=6.5, color=C["dark"],
            va="center")

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=7)
ax.set_xlabel("Normalized diagnostic value")
ax.set_xlim(0, 1.25)
ax.invert_yaxis()
ax.xaxis.grid(True, linewidth=0.2, color=C["gray_lt"], zorder=0)
ax.set_axisbelow(True)

# Remove x ticks since normalization makes them less meaningful
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(["0", "", "0.5", "", "1.0"])

out = "C:/Users/Jonaspetersen/dev/IndustrialJEPA/paper-neurips/figures/fig_verification.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
