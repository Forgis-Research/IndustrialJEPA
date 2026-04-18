#!/usr/bin/env python3
"""
FIGURE DESIGN
Message: "Pretraining loss converges within ~20 epochs; probe RMSE tracks loss closely,
         confirming the prediction objective surfaces degradation-relevant features."
Type: Dual-axis line plot (loss + probe RMSE)
Layout: Single panel, two y-axes
Colors: blue=loss, orange=probe RMSE
Size: NeurIPS single column (5.5in wide, ~2.0in tall)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json

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
    "axes.spines.right": True,  # need right spine for dual axis
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

# Load data
with open("C:/Users/Jonaspetersen/dev/IndustrialJEPA/mechanical-jepa/experiments/v11/pretrain_history_L1_v2.json") as f:
    data = json.load(f)

loss = np.array(data["loss"])
pred_loss = np.array(data["pred_loss"])
probe_rmse = np.array(data["probe_rmse"])
probe_epochs = np.array(data["probe_epochs"])
epochs = np.arange(1, len(loss) + 1)

# Smooth loss with exponential moving average for readability
def ema(x, alpha=0.15):
    s = np.zeros_like(x)
    s[0] = x[0]
    for i in range(1, len(x)):
        s[i] = alpha * x[i] + (1 - alpha) * s[i-1]
    return s

loss_smooth = ema(loss)

fig, ax1 = plt.subplots(figsize=(5.5, 2.0))

# Left y-axis: pretraining loss
ax1.plot(epochs, loss, color=C["blue"], alpha=0.25, linewidth=0.5, zorder=2)
ax1.plot(epochs, loss_smooth, color=C["blue"], linewidth=1.2, zorder=3,
         label="Pretraining loss (EMA)")
ax1.set_xlabel("Pretraining epoch")
ax1.set_ylabel("L1 prediction loss", color=C["blue"])
ax1.tick_params(axis="y", labelcolor=C["blue"])
ax1.set_ylim(0.012, 0.055)
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)

# Right y-axis: probe RMSE
ax2 = ax1.twinx()
ax2.plot(probe_epochs, probe_rmse, color=C["orange"], linestyle="--", marker="s",
         markersize=3.5, linewidth=0.9, zorder=4, label="Frozen probe RMSE")
ax2.set_ylabel("Probe RMSE (cycles)", color=C["orange"])
ax2.tick_params(axis="y", labelcolor=C["orange"])
ax2.set_ylim(10, 30)
ax2.spines["top"].set_visible(False)

# Best probe annotation
best_idx = np.argmin(probe_rmse)
best_epoch = probe_epochs[best_idx]
best_rmse = probe_rmse[best_idx]
ax2.annotate(f"best: {best_rmse:.1f}",
             xy=(best_epoch, best_rmse),
             xytext=(best_epoch + 12, best_rmse - 2.5),
             fontsize=6.5, color=C["orange"],
             arrowprops=dict(arrowstyle="-", color=C["orange"], lw=0.4),
             va="center")

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
           fontsize=7, handlelength=2.0)

ax1.set_xlim(0, 102)

out = "C:/Users/Jonaspetersen/dev/IndustrialJEPA/paper-neurips/figures/fig_pretraining_curve.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
