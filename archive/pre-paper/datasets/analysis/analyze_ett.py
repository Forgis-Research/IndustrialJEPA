"""
ETT Dataset Analysis — Generate overview figure.

Produces: datasets/analysis/figures/ett_overview.png

Run: python datasets/analysis/analyze_ett.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "ett"
OUT_DIR  = Path(__file__).parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PHYSICS_GROUPS = {
    "HV_power":  ["HUFL", "HULL"],
    "MV_power":  ["MUFL", "MULL"],
    "LV_power":  ["LUFL", "LULL"],
    "thermal":   ["OT"],
}
GROUP_COLORS = {
    "HV_power": "#e74c3c",
    "MV_power": "#f39c12",
    "LV_power": "#2ecc71",
    "thermal":  "#9b59b6",
}
FEATURE_COLORS = {
    "HUFL": "#e74c3c", "HULL": "#c0392b",
    "MUFL": "#f39c12", "MULL": "#d68910",
    "LUFL": "#2ecc71", "LULL": "#1e8449",
    "OT":   "#9b59b6",
}


def load_datasets():
    dfs = {}
    for variant in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        fpath = DATA_DIR / f"{variant}.csv"
        if fpath.exists():
            df = pd.read_csv(fpath, parse_dates=["date"])
            dfs[variant] = df
    return dfs


def main():
    dfs = load_datasets()
    if not dfs:
        print(f"No data found in {DATA_DIR}")
        print("Run: python datasets/downloaders/download_ett.py")
        sys.exit(1)

    df_h1 = dfs["ETTh1"]
    df_h2 = dfs["ETTh2"]
    feature_cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("ETT Dataset (Electricity Transformer Temperature) — Analysis Overview",
                 fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 0: Big Picture ────────────────────────────────────────────────
    # 0a: Full ETTh1 time series (all 7 channels)
    ax0a = fig.add_subplot(gs[0, :2])
    for col in feature_cols:
        color = FEATURE_COLORS[col]
        x = np.arange(len(df_h1))
        ax0a.plot(x, df_h1[col].values, color=color, alpha=0.8, linewidth=0.5,
                  label=col)
    ax0a.set_title("ETTh1: Full time series (17,420 timesteps)", fontweight="bold")
    ax0a.set_xlabel("Timestep (hourly)")
    ax0a.set_ylabel("Value")
    ax0a.legend(loc="upper right", fontsize=7, ncol=2)
    ax0a.set_xlim(0, len(df_h1))

    # 0b: Missing values heatmap (should be zero for ETT)
    ax0b = fig.add_subplot(gs[0, 2])
    missing = pd.DataFrame({
        v: dfs[v][feature_cols].isnull().sum()
        for v in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"] if v in dfs
    })
    im = ax0b.imshow(missing.values, aspect="auto", cmap="Reds", vmin=0, vmax=100)
    ax0b.set_xticks(range(len(missing.columns)))
    ax0b.set_xticklabels(missing.columns, rotation=45, ha="right", fontsize=8)
    ax0b.set_yticks(range(len(feature_cols)))
    ax0b.set_yticklabels(feature_cols, fontsize=8)
    ax0b.set_title("Missing Values\n(all 4 variants)", fontweight="bold")
    plt.colorbar(im, ax=ax0b, shrink=0.7)

    # 0c: Dataset sizes comparison
    ax0c = fig.add_subplot(gs[0, 3])
    variants = list(dfs.keys())
    sizes = [len(dfs[v]) for v in variants]
    bars = ax0c.bar(variants, sizes, color=["#3498db", "#2980b9", "#1abc9c", "#16a085"])
    ax0c.set_title("Dataset Sizes\n(timesteps)", fontweight="bold")
    ax0c.set_ylabel("Timesteps")
    ax0c.tick_params(axis="x", labelsize=8)
    for bar, size in zip(bars, sizes):
        ax0c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                  f"{size:,}", ha="center", va="bottom", fontsize=8)
    ax0c.set_ylim(0, max(sizes) * 1.15)

    # ── Row 1: Feature Distributions & Correlations ───────────────────────
    # 1a: Box plots by physics group
    ax1a = fig.add_subplot(gs[1, :2])
    data_by_group = {}
    all_data = []
    all_labels = []
    all_colors_bp = []
    for group, cols in PHYSICS_GROUPS.items():
        color = GROUP_COLORS[group]
        for col in cols:
            all_data.append(df_h1[col].values)
            all_labels.append(f"{col}\n({group[:2]})")
            all_colors_bp.append(color)

    bp = ax1a.boxplot(all_data, labels=all_labels, patch_artist=True, showfliers=False,
                      medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], all_colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1a.set_title("Feature Distributions by Physics Group (ETTh1)", fontweight="bold")
    ax1a.set_ylabel("Value")
    ax1a.tick_params(axis="x", labelsize=7)

    # Add group legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=c, label=g, alpha=0.7)
                      for g, c in GROUP_COLORS.items()]
    ax1a.legend(handles=legend_handles, loc="upper right", fontsize=7)

    # 1b: Correlation heatmap
    ax1b = fig.add_subplot(gs[1, 2])
    corr = df_h1[feature_cols].corr()
    im2 = ax1b.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax1b.set_xticks(range(len(feature_cols)))
    ax1b.set_xticklabels(feature_cols, rotation=45, ha="right", fontsize=7)
    ax1b.set_yticks(range(len(feature_cols)))
    ax1b.set_yticklabels(feature_cols, fontsize=7)
    ax1b.set_title("ETTh1 Pearson\nCorrelation", fontweight="bold")
    for i in range(len(feature_cols)):
        for j in range(len(feature_cols)):
            ax1b.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                      fontsize=5.5, color="white" if abs(corr.values[i,j]) > 0.5 else "black")
    plt.colorbar(im2, ax=ax1b, shrink=0.7)

    # 1c: ETTh1 vs ETTh2 OT (oil temperature) comparison — transfer test
    ax1c = fig.add_subplot(gs[1, 3])
    x_h1 = np.arange(len(df_h1))
    x_h2 = np.arange(len(df_h2))
    ax1c.plot(x_h1, df_h1["OT"].values, color="#9b59b6", alpha=0.7,
              linewidth=0.8, label="ETTh1 OT")
    ax1c.plot(x_h2, df_h2["OT"].values, color="#e74c3c", alpha=0.7,
              linewidth=0.8, label="ETTh2 OT")
    ax1c.set_title("OT: ETTh1→ETTh2\n(Transfer Test Pair)", fontweight="bold")
    ax1c.set_xlabel("Timestep")
    ax1c.set_ylabel("Oil Temperature")
    ax1c.legend(fontsize=7)

    # ── Row 2: Time Series Examples ───────────────────────────────────────
    # Show 3 representative 200-step windows from different periods
    windows = [
        (0,    200,  "Early (summer 2016)"),
        (4000, 4200, "Mid (spring 2017)"),
        (8000, 8200, "Late (winter 2017)"),
    ]

    for idx, (start, end, title) in enumerate(windows):
        ax = fig.add_subplot(gs[2, idx])
        window = df_h1.iloc[start:end]
        for col in feature_cols:
            ax.plot(range(end-start), window[col].values,
                    color=FEATURE_COLORS[col], alpha=0.8, linewidth=0.8,
                    label=col if idx == 0 else "_nolegend_")
        ax.set_title(f"ETTh1: {title}", fontweight="bold", fontsize=8)
        ax.set_xlabel("Relative timestep")
        ax.set_ylabel("Value" if idx == 0 else "")
        if idx == 0:
            ax.legend(fontsize=6, ncol=2, loc="upper right")

    # Physics grouping diagram (last cell)
    ax_info = fig.add_subplot(gs[2, 3])
    ax_info.axis("off")
    info_text = (
        "Physics Groups (ETTh1)\n"
        "─────────────────────\n"
        "HV_power: HUFL, HULL\n"
        "MV_power: MUFL, MULL\n"
        "LV_power: LUFL, LULL\n"
        "thermal:  OT\n\n"
        "Tier 3 in Rapid Eval Suite\n\n"
        "Transfer Test:\n"
        "  ETTh1 → ETTh2\n\n"
        "IndustrialJEPA (Exp 46):\n"
        "  PhysMask vs Full-Attn\n"
        "  = -1.3% (NEGATIVE)\n\n"
        "Reason: OT (thermal) couples\n"
        "to all load groups → masking\n"
        "blocks useful cross-group info\n\n"
        "Published SOTA (H=96 MSE):\n"
        "  Moirai/TimesFM: ~0.35\n"
        "  PatchTST:       ~0.37\n"
        "  iTransformer:   ~0.39"
    )
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                 fontsize=7.5, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.savefig(OUT_DIR / "ett_overview.png", dpi=120, bbox_inches="tight",
                facecolor="white")
    print(f"Saved: {OUT_DIR / 'ett_overview.png'}")


if __name__ == "__main__":
    main()
