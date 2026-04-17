"""
UCI Hydraulic System Dataset Analysis — Generate overview figure.

Produces: datasets/analysis/figures/hydraulic_overview.png

Run: python datasets/analysis/analyze_hydraulic.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data" / "hydraulic"
OUT_DIR  = Path(__file__).parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SENSOR_META = {
    "PS1":  {"rate": 100, "unit": "bar",   "group": "pressure",   "color": "#e74c3c"},
    "PS2":  {"rate": 100, "unit": "bar",   "group": "pressure",   "color": "#c0392b"},
    "PS3":  {"rate": 100, "unit": "bar",   "group": "pressure",   "color": "#e67e22"},
    "PS4":  {"rate": 100, "unit": "bar",   "group": "pressure",   "color": "#d35400"},
    "PS5":  {"rate": 100, "unit": "bar",   "group": "pressure",   "color": "#f39c12"},
    "PS6":  {"rate": 100, "unit": "bar",   "group": "pressure",   "color": "#d68910"},
    "EPS1": {"rate": 100, "unit": "W",     "group": "flow_power", "color": "#3498db"},
    "FS1":  {"rate": 10,  "unit": "l/min", "group": "flow_power", "color": "#2980b9"},
    "FS2":  {"rate": 10,  "unit": "l/min", "group": "flow_power", "color": "#1a5276"},
    "TS1":  {"rate": 1,   "unit": "°C",    "group": "thermal",    "color": "#2ecc71"},
    "TS2":  {"rate": 1,   "unit": "°C",    "group": "thermal",    "color": "#27ae60"},
    "TS3":  {"rate": 1,   "unit": "°C",    "group": "thermal",    "color": "#1e8449"},
    "TS4":  {"rate": 1,   "unit": "°C",    "group": "thermal",    "color": "#196f3d"},
    "VS1":  {"rate": 1,   "unit": "mm/s",  "group": "mechanical", "color": "#9b59b6"},
    "CE":   {"rate": 1,   "unit": "%",     "group": "thermal",    "color": "#76d7c4"},
    "CP":   {"rate": 1,   "unit": "kW",    "group": "thermal",    "color": "#45b39d"},
    "SE":   {"rate": 1,   "unit": "%",     "group": "thermal",    "color": "#16a085"},
}

PHYSICS_GROUPS = {
    "pressure":    ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6"],
    "flow_power":  ["EPS1", "FS1", "FS2"],
    "thermal":     ["TS1", "TS2", "TS3", "TS4", "CE", "CP", "SE"],
    "mechanical":  ["VS1"],
}
GROUP_COLORS = {"pressure": "#e74c3c", "flow_power": "#3498db",
                "thermal": "#2ecc71", "mechanical": "#9b59b6"}

FAULT_LABELS = ["cooler", "valve", "pump", "accumulator", "stable"]


def load_data():
    """Load all sensors at 100 Hz (PS1-PS6, EPS1) and labels."""
    sensors = {}
    for name in SENSOR_META:
        fpath = DATA_DIR / f"{name}.txt"
        if fpath.exists():
            sensors[name] = np.loadtxt(str(fpath))

    profile_path = DATA_DIR / "profile.txt"
    labels = None
    if profile_path.exists():
        labels = np.loadtxt(str(profile_path)).astype(int)

    return sensors, labels


def main():
    sensors, labels = load_data()
    if not sensors:
        print(f"No data found in {DATA_DIR}")
        print("Run: python datasets/downloaders/download_hydraulic.py")
        sys.exit(1)

    n_cycles = sensors["PS1"].shape[0]
    print(f"Loaded {len(sensors)} sensors, {n_cycles} cycles")

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("UCI Hydraulic System Dataset — Analysis Overview",
                 fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 0: Big Picture ────────────────────────────────────────────────
    # 0a: Mean sensor value per cycle for pressure sensors
    ax0a = fig.add_subplot(gs[0, :2])
    for name in ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6"]:
        if name in sensors:
            means = sensors[name].mean(axis=1)
            ax0a.plot(means, color=SENSOR_META[name]["color"],
                      alpha=0.8, linewidth=0.7, label=name)
    ax0a.set_title(f"Pressure Sensors: Mean per Cycle ({n_cycles} cycles)",
                   fontweight="bold")
    ax0a.set_xlabel("Cycle index")
    ax0a.set_ylabel("Mean pressure (bar)")
    ax0a.legend(fontsize=7, ncol=3)

    # 0b: Fault condition distribution (label 4-column profile)
    ax0b = fig.add_subplot(gs[0, 2])
    if labels is not None:
        # Column 0: cooler (3%, 20%, 100%)
        cooler_vals = labels[:, 0]
        unique, counts = np.unique(cooler_vals, return_counts=True)
        ax0b.bar([f"Cooler\n{v}%" for v in unique], counts,
                 color=["#e74c3c", "#f39c12", "#2ecc71"])
        ax0b.set_title("Cooler Condition\nDistribution", fontweight="bold")
        ax0b.set_ylabel("Count")
        ax0b.tick_params(axis="x", labelsize=7)

    # 0c: Valve condition distribution
    ax0c = fig.add_subplot(gs[0, 3])
    if labels is not None:
        valve_vals = labels[:, 1]
        unique_v, counts_v = np.unique(valve_vals, return_counts=True)
        ax0c.bar([f"Valve\n{v}%" for v in unique_v], counts_v,
                 color=["#e74c3c", "#f39c12", "#2ecc71", "#3498db"])
        ax0c.set_title("Valve Condition\nDistribution", fontweight="bold")
        ax0c.set_ylabel("Count")
        ax0c.tick_params(axis="x", labelsize=7)

    # ── Row 1: Feature Distributions & Correlations ───────────────────────
    # 1a: Multi-sensor std (cycle-level variability) by physics group
    ax1a = fig.add_subplot(gs[1, :2])
    group_data = {}
    for group, sensor_list in PHYSICS_GROUPS.items():
        for name in sensor_list:
            if name in sensors:
                cycle_means = sensors[name].mean(axis=1)
                if group not in group_data:
                    group_data[group] = []
                group_data[group].append(cycle_means)

    bp_data = []
    bp_labels = []
    bp_colors = []
    for group, arrays in group_data.items():
        for i, arr in enumerate(arrays):
            bp_data.append(arr)
            sensor_name = PHYSICS_GROUPS[group][i] if i < len(PHYSICS_GROUPS[group]) else f"{group}_{i}"
            bp_labels.append(sensor_name)
            bp_colors.append(GROUP_COLORS[group])

    bp = ax1a.boxplot(bp_data, labels=bp_labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], bp_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax1a.set_title("Sensor Distributions Across Cycles (mean/cycle)",
                   fontweight="bold")
    ax1a.set_ylabel("Mean sensor value (cycle)")
    ax1a.tick_params(axis="x", labelsize=6, rotation=30)

    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=g, alpha=0.7)
               for g, c in GROUP_COLORS.items()]
    ax1a.legend(handles=handles, fontsize=7)

    # 1b: Correlation of cycle-mean features
    ax1b = fig.add_subplot(gs[1, 2])
    selected = ["PS1", "PS2", "EPS1", "FS1", "TS1", "TS2", "VS1"]
    cycle_means_mat = np.array([
        sensors[s].mean(axis=1) for s in selected if s in sensors
    ]).T
    labels_sel = [s for s in selected if s in sensors]
    corr = np.corrcoef(cycle_means_mat.T)
    im = ax1b.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax1b.set_xticks(range(len(labels_sel)))
    ax1b.set_xticklabels(labels_sel, rotation=45, ha="right", fontsize=7)
    ax1b.set_yticks(range(len(labels_sel)))
    ax1b.set_yticklabels(labels_sel, fontsize=7)
    ax1b.set_title("Cross-Sensor Correlation\n(cycle-mean values)", fontweight="bold")
    for i in range(len(labels_sel)):
        for j in range(len(labels_sel)):
            ax1b.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                      fontsize=6, color="white" if abs(corr[i,j]) > 0.5 else "black")
    plt.colorbar(im, ax=ax1b, shrink=0.7)

    # 1c: EPS1 (motor power) for healthy vs faulty cooler
    ax1c = fig.add_subplot(gs[1, 3])
    if labels is not None and "EPS1" in sensors:
        eps1_means = sensors["EPS1"].mean(axis=1)
        cooler_good = eps1_means[labels[:, 0] == 100]   # cooler 100% (healthy)
        cooler_bad  = eps1_means[labels[:, 0] == 3]     # cooler 3% (degraded)
        ax1c.hist(cooler_good, bins=30, alpha=0.6, color="#2ecc71",
                  label="Cooler 100% (OK)")
        ax1c.hist(cooler_bad,  bins=30, alpha=0.6, color="#e74c3c",
                  label="Cooler 3% (fail)")
        ax1c.set_title("Motor Power (EPS1):\nHealthy vs Cooler Fault",
                       fontweight="bold")
        ax1c.set_xlabel("Mean EPS1 (W)")
        ax1c.set_ylabel("Count")
        ax1c.legend(fontsize=7)

    # ── Row 2: Granular Time Series ───────────────────────────────────────
    # 2a: Within one healthy cycle — pressure dynamics at 100 Hz
    ax2a = fig.add_subplot(gs[2, 0])
    if labels is not None:
        healthy_idx = np.where(
            (labels[:, 0] == 100) & (labels[:, 1] == 100)
        )[0][0]
    else:
        healthy_idx = 0
    t = np.linspace(0, 60, 6000)  # 60 seconds at 100 Hz
    for name in ["PS1", "PS3", "PS5"]:
        if name in sensors:
            ax2a.plot(t, sensors[name][healthy_idx, :],
                      color=SENSOR_META[name]["color"],
                      alpha=0.8, linewidth=0.6, label=name)
    ax2a.set_title(f"Pressure: Healthy Cycle\n(cycle {healthy_idx})",
                   fontweight="bold", fontsize=8)
    ax2a.set_xlabel("Time (s)")
    ax2a.set_ylabel("Pressure (bar)")
    ax2a.legend(fontsize=6)

    # 2b: Same cycle — temperature (low rate)
    ax2b = fig.add_subplot(gs[2, 1])
    t_temp = np.linspace(0, 60, 60)  # 60 seconds at 1 Hz
    for name in ["TS1", "TS2", "TS3"]:
        if name in sensors:
            ax2b.plot(t_temp, sensors[name][healthy_idx, :],
                      color=SENSOR_META[name]["color"],
                      alpha=0.8, linewidth=1.5, marker="o", markersize=3,
                      label=name)
    ax2b.set_title(f"Temperature: Healthy Cycle\n(cycle {healthy_idx})",
                   fontweight="bold", fontsize=8)
    ax2b.set_xlabel("Time (s)")
    ax2b.set_ylabel("Temperature (°C)")
    ax2b.legend(fontsize=6)

    # 2c: Faulty cooler cycle comparison
    ax2c = fig.add_subplot(gs[2, 2])
    if labels is not None:
        fault_idx = np.where(labels[:, 0] == 3)[0][0]  # cooler 3% (broken)
    else:
        fault_idx = 100
    for name in ["TS1", "TS2"]:
        if name in sensors:
            ax2c.plot(t_temp, sensors[name][healthy_idx, :],
                      linestyle="--", alpha=0.6,
                      color=SENSOR_META[name]["color"],
                      linewidth=1.2, label=f"{name} healthy")
            ax2c.plot(t_temp, sensors[name][fault_idx, :],
                      linestyle="-", alpha=0.9,
                      color=SENSOR_META[name]["color"],
                      linewidth=1.8, label=f"{name} fault")
    ax2c.set_title("Temperature: Healthy vs\nFaulty Cooler", fontweight="bold",
                   fontsize=8)
    ax2c.set_xlabel("Time (s)")
    ax2c.set_ylabel("Temperature (°C)")
    ax2c.legend(fontsize=6)

    # 2d: Dataset info box
    ax2d = fig.add_subplot(gs[2, 3])
    ax2d.axis("off")
    info_text = (
        "Hydraulic System Summary\n"
        "─────────────────────────\n"
        "Real lab test rig data\n"
        "2,205 cycles × 60 seconds\n"
        "17 sensors, mixed rates\n\n"
        "Physics Groups:\n"
        "  pressure:   PS1-PS6\n"
        "  flow/power: EPS1, FS1, FS2\n"
        "  thermal:    TS1-4, CE, CP, SE\n"
        "  mechanical: VS1\n\n"
        "Fault Labels (multi-label):\n"
        "  Cooler: 3%, 20%, 100%\n"
        "  Valve: 73%, 80%, 90%, 100%\n"
        "  Pump: leak 0/1/2 ml/min\n"
        "  Accumulator: 90-130 bar\n\n"
        "Tier 2 Assessment:\n"
        "  POSSIBLE FALLBACK\n"
        "  Pro: real, clear groups\n"
        "  Con: small (2205 cycles),\n"
        "  mixed sampling rates,\n"
        "  no forecasting SOTA"
    )
    ax2d.text(0.05, 0.95, info_text, transform=ax2d.transAxes,
              fontsize=7.5, verticalalignment="top", fontfamily="monospace",
              bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8))

    plt.savefig(OUT_DIR / "hydraulic_overview.png", dpi=120,
                bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT_DIR / 'hydraulic_overview.png'}")


if __name__ == "__main__":
    main()
