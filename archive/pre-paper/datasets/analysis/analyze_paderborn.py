"""
Paderborn Bearing Dataset Analysis — Generate overview figure.

Produces: datasets/analysis/figures/paderborn_overview.png

Run: python datasets/analysis/analyze_paderborn.py
     (requires sample: python datasets/downloaders/download_paderborn.py --sample)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data" / "paderborn"
OUT_DIR  = Path(__file__).parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Actual channel structure discovered from real .mat file inspection
# Y['Name'] channels: force, phase_current_1, phase_current_2, speed, temp_2_bearing_module, torque, vibration_1
CHANNEL_META = {
    0: {"name": "force",               "group": "force",     "rate": "low",  "color": "#e74c3c"},
    1: {"name": "phase_current_1",     "group": "current",   "rate": "high", "color": "#3498db"},
    2: {"name": "phase_current_2",     "group": "current",   "rate": "high", "color": "#2980b9"},
    3: {"name": "speed",               "group": "speed",     "rate": "low",  "color": "#f39c12"},
    4: {"name": "temp_2_bearing",      "group": "thermal",   "rate": "very_low", "color": "#2ecc71"},
    5: {"name": "torque",              "group": "torque",    "rate": "low",  "color": "#9b59b6"},
    6: {"name": "vibration_1",         "group": "vibration", "rate": "high", "color": "#e67e22"},
}

PHYSICS_GROUPS = {
    "vibration":  [6],          # vibration_1 (64 kHz)
    "current":    [1, 2],       # phase_current_1, phase_current_2 (64 kHz)
    "mechanical": [0, 3, 5],    # force, speed, torque (~4 kHz)
    "thermal":    [4],          # temp_2_bearing (very low rate)
}
GROUP_COLORS = {
    "vibration": "#e67e22",
    "current":   "#3498db",
    "mechanical":"#9b59b6",
    "thermal":   "#2ecc71",
}

BEARING_CLASSES = {
    "K001": {"label": "Healthy",       "color": "#2ecc71"},
    "KA01": {"label": "Outer Race Fault", "color": "#e74c3c"},
    "KI01": {"label": "Inner Race Fault", "color": "#3498db"},
}


def load_bearing(bearing_id: str, n_files: int = 3) -> dict:
    """Load N .mat files for a bearing condition."""
    import scipy.io

    bearing_dir = DATA_DIR / bearing_id
    if not bearing_dir.exists():
        return None

    mat_files = sorted(bearing_dir.glob("*.mat"))[:n_files]
    channels_all = {i: [] for i in range(7)}

    for fpath in mat_files:
        mat = scipy.io.loadmat(str(fpath))
        key = fpath.stem
        if key not in mat:
            continue
        data = mat[key]
        Y = data["Y"][0, 0]
        data_arr = Y["Data"]
        for i in range(7):
            arr = data_arr[0, i].flatten()
            channels_all[i].append(arr)

    return {i: channels_all[i] for i in range(7) if channels_all[i]}


def main():
    bearings = {}
    for bid in ["K001", "KA01", "KI01"]:
        d = load_bearing(bid, n_files=3)
        if d:
            bearings[bid] = d
            print(f"  Loaded {bid}: {len(d[6])} files of vibration data")

    if not bearings:
        print(f"No data in {DATA_DIR}")
        print("Run: python datasets/downloaders/download_paderborn.py --sample")
        sys.exit(1)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Paderborn Bearing Dataset — Analysis Overview (Tier 2 Candidate)",
                 fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 0: Big Picture ─────────────────────────────────────────────
    # 0a: Vibration signal for all 3 conditions (first 5000 samples at 64 kHz)
    ax0a = fig.add_subplot(gs[0, :2])
    n_show = 5000
    fs_high = 64000.0
    t_ms = np.arange(n_show) / fs_high * 1000  # ms
    for bid, d in bearings.items():
        vibs = d.get(6, [])
        if vibs:
            sig = vibs[0][:n_show]
            color = BEARING_CLASSES[bid]["color"]
            label = BEARING_CLASSES[bid]["label"]
            ax0a.plot(t_ms, sig, color=color, alpha=0.7, linewidth=0.6, label=label)
    ax0a.set_title(f"Vibration Signal: 3 Bearing Conditions (first {n_show/fs_high*1000:.1f} ms)",
                   fontweight="bold")
    ax0a.set_xlabel("Time (ms)")
    ax0a.set_ylabel("Vibration (a.u.)")
    ax0a.legend(fontsize=8)

    # 0b: Motor current — distinguishes healthy vs faulty
    ax0b = fig.add_subplot(gs[0, 2])
    n_curr = 3000
    for bid, d in bearings.items():
        curr = d.get(1, [])
        if curr:
            sig = curr[0][:n_curr]
            color = BEARING_CLASSES[bid]["color"]
            label = BEARING_CLASSES[bid]["label"]
            ax0b.plot(np.arange(n_curr), sig, color=color, alpha=0.7,
                      linewidth=0.5, label=label)
    ax0b.set_title("Motor Current (Phase 1)\nvs Condition", fontweight="bold")
    ax0b.set_xlabel("Sample index")
    ax0b.set_ylabel("Current (a.u.)")
    ax0b.legend(fontsize=7)

    # 0c: Channel sampling rates (visual summary)
    ax0c = fig.add_subplot(gs[0, 3])
    ax0c.axis("off")
    channel_text = (
        "Paderborn Channel Structure\n"
        "──────────────────────────\n"
        "Ch  Name              Rate\n"
        "─  ────────────────   ────\n"
        "0  force              4kHz\n"
        "1  phase_current_1   64kHz\n"
        "2  phase_current_2   64kHz\n"
        "3  speed              4kHz\n"
        "4  temp_2_bearing   <<1 Hz\n"
        "5  torque             4kHz\n"
        "6  vibration_1       64kHz\n\n"
        "Physics Groups:\n"
        "  vibration:   [ch6]\n"
        "  current:     [ch1, ch2]\n"
        "  mechanical:  [ch0, ch3, ch5]\n"
        "  thermal:     [ch4]\n\n"
        "NOTE: 64kHz channels have\n"
        "256,823 samples per 4s file.\n"
        "Downsample to 1kHz for\n"
        "uniform time series."
    )
    ax0c.text(0.05, 0.95, channel_text, transform=ax0c.transAxes,
              fontsize=7.5, verticalalignment="top", fontfamily="monospace",
              bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # ── Row 1: FFT & Physics ────────────────────────────────────────────
    # 1a: FFT comparison (vibration)
    ax1a = fig.add_subplot(gs[1, :2])
    for bid, d in bearings.items():
        vibs = d.get(6, [])
        if vibs:
            sig = vibs[0]
            n_fft = min(131072, len(sig))
            fft_mag = np.abs(np.fft.rfft(sig[:n_fft]))
            freqs = np.fft.rfftfreq(n_fft, d=1.0/fs_high)
            mask = freqs <= 5000
            color = BEARING_CLASSES[bid]["color"]
            label = BEARING_CLASSES[bid]["label"]
            ax1a.semilogy(freqs[mask], fft_mag[mask] + 1e-8,
                          color=color, alpha=0.7, linewidth=0.6, label=label)

    # Mark shaft frequency and harmonics
    shaft_freq = 900 / 60  # ~15 Hz (900 RPM operating condition)
    for harm in [1, 2, 3, 4, 5]:
        ax1a.axvline(x=shaft_freq * harm, color="gray", linestyle=":",
                     alpha=0.4, linewidth=0.8)
    ax1a.set_title("Vibration FFT: Healthy vs Fault Conditions",
                   fontweight="bold")
    ax1a.set_xlabel("Frequency (Hz)")
    ax1a.set_ylabel("Magnitude (log)")
    ax1a.legend(fontsize=8)
    ax1a.set_xlim(0, 5000)

    # 1b: RMS energy per channel per condition
    ax1b = fig.add_subplot(gs[1, 2])
    high_rate_channels = [1, 2, 6]  # current × 2, vibration
    channel_names_short = ["current_1", "current_2", "vibration"]
    x_pos = np.arange(len(high_rate_channels))
    width = 0.25
    for i, (bid, d) in enumerate(bearings.items()):
        rms_vals = []
        for ch in high_rate_channels:
            arr_list = d.get(ch, [])
            if arr_list:
                rms = np.sqrt(np.mean(arr_list[0]**2))
            else:
                rms = 0.0
            rms_vals.append(rms)
        color = BEARING_CLASSES[bid]["color"]
        label = BEARING_CLASSES[bid]["label"]
        ax1b.bar(x_pos + i * width, rms_vals, width, color=color,
                 alpha=0.8, label=label)
    ax1b.set_title("RMS per Channel\nby Fault Condition", fontweight="bold")
    ax1b.set_xticks(x_pos + width)
    ax1b.set_xticklabels(channel_names_short, fontsize=8)
    ax1b.set_ylabel("RMS amplitude")
    ax1b.legend(fontsize=7)

    # 1c: Cross-channel correlation within one bearing condition
    ax1c = fig.add_subplot(gs[1, 3])
    healthy = bearings.get("K001", {})
    if healthy:
        # Downsample all channels to same length (use 4kHz equivalent)
        n_low = 16008  # force/speed/torque length
        n_high = 256823  # vibration/current length
        downsample_factor = n_high // n_low
        ch_arrays = []
        ch_labels = []
        for ch_idx in [0, 1, 2, 3, 5, 6]:  # skip temp (only 5 points)
            arr_list = healthy.get(ch_idx, [])
            if arr_list:
                arr = arr_list[0]
                if len(arr) > n_low:
                    arr = arr[:n_high:downsample_factor]  # downsample
                arr = arr[:n_low]
                if len(arr) == n_low:
                    ch_arrays.append(arr)
                    ch_labels.append(CHANNEL_META[ch_idx]["name"][:10])
        if len(ch_arrays) >= 2:
            corr_mat = np.corrcoef(np.array(ch_arrays))
            im = ax1c.imshow(corr_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax1c.set_xticks(range(len(ch_labels)))
            ax1c.set_xticklabels(ch_labels, rotation=45, ha="right", fontsize=7)
            ax1c.set_yticks(range(len(ch_labels)))
            ax1c.set_yticklabels(ch_labels, fontsize=7)
            ax1c.set_title("Cross-Channel Correlation\n(K001 Healthy)", fontweight="bold")
            for i in range(len(ch_labels)):
                for j in range(len(ch_labels)):
                    ax1c.text(j, i, f"{corr_mat[i,j]:.2f}", ha="center",
                              va="center", fontsize=6,
                              color="white" if abs(corr_mat[i,j]) > 0.5 else "black")
            plt.colorbar(im, ax=ax1c, shrink=0.7)

    # ── Row 2: Granular Windows ────────────────────────────────────────
    # 2a: Current signal zoom (shows bearing fault modulation)
    ax2a = fig.add_subplot(gs[2, 0])
    n_zoom_curr = 6400  # 100ms at 64kHz
    t_curr = np.arange(n_zoom_curr) / fs_high * 1000
    for bid, d in bearings.items():
        curr_list = d.get(1, [])
        if curr_list:
            sig = curr_list[0][:n_zoom_curr]
            ax2a.plot(t_curr, sig, color=BEARING_CLASSES[bid]["color"],
                      alpha=0.7, linewidth=0.5, label=BEARING_CLASSES[bid]["label"])
    ax2a.set_title("Phase Current 1:\n100ms Detail Window", fontweight="bold", fontsize=8)
    ax2a.set_xlabel("Time (ms)")
    ax2a.set_ylabel("Current (a.u.)")
    ax2a.legend(fontsize=6)

    # 2b: Vibration zoom (show impulse differences)
    ax2b = fig.add_subplot(gs[2, 1])
    n_zoom_vib = 12800  # 200ms
    t_vib = np.arange(n_zoom_vib) / fs_high * 1000
    for bid, d in bearings.items():
        vib_list = d.get(6, [])
        if vib_list:
            sig = vib_list[0][:n_zoom_vib]
            ax2b.plot(t_vib, sig, color=BEARING_CLASSES[bid]["color"],
                      alpha=0.7, linewidth=0.5, label=BEARING_CLASSES[bid]["label"])
    ax2b.set_title("Vibration:\n200ms Detail Window", fontweight="bold", fontsize=8)
    ax2b.set_xlabel("Time (ms)")
    ax2b.set_ylabel("Vibration (a.u.)")
    ax2b.legend(fontsize=6)

    # 2c: Force + torque (low-rate mechanical channels)
    ax2c = fig.add_subplot(gs[2, 2])
    for bid, d in bearings.items():
        force_list = d.get(0, [])
        if force_list:
            n_show_low = min(1000, len(force_list[0]))
            ax2c.plot(force_list[0][:n_show_low],
                      color=BEARING_CLASSES[bid]["color"],
                      alpha=0.7, linewidth=0.8,
                      label=BEARING_CLASSES[bid]["label"])
    ax2c.set_title("Force (radial load)\nper Condition", fontweight="bold", fontsize=8)
    ax2c.set_xlabel("Sample index")
    ax2c.set_ylabel("Force (a.u.)")
    ax2c.legend(fontsize=6)

    # 2d: Tier 2 recommendation box
    ax2d = fig.add_subplot(gs[2, 3])
    ax2d.axis("off")
    tier2_text = (
        "TIER 2 RECOMMENDATION\n"
        "══════════════════════\n"
        "VERDICT: STRONG CANDIDATE\n\n"
        "Strengths:\n"
        "  + Real mechanical data\n"
        "  + 4 physical modalities:\n"
        "    vibration, current,\n"
        "    mechanical, thermal\n"
        "  + 33 bearing conditions\n"
        "  + No registration required\n"
        "  + Direct HTTP download\n\n"
        "Weaknesses:\n"
        "  - No forecasting SOTA\n"
        "    (must define own baseline)\n"
        "  - Mixed sampling rates\n"
        "    (need consistent downsampling)\n"
        "  - Only 7 channels after\n"
        "    dropping temperature\n\n"
        "Transfer Test Proposal:\n"
        "  K001 (healthy) → KA01\n"
        "  (outer race fault)\n"
        "  K001 → KI01\n"
        "  (inner race fault)"
    )
    ax2d.text(0.05, 0.95, tier2_text, transform=ax2d.transAxes,
              fontsize=7.5, verticalalignment="top", fontfamily="monospace",
              bbox=dict(boxstyle="round", facecolor="#d5e8d4", alpha=0.8))

    plt.savefig(OUT_DIR / "paderborn_overview.png", dpi=120,
                bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT_DIR / 'paderborn_overview.png'}")


if __name__ == "__main__":
    main()
