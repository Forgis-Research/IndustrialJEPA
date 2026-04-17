"""
CWRU Bearing Dataset Analysis — Generate overview figure.

Produces: datasets/analysis/figures/cwru_overview.png

Run: python datasets/analysis/analyze_cwru.py
     (requires sample downloaded: python datasets/downloaders/download_cwru.py --sample)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data" / "cwru"
OUT_DIR  = Path(__file__).parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILE_META = {
    "normal_0hp.mat":  {"fault": "normal",     "color": "#2ecc71", "load": 0},
    "IR007_0hp.mat":   {"fault": "inner_race",  "color": "#e74c3c", "load": 0},
    "B007_0hp.mat":    {"fault": "ball",        "color": "#3498db", "load": 0},
    "OR007_6_0hp.mat": {"fault": "outer_race",  "color": "#f39c12", "load": 0},
}


def load_mat_file(path: Path) -> dict:
    """Load CWRU .mat file and extract the time series arrays."""
    import scipy.io
    mat = scipy.io.loadmat(str(path))
    data_keys = [k for k in mat.keys() if not k.startswith("__")]
    result = {}
    for k in data_keys:
        arr = mat[k]
        if isinstance(arr, np.ndarray) and arr.ndim >= 1 and arr.size > 1000:
            result[k] = arr.flatten()
    return result


def compute_fft(signal: np.ndarray, fs: float = 12000.0) -> tuple:
    """Compute power spectral density."""
    n = len(signal)
    fft_vals = np.abs(np.fft.rfft(signal[:n]))
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    return freqs, fft_vals


def main():
    # Load all 4 sample files
    loaded = {}
    for fname, meta in FILE_META.items():
        fpath = DATA_DIR / fname
        if fpath.exists():
            try:
                arrays = load_mat_file(fpath)
                de_key = [k for k in arrays if "DE_time" in k or "DE" in k.upper()]
                if de_key:
                    loaded[fname] = {
                        "signal": arrays[de_key[0]],
                        "fault":  meta["fault"],
                        "color":  meta["color"],
                        "all_channels": arrays,
                    }
            except Exception as e:
                print(f"  [WARN] Could not load {fname}: {e}")

    if not loaded:
        print(f"No data found in {DATA_DIR}")
        print("Run: python datasets/downloaders/download_cwru.py --sample")
        sys.exit(1)

    print(f"Loaded {len(loaded)} files")

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("CWRU Bearing Dataset — Analysis Overview",
                 fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 0: Big Picture ────────────────────────────────────────────────
    # 0a: Full time series for all 4 conditions (first 5000 samples)
    ax0a = fig.add_subplot(gs[0, :2])
    n_show = 5000
    fs = 12000.0
    t = np.arange(n_show) / fs * 1000  # milliseconds
    for fname, d in loaded.items():
        sig = d["signal"][:n_show]
        ax0a.plot(t, sig, color=d["color"], alpha=0.7, linewidth=0.6,
                  label=d["fault"].replace("_", " ").title())
    ax0a.set_title(f"Drive End Vibration: All Fault Types (first {n_show/fs*1000:.0f} ms)",
                   fontweight="bold")
    ax0a.set_xlabel("Time (ms)")
    ax0a.set_ylabel("Vibration (g)")
    ax0a.legend(fontsize=8)

    # 0b: RMS amplitude per condition (using all available data)
    ax0b = fig.add_subplot(gs[0, 2])
    fault_names = []
    rms_values = []
    colors_rms = []
    for fname, d in loaded.items():
        rms = np.sqrt(np.mean(d["signal"]**2))
        fault_names.append(d["fault"].replace("_", "\n").replace("race", "").strip())
        rms_values.append(rms)
        colors_rms.append(d["color"])
    bars = ax0b.bar(fault_names, rms_values, color=colors_rms, alpha=0.8)
    ax0b.set_title("RMS Amplitude\nby Fault Type", fontweight="bold")
    ax0b.set_ylabel("RMS (g)")
    for bar, val in zip(bars, rms_values):
        ax0b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                  f"{val:.4f}", ha="center", va="bottom", fontsize=7)

    # 0c: Channel count per file
    ax0c = fig.add_subplot(gs[0, 3])
    labels_ch = []
    n_channels = []
    colors_ch = []
    for fname, d in loaded.items():
        n_ch = len(d["all_channels"])
        labels_ch.append(d["fault"].replace("_", "\n").replace("race", ""))
        n_channels.append(n_ch)
        colors_ch.append(d["color"])
    ax0c.bar(labels_ch, n_channels, color=colors_ch, alpha=0.8)
    ax0c.set_title("Channels per File\n(DE, FE, BA)", fontweight="bold")
    ax0c.set_ylabel("Number of channels")
    ax0c.set_ylim(0, 5)
    ax0c.axhline(y=2, color="red", linestyle="--", alpha=0.5, label="Min (2)")
    ax0c.axhline(y=4, color="blue", linestyle="--", alpha=0.5, label="Max (4)")
    ax0c.legend(fontsize=7)

    # ── Row 1: Frequency Analysis ─────────────────────────────────────────
    # 1a–c: FFT for each fault type (DE channel)
    bearing_freqs = {
        "BPFO": 107.36,  # Ball Pass Frequency Outer Race (6205, 1797 RPM)
        "BPFI": 162.18,  # Ball Pass Frequency Inner Race
        "BSF":  71.41,   # Ball Spin Frequency
        "FTF":  14.94,   # Fundamental Train Frequency
    }

    for idx, (fname, d) in enumerate(loaded.items()):
        if idx >= 3:
            break
        ax = fig.add_subplot(gs[1, idx])
        sig = d["signal"]
        # Use first 65536 samples for FFT resolution
        n_fft = min(65536, len(sig))
        freqs, psd = compute_fft(sig[:n_fft], fs=fs)
        # Only show 0–2000 Hz (relevant bearing frequencies)
        mask = freqs <= 2000
        ax.semilogy(freqs[mask], psd[mask] + 1e-10, color=d["color"],
                    alpha=0.8, linewidth=0.6)
        # Mark bearing defect frequencies
        for freq_name, freq_val in bearing_freqs.items():
            if freq_val <= 2000:
                ax.axvline(x=freq_val, color="gray", linestyle="--",
                           alpha=0.5, linewidth=0.8)
                ax.text(freq_val, psd[mask].max() * 1.5,
                        freq_name, rotation=90, fontsize=5.5,
                        color="gray", va="bottom")
        ax.set_title(f"FFT: {d['fault'].replace('_', ' ').title()}",
                     fontweight="bold", fontsize=8)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")

    # 1d: Info box
    ax1d = fig.add_subplot(gs[1, 3])
    ax1d.axis("off")
    info_text = (
        "Bearing Fault Frequencies\n"
        "(6205 bearing, 1797 RPM)\n"
        "─────────────────────────\n"
        f"BPFO: {bearing_freqs['BPFO']:.1f} Hz  Outer race\n"
        f"BPFI: {bearing_freqs['BPFI']:.1f} Hz  Inner race\n"
        f"BSF:  {bearing_freqs['BSF']:.1f} Hz  Ball spin\n"
        f"FTF:  {bearing_freqs['FTF']:.1f} Hz  Cage\n\n"
        "Dataset Stats:\n"
        "  ~500 total files\n"
        "  2–4 channels/file\n"
        "  12,000–48,000 Hz\n"
        "  20,480–243,938 samples/file\n\n"
        "Physics Grouping:\n"
        "  WEAK (vibration only)\n"
        "  Groups: {DE, FE, BA}\n"
        "  All same modality!\n\n"
        "IndustrialJEPA Verdict:\n"
        "  NOT recommended for\n"
        "  Tier 2 (too few channels)"
    )
    ax1d.text(0.05, 0.95, info_text, transform=ax1d.transAxes,
              fontsize=7.5, verticalalignment="top", fontfamily="monospace",
              bbox=dict(boxstyle="round", facecolor="#ffeeba", alpha=0.8))

    # ── Row 2: Granular Time Series ───────────────────────────────────────
    for idx, (fname, d) in enumerate(list(loaded.items())[:3]):
        ax = fig.add_subplot(gs[2, idx])
        # Show 5 ms window at highest temporal detail
        n_zoom = 300  # 25 ms at 12 kHz
        t_zoom = np.arange(n_zoom) / fs * 1000  # ms
        ax.plot(t_zoom, d["signal"][:n_zoom], color=d["color"],
                linewidth=0.8, alpha=0.9)
        ax.set_title(f"{d['fault'].replace('_', ' ').title()}\n(25 ms detail)",
                     fontweight="bold", fontsize=8)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Vibration (g)")

    # Last panel: Publication SOTA comparison
    ax_sota = fig.add_subplot(gs[2, 3])
    ax_sota.axis("off")
    sota_text = (
        "Published SOTA (Fault Classification)\n"
        "──────────────────────────────────────\n"
        "Method            Accuracy   Year\n"
        "CNN-1D            99.7%      2017\n"
        "Res-Net           99.4%      2018\n"
        "WDCNN             99.5%      2017\n"
        "Transformer       99%+       2021\n\n"
        "IMPORTANT NOTE:\n"
        "  SOTA above = fault classification\n"
        "  (10-class: normal + 3 fault types\n"
        "   × 3 severity levels)\n\n"
        "  NO published forecasting SOTA\n"
        "  on CWRU (not a standard\n"
        "  forecasting benchmark)\n\n"
        "Recommendation:\n"
        "  Use only for ablation/comparison\n"
        "  NOT as Tier 2 dataset"
    )
    ax_sota.text(0.05, 0.95, sota_text, transform=ax_sota.transAxes,
                 fontsize=7, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="#d5e8d4", alpha=0.8))

    plt.savefig(OUT_DIR / "cwru_overview.png", dpi=120,
                bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT_DIR / 'cwru_overview.png'}")


if __name__ == "__main__":
    main()
