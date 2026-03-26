"""
Download UCI Hydraulic System Condition Monitoring Dataset.

Real hydraulic test rig, 17 sensors, 2,205 cycles.
Direct download, no registration required.
Saves to datasets/data/hydraulic/

Usage:
    python datasets/downloaders/download_hydraulic.py
    python datasets/downloaders/download_hydraulic.py --sample  # First 200 cycles only
"""

import argparse
import io
import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve, urlopen
from urllib.error import URLError

DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/static/public/447/"
    "condition+monitoring+of+hydraulic+systems.zip"
)

# Sensor files with sampling rates
SENSOR_FILES = {
    "PS1.txt":   {"rate": 100, "unit": "bar",   "desc": "Pressure 1"},
    "PS2.txt":   {"rate": 100, "unit": "bar",   "desc": "Pressure 2"},
    "PS3.txt":   {"rate": 100, "unit": "bar",   "desc": "Pressure 3"},
    "PS4.txt":   {"rate": 100, "unit": "bar",   "desc": "Pressure 4"},
    "PS5.txt":   {"rate": 100, "unit": "bar",   "desc": "Pressure 5"},
    "PS6.txt":   {"rate": 100, "unit": "bar",   "desc": "Pressure 6"},
    "EPS1.txt":  {"rate": 100, "unit": "W",     "desc": "Motor power"},
    "FS1.txt":   {"rate": 10,  "unit": "l/min", "desc": "Flow sensor 1"},
    "FS2.txt":   {"rate": 10,  "unit": "l/min", "desc": "Flow sensor 2"},
    "TS1.txt":   {"rate": 1,   "unit": "°C",    "desc": "Temperature 1"},
    "TS2.txt":   {"rate": 1,   "unit": "°C",    "desc": "Temperature 2"},
    "TS3.txt":   {"rate": 1,   "unit": "°C",    "desc": "Temperature 3"},
    "TS4.txt":   {"rate": 1,   "unit": "°C",    "desc": "Temperature 4"},
    "VS1.txt":   {"rate": 1,   "unit": "mm/s",  "desc": "Vibration"},
    "CE.txt":    {"rate": 1,   "unit": "%",     "desc": "Cooling efficiency"},
    "CP.txt":    {"rate": 1,   "unit": "kW",    "desc": "Cooling power"},
    "SE.txt":    {"rate": 1,   "unit": "%",     "desc": "System efficiency"},
    "profile.txt": {"rate": None, "unit": None, "desc": "Target labels"},
}

PHYSICS_GROUPS = {
    "pressure":    ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6"],
    "flow_power":  ["EPS1", "FS1", "FS2"],
    "thermal":     ["TS1", "TS2", "TS3", "TS4", "CE", "CP", "SE"],
    "mechanical":  ["VS1"],
}


def download_and_extract(output_dir: Path, verbose: bool = True) -> bool:
    """Download the ZIP archive and extract its contents."""
    zip_path = output_dir / "hydraulic_raw.zip"

    if not zip_path.exists():
        if verbose:
            print(f"  [DOWN] Downloading ZIP archive (~73 MB)...")
        try:
            urlretrieve(DOWNLOAD_URL, zip_path)
            size_mb = zip_path.stat().st_size / 1024 / 1024
            if verbose:
                print(f"         -> {size_mb:.1f} MB saved")
        except URLError as e:
            print(f"  [ERR]  Download failed: {e}")
            return False

    # Extract
    if verbose:
        print(f"  [EXTR] Extracting archive...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if verbose:
                    print(f"         {name}")
                zf.extract(name, output_dir)
        return True
    except zipfile.BadZipFile as e:
        print(f"  [ERR]  Bad ZIP file: {e}")
        return False


def verify_and_stats(output_dir: Path) -> dict:
    """Load sensors and verify dataset integrity."""
    try:
        import numpy as np

        stats = {}
        for fname, meta in SENSOR_FILES.items():
            fpath = output_dir / fname
            if not fpath.exists():
                stats[fname] = {"status": "missing"}
                continue
            data = np.loadtxt(str(fpath))
            stats[fname] = {
                "status": "ok",
                "shape": data.shape,
                "rate": meta["rate"],
                "unit": meta["unit"],
                "mean": float(data.mean()) if data.ndim > 0 else None,
                "std": float(data.std()) if data.ndim > 0 else None,
            }
        return stats
    except Exception as e:
        return {"status": "error", "error": str(e)}


def create_sample(output_dir: Path, sample_dir: Path, n_cycles: int = 200):
    """Extract first N cycles from each sensor file."""
    import numpy as np

    sample_dir.mkdir(parents=True, exist_ok=True)
    for fname in SENSOR_FILES:
        src = output_dir / fname
        dst = sample_dir / fname
        if not src.exists():
            continue
        if fname == "profile.txt":
            data = np.loadtxt(str(src))
            np.savetxt(str(dst), data[:n_cycles])
        else:
            data = np.loadtxt(str(src))
            if data.ndim == 2:
                np.savetxt(str(dst), data[:n_cycles])
            else:
                np.savetxt(str(dst), data[:n_cycles])
    print(f"  [SAMP] Sample ({n_cycles} cycles) saved to {sample_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download UCI Hydraulic System Dataset")
    parser.add_argument("--sample", action="store_true",
                        help="Extract only first 200 cycles after download")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent.parent / "data" / "hydraulic"),
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nUCI Hydraulic System Dataset Downloader")
    print(f"Output: {output_dir}")
    print("-" * 60)

    success = download_and_extract(output_dir)

    if success:
        try:
            import numpy as np
            stats = verify_and_stats(output_dir)
            print("\nSensor verification:")
            for fname, info in stats.items():
                if isinstance(info, dict) and info.get("status") == "ok":
                    shape = info.get("shape", "?")
                    rate = info.get("rate")
                    unit = info.get("unit", "")
                    mean = info.get("mean")
                    rate_str = f"{rate} Hz" if rate else "labels"
                    mean_str = f"{mean:.2f}" if mean is not None else "?"
                    print(f"  {fname:<12} shape={shape!s:<14} {rate_str:<8} "
                          f"mean={mean_str} {unit}")
                elif isinstance(info, dict) and info.get("status") == "missing":
                    print(f"  {fname:<12} MISSING")

            if args.sample:
                sample_dir = output_dir.parent / "hydraulic_sample"
                create_sample(output_dir, sample_dir, n_cycles=200)

        except ImportError:
            print("\n[WARN] numpy not available; skipping verification")

    print("\nDataset Summary:")
    print("  Physical system: Hydraulic test rig (pump + valves + cooler)")
    print("  Instances: 2,205 cycles × 60 seconds each")
    print("  Sensors: 17 (pressure, flow, power, temperature, vibration)")
    print("  Sampling: Mixed (100 Hz pressure, 10 Hz flow, 1 Hz temperature)")
    print("  Labels: Cooler/valve/pump/accumulator degradation (multi-label)")
    print("\nPhysics Groups:")
    for group, sensors in PHYSICS_GROUPS.items():
        print(f"  {group}: {sensors}")
    print(f"\nTier 2 assessment: POSSIBLE FALLBACK")
    print(f"  Pro: Real data, clear physics groups, no registration")
    print(f"  Con: Small (2205 cycles), mixed sampling rates, no forecasting SOTA")
    print(f"\nData saved to: {output_dir}")


if __name__ == "__main__":
    main()
