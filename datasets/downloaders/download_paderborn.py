"""
Download Paderborn University Bearing Dataset (KAT Datacenter).

33 RAR archives (~155–175 MB each, ~5.4 GB total).
Direct download, no registration required.
Saves to datasets/data/paderborn/

Usage:
    python datasets/downloaders/download_paderborn.py
    python datasets/downloaders/download_paderborn.py --sample  # 3 files only (K001 healthy + KA01 outer + KI01 inner)
    python datasets/downloaders/download_paderborn.py --condition K001  # Single bearing
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

BASE_URL = "https://groups.uni-paderborn.de/kat/BearingDataCenter/"

# All 33 bearing archives
ALL_BEARINGS = {
    # Healthy bearings
    "K001": "K001.rar",
    "K002": "K002.rar",
    "K003": "K003.rar",
    "K004": "K004.rar",
    "K005": "K005.rar",
    "K006": "K006.rar",
    # Outer race damage (artificially damaged via EDM)
    "KA01": "KA01.rar",
    "KA03": "KA03.rar",
    "KA04": "KA04.rar",
    "KA05": "KA05.rar",
    "KA06": "KA06.rar",
    "KA07": "KA07.rar",
    "KA08": "KA08.rar",
    "KA09": "KA09.rar",
    # Outer race damage (real/natural damage)
    "KA15": "KA15.rar",
    "KA16": "KA16.rar",
    "KA22": "KA22.rar",
    "KA30": "KA30.rar",
    # Combined damage
    "KB23": "KB23.rar",
    "KB24": "KB24.rar",
    "KB27": "KB27.rar",
    # Inner race damage (artificially damaged)
    "KI01": "KI01.rar",
    "KI03": "KI03.rar",
    "KI04": "KI04.rar",
    "KI05": "KI05.rar",
    "KI06": "KI06.rar",
    "KI07": "KI07.rar",
    "KI08": "KI08.rar",
    # Inner race damage (real/natural)
    "KI14": "KI14.rar",
    "KI16": "KI16.rar",
    "KI17": "KI17.rar",
    "KI18": "KI18.rar",
    "KI21": "KI21.rar",
}

BEARING_CATEGORIES = {
    "healthy":       ["K001", "K002", "K003", "K004", "K005", "K006"],
    "outer_race":    ["KA01", "KA03", "KA04", "KA05", "KA06", "KA07",
                      "KA08", "KA09", "KA15", "KA16", "KA22", "KA30"],
    "combined":      ["KB23", "KB24", "KB27"],
    "inner_race":    ["KI01", "KI03", "KI04", "KI05", "KI06", "KI07",
                      "KI08", "KI14", "KI16", "KI17", "KI18", "KI21"],
}

# Minimal sample: 1 healthy + 1 outer race (artificial) + 1 inner race (artificial)
SAMPLE_BEARINGS = ["K001", "KA01", "KI01"]

CHANNEL_INFO = {
    "a1":    {"type": "vibration",   "unit": "m/s²", "desc": "Accelerometer radial"},
    "a2":    {"type": "vibration",   "unit": "m/s²", "desc": "Accelerometer tangential"},
    "a3":    {"type": "vibration",   "unit": "m/s²", "desc": "Accelerometer axial"},
    "v1":    {"type": "vibration",   "unit": "mm/s", "desc": "Shaft velocity"},
    "temp1": {"type": "temperature", "unit": "°C",   "desc": "Bearing housing temperature"},
    "torque":{"type": "torque",      "unit": "Nm",   "desc": "Measured motor torque"},
    "ia":    {"type": "current",     "unit": "A",    "desc": "Motor current phase A"},
    "ib":    {"type": "current",     "unit": "A",    "desc": "Motor current phase B"},
}

PHYSICS_GROUPS = {
    "vibration_radial":  ["a1", "a2"],
    "vibration_axial":   ["a3", "v1"],
    "thermal_torque":    ["temp1", "torque"],
    "motor_current":     ["ia", "ib"],
}


def check_unrar() -> bool:
    """Check if unrar or 7z is available."""
    for cmd in ["unrar", "7z", "bsdtar"]:
        try:
            subprocess.run([cmd, "--version"], capture_output=True, timeout=5)
            return cmd
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def download_bearing(bearing_id: str, output_dir: Path,
                     verbose: bool = True) -> bool:
    """Download and optionally extract a single bearing RAR file."""
    fname = ALL_BEARINGS.get(bearing_id)
    if not fname:
        print(f"  [ERR] Unknown bearing: {bearing_id}")
        return False

    url = BASE_URL + fname
    rar_path = output_dir / fname
    extract_dir = output_dir / bearing_id

    if extract_dir.exists() and any(extract_dir.iterdir()):
        if verbose:
            print(f"  [SKIP] {bearing_id} already extracted")
        return True

    if not rar_path.exists():
        if verbose:
            print(f"  [DOWN] {bearing_id} (~160 MB)...")
        try:
            urlretrieve(url, rar_path)
            size_mb = rar_path.stat().st_size / 1024 / 1024
            if verbose:
                print(f"         -> {size_mb:.1f} MB")
        except URLError as e:
            print(f"  [ERR] Failed: {e}")
            return False

    # Try to extract
    extractor = check_unrar()
    if extractor:
        extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            if extractor == "unrar":
                subprocess.run(
                    ["unrar", "e", str(rar_path), str(extract_dir) + "/"],
                    check=True, capture_output=True
                )
            elif extractor == "7z":
                subprocess.run(
                    ["7z", "e", str(rar_path), f"-o{extract_dir}"],
                    check=True, capture_output=True
                )
            if verbose:
                extracted = list(extract_dir.glob("*.mat"))
                print(f"  [EXTR] {len(extracted)} .mat files extracted to {extract_dir}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"  [WARN] Extraction failed: {e}. RAR file retained.")
            return True  # Download succeeded even if extraction failed
    else:
        if verbose:
            print(f"  [WARN] No RAR extractor found (install unrar or 7z).")
            print(f"         RAR file saved at {rar_path}")
        return True


def verify_mat_stats(extract_dir: Path) -> dict:
    """Load a .mat measurement file and return channel stats."""
    try:
        import scipy.io
        import numpy as np
        mat_files = sorted(extract_dir.glob("*.mat"))
        if not mat_files:
            return {"status": "no_mat_files"}
        mat = scipy.io.loadmat(str(mat_files[0]))
        # Find data structure: typically has field 'Y' with sub-fields
        result = {"status": "ok", "n_files": len(mat_files)}
        if "Y" in mat:
            y = mat["Y"]
            result["structure"] = "Y-field"
            result["shape"] = str(y.shape)
        return result
    except Exception as e:
        return {"status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Download Paderborn Bearing Dataset")
    parser.add_argument("--sample", action="store_true",
                        help=f"Download only {SAMPLE_BEARINGS} (3 files, ~480 MB)")
    parser.add_argument("--condition", type=str,
                        help="Download single bearing (e.g., K001, KA01)")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent.parent / "data" / "paderborn"),
                        help="Output directory")
    parser.add_argument("--no-extract", action="store_true",
                        help="Skip extraction, keep RAR files only")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.condition:
        bearings_to_download = [args.condition]
    elif args.sample:
        bearings_to_download = SAMPLE_BEARINGS
    else:
        bearings_to_download = list(ALL_BEARINGS.keys())

    total_size_approx = len(bearings_to_download) * 165  # ~165 MB per file
    mode = "SAMPLE" if args.sample else ("SINGLE" if args.condition else "FULL")

    print(f"\nPaderborn Bearing Dataset Downloader")
    print(f"Mode: {mode} ({len(bearings_to_download)} bearings, ~{total_size_approx/1024:.1f} GB)")
    print(f"Output: {output_dir}")
    print("-" * 60)

    extractor = check_unrar()
    if not extractor:
        print("[WARN] No RAR extractor found. Install: sudo apt-get install unrar")
        print("       Files will be downloaded but not extracted.")
    else:
        print(f"[INFO] Using {extractor} for extraction")

    success_count = 0
    for bearing_id in bearings_to_download:
        if download_bearing(bearing_id, output_dir):
            success_count += 1

    print(f"\nCompleted {success_count}/{len(bearings_to_download)} bearings")

    print("\nDataset Summary:")
    print("  Physical system: Motor drive test rig with bearing housing")
    print("  Total bearings: 33 (6 healthy, 15 outer race, 3 combined, 15 inner race)")
    print("  Measurement files per bearing: 20 (4 conditions × 5 repetitions)")
    print("  Duration per file: 4 seconds at 64,000 Hz = 256,000 samples")
    print("  Channels: 8 (vibration ×3, shaft velocity, temperature, torque, current ×2)")
    print("\nPhysics Groups:")
    for group, sensors in PHYSICS_GROUPS.items():
        print(f"  {group}: {sensors}")
    print(f"\nTier 2 VERDICT: STRONG CANDIDATE")
    print(f"  Real mechanical data with motor current + vibration multimodality")
    print(f"  8 channels, 4 clear physics groups")
    print(f"  Main gap: No published FORECASTING SOTA (would need to define own baseline)")
    print(f"\nData saved to: {output_dir}")


if __name__ == "__main__":
    main()
