"""
Download FEMTO/PRONOSTIA Bearing Dataset (PHM IEEE 2012 Challenge).

Run-to-failure bearing data from FEMTO-ST Institute.
- 17 bearings: 6 training + 11 test
- Horizontal/vertical vibration at 25.6 kHz
- Temperature measurements
- 3 operating conditions (speed/load combinations)

Sources:
- Primary: https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip
- Mirror: https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset
- Kaggle: https://www.kaggle.com/datasets/alanhabrony/ieee-phm-2012-data-challenge

Usage:
    python datasets/downloaders/download_femto.py
    python datasets/downloaders/download_femto.py --sample  # Download subset
"""

import argparse
import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError
import shutil

# Primary download URL (PHM Society S3 mirror)
PRIMARY_URL = "https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip"

# Dataset structure after extraction
OPERATING_CONDITIONS = {
    "Condition_1": {"speed_rpm": 1800, "load_N": 4000},
    "Condition_2": {"speed_rpm": 1650, "load_N": 4200},
    "Condition_3": {"speed_rpm": 1500, "load_N": 5000},
}

TRAINING_BEARINGS = {
    "Condition_1": ["Bearing1_1", "Bearing1_2"],
    "Condition_2": ["Bearing2_1", "Bearing2_2"],
    "Condition_3": ["Bearing3_1", "Bearing3_2"],
}

TEST_BEARINGS = {
    "Condition_1": ["Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7"],
    "Condition_2": ["Bearing2_3", "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7"],
    "Condition_3": ["Bearing3_3"],
}


def download_progress(block_num, block_size, total_size):
    """Progress callback for urlretrieve."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)


def download_femto(output_dir: Path, sample: bool = False) -> bool:
    """Download the FEMTO dataset."""
    zip_path = output_dir / "femto_bearing.zip"

    # Check if already extracted
    extracted_marker = output_dir / ".extracted"
    if extracted_marker.exists():
        print("  [SKIP] Dataset already downloaded and extracted")
        return True

    # Download if needed
    if not zip_path.exists():
        print(f"  [DOWN] Downloading FEMTO dataset (~700 MB)...")
        print(f"         URL: {PRIMARY_URL}")
        try:
            urlretrieve(PRIMARY_URL, zip_path, download_progress)
            print()  # newline after progress
        except URLError as e:
            print(f"\n  [ERR] Download failed: {e}")
            print("  [TIP] Try downloading manually from:")
            print("        - https://www.kaggle.com/datasets/alanhabrony/ieee-phm-2012-data-challenge")
            print("        - https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset")
            return False
    else:
        print(f"  [SKIP] ZIP already exists: {zip_path}")

    # Extract
    print("  [EXTR] Extracting...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            if sample:
                # Extract only one bearing from each condition for sample
                sample_bearings = ["Bearing1_1", "Bearing2_1", "Bearing3_1"]
                for member in zf.namelist():
                    if any(b in member for b in sample_bearings):
                        zf.extract(member, output_dir)
                        print(f"    Extracted: {member}")
            else:
                zf.extractall(output_dir)
        extracted_marker.touch()
        print("  [OK] Extraction complete")
        return True
    except Exception as e:
        print(f"  [ERR] Extraction failed: {e}")
        return False


def verify_femto(output_dir: Path) -> dict:
    """Verify the downloaded FEMTO dataset and return statistics."""
    stats = {
        "bearings": 0,
        "files": 0,
        "total_samples": 0,
        "conditions": [],
    }

    # Look for data directories
    for condition_name, bearings in {**TRAINING_BEARINGS, **TEST_BEARINGS}.items():
        condition_dir = None
        # Search for condition directory (may be nested)
        for root, dirs, files in os.walk(output_dir):
            for d in dirs:
                if condition_name.lower().replace("_", "") in d.lower().replace("_", ""):
                    condition_dir = Path(root) / d
                    break

        if condition_dir and condition_dir.exists():
            stats["conditions"].append(condition_name)

            for bearing in bearings:
                bearing_path = None
                for root, dirs, files in os.walk(condition_dir):
                    if bearing.lower() in Path(root).name.lower():
                        bearing_path = Path(root)
                        break

                if bearing_path and bearing_path.exists():
                    csv_files = list(bearing_path.glob("*.csv"))
                    stats["bearings"] += 1
                    stats["files"] += len(csv_files)

                    # Count samples in first file as estimate
                    if csv_files:
                        try:
                            with open(csv_files[0], 'r') as f:
                                lines = sum(1 for _ in f) - 1  # minus header
                            stats["total_samples"] += lines * len(csv_files)
                        except:
                            pass

    return stats


def load_femto_bearing(data_dir: Path, bearing_name: str):
    """
    Load a single bearing's data.

    Returns:
        dict with keys:
        - 'vibration_h': horizontal vibration (2560 samples per file, 25.6 kHz)
        - 'vibration_v': vertical vibration
        - 'timestamps': list of file timestamps
        - 'metadata': operating condition info
    """
    import numpy as np

    # Find bearing directory
    bearing_path = None
    for root, dirs, files in os.walk(data_dir):
        if bearing_name.lower() in Path(root).name.lower():
            bearing_path = Path(root)
            break

    if not bearing_path:
        raise FileNotFoundError(f"Bearing {bearing_name} not found in {data_dir}")

    csv_files = sorted(bearing_path.glob("*.csv"))

    vibration_h = []
    vibration_v = []
    timestamps = []

    for f in csv_files:
        try:
            data = np.loadtxt(f, delimiter=',', skiprows=0)
            if data.ndim == 2 and data.shape[1] >= 2:
                vibration_h.append(data[:, 0])
                vibration_v.append(data[:, 1])
                timestamps.append(f.stem)
        except Exception as e:
            print(f"  [WARN] Could not load {f}: {e}")

    return {
        'vibration_h': np.array(vibration_h),
        'vibration_v': np.array(vibration_v),
        'timestamps': timestamps,
        'n_files': len(csv_files),
        'sampling_rate': 25600,  # Hz
        'samples_per_file': 2560,
    }


def main():
    parser = argparse.ArgumentParser(description="Download FEMTO/PRONOSTIA Bearing Dataset")
    parser.add_argument("--sample", action="store_true",
                        help="Download only 3 bearings (one per condition) instead of all 17")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent.parent / "data" / "femto"),
                        help="Output directory")
    parser.add_argument("--verify", action="store_true",
                        help="Verify downloaded data")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nFEMTO/PRONOSTIA Bearing Dataset Downloader")
    print("=" * 60)
    print(f"Source: PHM IEEE 2012 Challenge (FEMTO-ST Institute)")
    print(f"Output: {output_dir}")
    print(f"Mode: {'SAMPLE (3 bearings)' if args.sample else 'FULL (17 bearings)'}")
    print("-" * 60)

    success = download_femto(output_dir, args.sample)

    if success and args.verify:
        print("\nVerifying downloaded data...")
        stats = verify_femto(output_dir)
        print(f"  Bearings found: {stats['bearings']}")
        print(f"  Files: {stats['files']}")
        print(f"  Estimated samples: {stats['total_samples']:,}")
        print(f"  Conditions: {stats['conditions']}")

    print("\n" + "=" * 60)
    print("Dataset Summary:")
    print("-" * 60)
    print("  Task: Run-to-failure prognostics (RUL prediction)")
    print("  Bearings: 17 total (6 training, 11 test)")
    print("  Operating conditions: 3 (different speed/load)")
    print("  Channels: 2 (horizontal + vertical vibration)")
    print("  Sampling rate: 25,600 Hz")
    print("  Sample duration: 0.1 seconds every 10 seconds")
    print("  Total size: ~700 MB (full), ~120 MB (sample)")
    print("-" * 60)
    print("Physics groups: MINIMAL (2 vibration channels)")
    print("Best for: RUL prediction, degradation modeling")
    print("=" * 60)


if __name__ == "__main__":
    main()
