"""
Download CWRU (Case Western Reserve University) Bearing Dataset.

Data is in MATLAB .mat format. No registration required.
Saves to datasets/data/cwru/

Usage:
    python datasets/downloaders/download_cwru.py
    python datasets/downloaders/download_cwru.py --sample   # Download only 4 representative files
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve, urlopen
from urllib.error import URLError

# ---------------------------------------------------------------------------
# File manifest: (local_filename, remote_id, condition, fault, load_hp, description)
# Correct URL pattern: https://engineering.case.edu/sites/default/files/{id}.mat
# ---------------------------------------------------------------------------
BASE_URL = "https://engineering.case.edu/sites/default/files/"

FULL_MANIFEST = [
    # Normal baseline
    ("normal_0hp.mat", "97",   "normal", "none",       0, "Normal baseline, 0 HP"),
    ("normal_1hp.mat", "98",   "normal", "none",       1, "Normal baseline, 1 HP"),
    ("normal_2hp.mat", "99",   "normal", "none",       2, "Normal baseline, 2 HP"),
    ("normal_3hp.mat", "100",  "normal", "none",       3, "Normal baseline, 3 HP"),
    # 12k Drive End — Inner Race faults (0.007")
    ("IR007_0hp.mat",  "105",  "12k_DE", "inner_race", 0, "Inner race, 0.007in, 0HP"),
    ("IR007_1hp.mat",  "106",  "12k_DE", "inner_race", 1, "Inner race, 0.007in, 1HP"),
    ("IR007_2hp.mat",  "107",  "12k_DE", "inner_race", 2, "Inner race, 0.007in, 2HP"),
    ("IR007_3hp.mat",  "108",  "12k_DE", "inner_race", 3, "Inner race, 0.007in, 3HP"),
    # Inner Race faults (0.014")
    ("IR014_0hp.mat",  "169",  "12k_DE", "inner_race", 0, "Inner race, 0.014in, 0HP"),
    # Inner Race faults (0.021")
    ("IR021_0hp.mat",  "209",  "12k_DE", "inner_race", 0, "Inner race, 0.021in, 0HP"),
    # 12k Drive End — Ball faults (0.007")
    ("B007_0hp.mat",   "118",  "12k_DE", "ball",       0, "Ball fault, 0.007in, 0HP"),
    ("B007_1hp.mat",   "119",  "12k_DE", "ball",       1, "Ball fault, 0.007in, 1HP"),
    ("B007_2hp.mat",   "120",  "12k_DE", "ball",       2, "Ball fault, 0.007in, 2HP"),
    # Ball faults (0.014", 0.021")
    ("B014_0hp.mat",   "185",  "12k_DE", "ball",       0, "Ball fault, 0.014in, 0HP"),
    ("B021_0hp.mat",   "222",  "12k_DE", "ball",       0, "Ball fault, 0.021in, 0HP"),
    # 12k Drive End — Outer Race faults @6:00 (0.007")
    ("OR007_6_0hp.mat","130",  "12k_DE", "outer_race", 0, "Outer race, 0.007in, @6, 0HP"),
    ("OR007_6_1hp.mat","131",  "12k_DE", "outer_race", 1, "Outer race, 0.007in, @6, 1HP"),
    # Outer Race faults @6:00 (0.014", 0.021")
    ("OR014_6_0hp.mat","197",  "12k_DE", "outer_race", 0, "Outer race, 0.014in, @6, 0HP"),
    ("OR021_6_0hp.mat","234",  "12k_DE", "outer_race", 0, "Outer race, 0.021in, @6, 0HP"),
]

# Minimal sample: 1 normal + 1 each of inner/ball/outer race fault at 0 HP
SAMPLE_MANIFEST = [
    ("normal_0hp.mat", "97",  "normal", "none",       0, "Normal baseline, 0 HP"),
    ("IR007_0hp.mat",  "105", "12k_DE", "inner_race", 0, "Inner race, 0.007in, 0HP"),
    ("B007_0hp.mat",   "118", "12k_DE", "ball",       0, "Ball fault, 0.007in, 0HP"),
    ("OR007_6_0hp.mat","130", "12k_DE", "outer_race", 0, "Outer race, 0.007in, @6, 0HP"),
]


def download_file(url: str, dest: Path, verbose: bool = True) -> bool:
    """Download a single file with progress reporting."""
    if dest.exists():
        if verbose:
            print(f"  [SKIP] {dest.name} already exists")
        return True

    try:
        if verbose:
            print(f"  [DOWN] {dest.name} from {url}")
        urlretrieve(url, dest)
        size_kb = dest.stat().st_size / 1024
        if verbose:
            print(f"         -> {size_kb:.0f} KB saved")
        return True
    except URLError as e:
        print(f"  [ERR]  Failed to download {dest.name}: {e}")
        return False


def verify_mat_file(path: Path) -> dict:
    """Load a .mat file and return basic stats."""
    try:
        import scipy.io
        mat = scipy.io.loadmat(str(path))
        # Find the data keys (skip metadata keys that start with '__')
        data_keys = [k for k in mat.keys() if not k.startswith("__")]
        total_samples = 0
        channels = []
        for k in data_keys:
            arr = mat[k]
            if hasattr(arr, "shape") and len(arr.shape) >= 1:
                total_samples = max(total_samples, arr.shape[0])
                channels.append(k)
        return {
            "keys": data_keys,
            "channels": channels,
            "n_samples": total_samples,
            "status": "ok",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Download CWRU Bearing Dataset")
    parser.add_argument("--sample", action="store_true",
                        help="Download only 4 representative files instead of full dataset")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent.parent / "data" / "cwru"),
                        help="Output directory")
    parser.add_argument("--verify", action="store_true",
                        help="Verify downloaded files by loading them")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = SAMPLE_MANIFEST if args.sample else FULL_MANIFEST
    mode = "SAMPLE" if args.sample else "FULL"

    print(f"\nCWRU Bearing Dataset Downloader")
    print(f"Mode: {mode} ({len(manifest)} files)")
    print(f"Output: {output_dir}")
    print("-" * 60)

    success_count = 0
    for local_name, file_id, condition, fault, load_hp, desc in manifest:
        url = BASE_URL + file_id + ".mat"
        dest = output_dir / local_name
        if download_file(url, dest):
            success_count += 1

    print(f"\nDownloaded {success_count}/{len(manifest)} files")

    if args.verify and success_count > 0:
        print("\nVerifying downloaded files...")
        for local_name, _, condition, fault, load_hp, desc in manifest:
            path = output_dir / local_name
            if path.exists():
                info = verify_mat_file(path)
                if info["status"] == "ok":
                    print(f"  [OK] {local_name}: {info['n_samples']} samples, "
                          f"channels={info['channels'][:3]}")
                else:
                    print(f"  [ERR] {local_name}: {info.get('error')}")

    # Print summary statistics
    print("\nSummary:")
    print(f"  Dataset: CWRU Bearing Fault")
    print(f"  Sampling rate: 12,000 Hz (standard) or 48,000 Hz (high-res)")
    print(f"  Channels per file: 2–4 (DE, FE, BA accelerometers)")
    print(f"  Samples per file: 20,480–122,571")
    print(f"  Fault types: normal, inner_race, ball, outer_race")
    print(f"  Load conditions: 0–3 HP")
    print(f"\nPhysics grouping: MINIMAL (2-4 vibration channels only)")
    print(f"Tier 2 recommendation: NOT RECOMMENDED (too few channels)")
    print(f"\nData saved to: {output_dir}")


if __name__ == "__main__":
    main()
