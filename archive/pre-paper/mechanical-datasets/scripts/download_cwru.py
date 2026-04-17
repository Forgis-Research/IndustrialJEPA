"""
Download CWRU Bearing Dataset

Source: https://engineering.case.edu/bearingdatacenter
The most widely used bearing fault diagnosis benchmark.

Sampling: 12kHz (drive-end) and 48kHz (fan-end)
Bearings: SKF 6205-2RS deep groove ball bearings
Faults: Inner race, outer race, ball faults at 0.007", 0.014", 0.021", 0.028"
Loads: 0, 1, 2, 3 HP (1797, 1772, 1750, 1730 RPM)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from project root .env
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)
import requests
from pathlib import Path
from tqdm import tqdm
import time

# CWRU data files are available at individual URLs
# Base URL pattern (12kHz drive-end data)
BASE_URL = "https://engineering.case.edu/sites/default/files/"

# File mapping: filename -> (fault_type, fault_diameter, load_hp, rpm)
# This is a subset - full list at https://engineering.case.edu/bearingdatacenter/download-data-file
CWRU_FILES = {
    # Normal baseline
    "97.mat": ("normal", 0, 0, 1797),
    "98.mat": ("normal", 0, 1, 1772),
    "99.mat": ("normal", 0, 2, 1750),
    "100.mat": ("normal", 0, 3, 1730),

    # Inner race faults - 0.007"
    "105.mat": ("inner_race", 0.007, 0, 1797),
    "106.mat": ("inner_race", 0.007, 1, 1772),
    "107.mat": ("inner_race", 0.007, 2, 1750),
    "108.mat": ("inner_race", 0.007, 3, 1730),

    # Inner race faults - 0.014"
    "169.mat": ("inner_race", 0.014, 0, 1797),
    "170.mat": ("inner_race", 0.014, 1, 1772),
    "171.mat": ("inner_race", 0.014, 2, 1750),
    "172.mat": ("inner_race", 0.014, 3, 1730),

    # Inner race faults - 0.021"
    "209.mat": ("inner_race", 0.021, 0, 1797),
    "210.mat": ("inner_race", 0.021, 1, 1772),
    "211.mat": ("inner_race", 0.021, 2, 1750),
    "212.mat": ("inner_race", 0.021, 3, 1730),

    # Ball faults - 0.007"
    "118.mat": ("ball", 0.007, 0, 1797),
    "119.mat": ("ball", 0.007, 1, 1772),
    "120.mat": ("ball", 0.007, 2, 1750),
    "121.mat": ("ball", 0.007, 3, 1730),

    # Ball faults - 0.014"
    "185.mat": ("ball", 0.014, 0, 1797),
    "186.mat": ("ball", 0.014, 1, 1772),
    "187.mat": ("ball", 0.014, 2, 1750),
    "188.mat": ("ball", 0.014, 3, 1730),

    # Ball faults - 0.021"
    "222.mat": ("ball", 0.021, 0, 1797),
    "223.mat": ("ball", 0.021, 1, 1772),
    "224.mat": ("ball", 0.021, 2, 1750),
    "225.mat": ("ball", 0.021, 3, 1730),

    # Outer race faults - 0.007" (centered, 6 o'clock)
    "130.mat": ("outer_race", 0.007, 0, 1797),
    "131.mat": ("outer_race", 0.007, 1, 1772),
    "132.mat": ("outer_race", 0.007, 2, 1750),
    "133.mat": ("outer_race", 0.007, 3, 1730),

    # Outer race faults - 0.014"
    "197.mat": ("outer_race", 0.014, 0, 1797),
    "198.mat": ("outer_race", 0.014, 1, 1772),
    "199.mat": ("outer_race", 0.014, 2, 1750),
    "200.mat": ("outer_race", 0.014, 3, 1730),

    # Outer race faults - 0.021"
    "234.mat": ("outer_race", 0.021, 0, 1797),
    "235.mat": ("outer_race", 0.021, 1, 1772),
    "236.mat": ("outer_race", 0.021, 2, 1750),
    "237.mat": ("outer_race", 0.021, 3, 1730),
}


def download_file(url: str, dest: Path, timeout: int = 30) -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(dest, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest.name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True

    except Exception as e:
        print(f"  ERROR downloading {url}: {e}")
        return False


def download_cwru(output_dir: Path, delay: float = 0.5):
    """Download all CWRU files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading CWRU dataset to {output_dir}")
    print(f"Files to download: {len(CWRU_FILES)}")

    success = 0
    failed = []

    for filename, metadata in CWRU_FILES.items():
        dest = output_dir / filename

        if dest.exists():
            print(f"  {filename} already exists, skipping")
            success += 1
            continue

        url = BASE_URL + filename
        print(f"  Downloading {filename}...")

        if download_file(url, dest):
            success += 1
        else:
            failed.append(filename)

        time.sleep(delay)  # Be nice to the server

    print(f"\nDownload complete: {success}/{len(CWRU_FILES)} files")
    if failed:
        print(f"Failed files: {failed}")

    # Save metadata
    import json
    meta_file = output_dir / "cwru_metadata.json"
    with open(meta_file, 'w') as f:
        json.dump({
            fname: {
                "fault_type": meta[0],
                "fault_diameter_in": meta[1],
                "load_hp": meta[2],
                "rpm": meta[3],
                "sampling_rate_hz": 12000,
                "sensor": "drive_end_accelerometer",
            }
            for fname, meta in CWRU_FILES.items()
        }, f, indent=2)
    print(f"Metadata saved to {meta_file}")


if __name__ == "__main__":
    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / "raw" / "cwru"
    download_cwru(output_dir)
