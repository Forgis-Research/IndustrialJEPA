"""
Download MFPT (Machinery Failure Prevention Technology) Bearing Dataset.

Bearing fault data from MFPT Society's Condition Based Maintenance Fault Database.
- Baseline + fault conditions (inner race, outer race)
- Real-world fault examples from wind turbines
- High sampling rates: 48,828 Hz and 97,656 Hz

Source: https://www.mfpt.org/fault-data-sets/

Usage:
    python datasets/downloaders/download_mfpt.py
    python datasets/downloaders/download_mfpt.py --verify
"""

import argparse
import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve, Request, urlopen
from urllib.error import URLError
import shutil

# MFPT dataset files
# Note: The MFPT website hosts multiple datasets
MFPT_URLS = {
    "main": "https://www.mfpt.org/wp-content/uploads/2020/02/MFPT-Fault-Data-Sets-20200227T131140Z-001.zip",
}

# Dataset structure
DATASET_INFO = {
    "baseline": {
        "description": "Normal operation",
        "files": 3,
        "load_lbs": 270,
        "shaft_hz": 25,
        "sample_rate": 97656,
        "duration_sec": 6,
    },
    "outer_race": {
        "description": "Outer race fault",
        "files": 3,
        "load_lbs": 270,
        "shaft_hz": 25,
        "sample_rate": 97656,
        "duration_sec": 6,
    },
    "inner_race": {
        "description": "Inner race fault at various loads",
        "files": 7,
        "load_lbs": [0, 50, 100, 150, 200, 250, 300],
        "shaft_hz": 25,
        "sample_rate": 48828,
        "duration_sec": 3,
    },
    "real_world": {
        "description": "Real-world faults from wind turbines",
        "files": 3,
        "examples": [
            "Intermediate shaft bearing",
            "Oil pump shaft bearing",
            "Planet bearing fault",
        ],
    },
}


def download_progress(block_num, block_size, total_size):
    """Progress callback for urlretrieve."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)
    else:
        mb_downloaded = downloaded / (1024 * 1024)
        print(f"\r  Downloaded: {mb_downloaded:.1f} MB", end="", flush=True)


def download_mfpt(output_dir: Path) -> bool:
    """Download the MFPT dataset."""
    zip_path = output_dir / "mfpt_bearing.zip"

    # Check if already extracted
    extracted_marker = output_dir / ".extracted"
    if extracted_marker.exists():
        print("  [SKIP] Dataset already downloaded and extracted")
        return True

    # Download if needed
    if not zip_path.exists():
        print(f"  [DOWN] Downloading MFPT dataset...")
        url = MFPT_URLS["main"]
        print(f"         URL: {url}")

        try:
            # MFPT website may need headers
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urlopen(req) as response:
                total_size = int(response.headers.get('content-length', 0))

                with open(zip_path, 'wb') as f:
                    downloaded = 0
                    block_size = 8192
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        f.write(buffer)
                        downloaded += len(buffer)
                        if total_size > 0:
                            percent = min(100, downloaded * 100 / total_size)
                            print(f"\r  Progress: {percent:.1f}%", end="", flush=True)
                        else:
                            print(f"\r  Downloaded: {downloaded / (1024*1024):.1f} MB", end="", flush=True)
            print()  # newline after progress

        except URLError as e:
            print(f"\n  [ERR] Download failed: {e}")
            print("  [TIP] Try downloading manually from:")
            print("        https://www.mfpt.org/fault-data-sets/")
            return False
    else:
        print(f"  [SKIP] ZIP already exists: {zip_path}")

    # Extract
    print("  [EXTR] Extracting...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(output_dir)
        extracted_marker.touch()
        print("  [OK] Extraction complete")
        return True
    except Exception as e:
        print(f"  [ERR] Extraction failed: {e}")
        return False


def verify_mfpt(output_dir: Path) -> dict:
    """Verify the downloaded MFPT dataset and return statistics."""
    stats = {
        "mat_files": 0,
        "total_samples": 0,
        "categories": [],
    }

    # Find all .mat files
    mat_files = list(output_dir.rglob("*.mat"))
    stats["mat_files"] = len(mat_files)

    # Categorize files
    categories = set()
    for f in mat_files:
        name = f.stem.lower()
        if "baseline" in name or "normal" in name:
            categories.add("baseline")
        elif "inner" in name:
            categories.add("inner_race")
        elif "outer" in name:
            categories.add("outer_race")
        else:
            categories.add("other")

    stats["categories"] = list(categories)

    # Try to load a file to count samples
    if mat_files:
        try:
            import scipy.io
            mat = scipy.io.loadmat(str(mat_files[0]))
            # Find data array
            for key, val in mat.items():
                if not key.startswith("_") and hasattr(val, "shape"):
                    if len(val.shape) >= 1 and val.shape[0] > 1000:
                        stats["sample_example"] = val.shape[0]
                        break
        except ImportError:
            stats["sample_example"] = "scipy not available"
        except Exception as e:
            stats["sample_example"] = f"Error: {e}"

    return stats


def load_mfpt_file(file_path: Path) -> dict:
    """
    Load a single MFPT .mat file.

    The MFPT data structure typically contains:
    - gs: vibration signal (g values)
    - sr: sample rate
    - load: load in pounds
    - rate: shaft rate in Hz

    Returns:
        dict with signal data and metadata
    """
    import scipy.io
    import numpy as np

    mat = scipy.io.loadmat(str(file_path), squeeze_me=True)

    result = {
        'file': file_path.name,
        'signal': None,
        'sample_rate': None,
        'load': None,
        'shaft_rate': None,
    }

    # Extract data based on MFPT structure
    for key, val in mat.items():
        if key.startswith("_"):
            continue

        # The main structure might have nested fields
        if hasattr(val, 'dtype') and val.dtype.names:
            # Structured array
            for field in val.dtype.names:
                field_val = val[field]
                if isinstance(field_val, np.ndarray) and field_val.size > 0:
                    field_val = field_val.item() if field_val.ndim == 0 else field_val

                if field.lower() in ['gs', 'signal', 'data']:
                    result['signal'] = np.array(field_val).flatten()
                elif field.lower() in ['sr', 'samplerate', 'fs']:
                    result['sample_rate'] = int(field_val)
                elif field.lower() == 'load':
                    result['load'] = float(field_val)
                elif field.lower() in ['rate', 'shaft', 'rpm']:
                    result['shaft_rate'] = float(field_val)
        else:
            # Direct array
            if hasattr(val, 'shape') and len(val.shape) >= 1 and val.shape[0] > 100:
                if result['signal'] is None:
                    result['signal'] = np.array(val).flatten()

    return result


def main():
    parser = argparse.ArgumentParser(description="Download MFPT Bearing Dataset")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent.parent / "data" / "mfpt"),
                        help="Output directory")
    parser.add_argument("--verify", action="store_true",
                        help="Verify downloaded data")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nMFPT Bearing Dataset Downloader")
    print("=" * 60)
    print("Source: Society for Machinery Failure Prevention Technology")
    print(f"Output: {output_dir}")
    print("-" * 60)

    success = download_mfpt(output_dir)

    if success and args.verify:
        print("\nVerifying downloaded data...")
        stats = verify_mfpt(output_dir)
        print(f"  MAT files found: {stats['mat_files']}")
        print(f"  Categories: {stats['categories']}")
        if 'sample_example' in stats:
            print(f"  Samples per file (example): {stats['sample_example']}")

    print("\n" + "=" * 60)
    print("Dataset Summary:")
    print("-" * 60)
    print("  Task: Bearing fault classification")
    print("  Conditions:")
    print("    - Baseline: 3 files, 97,656 Hz, 6 sec, 270 lbs")
    print("    - Outer race: 3 files, 97,656 Hz, 6 sec, 270 lbs")
    print("    - Inner race: 7 files, 48,828 Hz, 3 sec, 0-300 lbs")
    print("    - Real-world: 3 files (wind turbine faults)")
    print("  Channels: 1 (vibration)")
    print("  Total size: ~100 MB")
    print("-" * 60)
    print("Physics groups: MINIMAL (single channel)")
    print("Best for: Fault classification, load variation studies")
    print("=" * 60)


if __name__ == "__main__":
    main()
