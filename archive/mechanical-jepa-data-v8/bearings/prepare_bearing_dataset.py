"""
Prepare Bearing Fault Datasets from multiple sources.

Supports:
- Paderborn (8 channels, multimodal) - Direct download
- CWRU (2-4 channels, vibration) - Direct download
- IMS (4-8 channels, run-to-failure) - Kaggle (manual download)
- XJTU-SY (2 channels, progressive degradation) - IEEE DataPort (manual download)

Usage:
    # Download Paderborn sample
    python prepare_bearing_dataset.py --download --sample --dataset paderborn

    # Download CWRU sample
    python prepare_bearing_dataset.py --download --sample --dataset cwru

    # Download all auto-downloadable datasets
    python prepare_bearing_dataset.py --download --dataset all

    # Process all available datasets
    python prepare_bearing_dataset.py --process

    # Verify
    python prepare_bearing_dataset.py --verify
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ============================================================
# Paderborn Configuration (Direct download, ~170MB per bearing)
# ============================================================

PADERBORN_URL = "https://groups.uni-paderborn.de/kat/BearingDataCenter/"

PADERBORN_BEARINGS = {
    "K001": "healthy", "K002": "healthy", "K003": "healthy",
    "K004": "healthy", "K005": "healthy", "K006": "healthy",
    "KA01": "outer_race", "KA03": "outer_race", "KA04": "outer_race",
    "KA05": "outer_race", "KA06": "outer_race", "KA07": "outer_race",
    "KA08": "outer_race", "KA09": "outer_race", "KA15": "outer_race",
    "KA16": "outer_race", "KA22": "outer_race", "KA30": "outer_race",
    "KB23": "combined", "KB24": "combined", "KB27": "combined",
    "KI01": "inner_race", "KI03": "inner_race", "KI04": "inner_race",
    "KI05": "inner_race", "KI06": "inner_race", "KI07": "inner_race",
    "KI08": "inner_race", "KI14": "inner_race", "KI16": "inner_race",
    "KI17": "inner_race", "KI18": "inner_race", "KI21": "inner_race",
}

PADERBORN_SAMPLE = ["K001", "KA01", "KI01"]

PADERBORN_CHANNELS = ["a1", "a2", "a3", "v1", "temp1", "torque", "ia", "ib"]

PADERBORN_PHYSICS_GROUPS = {
    "vibration_radial": ["a1", "a2"],
    "vibration_axial": ["a3", "v1"],
    "thermal_torque": ["temp1", "torque"],
    "motor_current": ["ia", "ib"],
}

# ============================================================
# CWRU Configuration (Direct download, ~3MB per file)
# ============================================================

CWRU_FILES = {
    # Normal baseline (12kHz)
    "normal_0": {"url": "97.mat", "fault": "healthy", "load": 0, "rpm": 1797},
    "normal_1": {"url": "98.mat", "fault": "healthy", "load": 1, "rpm": 1772},
    "normal_2": {"url": "99.mat", "fault": "healthy", "load": 2, "rpm": 1750},
    "normal_3": {"url": "100.mat", "fault": "healthy", "load": 3, "rpm": 1730},
    # Inner race 0.007"
    "IR007_0": {"url": "105.mat", "fault": "inner_race", "load": 0, "size": 0.007, "rpm": 1797},
    "IR007_1": {"url": "106.mat", "fault": "inner_race", "load": 1, "size": 0.007, "rpm": 1772},
    "IR007_2": {"url": "107.mat", "fault": "inner_race", "load": 2, "size": 0.007, "rpm": 1750},
    "IR007_3": {"url": "108.mat", "fault": "inner_race", "load": 3, "size": 0.007, "rpm": 1730},
    # Inner race 0.014"
    "IR014_0": {"url": "169.mat", "fault": "inner_race", "load": 0, "size": 0.014, "rpm": 1797},
    "IR014_1": {"url": "170.mat", "fault": "inner_race", "load": 1, "size": 0.014, "rpm": 1772},
    "IR014_2": {"url": "171.mat", "fault": "inner_race", "load": 2, "size": 0.014, "rpm": 1750},
    "IR014_3": {"url": "172.mat", "fault": "inner_race", "load": 3, "size": 0.014, "rpm": 1730},
    # Inner race 0.021"
    "IR021_0": {"url": "209.mat", "fault": "inner_race", "load": 0, "size": 0.021, "rpm": 1797},
    "IR021_1": {"url": "210.mat", "fault": "inner_race", "load": 1, "size": 0.021, "rpm": 1772},
    "IR021_2": {"url": "211.mat", "fault": "inner_race", "load": 2, "size": 0.021, "rpm": 1750},
    "IR021_3": {"url": "212.mat", "fault": "inner_race", "load": 3, "size": 0.021, "rpm": 1730},
    # Ball 0.007"
    "B007_0": {"url": "118.mat", "fault": "ball", "load": 0, "size": 0.007, "rpm": 1797},
    "B007_1": {"url": "119.mat", "fault": "ball", "load": 1, "size": 0.007, "rpm": 1772},
    "B007_2": {"url": "120.mat", "fault": "ball", "load": 2, "size": 0.007, "rpm": 1750},
    "B007_3": {"url": "121.mat", "fault": "ball", "load": 3, "size": 0.007, "rpm": 1730},
    # Ball 0.014"
    "B014_0": {"url": "185.mat", "fault": "ball", "load": 0, "size": 0.014, "rpm": 1797},
    "B014_1": {"url": "186.mat", "fault": "ball", "load": 1, "size": 0.014, "rpm": 1772},
    "B014_2": {"url": "187.mat", "fault": "ball", "load": 2, "size": 0.014, "rpm": 1750},
    "B014_3": {"url": "188.mat", "fault": "ball", "load": 3, "size": 0.014, "rpm": 1730},
    # Ball 0.021"
    "B021_0": {"url": "222.mat", "fault": "ball", "load": 0, "size": 0.021, "rpm": 1797},
    "B021_1": {"url": "223.mat", "fault": "ball", "load": 1, "size": 0.021, "rpm": 1772},
    "B021_2": {"url": "224.mat", "fault": "ball", "load": 2, "size": 0.021, "rpm": 1750},
    "B021_3": {"url": "225.mat", "fault": "ball", "load": 3, "size": 0.021, "rpm": 1730},
    # Outer race 0.007" (centered)
    "OR007_0": {"url": "130.mat", "fault": "outer_race", "load": 0, "size": 0.007, "rpm": 1797},
    "OR007_1": {"url": "131.mat", "fault": "outer_race", "load": 1, "size": 0.007, "rpm": 1772},
    "OR007_2": {"url": "132.mat", "fault": "outer_race", "load": 2, "size": 0.007, "rpm": 1750},
    "OR007_3": {"url": "133.mat", "fault": "outer_race", "load": 3, "size": 0.007, "rpm": 1730},
    # Outer race 0.014"
    "OR014_0": {"url": "197.mat", "fault": "outer_race", "load": 0, "size": 0.014, "rpm": 1797},
    "OR014_1": {"url": "198.mat", "fault": "outer_race", "load": 1, "size": 0.014, "rpm": 1772},
    "OR014_2": {"url": "199.mat", "fault": "outer_race", "load": 2, "size": 0.014, "rpm": 1750},
    "OR014_3": {"url": "200.mat", "fault": "outer_race", "load": 3, "size": 0.014, "rpm": 1730},
    # Outer race 0.021"
    "OR021_0": {"url": "234.mat", "fault": "outer_race", "load": 0, "size": 0.021, "rpm": 1797},
    "OR021_1": {"url": "235.mat", "fault": "outer_race", "load": 1, "size": 0.021, "rpm": 1772},
    "OR021_2": {"url": "236.mat", "fault": "outer_race", "load": 2, "size": 0.021, "rpm": 1750},
    "OR021_3": {"url": "237.mat", "fault": "outer_race", "load": 3, "size": 0.021, "rpm": 1730},
}

CWRU_SAMPLE = ["normal_0", "IR007_0", "B007_0", "OR007_0"]

CWRU_CHANNELS = ["DE", "FE", "BA"]  # Drive End, Fan End, Base Accelerometer

# ============================================================
# IMS Configuration (Kaggle download required)
# ============================================================

IMS_INFO = {
    "url": "https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset",
    "description": "NASA IMS Bearing Run-to-Failure Dataset",
    "sets": {
        "1st_test": {"channels": 8, "files": 2156, "failure": "Bearing 3 inner race, Bearing 4 roller"},
        "2nd_test": {"channels": 4, "files": 984, "failure": "Bearing 1 outer race"},
        "3rd_test": {"channels": 4, "files": 4448, "failure": "Bearing 3 outer race"},
    },
    "sampling_rate": 20000,
    "samples_per_file": 20480,
}

IMS_CHANNELS = {
    "1st_test": ["b1_x", "b1_y", "b2_x", "b2_y", "b3_x", "b3_y", "b4_x", "b4_y"],
    "2nd_test": ["b1", "b2", "b3", "b4"],
    "3rd_test": ["b1", "b2", "b3", "b4"],
}

# ============================================================
# XJTU-SY Configuration (IEEE DataPort download required)
# ============================================================

XJTU_INFO = {
    "url": "https://ieee-dataport.org/open-access/xjtu-sy-bearing-datasets",
    "description": "XJTU-SY Bearing Degradation Dataset",
    "conditions": {
        "35Hz12kN": {"rpm": 2100, "load_kn": 12, "bearings": 5},
        "37.5Hz11kN": {"rpm": 2250, "load_kn": 11, "bearings": 5},
        "40Hz10kN": {"rpm": 2400, "load_kn": 10, "bearings": 5},
    },
    "sampling_rate": 25600,
    "samples_per_file": 32768,
    "channels": ["horizontal", "vertical"],
}

# ============================================================
# Fault Labels
# ============================================================

FAULT_LABELS = {
    "healthy": 0,
    "outer_race": 1,
    "inner_race": 2,
    "ball": 3,
    "combined": 4,
}


def check_unrar() -> Optional[str]:
    """Check if unrar or 7z is available."""
    import platform

    # Standard commands to try
    commands = ["unrar", "7z", "bsdtar"]

    # On Windows, also check common 7-Zip installation paths
    if platform.system() == "Windows":
        win_paths = [
            r"C:\Program Files\7-Zip\7z.exe",
            r"C:\Program Files (x86)\7-Zip\7z.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\7-Zip\7z.exe"),
        ]
        for path in win_paths:
            if os.path.exists(path):
                return path

    for cmd in commands:
        try:
            subprocess.run([cmd], capture_output=True)
            return cmd
        except FileNotFoundError:
            continue
    return None


def extract_paderborn_rar(rar_path: Path, output_dir: Path, unrar_cmd: str) -> bool:
    """Extract a single Paderborn RAR file."""
    bearing_name = rar_path.stem
    bearing_dir = output_dir / bearing_name

    if bearing_dir.exists() and any(bearing_dir.iterdir()):
        return True  # Already extracted

    print(f"  {bearing_name}: Extracting...")
    try:
        if "7z" in unrar_cmd.lower():
            # 7z syntax (works for 7z.exe on Windows too)
            result = subprocess.run(
                [unrar_cmd, "x", f"-o{output_dir}", str(rar_path), "-y"],
                capture_output=True, text=True
            )
        elif unrar_cmd == "unrar":
            result = subprocess.run(
                ["unrar", "x", "-o+", str(rar_path), str(output_dir) + "/"],
                capture_output=True, text=True
            )
        elif unrar_cmd == "bsdtar":
            bearing_dir.mkdir(exist_ok=True)
            result = subprocess.run(
                ["bsdtar", "-xf", str(rar_path), "-C", str(bearing_dir)],
                capture_output=True, text=True
            )
        else:
            print(f"  {bearing_name}: Unknown extraction tool {unrar_cmd}")
            return False

        if result.returncode != 0:
            print(f"  {bearing_name}: Extraction failed - {result.stderr[:200]}")
            return False
        return True
    except Exception as e:
        print(f"  {bearing_name}: Extraction error - {e}")
        return False


# ============================================================
# Paderborn Download/Processing
# ============================================================

def download_paderborn(output_dir: Path, sample: bool = False) -> List[str]:
    """Download Paderborn bearing dataset."""
    raw_dir = output_dir / "raw" / "paderborn"
    raw_dir.mkdir(parents=True, exist_ok=True)

    bearings = PADERBORN_SAMPLE if sample else list(PADERBORN_BEARINGS.keys())
    print(f"Downloading Paderborn: {len(bearings)} bearings...")

    unrar_cmd = check_unrar()
    if not unrar_cmd:
        print("  WARNING: No RAR extraction tool found. Install unrar, 7z, or bsdtar.")
        print("           Files will be downloaded but not extracted.")

    downloaded = []
    for bearing in bearings:
        filename = f"{bearing}.rar"
        url = PADERBORN_URL + filename
        output_path = raw_dir / filename

        if output_path.exists():
            print(f"  {bearing}: Already downloaded")
            downloaded.append(bearing)
        else:
            print(f"  {bearing}: Downloading (~170MB)...")
            try:
                urlretrieve(url, output_path)
                downloaded.append(bearing)
            except URLError as e:
                print(f"  {bearing}: FAILED - {e}")
                continue

        # Extract if possible
        if unrar_cmd:
            extract_paderborn_rar(output_path, raw_dir, unrar_cmd)

    return downloaded


def extract_all_paderborn(output_dir: Path) -> int:
    """Extract all downloaded Paderborn RAR files."""
    raw_dir = output_dir / "raw" / "paderborn"
    if not raw_dir.exists():
        print("Paderborn directory not found. Run --download --dataset paderborn first.")
        return 0

    unrar_cmd = check_unrar()
    if not unrar_cmd:
        print("ERROR: No RAR extraction tool found.")
        print("  Windows: Install 7-Zip from https://www.7-zip.org/")
        print("  Linux:   sudo apt-get install unrar  OR  sudo apt-get install p7zip-full")
        print("  macOS:   brew install unar  OR  brew install p7zip")
        return 0

    print(f"Using extraction tool: {unrar_cmd}")

    rar_files = list(raw_dir.glob("*.rar"))
    if not rar_files:
        print("No RAR files found in Paderborn directory.")
        return 0

    print(f"Found {len(rar_files)} RAR files to extract...")
    extracted = 0
    for rar_path in sorted(rar_files):
        if extract_paderborn_rar(rar_path, raw_dir, unrar_cmd):
            extracted += 1

    print(f"Extracted {extracted}/{len(rar_files)} bearings.")
    return extracted


def process_paderborn(output_dir: Path) -> List[Dict]:
    """Process Paderborn MAT files."""
    raw_dir = output_dir / "raw" / "paderborn"
    if not raw_dir.exists():
        return []

    try:
        from scipy.io import loadmat
    except ImportError:
        print("  scipy not installed, skipping Paderborn")
        return []

    episodes = []
    bearing_dirs = [d for d in raw_dir.iterdir() if d.is_dir() and d.name in PADERBORN_BEARINGS]

    if not bearing_dirs:
        print("  Paderborn: No extracted directories found")
        return []

    for bearing_dir in bearing_dirs:
        bearing = bearing_dir.name
        fault_type = PADERBORN_BEARINGS[bearing]

        for mat_file in bearing_dir.rglob("*.mat"):
            try:
                data = loadmat(str(mat_file), squeeze_me=True, struct_as_record=False)

                # Extract sensor data (Paderborn structure has .Y attribute)
                sensor_data = None
                for key in data.keys():
                    if not key.startswith('_'):
                        val = data[key]
                        if hasattr(val, 'Y'):
                            sensor_data = val.Y
                            break

                if sensor_data is not None and sensor_data.ndim == 2:
                    n_channels = min(sensor_data.shape[1], 8)
                    episodes.append({
                        'dataset': 'paderborn',
                        'bearing_id': bearing,
                        'measurement_id': mat_file.stem,
                        'fault_type': fault_type,
                        'fault_label': FAULT_LABELS[fault_type],
                        'n_samples': sensor_data.shape[0],
                        'n_channels': n_channels,
                        'sampling_rate': 64000,
                        'channels': PADERBORN_CHANNELS[:n_channels],
                    })
            except Exception as e:
                pass  # Silently skip problematic files

    print(f"  Paderborn: {len(episodes)} measurements from {len(bearing_dirs)} bearings")
    return episodes


# ============================================================
# CWRU Download/Processing
# ============================================================

def download_cwru(output_dir: Path, sample: bool = False) -> List[str]:
    """Download CWRU bearing dataset."""
    raw_dir = output_dir / "raw" / "cwru"
    raw_dir.mkdir(parents=True, exist_ok=True)

    files = CWRU_SAMPLE if sample else list(CWRU_FILES.keys())
    print(f"Downloading CWRU: {len(files)} files...")

    downloaded = []
    for file_id in files:
        info = CWRU_FILES[file_id]
        filename = info['url']
        url = f"https://engineering.case.edu/sites/default/files/{filename}"
        output_path = raw_dir / f"{file_id}.mat"

        if output_path.exists():
            print(f"  {file_id}: Already downloaded")
            downloaded.append(file_id)
            continue

        print(f"  {file_id}: Downloading...")
        try:
            urlretrieve(url, output_path)
            downloaded.append(file_id)
        except URLError as e:
            print(f"  {file_id}: FAILED - {e}")

    return downloaded


def process_cwru(output_dir: Path) -> List[Dict]:
    """Process CWRU MAT files."""
    raw_dir = output_dir / "raw" / "cwru"
    if not raw_dir.exists():
        return []

    try:
        from scipy.io import loadmat
    except ImportError:
        print("  scipy not installed, skipping CWRU")
        return []

    episodes = []
    for file_id, info in CWRU_FILES.items():
        mat_file = raw_dir / f"{file_id}.mat"
        if not mat_file.exists():
            continue

        try:
            data = loadmat(str(mat_file), squeeze_me=True)

            # CWRU files have keys like 'X097_DE_time', 'X097_FE_time', 'X097_BA_time'
            channels = set()
            n_samples = 0
            for key in data.keys():
                if '_DE_time' in key:
                    channels.add('DE')
                    n_samples = max(n_samples, len(data[key]))
                elif '_FE_time' in key:
                    channels.add('FE')
                    n_samples = max(n_samples, len(data[key]))
                elif '_BA_time' in key:
                    channels.add('BA')
                    n_samples = max(n_samples, len(data[key]))
            channels = sorted(list(channels))

            if channels:
                episodes.append({
                    'dataset': 'cwru',
                    'bearing_id': file_id,
                    'measurement_id': file_id,
                    'fault_type': info['fault'],
                    'fault_label': FAULT_LABELS[info['fault']],
                    'n_samples': n_samples,
                    'n_channels': len(channels),
                    'sampling_rate': 12000,
                    'channels': channels,
                    'load_hp': info.get('load'),
                    'fault_size_inch': info.get('size'),
                    'rpm': info.get('rpm'),
                })
        except Exception as e:
            print(f"    Failed to load {file_id}: {e}")

    print(f"  CWRU: {len(episodes)} files")
    return episodes


# ============================================================
# IMS Download/Processing (Manual download from Kaggle)
# ============================================================

def download_ims(output_dir: Path, sample: bool = False) -> List[str]:
    """Download IMS dataset via Kaggle CLI or provide manual instructions."""
    raw_dir = output_dir / "raw" / "ims"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    test_dirs = ["1st_test", "2nd_test", "3rd_test"]
    found = [d for d in test_dirs if (raw_dir / d).exists()]
    if found:
        print(f"IMS: Already downloaded: {found}")
        return found

    # Try Kaggle CLI first
    try:
        result = subprocess.run(["kaggle", "--version"], capture_output=True)
        if result.returncode == 0:
            print("IMS: Downloading via Kaggle CLI...")
            subprocess.run([
                "kaggle", "datasets", "download",
                "-d", "vinayak123tyagi/bearing-dataset",
                "-p", str(raw_dir), "--unzip"
            ], check=True)
            # Check if download succeeded
            found = [d for d in test_dirs if (raw_dir / d).exists()]
            if found:
                print(f"  Downloaded: {found}")
                return found
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Manual instructions
    print("IMS Dataset (Kaggle):")
    print(f"  URL: {IMS_INFO['url']}")
    print("\n  Option A - Kaggle CLI (recommended for VMs):")
    print("    pip install kaggle")
    print("    # Get API key from kaggle.com → Account → Create API Token")
    print("    # Save to ~/.kaggle/kaggle.json")
    print("    kaggle datasets download -d vinayak123tyagi/bearing-dataset")
    print(f"    unzip bearing-dataset.zip -d {raw_dir}")
    print("\n  Option B - Manual download:")
    print("    1. Download ZIP from Kaggle website (~1.67 GB)")
    print(f"    2. Extract to: {raw_dir}")
    print("\n  Expected structure:")
    print("       raw/ims/1st_test/*.txt")
    print("       raw/ims/2nd_test/*.txt")
    print("       raw/ims/3rd_test/*.txt")
    print("  Then run: python prepare_bearing_dataset.py --process")
    return []


def process_ims(output_dir: Path) -> List[Dict]:
    """Process IMS text files."""
    raw_dir = output_dir / "raw" / "ims"
    if not raw_dir.exists():
        return []

    episodes = []
    test_dirs = ["1st_test", "2nd_test", "3rd_test"]

    for test_name in test_dirs:
        test_dir = raw_dir / test_name
        if not test_dir.exists():
            continue

        channels = IMS_CHANNELS[test_name]
        # IMS files have no extension, names are timestamps like "2003.10.22.12.06.24"
        txt_files = sorted([f for f in test_dir.iterdir()
                           if f.is_file() and f.name[0].isdigit()])

        if not txt_files:
            # Try .txt extension as fallback
            txt_files = sorted(test_dir.glob("*.txt"))

        for txt_file in txt_files:
            try:
                # IMS files are space-delimited, one column per channel
                data = np.loadtxt(txt_file)
                if data.ndim == 1:
                    data = data.reshape(-1, 1)

                n_samples, n_channels = data.shape

                # Determine fault label based on test and file position
                # Early files = healthy, late files = degraded/faulty
                file_idx = int(txt_file.stem) if txt_file.stem.isdigit() else 0
                total_files = IMS_INFO["sets"][test_name]["files"]

                # Last 10% of files considered "faulty" for this test
                if file_idx > total_files * 0.9:
                    fault_type = "outer_race" if "outer" in IMS_INFO["sets"][test_name]["failure"].lower() else "inner_race"
                else:
                    fault_type = "healthy"

                episodes.append({
                    'dataset': 'ims',
                    'bearing_id': f"ims_{test_name}",
                    'measurement_id': txt_file.stem,
                    'fault_type': fault_type,
                    'fault_label': FAULT_LABELS.get(fault_type, 0),
                    'n_samples': n_samples,
                    'n_channels': n_channels,
                    'sampling_rate': 20000,
                    'channels': channels[:n_channels],
                    'test_set': test_name,
                })
            except Exception as e:
                pass  # Skip problematic files

    print(f"  IMS: {len(episodes)} files")
    return episodes


# ============================================================
# XJTU-SY Download/Processing (Manual download from IEEE)
# ============================================================

def download_xjtu(output_dir: Path, sample: bool = False) -> List[str]:
    """Provide instructions for XJTU-SY download (IEEE requires registration)."""
    raw_dir = output_dir / "raw" / "xjtu"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("XJTU-SY Dataset (IEEE DataPort - requires registration):")
    print(f"  URL: {XJTU_INFO['url']}")
    print("  Steps:")
    print("    1. Create free IEEE account")
    print("    2. Download the dataset")
    print(f"    3. Extract to: {raw_dir}")
    print("    4. Expected structure:")
    print("       raw/xjtu/35Hz12kN/Bearing1_1/*.csv")
    print("       raw/xjtu/37.5Hz11kN/Bearing2_1/*.csv")
    print("       raw/xjtu/40Hz10kN/Bearing3_1/*.csv")
    print("    5. Run --process to include XJTU-SY data")

    # Check if already downloaded
    found = [d.name for d in raw_dir.iterdir() if d.is_dir()]
    if found:
        print(f"  Found existing: {found}")
        return found
    return []


def process_xjtu(output_dir: Path) -> List[Dict]:
    """Process XJTU-SY CSV files."""
    raw_dir = output_dir / "raw" / "xjtu"
    if not raw_dir.exists():
        return []

    episodes = []
    channels = XJTU_INFO["channels"]

    for condition_dir in raw_dir.iterdir():
        if not condition_dir.is_dir():
            continue

        condition = condition_dir.name
        for bearing_dir in condition_dir.iterdir():
            if not bearing_dir.is_dir():
                continue

            csv_files = sorted(bearing_dir.glob("*.csv"))
            total_files = len(csv_files)

            for i, csv_file in enumerate(csv_files):
                try:
                    data = pd.read_csv(csv_file, header=None).values
                    n_samples, n_channels = data.shape

                    # Last 10% considered degraded
                    fault_type = "inner_race" if i > total_files * 0.9 else "healthy"

                    episodes.append({
                        'dataset': 'xjtu',
                        'bearing_id': bearing_dir.name,
                        'measurement_id': csv_file.stem,
                        'fault_type': fault_type,
                        'fault_label': FAULT_LABELS.get(fault_type, 0),
                        'n_samples': n_samples,
                        'n_channels': n_channels,
                        'sampling_rate': 25600,
                        'channels': channels[:n_channels],
                        'condition': condition,
                    })
                except Exception:
                    pass

    print(f"  XJTU-SY: {len(episodes)} files")
    return episodes


# ============================================================
# Main Processing
# ============================================================

def process_all(output_dir: Path, window_size: int = 4096, stride: int = 2048):
    """Process all downloaded datasets into unified format."""
    print("Processing all datasets...")

    all_episodes = []

    # Process each dataset
    all_episodes.extend(process_paderborn(output_dir))
    all_episodes.extend(process_cwru(output_dir))
    all_episodes.extend(process_ims(output_dir))
    all_episodes.extend(process_xjtu(output_dir))

    if not all_episodes:
        print("No data found. Run --download first.")
        return

    # Save episodes
    episodes_df = pd.DataFrame(all_episodes)
    episodes_file = output_dir / "bearing_episodes.parquet"
    episodes_df.to_parquet(episodes_file, index=False)
    print(f"\nSaved {len(episodes_df)} episodes to {episodes_file}")

    # Compute statistics
    stats = {
        'n_episodes': len(all_episodes),
        'datasets': list(episodes_df['dataset'].unique()),
        'fault_distribution': episodes_df['fault_type'].value_counts().to_dict(),
        'by_dataset': {},
        'window_size': window_size,
        'stride': stride,
    }

    for ds in episodes_df['dataset'].unique():
        ds_df = episodes_df[episodes_df['dataset'] == ds]
        stats['by_dataset'][ds] = {
            'n_episodes': len(ds_df),
            'n_samples_total': int(ds_df['n_samples'].sum()),
            'n_channels': int(ds_df['n_channels'].iloc[0]) if len(ds_df) > 0 else 0,
            'sampling_rate': int(ds_df['sampling_rate'].iloc[0]) if len(ds_df) > 0 else 0,
            'fault_types': ds_df['fault_type'].unique().tolist(),
        }

    stats_file = output_dir / "statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_file}")

    # Print summary
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    for ds, ds_stats in stats['by_dataset'].items():
        total_samples = ds_stats['n_samples_total']
        duration_sec = total_samples / ds_stats['sampling_rate'] if ds_stats['sampling_rate'] > 0 else 0
        print(f"\n{ds.upper()}:")
        print(f"  Episodes: {ds_stats['n_episodes']}")
        print(f"  Samples: {total_samples:,} ({duration_sec/3600:.1f} hours)")
        print(f"  Channels: {ds_stats['n_channels']}")
        print(f"  Sampling: {ds_stats['sampling_rate']/1000:.0f} kHz")
        print(f"  Faults: {ds_stats['fault_types']}")
    print("="*60)


def verify_dataset(output_dir: Path):
    """Verify dataset integrity."""
    print("Verifying dataset...")

    for f in ["bearing_episodes.parquet", "statistics.json"]:
        path = output_dir / f
        if not path.exists():
            print(f"  MISSING: {f}")
        else:
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"  OK: {f} ({size_mb:.2f} MB)")

    stats_file = output_dir / "statistics.json"
    if stats_file.exists():
        with open(stats_file) as f:
            stats = json.load(f)
        print(f"\nDataset Statistics:")
        print(f"  Total episodes: {stats['n_episodes']}")
        print(f"  Datasets: {stats['datasets']}")
        print(f"  Fault distribution: {stats['fault_distribution']}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Bearing Fault Datasets")
    parser.add_argument('--download', action='store_true', help='Download raw data')
    parser.add_argument('--sample', action='store_true', help='Download sample only')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['paderborn', 'cwru', 'ims', 'xjtu', 'all'],
                        help='Dataset to download')
    parser.add_argument('--extract', action='store_true',
                        help='Extract Paderborn RAR files (requires unrar/7z)')
    parser.add_argument('--process', action='store_true', help='Process into unified format')
    parser.add_argument('--verify', action='store_true', help='Verify dataset')
    parser.add_argument('--all', action='store_true', help='Run all steps')

    args = parser.parse_args()
    output_dir = Path(__file__).parent

    if args.all or args.download:
        if args.dataset in ['paderborn', 'all']:
            download_paderborn(output_dir, sample=args.sample)
        if args.dataset in ['cwru', 'all']:
            download_cwru(output_dir, sample=args.sample)
        if args.dataset in ['ims', 'all']:
            download_ims(output_dir, sample=args.sample)
        if args.dataset in ['xjtu', 'all']:
            download_xjtu(output_dir, sample=args.sample)

    if args.extract:
        extract_all_paderborn(output_dir)

    if args.all or args.process:
        process_all(output_dir)

    if args.all or args.verify:
        verify_dataset(output_dir)

    if not any([args.download, args.extract, args.process, args.verify, args.all]):
        parser.print_help()


if __name__ == "__main__":
    main()
