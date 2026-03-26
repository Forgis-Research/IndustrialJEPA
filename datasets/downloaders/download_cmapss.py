"""
Download C-MAPSS (NASA CMAPSS Turbofan Engine) Dataset.

4 sub-datasets (FD001–FD004), 21 sensors + 3 operational settings.
Uses Kaggle mirror (requires kaggle API) or direct NASA URL.
Saves to datasets/data/cmapss/

Usage:
    python datasets/downloaders/download_cmapss.py
    python datasets/downloaders/download_cmapss.py --sample  # FD001 only
    python datasets/downloaders/download_cmapss.py --source kaggle
    python datasets/downloaders/download_cmapss.py --source nasa
"""

import argparse
import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

# Column names for C-MAPSS
COLUMN_NAMES = (
    ["unit_id", "cycle", "setting1", "setting2", "setting3"]
    + [f"s{i}" for i in range(1, 22)]
)

# Sensor name mapping (physical meaning)
SENSOR_DESCRIPTIONS = {
    "s1":  "T2 - Total temperature at fan inlet (°R)",
    "s2":  "T24 - Total temperature at LPC outlet (°R)",
    "s3":  "T30 - Total temperature at HPC outlet (°R)",
    "s4":  "T50 - Total temperature at LPT outlet (°R)",
    "s5":  "P2 - Pressure at fan inlet (psia)",
    "s6":  "P15 - Total pressure in bypass-duct (psia)",
    "s7":  "P30 - Total pressure at HPC outlet (psia)",
    "s8":  "Nf - Physical fan speed (rpm)",
    "s9":  "Nc - Physical core speed (rpm)",
    "s10": "epr - Engine pressure ratio (P50/P2)",
    "s11": "Ps30 - Static pressure at HPC outlet (psia)",
    "s12": "phi - Ratio of fuel flow to Ps30 (pps/psi)",
    "s13": "NRf - Corrected fan speed (rpm)",
    "s14": "NRc - Corrected core speed (rpm)",
    "s15": "BPR - Bypass ratio",
    "s16": "farB - Burner fuel-air ratio",
    "s17": "htBleed - Bleed enthalpy",
    "s18": "Nf_dmd - Demanded fan speed (rpm)",
    "s19": "PCNfR_dmd - Demanded corrected fan speed (rpm)",
    "s20": "W31 - HPT coolant bleed (lbm/s)",
    "s21": "W32 - LPT coolant bleed (lbm/s)",
}

PHYSICS_GROUPS = {
    "temperatures": ["s1", "s2", "s3", "s4"],
    "pressures":    ["s5", "s6", "s7", "s11"],
    "speeds":       ["s8", "s9", "s13", "s14"],
    "flow_rates":   ["s20", "s21"],
    "other":        ["s10", "s12", "s15", "s16", "s17", "s18", "s19"],
}

# NOTE: 14 of 21 sensors are nearly constant (zero variance) in C-MAPSS.
NEAR_CONSTANT_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]

# Direct download from NASA (may redirect)
NASA_URL = "https://data.nasa.gov/api/views/ff5v-kuh6/rows.csv?accessType=DOWNLOAD"

# Alternative: manual download from Kaggle
KAGGLE_DATASET = "behrad3d/nasa-cmaps"


def try_direct_download(output_dir: Path) -> bool:
    """Try to download from NASA data portal directly."""
    dest = output_dir / "CMAPSSData.zip"
    if dest.exists():
        print("  [SKIP] CMAPSSData.zip already exists")
        return True
    try:
        print(f"  [DOWN] Attempting NASA direct download...")
        urlretrieve(NASA_URL, dest)
        return True
    except URLError as e:
        print(f"  [FAIL] NASA direct download failed: {e}")
        return False


def try_kaggle_download(output_dir: Path) -> bool:
    """Try to download via Kaggle API."""
    try:
        import kaggle
        print(f"  [DOWN] Downloading via Kaggle API ({KAGGLE_DATASET})...")
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET, path=str(output_dir), unzip=True
        )
        return True
    except ImportError:
        print("  [SKIP] kaggle package not installed. Run: pip install kaggle")
        return False
    except Exception as e:
        print(f"  [FAIL] Kaggle download failed: {e}")
        return False


def create_sample_from_txt(output_dir: Path, sample_dir: Path,
                            n_units: int = 20) -> bool:
    """Keep only first N units from FD001 as a sample."""
    import pandas as pd

    sample_dir.mkdir(parents=True, exist_ok=True)
    src = output_dir / "train_FD001.txt"
    if not src.exists():
        print(f"  [ERR] {src} not found for sample creation")
        return False

    df = pd.read_csv(src, sep=" ", header=None, index_col=False)
    df = df.dropna(axis=1, how="all")
    df.columns = COLUMN_NAMES[:len(df.columns)]
    sample = df[df["unit_id"] <= n_units]
    dst = sample_dir / "train_FD001_sample.csv"
    sample.to_csv(dst, index=False)
    print(f"  [SAMP] {n_units} units saved to {dst} ({len(sample)} rows)")
    return True


def verify_and_stats(output_dir: Path) -> dict:
    """Load all 4 training files and report stats."""
    try:
        import pandas as pd
        import numpy as np

        results = {}
        for fd in ["FD001", "FD002", "FD003", "FD004"]:
            fpath = output_dir / f"train_{fd}.txt"
            if not fpath.exists():
                results[fd] = {"status": "missing"}
                continue
            df = pd.read_csv(fpath, sep=" ", header=None, index_col=False)
            df = df.dropna(axis=1, how="all")
            df.columns = COLUMN_NAMES[:len(df.columns)]
            n_units = df["unit_id"].nunique()
            avg_cycles = df.groupby("unit_id")["cycle"].max().mean()
            # Find near-constant sensors
            sensor_cols = [c for c in df.columns if c.startswith("s")]
            stds = df[sensor_cols].std()
            constant = stds[stds < 1e-6].index.tolist()

            results[fd] = {
                "status": "ok",
                "rows": len(df),
                "n_units": n_units,
                "avg_cycles": round(avg_cycles, 1),
                "near_constant_sensors": constant,
            }
        return results
    except ImportError:
        return {"status": "import_error", "msg": "pandas not available"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Download C-MAPSS Dataset")
    parser.add_argument("--sample", action="store_true",
                        help="Create a small sample (FD001, first 20 units)")
    parser.add_argument("--source", choices=["nasa", "kaggle", "auto"],
                        default="auto",
                        help="Download source (auto tries NASA then Kaggle)")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent.parent / "data" / "cmapss"),
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nC-MAPSS Dataset Downloader")
    print(f"Output: {output_dir}")
    print("-" * 60)

    # Check if already extracted
    already_have = list(output_dir.glob("train_FD*.txt"))
    if already_have:
        print(f"  [SKIP] Found {len(already_have)} training files already")
        success = True
    elif args.source in ("nasa", "auto"):
        success = try_direct_download(output_dir)
        if success:
            zip_path = output_dir / "CMAPSSData.zip"
            if zip_path.exists():
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(output_dir)
    if args.source == "kaggle" or (args.source == "auto" and not success):
        success = try_kaggle_download(output_dir)

    if not success:
        print("\n[MANUAL] Download instructions:")
        print("  Option 1 (Kaggle): pip install kaggle && kaggle datasets download behrad3d/nasa-cmaps")
        print("  Option 2 (NASA): https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6")
        print("  Place train_FD001.txt through train_FD004.txt in:", output_dir)
        return

    stats = verify_and_stats(output_dir)
    print("\nDataset verification:")
    for fd, info in stats.items():
        if isinstance(info, dict) and info.get("status") == "ok":
            print(f"  {fd}: {info['n_units']} units, "
                  f"avg {info['avg_cycles']:.0f} cycles, "
                  f"{info['rows']} rows")
            if info.get("near_constant_sensors"):
                print(f"       Near-constant sensors: {info['near_constant_sensors']}")
        elif isinstance(info, dict):
            print(f"  {fd}: {info.get('status', 'unknown')}")

    if args.sample:
        sample_dir = output_dir.parent / "cmapss_sample"
        create_sample_from_txt(output_dir, sample_dir, n_units=20)

    print("\nDataset Summary:")
    print("  Physical system: Turbofan engine (MAPSS simulation)")
    print("  Sub-datasets: FD001–FD004 (1–6 op conditions, 1–2 fault modes)")
    print("  Sensors: 21 (14 are near-constant, effectively 7 informative)")
    print("  Task: RUL prediction (non-standard: sensor forecasting)")
    print("\nPhysics Groups:")
    for group, sensors in PHYSICS_GROUPS.items():
        print(f"  {group}: {sensors}")
    print(f"\nIndustrialJEPA Exp 48: physics mask ≈ random mask (p=0.528)")
    print(f"Reason: Correlated degradation — all sensors fail together.")
    print(f"\nRecommendation: Keep as 'correlated system' example in paper.")
    print(f"Do NOT use as the real-data Tier 2.")
    print(f"\nData saved to: {output_dir}")


if __name__ == "__main__":
    main()
