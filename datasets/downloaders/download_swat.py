"""
SWaT / WADI Dataset — Access Instructions.

SWaT requires a formal data request to iTrust, SUTD Singapore.
This script cannot automate the download but provides instructions,
validates the data format once obtained, and prepares it for use.

Usage:
    python datasets/downloaders/download_swat.py --instructions  # Print access instructions
    python datasets/downloaders/download_swat.py --validate /path/to/swat/  # Validate obtained data
    python datasets/downloaders/download_swat.py --prepare /path/to/swat/   # Prepare for training
"""

import argparse
from pathlib import Path

REQUEST_URL = "https://itrust.sutd.edu.sg/testbeds/secure-water-treatment-swat/"

EXPECTED_FILES = {
    "SWaT.A1 & A2_Jan 2016": {
        "normal": "SWaT_Dataset_Normal_v1.xlsx",
        "attack": "SWaT_Dataset_Attack_v0.xlsx",
    }
}

SWAT_CHANNELS = {
    "P1": ["FIT101", "LIT101", "MV101", "P101", "P102"],
    "P2": ["AIT201", "AIT202", "AIT203", "FIT201", "MV201",
           "P201", "P202", "P203", "P204", "P205", "P206"],
    "P3": ["DPIT301", "FIT301", "LIT301", "MV301", "MV302",
           "MV303", "MV304", "P301", "P302"],
    "P4": ["AIT401", "AIT402", "FIT401", "LIT401", "P401",
           "P402", "P403", "P404", "UV401"],
    "P5": ["AIT501", "AIT502", "AIT503", "AIT504", "FIT501",
           "FIT502", "FIT503", "FIT504", "P501", "P502",
           "PIT501", "PIT502", "PIT503"],
    "P6": ["FIT601", "P601", "P602", "P603"],
}

PHYSICS_GROUPS = {
    "P1_water_intake":      list(range(0, 5)),
    "P2_chemical_dosing":   list(range(5, 16)),
    "P3_uf_filtration":     list(range(16, 25)),
    "P4_dechlorination":    list(range(25, 34)),
    "P5_ro_desalination":   list(range(34, 47)),
    "P6_product_water":     list(range(47, 51)),
}


def print_instructions():
    print("""
=============================================================
SWaT Dataset Access Instructions
=============================================================

SWaT (Secure Water Treatment) requires a formal data request.
This is standard for security-sensitive industrial datasets.

STEP 1: Submit Request
  URL: https://itrust.sutd.edu.sg/testbeds/secure-water-treatment-swat/
  Fill out the "Request Dataset" form at the bottom of the page.
  Approval typically takes 3–7 business days for academic researchers.

STEP 2: Wait for Approval
  You will receive an email with a Google Drive link.
  The link contains:
    - SWaT_Dataset_Normal_v1.xlsx  (7 days normal operation, ~946k rows)
    - SWaT_Dataset_Attack_v0.xlsx  (4 days with 36 attack scenarios)

STEP 3: Download to Local Machine
  Place files in: datasets/data/swat/

STEP 4: Validate
  python datasets/downloaders/download_swat.py --validate datasets/data/swat/

=============================================================
WADI Dataset (Water Distribution, 127 channels)
=============================================================

Same process, but request at:
  URL: https://itrust.sutd.edu.sg/testbeds/water-distribution-wadi/
  Files: WADI.A1_9 Oct 2017.csv, WADI.A2_19 Nov 2019.csv

WADI is preferred for Brain-JEPA scale (127 channels vs 51 for SWaT).

=============================================================
Why SWaT/WADI for IndustrialJEPA?
=============================================================

SWaT: 51 sensors × 946,000 timesteps at 1 Hz
  - 6 physical process stages (P1–P6) with clear causal flow
  - Real industrial control system (water treatment plant)
  - Continuous operation (no episodes) — better for pretraining
  - Published anomaly detection SOTA available

WADI: 127 sensors × 1,209,600 timesteps at 1 Hz
  - Closest to Brain-JEPA channel count (127 vs 450)
  - 3 subsystems: chemical treatment, storage, distribution
  - 16 days of data including 2 attack periods

Physics Grouping (SWaT):
  P1 (Water intake):      FIT101, LIT101, MV101, P101, P102
  P2 (Chemical dosing):   AIT201-203, FIT201, MV201, P201-206
  P3 (UF filtration):     DPIT301, FIT301, LIT301, MV301-304, P301-302
  P4 (De-chlorination):   AIT401-402, FIT401, LIT401, P401-404, UV401
  P5 (RO desalination):   AIT501-504, FIT501-504, P501-502, PIT501-503
  P6 (Product water):     FIT601, P601-603

This process-flow grouping is IDEAL for physics-informed attention:
  - P1 → P2 → P3 → P4 → P5 → P6 is the causal flow
  - Cross-process attention can be masked to follow this direction
""")


def validate_swat(data_dir: Path):
    """Validate SWaT data once obtained."""
    try:
        import pandas as pd
        import numpy as np

        data_dir = Path(data_dir)
        normal_file = data_dir / "SWaT_Dataset_Normal_v1.xlsx"
        attack_file = data_dir / "SWaT_Dataset_Attack_v0.xlsx"

        results = {}
        for name, fpath in [("Normal", normal_file), ("Attack", attack_file)]:
            if not fpath.exists():
                print(f"  [MISS] {fpath.name} not found")
                continue
            print(f"  [LOAD] Loading {fpath.name}...")
            # SWaT Excel files have a header row to skip
            df = pd.read_excel(fpath, skiprows=1)
            print(f"  [OK]   {name}: {len(df)} rows × {len(df.columns)} columns")
            print(f"         Columns: {list(df.columns[:5])}...")
            if "Normal/Attack" in df.columns:
                label_counts = df["Normal/Attack"].value_counts()
                print(f"         Labels: {dict(label_counts)}")
            results[name] = {"rows": len(df), "cols": len(df.columns)}

        return results

    except ImportError:
        print("  [ERR] pandas and openpyxl required: pip install pandas openpyxl")
        return {}


def prepare_for_training(data_dir: Path, output_dir: Path, downsample: int = 1):
    """Convert SWaT Excel to clean numpy/CSV for fast loading."""
    try:
        import pandas as pd
        import numpy as np

        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for split in ["Normal", "Attack"]:
            fname = f"SWaT_Dataset_{split}_v{'1' if split == 'Normal' else '0'}.xlsx"
            fpath = data_dir / fname
            if not fpath.exists():
                continue
            print(f"  [PREP] Processing {fname}...")
            df = pd.read_excel(fpath, skiprows=1)

            # Remove timestamp and label columns for raw features
            label_col = "Normal/Attack" if "Normal/Attack" in df.columns else None
            timestamp_col = " Timestamp" if " Timestamp" in df.columns else None
            feature_cols = [c for c in df.columns
                            if c not in [label_col, timestamp_col, "Timestamp"]]
            X = df[feature_cols].values.astype(np.float32)
            y = (df[label_col].values != "Normal").astype(np.int8) if label_col else None

            # Downsample if requested
            if downsample > 1:
                X = X[::downsample]
                if y is not None:
                    y = y[::downsample]

            np.save(output_dir / f"swat_{split.lower()}_X.npy", X)
            if y is not None:
                np.save(output_dir / f"swat_{split.lower()}_y.npy", y)

            print(f"  [OK]   X shape: {X.shape}, anomaly rate: "
                  f"{y.mean():.3f}" if y is not None else f"  [OK]   X shape: {X.shape}")

        # Save channel metadata
        import json
        meta = {
            "channels": sum(SWAT_CHANNELS.values(), []),
            "physics_groups": {k: list(v) for k, v in PHYSICS_GROUPS.items()},
            "n_channels": 51,
            "sampling_hz": 1,
            "description": "SWaT water treatment plant, 6 process stages",
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\n  Data prepared and saved to {output_dir}")

    except Exception as e:
        print(f"  [ERR] Preparation failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="SWaT/WADI Dataset Handler")
    parser.add_argument("--instructions", action="store_true",
                        help="Print data access instructions")
    parser.add_argument("--validate", type=str, metavar="DATA_DIR",
                        help="Validate downloaded SWaT data")
    parser.add_argument("--prepare", type=str, metavar="DATA_DIR",
                        help="Convert SWaT data to numpy format for training")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent.parent / "data" / "swat"),
                        help="Output directory for prepared data")
    args = parser.parse_args()

    if args.instructions or (not args.validate and not args.prepare):
        print_instructions()

    if args.validate:
        print(f"\nValidating SWaT data at: {args.validate}")
        validate_swat(Path(args.validate))

    if args.prepare:
        print(f"\nPreparing SWaT data from: {args.prepare}")
        output_dir = Path(args.output_dir)
        prepare_for_training(Path(args.prepare), output_dir)


if __name__ == "__main__":
    main()
