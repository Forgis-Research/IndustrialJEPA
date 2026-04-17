"""
Download ETT (Electricity Transformer Temperature) Dataset.

4 variants: ETTh1, ETTh2 (hourly), ETTm1, ETTm2 (15-min).
Direct CSV download from GitHub — no registration required.
Saves to datasets/data/ett/

Usage:
    python datasets/downloaders/download_ett.py
    python datasets/downloaders/download_ett.py --sample   # Head only (first 2000 rows)
"""

import argparse
import io
import sys
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

# ---------------------------------------------------------------------------
# ETT files are in the ETDataset GitHub repo
# ---------------------------------------------------------------------------
BASE_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/"

FILES = {
    "ETTh1.csv": "ETTh1.csv",
    "ETTh2.csv": "ETTh2.csv",
    "ETTm1.csv": "ETTm1.csv",
    "ETTm2.csv": "ETTm2.csv",
}

COLUMNS = ["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

PHYSICS_GROUPS = {
    "HV_power":  ["HUFL", "HULL"],
    "MV_power":  ["MUFL", "MULL"],
    "LV_power":  ["LUFL", "LULL"],
    "thermal":   ["OT"],
}


def download_file(url: str, dest: Path, sample: bool = False,
                  sample_rows: int = 2000, verbose: bool = True) -> bool:
    """Download a CSV file, optionally keeping only the first N rows."""
    if dest.exists() and not sample:
        if verbose:
            print(f"  [SKIP] {dest.name} already exists")
        return True

    try:
        if verbose:
            mode = f"SAMPLE ({sample_rows} rows)" if sample else "FULL"
            print(f"  [DOWN] {dest.name} ({mode}) from {url}")
        response = urlopen(url)
        content = response.read().decode("utf-8")

        if sample:
            lines = content.splitlines()
            content = "\n".join(lines[:sample_rows + 1])  # +1 for header

        dest.write_text(content, encoding="utf-8")
        size_kb = dest.stat().st_size / 1024
        if verbose:
            print(f"         -> {size_kb:.1f} KB saved")
        return True
    except URLError as e:
        print(f"  [ERR]  Failed to download {dest.name}: {e}")
        return False


def verify_and_stats(path: Path) -> dict:
    """Load CSV and return basic statistics."""
    try:
        import pandas as pd
        df = pd.read_csv(path)
        stats = {
            "rows": len(df),
            "cols": len(df.columns),
            "columns": list(df.columns),
            "missing": int(df.isnull().sum().sum()),
            "date_range": f"{df['date'].iloc[0]} to {df['date'].iloc[-1]}" if "date" in df else "N/A",
            "status": "ok",
        }
        for col in df.select_dtypes("float").columns:
            stats[f"{col}_mean"] = f"{df[col].mean():.3f}"
            stats[f"{col}_std"] = f"{df[col].std():.3f}"
        return stats
    except Exception as e:
        return {"status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Download ETT Dataset")
    parser.add_argument("--sample", action="store_true",
                        help="Download only first 2000 rows of each file")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent.parent / "data" / "ett"),
                        help="Output directory")
    parser.add_argument("--variants", nargs="+",
                        default=["ETTh1", "ETTh2", "ETTm1", "ETTm2"],
                        help="Which variants to download")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = {f"{v}.csv": f"{v}.csv" for v in args.variants if f"{v}.csv" in FILES}
    mode = "SAMPLE" if args.sample else "FULL"

    print(f"\nETT Dataset Downloader")
    print(f"Mode: {mode} ({len(requested)} files)")
    print(f"Output: {output_dir}")
    print("-" * 60)

    success_count = 0
    stats_all = {}
    for local_name, remote_name in requested.items():
        url = BASE_URL + remote_name
        dest = output_dir / local_name
        if download_file(url, dest, sample=args.sample):
            success_count += 1
            stats = verify_and_stats(dest)
            stats_all[local_name] = stats
            if stats["status"] == "ok":
                print(f"  [STAT] {local_name}: {stats['rows']} rows, "
                      f"{stats['cols']} columns, "
                      f"{stats['missing']} missing values")
                print(f"         Date range: {stats['date_range']}")

    print(f"\nDownloaded {success_count}/{len(requested)} files")

    print("\nDataset Summary:")
    print(f"  ETTh1/h2: 17,420 timesteps at 1 sample/hour (~2 years)")
    print(f"  ETTm1/m2: 69,680 timesteps at 1 sample/15min (~2 years)")
    print(f"  Channels: 7 (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT)")
    print(f"\nPhysics Groups:")
    for group, cols in PHYSICS_GROUPS.items():
        print(f"  {group}: {cols}")

    print(f"\nPublished SOTA (ETTh1, H=96 MSE):")
    print(f"  PatchTST:     ~0.370")
    print(f"  iTransformer: ~0.386")
    print(f"  TimesNet:     ~0.384")
    print(f"  DLinear:      ~0.386")
    print(f"  Moirai/TimesFM: ~0.35 (approx, 2024)")

    print(f"\nIndustrialJEPA result (Exp 46): PhysMask vs Full-Attn = -1.3% (negative)")
    print(f"Interpretation: Cross-group thermal coupling makes masking harmful here.")
    print(f"\nData saved to: {output_dir}")


if __name__ == "__main__":
    main()
