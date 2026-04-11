"""
Compile all experiment results into a comprehensive comparison table.

Reads all result JSON files from the results/ directory
and produces a markdown table comparing with paper results.

Usage:
    python compile_results.py
    python compile_results.py --results_dir results/
"""

import json
import sys
import argparse
from pathlib import Path
import numpy as np


# Paper Table 3 results
PAPER_RESULTS = {
    "InfoTS": {
        "Bearing1_3": 0.0037, "Bearing1_4": 0.0566, "Bearing1_5": 0.0015,
        "Bearing1_6": 0.1095, "Bearing1_7": 0.0031,
        "Bearing2_3": 0.0805,
        "avg": None,
    },
    "USL": {
        "Bearing1_3": 0.0047, "Bearing1_4": 0.1003, "Bearing1_5": 0.0014,
        "Bearing1_6": 0.0449, "Bearing1_7": 0.0044,
        "Bearing2_3": 0.0406,
        "avg": None,
    },
    "CBHRL": {
        "Bearing1_3": 0.0052, "Bearing1_4": 0.0012, "Bearing1_5": 0.0005,
        "Bearing1_6": 0.0016, "Bearing1_7": 0.2782,
        "Bearing2_3": 0.0018, "Bearing2_4": 0.0229, "Bearing2_5": 0.0091,
        "Bearing2_6": 0.0425,
        "Bearing3_3": 0.0619,
        "avg": None,
    },
    "SimCLR_paper": {
        # CORRECTED from PDF (Shen et al. 2026, Table 3, col 4)
        "Bearing1_3": 0.0030, "Bearing1_4": 0.0560, "Bearing1_5": 0.0006,
        "Bearing1_6": 0.0904, "Bearing1_7": 0.0021,
        "Bearing2_3": 0.1849, "Bearing2_4": 0.2577, "Bearing2_5": 0.2782,
        "Bearing2_6": 0.0013, "Bearing2_7": 0.0089,
        "Bearing3_3": 0.0341,
        "avg": 0.0583,  # stated in paper footer (11-bearing simple mean = 0.0834)
    },
    "SupCon_paper": {
        # CORRECTED from PDF (Shen et al. 2026, Table 3, col 5)
        "Bearing1_3": 0.0213, "Bearing1_4": 0.0576, "Bearing1_5": 0.0046,
        "Bearing1_6": 0.0735, "Bearing1_7": 0.0038,
        "Bearing2_3": 0.0150, "Bearing2_4": 0.0017, "Bearing2_5": 0.2752,
        "Bearing2_6": 0.0014, "Bearing2_7": 0.0117,
        "Bearing3_3": 0.0619,
        "avg": 0.0480,  # stated in paper footer (11-bearing simple mean = 0.0480 ✓)
    },
    "DCSSL_paper": {
        "Bearing1_3": 0.0011, "Bearing1_4": 0.0476, "Bearing1_5": 0.0005,
        "Bearing1_6": 0.0892, "Bearing1_7": 0.0009,
        "Bearing2_3": 0.0027, "Bearing2_4": 0.0014, "Bearing2_5": 0.2538,
        "Bearing2_6": 0.0012, "Bearing2_7": 0.0075,
        "Bearing3_3": 0.0068,
        "avg": 0.0375,
    },
}

ALL_TEST_BEARINGS = [
    "Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7",
    "Bearing2_3", "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7",
    "Bearing3_3",
]


def load_our_results(results_dir: Path) -> dict:
    """Load all our experiment results."""
    our_results = {}

    # Look for per-experiment JSON files (skip top-level aggregates)
    for json_file in results_dir.rglob("*_results.json"):
        # Skip files in the top-level results dir (aggregates like all_results.json)
        if json_file.parent == results_dir:
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)

            model_name = data.get("model", json_file.parent.name.split("_")[0])
            per_bearing = data.get("per_bearing", {})

            if not per_bearing:
                continue  # skip empty results

            if model_name not in our_results:
                our_results[model_name] = {}

            for bearing_name, res in per_bearing.items():
                our_results[model_name][bearing_name] = res.get("mse", None)

        except Exception as e:
            print(f"  Warning: Could not load {json_file}: {e}")

    # Also check all_results.json
    all_results_path = results_dir / "all_results.json"
    if all_results_path.exists():
        try:
            with open(all_results_path) as f:
                all_data = json.load(f)
            for exp_key, exp_data in all_data.items():
                if "error" in exp_data:
                    continue
                model_name = exp_data.get("model", exp_key.split("_")[0])
                per_bearing = exp_data.get("per_bearing", {})
                if model_name not in our_results:
                    our_results[model_name] = {}
                for bearing_name, res in per_bearing.items():
                    our_results[model_name][bearing_name] = res.get("mse", None)
        except Exception as e:
            print(f"  Warning: Could not load all_results.json: {e}")

    # Check JEPA+HC results
    jepa_path = results_dir / "jepa_hc_all_results.json"
    if jepa_path.exists():
        try:
            with open(jepa_path) as f:
                jepa_data = json.load(f)
            jepa_mses = {}
            for cond_key, cond_data in jepa_data.items():
                per_bearing = cond_data.get("per_bearing", {})
                for bearing_name, res in per_bearing.items():
                    jepa_mses[bearing_name] = res.get("mse", None)
            if jepa_mses:
                our_results["jepa_hc"] = jepa_mses
        except Exception as e:
            print(f"  Warning: Could not load jepa_hc results: {e}")

    return our_results


def compute_averages(results_by_method: dict) -> dict:
    """Compute averages over all 11 test bearings."""
    avgs = {}
    for method, results in results_by_method.items():
        values = [results.get(b) for b in ALL_TEST_BEARINGS if results.get(b) is not None]
        if values:
            avgs[method] = np.mean(values)
    return avgs


def print_table(our_results: dict, results_dir: Path):
    """Print comprehensive results table."""
    # Combine paper and our results
    all_methods_data = {}
    for m, data in PAPER_RESULTS.items():
        all_methods_data[m] = data

    for m, data in our_results.items():
        all_methods_data[f"Ours_{m}"] = data

    our_method_names = [f"Ours_{m}" for m in our_results.keys()]

    print("\n" + "=" * 120)
    print("COMPREHENSIVE RESULTS TABLE (MSE)")
    print("=" * 120)

    # Column headers
    paper_methods = ["SimCLR_paper", "SupCon_paper", "DCSSL_paper"]
    our_display = our_method_names

    all_cols = paper_methods + our_display
    header = f"{'Bearing':<15}"
    for col in all_cols:
        display = col.replace("_paper", "").replace("Ours_", "Our_")
        header += f" {display:>12}"
    print(header)
    print("-" * 120)

    # Per-bearing rows
    method_mses = {m: [] for m in all_cols}
    for bearing in ALL_TEST_BEARINGS:
        row = f"{bearing:<15}"
        for col in all_cols:
            v = all_methods_data.get(col, {}).get(bearing, None)
            if v is not None:
                row += f" {v:>12.4f}"
                method_mses[col].append(v)
            else:
                row += f" {'—':>12}"
        print(row)

    print("-" * 120)

    # Average row
    avg_row = f"{'Average':<15}"
    for col in all_cols:
        if col in all_methods_data and "avg" in all_methods_data[col] and all_methods_data[col]["avg"] is not None:
            avg = all_methods_data[col]["avg"]
        elif method_mses[col]:
            avg = np.mean(method_mses[col])
        else:
            avg = None

        if avg is not None:
            avg_row += f" {avg:>12.4f}"
        else:
            avg_row += f" {'—':>12}"
    print(avg_row)
    print("=" * 120)

    # Summary stats
    print("\n\nSUMMARY STATISTICS")
    print("-" * 60)
    for method in our_method_names:
        clean_name = method.replace("Ours_", "")
        values = [all_methods_data.get(method, {}).get(b)
                  for b in ALL_TEST_BEARINGS
                  if all_methods_data.get(method, {}).get(b) is not None]
        if values:
            avg = np.mean(values)
            paper_dcssl = PAPER_RESULTS["DCSSL_paper"]["avg"]
            paper_simclr = PAPER_RESULTS["SimCLR_paper"]["avg"]
            pct_vs_dcssl = (avg / paper_dcssl - 1) * 100
            n = len(values)
            print(f"  {clean_name:<15}: avg={avg:.4f} ({n}/11 bearings) | "
                  f"vs DCSSL_paper: {pct_vs_dcssl:+.1f}%")

    # Save markdown table
    md_path = results_dir / "RESULTS_TABLE.md"
    with open(md_path, "w") as f:
        f.write("# DCSSL Replication Results\n\n")
        f.write("## MSE Comparison Table\n\n")
        f.write("| Bearing | SimCLR_paper | SupCon_paper | DCSSL_paper |")
        for method in our_method_names:
            display = method.replace("Ours_", "Our_")
            f.write(f" {display} |")
        f.write("\n|---|---|---|---|")
        for _ in our_method_names:
            f.write("---|")
        f.write("\n")

        for bearing in ALL_TEST_BEARINGS:
            row = f"| {bearing} |"
            for col in all_cols:
                v = all_methods_data.get(col, {}).get(bearing, None)
                if v is not None:
                    row += f" {v:.4f} |"
                else:
                    row += " — |"
            f.write(row + "\n")

        # Average
        avg_row_md = "| **Average** |"
        for col in all_cols:
            if col in all_methods_data and "avg" in all_methods_data[col] and all_methods_data[col]["avg"] is not None:
                avg = all_methods_data[col]["avg"]
            elif method_mses[col]:
                avg = np.mean(method_mses[col])
            else:
                avg = None
            if avg is not None:
                avg_row_md += f" **{avg:.4f}** |"
            else:
                avg_row_md += " — |"
        f.write(avg_row_md + "\n")

    print(f"\n  Saved markdown table to: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Compile DCSSL replication results")
    parser.add_argument("--results_dir", type=str,
                        default="/home/sagemaker-user/IndustrialJEPA/dcssl-replication/results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    print(f"Loading results from: {results_dir}")

    our_results = load_our_results(results_dir)

    if not our_results:
        print("No results found yet. Run experiments first.")
        print("Available files:")
        for f in results_dir.rglob("*.json"):
            print(f"  {f}")
        return

    print(f"\nFound results for {len(our_results)} methods: {list(our_results.keys())}")
    print_table(our_results, results_dir)


if __name__ == "__main__":
    main()
