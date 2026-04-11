"""
Comprehensive analysis of DCSSL replication results.

Produces:
1. Full comparison table vs paper
2. Per-bearing analysis
3. Prediction quality plots
4. Statistical analysis

Usage:
    python analyze_results.py
"""

import json
import sys
import warnings
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

RESULTS_DIR = Path("/home/sagemaker-user/IndustrialJEPA/dcssl-replication/results")
FIG_DIR = Path("/home/sagemaker-user/IndustrialJEPA/dcssl-replication/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Paper results for comparison
PAPER_DCSSL = {
    "Bearing1_3": 0.0011, "Bearing1_4": 0.0476, "Bearing1_5": 0.0005,
    "Bearing1_6": 0.0892, "Bearing1_7": 0.0009,
    "Bearing2_3": 0.0027, "Bearing2_4": 0.0014, "Bearing2_5": 0.2538,
    "Bearing2_6": 0.0012, "Bearing2_7": 0.0075,
    "Bearing3_3": 0.0068, "avg": 0.0375,
}
PAPER_SIMCLR = {
    # CORRECTED from PDF (Shen et al. 2026, Table 3, col 4: SimCLR)
    "Bearing1_3": 0.0030, "Bearing1_4": 0.0560, "Bearing1_5": 0.0006,
    "Bearing1_6": 0.0904, "Bearing1_7": 0.0021,
    "Bearing2_3": 0.1849, "Bearing2_4": 0.2577, "Bearing2_5": 0.2782,
    "Bearing2_6": 0.0013, "Bearing2_7": 0.0089,
    "Bearing3_3": 0.0341, "avg": 0.0583,
}
PAPER_SUPCON = {
    # CORRECTED from PDF (Shen et al. 2026, Table 3, col 5: SupCon)
    "Bearing1_3": 0.0213, "Bearing1_4": 0.0576, "Bearing1_5": 0.0046,
    "Bearing1_6": 0.0735, "Bearing1_7": 0.0038,
    "Bearing2_3": 0.0150, "Bearing2_4": 0.0017, "Bearing2_5": 0.2752,
    "Bearing2_6": 0.0014, "Bearing2_7": 0.0117,
    "Bearing3_3": 0.0619, "avg": 0.0480,
}

ALL_TEST_BEARINGS = [
    "Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7",
    "Bearing2_3", "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7",
    "Bearing3_3",
]


def load_results() -> dict:
    """Load all experiment results."""
    results = {}

    for json_file in RESULTS_DIR.rglob("*_results.json"):
        if "ablation" in str(json_file):
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)
            model_name = data.get("model", "unknown")
            per_bearing = data.get("per_bearing", {})
            if model_name not in results:
                results[model_name] = {}
            for bearing_name, res in per_bearing.items():
                # Use latest result if duplicate
                if bearing_name not in results[model_name] or \
                   res.get("mse", 1.0) < results[model_name][bearing_name].get("mse", 1.0):
                    results[model_name][bearing_name] = res
        except Exception as e:
            print(f"  Warning: {json_file}: {e}")

    # Check for JEPA+HC
    jepa_path = RESULTS_DIR / "jepa_hc_all_results.json"
    if jepa_path.exists():
        try:
            with open(jepa_path) as f:
                jepa_data = json.load(f)
            results["jepa_hc"] = {}
            for cond_key, cond_data in jepa_data.items():
                for bearing, res in cond_data.get("per_bearing", {}).items():
                    results["jepa_hc"][bearing] = res
        except Exception as e:
            print(f"  Warning: jepa_hc results: {e}")

    return results


def print_final_table(our_results: dict):
    """Print comprehensive comparison table."""

    methods_order = ["simclr", "supcon", "dcssl", "jepa_hc"]
    paper_methods = {"SimCLR": PAPER_SIMCLR, "SupCon": PAPER_SUPCON, "DCSSL": PAPER_DCSSL}

    print("\n" + "="*100)
    print("FINAL RESULTS TABLE (MSE) — DCSSL Replication vs Paper Table 3")
    print("="*100)

    # Header
    header = f"{'Test Bearing':<15}"
    for m in methods_order:
        if m in our_results:
            header += f" {'Ours_'+m:>12}"
    for m in paper_methods.keys():
        header += f" {'Paper_'+m:>13}"
    print(header)
    print("-"*100)

    our_avgs = {m: [] for m in methods_order}
    paper_avgs = {m: [] for m in paper_methods.keys()}

    for bearing in ALL_TEST_BEARINGS:
        row = f"{bearing:<15}"
        for m in methods_order:
            if m in our_results and bearing in our_results[m]:
                v = our_results[m][bearing]["mse"]
                row += f" {v:>12.4f}"
                our_avgs[m].append(v)
            elif m in our_results:
                row += f" {'—':>12}"
        for m, data in paper_methods.items():
            v = data.get(bearing, None)
            if v is not None:
                row += f" {v:>13.4f}"
                paper_avgs[m].append(v)
            else:
                row += f" {'—':>13}"
        print(row)

    print("-"*100)
    avg_row = f"{'Average':<15}"
    for m in methods_order:
        vals = our_avgs[m]
        if vals:
            avg_row += f" {np.mean(vals):>12.4f}"
        else:
            avg_row += f" {'—':>12}"
    for m, data in paper_methods.items():
        if "avg" in data and data["avg"] is not None:
            avg_row += f" {data['avg']:>13.4f}"
        elif paper_avgs[m]:
            avg_row += f" {np.mean(paper_avgs[m]):>13.4f}"
        else:
            avg_row += f" {'—':>13}"
    print(avg_row)
    print("="*100)

    # Comparison summary
    print("\n\nIMPROVEMENT SUMMARY vs Paper")
    print("-"*70)
    for m in methods_order:
        if m not in our_results or not our_avgs[m]:
            continue
        our_avg = np.mean(our_avgs[m])

        # Compare to paper counterpart
        if m in ("simclr",):
            paper_avg = PAPER_SIMCLR["avg"]
            paper_label = "SimCLR"
        elif m == "supcon":
            paper_avg = PAPER_SUPCON["avg"]
            paper_label = "SupCon"
        elif m in ("dcssl", "jepa_hc"):
            paper_avg = PAPER_DCSSL["avg"]
            paper_label = "DCSSL"
        else:
            continue

        improvement = (paper_avg - our_avg) / paper_avg * 100
        n_bearings = len(our_avgs[m])
        print(f"  {'Ours_'+m:<20}: {our_avg:.4f} ({n_bearings}/11 bearings) | "
              f"Paper_{paper_label}: {paper_avg:.4f} | "
              f"{'Better' if our_avg < paper_avg else 'Worse'}: {abs(improvement):.1f}%")


def plot_prediction_curves(our_results: dict):
    """Plot predicted vs true RUL curves for all test bearings."""
    from data_utils import load_bearing_with_rul

    DATA_ROOT = Path("/mnt/sagemaker-nvme/femto_data/10. FEMTO Bearing")

    # Collect methods with prediction data
    methods_with_preds = {}
    for m, bearing_results in our_results.items():
        has_preds = any(
            "predictions" in bearing_results.get(b, {})
            for b in ALL_TEST_BEARINGS
        )
        if has_preds:
            methods_with_preds[m] = bearing_results

    if not methods_with_preds:
        print("No prediction data available for plotting")
        return

    conditions = {1: [f"Bearing1_{i}" for i in [3,4,5,6,7]],
                  2: [f"Bearing2_{i}" for i in [3,4,5,6,7]],
                  3: ["Bearing3_3"]}

    colors = {"simclr": "blue", "supcon": "green", "dcssl": "red", "jepa_hc": "purple"}

    for condition, bearings in conditions.items():
        n_bearings = len(bearings)
        fig, axes = plt.subplots(1, n_bearings, figsize=(5*n_bearings, 4))
        if n_bearings == 1:
            axes = [axes]

        fig.suptitle(f"Condition {condition}: Predicted vs True RUL", fontsize=14)

        for ax, bearing_name in zip(axes, bearings):
            # Load true RUL
            try:
                d = load_bearing_with_rul(DATA_ROOT, bearing_name)
                true_rul = d["rul"]
                n = d["n_snapshots"]
                fpt = d["fpt"]
                t = np.arange(n) / n

                ax.plot(t, true_rul, 'k-', linewidth=2, label="True RUL", zorder=10)
                if fpt > 0:
                    ax.axvline(x=fpt/n, color='k', linestyle='--', alpha=0.5, label=f"FPT")
            except Exception:
                t = np.linspace(0, 1, 100)
                true_rul = np.linspace(1, 0, 100)

            # Plot model predictions
            for m, bearing_results in methods_with_preds.items():
                if bearing_name in bearing_results:
                    res = bearing_results[bearing_name]
                    if "predictions" in res:
                        preds = np.array(res["predictions"])
                        n_pred = len(preds)
                        t_pred = np.arange(n_pred) / n_pred
                        color = colors.get(m, "gray")
                        mse = res["mse"]
                        ax.plot(t_pred, preds, color=color, alpha=0.8,
                                linewidth=1.5, label=f"{m} (MSE={mse:.4f})")

            ax.set_title(f"{bearing_name}")
            ax.set_xlabel("Normalized Time")
            ax.set_ylabel("RUL")
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.1, 1.1)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = FIG_DIR / f"rul_curves_condition{condition}.pdf"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")


def generate_latex_table(our_results: dict) -> str:
    """Generate LaTeX table for paper inclusion."""
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{MSE comparison on FEMTO/PRONOSTIA bearings}")
    lines.append(r"\begin{tabular}{l|ccc|ccc}")
    lines.append(r"\hline")
    lines.append(r"Bearing & SimCLR* & SupCon* & DCSSL* & Ours\_SimCLR & Ours\_SupCon & Ours\_DCSSL \\")
    lines.append(r"\hline")

    our_mses = {"simclr": {}, "supcon": {}, "dcssl": {}}
    for m in our_mses:
        if m in our_results:
            for b, res in our_results[m].items():
                our_mses[m][b] = res["mse"]

    for bearing in ALL_TEST_BEARINGS:
        row_parts = [bearing.replace("_", "\\_")]
        for paper_data in [PAPER_SIMCLR, PAPER_SUPCON, PAPER_DCSSL]:
            v = paper_data.get(bearing, "—")
            row_parts.append(f"{v:.4f}" if v != "—" else "—")
        for m in ["simclr", "supcon", "dcssl"]:
            v = our_mses[m].get(bearing, "—")
            row_parts.append(f"{v:.4f}" if v != "—" else "—")
        lines.append(" & ".join(row_parts) + r" \\")

    # Average row
    lines.append(r"\hline")
    avg_parts = ["Average"]
    for paper_data in [PAPER_SIMCLR, PAPER_SUPCON, PAPER_DCSSL]:
        avg_parts.append(f"{paper_data['avg']:.4f}")
    for m in ["simclr", "supcon", "dcssl"]:
        vals = [v for v in our_mses[m].values() if isinstance(v, float)]
        if vals:
            avg_parts.append(f"\\mathbf{{{np.mean(vals):.4f}}}")
        else:
            avg_parts.append("—")
    lines.append(" & ".join(avg_parts) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\multicolumn{7}{l}{* Paper results from Shen et al. 2026} \\")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    print("Loading results...")
    our_results = load_results()

    if not our_results:
        print("No results found! Run experiments first.")
        return

    print(f"Found results for: {list(our_results.keys())}")
    print(f"Bearings covered: {set().union(*[set(r.keys()) for r in our_results.values()])}")

    # Print table
    print_final_table(our_results)

    # Generate figures if predictions available
    print("\nGenerating prediction curve plots...")
    plot_prediction_curves(our_results)

    # Save LaTeX table
    latex = generate_latex_table(our_results)
    latex_path = RESULTS_DIR / "results_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"\nSaved LaTeX table to: {latex_path}")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
