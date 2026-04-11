"""
Post-experiment analysis script.
Run after all experiments complete to generate full comparison and figures.

Usage:
    python post_experiment_analysis.py
"""
import json
import sys
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

RESULTS_DIR = Path("/home/sagemaker-user/IndustrialJEPA/dcssl-replication/results")

# Paper Table 3 results (corrected from PDF)
PAPER = {
    "simclr": {
        "Bearing1_3": 0.0030, "Bearing1_4": 0.0560, "Bearing1_5": 0.0006,
        "Bearing1_6": 0.0904, "Bearing1_7": 0.0021,
        "Bearing2_3": 0.1849, "Bearing2_4": 0.2577, "Bearing2_5": 0.2782,
        "Bearing2_6": 0.0013, "Bearing2_7": 0.0089,
        "Bearing3_3": 0.0341, "avg": 0.0583,
    },
    "supcon": {
        "Bearing1_3": 0.0213, "Bearing1_4": 0.0576, "Bearing1_5": 0.0046,
        "Bearing1_6": 0.0735, "Bearing1_7": 0.0038,
        "Bearing2_3": 0.0150, "Bearing2_4": 0.0017, "Bearing2_5": 0.2752,
        "Bearing2_6": 0.0014, "Bearing2_7": 0.0117,
        "Bearing3_3": 0.0619, "avg": 0.0480,
    },
    "dcssl": {
        "Bearing1_3": 0.0011, "Bearing1_4": 0.0476, "Bearing1_5": 0.0005,
        "Bearing1_6": 0.0892, "Bearing1_7": 0.0009,
        "Bearing2_3": 0.0027, "Bearing2_4": 0.0014, "Bearing2_5": 0.2538,
        "Bearing2_6": 0.0012, "Bearing2_7": 0.0075,
        "Bearing3_3": 0.0068, "avg": 0.0375,
    },
}

TRIVIAL_BASELINE = {
    "Bearing1_3": 0.0933, "Bearing1_4": 0.0655, "Bearing1_5": 0.0070,
    "Bearing1_6": 0.0834, "Bearing1_7": 0.0086,
    "Bearing2_3": 0.1013, "Bearing2_4": 0.1027, "Bearing2_5": 0.0834,
    "Bearing2_6": 0.0068, "Bearing2_7": 0.0121,
    "Bearing3_3": 0.0716,
}

ALL_BEARINGS = [
    "Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7",
    "Bearing2_3", "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7",
    "Bearing3_3",
]

FPT = {
    "Bearing1_3": 60, "Bearing1_4": 76, "Bearing1_5": 98,
    "Bearing1_6": 67, "Bearing1_7": 97,
    "Bearing2_3": 13, "Bearing2_4": 52, "Bearing2_5": 0,
    "Bearing2_6": 98, "Bearing2_7": 97,
    "Bearing3_3": 73,
}


def load_results():
    """Load all completed experiment results."""
    results = {}

    for jf in sorted(RESULTS_DIR.rglob("*_results.json")):
        if jf.parent == RESULTS_DIR:
            continue
        try:
            with open(jf) as f:
                d = json.load(f)
            model = d.get("model")
            per_b = d.get("per_bearing", {})
            if not per_b or not model:
                continue
            if model not in results:
                results[model] = {}
            for bname, res in per_b.items():
                results[model][bname] = res.get("mse")
        except Exception as e:
            print(f"  Warning: {jf}: {e}")

    # JEPA+HC
    jepa_path = RESULTS_DIR / "jepa_hc_all_results.json"
    if jepa_path.exists():
        with open(jepa_path) as f:
            d = json.load(f)
        jepa = {}
        for cond_d in d.values():
            for bname, r in cond_d.get("per_bearing", {}).items():
                jepa[bname] = r.get("mse")
        if jepa:
            results["jepa_hc"] = jepa

    return results


def print_full_table(results):
    """Print comprehensive comparison table."""
    methods_order = ["simclr", "supcon", "dcssl", "jepa_hc"]
    present = [m for m in methods_order if m in results]

    print("\n" + "=" * 120)
    print("FULL COMPARISON TABLE: Ours vs Paper (MSE)")
    print("=" * 120)

    # Header
    header = f"{'Bearing':<15} {'FPT%':>5}"
    header += f"  {'Trivial':>8}"
    for m in ["simclr", "supcon", "dcssl"]:
        header += f"  {'Paper_'+m:>13}"
    for m in present:
        header += f"  {'Ours_'+m:>12}"
    print(header)
    print("-" * 120)

    wins = {m: 0 for m in present}
    our_avgs = {m: [] for m in present}

    for b in ALL_BEARINGS:
        row = f"{b:<15} {FPT[b]:>4}%"
        triv = TRIVIAL_BASELINE.get(b, float('nan'))
        row += f"  {triv:>8.4f}"
        for m in ["simclr", "supcon", "dcssl"]:
            v = PAPER[m].get(b)
            row += f"  {v:>13.4f}" if v is not None else f"  {'—':>13}"
        for m in present:
            v = results[m].get(b)
            if v is not None:
                our_avgs[m].append(v)
                # Mark wins vs paper DCSSL
                paper_dcssl = PAPER["dcssl"].get(b, float("inf"))
                marker = "*" if v < paper_dcssl else " "
                row += f"  {v:>10.4f}{marker} "
            else:
                row += f"  {'—':>12} "
        print(row)

    print("-" * 120)

    # Averages
    avg_row = f"{'Average':<15} {'':>5}  {np.mean(list(TRIVIAL_BASELINE.values())):>8.4f}"
    for m in ["simclr", "supcon", "dcssl"]:
        avg = PAPER[m].get("avg") or np.mean([PAPER[m].get(b, float('nan')) for b in ALL_BEARINGS
                                               if PAPER[m].get(b) is not None])
        avg_row += f"  {avg:>13.4f}"
    for m in present:
        avg = np.mean(our_avgs[m]) if our_avgs[m] else float('nan')
        n = len(our_avgs[m])
        avg_row += f"  {avg:>10.4f}  " if not np.isnan(avg) else f"  {'—':>10}  "
    print(avg_row)
    print("=" * 120)
    print("  * = beats paper DCSSL on this bearing")

    # Summary
    print("\nSUMMARY")
    print("-" * 80)
    trivial_avg = np.mean(list(TRIVIAL_BASELINE.values()))
    dcssl_paper_avg = PAPER["dcssl"]["avg"]
    for m in present:
        if not our_avgs[m]:
            continue
        avg = np.mean(our_avgs[m])
        n = len(our_avgs[m])
        vs_trivial = (trivial_avg - avg) / trivial_avg * 100
        vs_dcssl = (dcssl_paper_avg - avg) / dcssl_paper_avg * 100
        n_wins = sum(1 for b in ALL_BEARINGS if results[m].get(b, float('inf')) < PAPER["dcssl"].get(b, float('inf')))
        status = "BETTER" if avg < dcssl_paper_avg else "WORSE"
        print(f"  {m:<12}: avg={avg:.4f} ({n}/11 bearings)")
        print(f"    vs trivial: {vs_trivial:+.1f}% | vs paper DCSSL: {vs_dcssl:+.1f}% [{status}]")
        print(f"    Wins vs paper DCSSL: {n_wins}/11 bearings")

    return results, our_avgs


def check_sanity(results, our_avgs):
    """Run 5-minute sanity check on all results."""
    print("\n5-MINUTE SANITY CHECK")
    print("-" * 60)
    trivial_avg = np.mean(list(TRIVIAL_BASELINE.values()))

    all_ok = True
    for m, bearings in results.items():
        if not bearings:
            continue
        avg = np.mean([v for v in bearings.values() if v is not None])

        # Check 1: beats trivial?
        beats_trivial = avg < trivial_avg
        # Check 2: reasonable range?
        reasonable = 0.0 < avg < 1.0
        # Check 3: per-bearing reasonable?
        per_bearing_ok = all(0 <= v <= 1.5 for v in bearings.values() if v is not None)

        status = "OK" if (beats_trivial and reasonable and per_bearing_ok) else "WARN"
        if status == "WARN":
            all_ok = False

        print(f"  {m:<12}: avg={avg:.4f} | beats_trivial={beats_trivial} | "
              f"reasonable={reasonable} | per_bearing_ok={per_bearing_ok} [{status}]")

    if all_ok:
        print("  ALL SANITY CHECKS PASSED")
    else:
        print("  WARNING: Some checks failed — review before claiming results")

    return all_ok


def generate_per_condition_summary(results):
    """Summarize by condition."""
    print("\nPER-CONDITION BREAKDOWN")
    print("-" * 80)
    conds = {
        1: ["Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7"],
        2: ["Bearing2_3", "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7"],
        3: ["Bearing3_3"],
    }
    methods_present = [m for m in ["simclr", "supcon", "dcssl", "jepa_hc"] if m in results]

    for cond, bearings in conds.items():
        print(f"\nCondition {cond} (FPT range: {[FPT[b] for b in bearings]}%)")
        for m in methods_present:
            vals = [results[m].get(b) for b in bearings if results[m].get(b) is not None]
            if vals:
                paper_dcssl_vals = [PAPER["dcssl"].get(b, float('inf')) for b in bearings
                                    if results[m].get(b) is not None]
                avg = np.mean(vals)
                paper_avg = np.mean(paper_dcssl_vals)
                n_wins = sum(1 for v, pv in zip(vals, paper_dcssl_vals) if v < pv)
                print(f"  {m:<12}: avg={avg:.4f} | paper_dcssl_avg={paper_avg:.4f} | wins={n_wins}/{len(vals)}")


def main():
    print("=" * 60)
    print("POST-EXPERIMENT ANALYSIS")
    print("=" * 60)

    results = load_results()
    if not results:
        print("No results found.")
        return

    print(f"\nFound results for: {list(results.keys())}")
    methods_status = {m: len(v) for m, v in results.items()}
    print(f"Coverage: {methods_status}")

    # Check completeness
    missing = []
    for m in ["simclr", "supcon", "dcssl", "jepa_hc"]:
        if m not in results:
            missing.append(m)
        elif len(results[m]) < 11:
            missing.append(f"{m} (only {len(results[m])}/11 bearings)")
    if missing:
        print(f"\nWARNING: Incomplete results for: {missing}")
    else:
        print("\nAll experiments complete!")

    results, our_avgs = print_full_table(results)
    check_sanity(results, our_avgs)
    generate_per_condition_summary(results)

    # Save JSON summary
    summary = {
        "our_results": {m: {b: v for b, v in bdata.items()} for m, bdata in results.items()},
        "paper_results": {m: dict(v) for m, v in PAPER.items()},
        "trivial_baseline": TRIVIAL_BASELINE,
    }
    with open(RESULTS_DIR / "post_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to: {RESULTS_DIR}/post_analysis_summary.json")


if __name__ == "__main__":
    main()
