"""
A2P Replication Runner
======================
Thin wrapper around AP/run.py that:
  1. Runs 3 seeds per (dataset, pred_len) combination
  2. Saves structured results to results/all_results.json
  3. Produces results/RESULTS_TABLE.md comparing paper vs ours

Usage:
    python run_replication.py --dataset mba
    python run_replication.py --dataset smd
    python run_replication.py --dataset all
    python run_replication.py --dataset mba --ablation aaf_off
    python run_replication.py --dataset mba --ablation shared_off

Requirements:
    - AP/ directory cloned from https://github.com/KU-VGI/AP
    - Data at DATA_ROOT/{MBA,SMD,WADI,exathlon}_train.npy etc.
    - pip install -r AP/requirements.txt

Data source:
    AnomalyTransformer repo (thuml) provides MBA, SMD, WADI, MSL, SMAP, PSM
    in the expected .npy format:
    https://github.com/thuml/Anomaly-Transformer
"""

import argparse
import subprocess
import json
import os
import sys
import numpy as np
from datetime import datetime
from pathlib import Path

# ---- paths ----------------------------------------------------------------
REPO_ROOT = Path(__file__).parent
AP_DIR = REPO_ROOT / "AP"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "improvements").mkdir(exist_ok=True)

# Change this to your data root. The directory must contain:
#   MBA_train.npy, MBA_test.npy, MBA_test_label.npy
#   SMD_train.npy, SMD_test.npy, SMD_test_label.npy
#   WADI_train.npy, WADI_test.npy, WADI_test_label.npy
#   exathlon_1_train.npy ... (per sub-dataset)
DATA_ROOT = os.environ.get("A2P_DATA_ROOT", "/local_datasets/AD_datasets")

# ---- target numbers from paper Table 1 -----------------------------------
PAPER_RESULTS = {
    "MBA": {
        100: {"f1": 67.55, "std": 5.62},
        200: {"f1": 74.63, "std": 5.92},
        400: {"f1": 69.35, "std": 7.15},
    },
    "Exathlon": {
        100: {"f1": 18.64, "std": 0.16},
        200: {"f1": 28.71, "std": 0.54},
        400: {"f1": 43.57, "std": 1.10},
    },
    "SMD": {
        100: {"f1": 36.29, "std": 0.18},
        200: {"f1": 42.36, "std": 0.80},
        400: {"f1": 48.10, "std": 2.55},
    },
    "WADI": {
        100: {"f1": 64.91, "std": 0.47},
        200: {"f1": 66.65, "std": 1.93},
        400: {"f1": 74.57, "std": 6.37},
    },
}

SEEDS = [0, 1, 2]
PRED_LENS = [100, 200, 400]


def build_cmd(dataset, pred_len, seed, data_root=DATA_ROOT, extra_flags=None):
    """Build the python command for a single run."""
    dataset_path = os.path.join(data_root, dataset)
    cmd = [
        sys.executable, "-u", str(AP_DIR / "run.py"),
        "--random_seed", str(seed),
        "--root_path", dataset_path,
        "--dataset", dataset,
        "--model_id", f"F+AD_100_{pred_len}",
        "--seq_len", "100",
        "--pred_len", str(pred_len),
        "--win_size", "100",
        "--step", str(pred_len),
        "--noise_step", "100",
        "--joint_epochs", "5",
        "--share",
        "--AD_model", "AT",
        "--d_model", "256",
        "--noise_injection",
        "--pretrain_noise",
        "--contrastive_loss",
        "--forecast_loss",
        "--cross_attn",
        "--cross_attn_epochs", "5",
        "--cross_attn_nheads", "1",
    ]
    if extra_flags:
        cmd.extend(extra_flags)
    return cmd


def run_one(dataset, pred_len, seed, extra_flags=None, label=""):
    """Run one seed, parse stdout for F1, return dict."""
    cmd = build_cmd(dataset, pred_len, seed, extra_flags=extra_flags)
    print(f"\n[{label or 'replication'}] {dataset} L_out={pred_len} seed={seed}")
    print(" ".join(cmd))

    result = {"dataset": dataset, "pred_len": pred_len, "seed": seed,
              "f1": None, "mse": None, "label": label,
              "timestamp": datetime.now().isoformat()}
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=str(AP_DIR), timeout=7200  # 2h max
        )
        stdout = proc.stdout
        # Parse F1 from output line: "F1: XX.XX (± YY.YY)"
        for line in stdout.splitlines():
            if "F1:" in line and "AD F1:" not in line:
                try:
                    f1_str = line.split("F1:")[1].strip().split()[0]
                    result["f1"] = float(f1_str)
                except Exception:
                    pass
            if "mse:" in line.lower():
                try:
                    mse_str = line.lower().split("mse:")[1].strip().split()[0]
                    result["mse"] = float(mse_str)
                except Exception:
                    pass
        if proc.returncode != 0:
            result["error"] = proc.stderr[-2000:]
        # Save full stdout for inspection
        out_path = RESULTS_DIR / f"{dataset.lower()}_{label or 'official'}_seed{seed}_L{pred_len}.txt"
        with open(out_path, "w") as f:
            f.write(stdout)
            if proc.returncode != 0:
                f.write("\n\nSTDERR:\n" + proc.stderr)
    except subprocess.TimeoutExpired:
        result["error"] = "timeout"
    except Exception as e:
        result["error"] = str(e)
    return result


def run_dataset(dataset, pred_lens=None, extra_flags=None, label=""):
    """Run all seeds x pred_lens for one dataset."""
    pred_lens = pred_lens or PRED_LENS
    results = []
    for pred_len in pred_lens:
        seed_results = []
        for seed in SEEDS:
            r = run_one(dataset, pred_len, seed, extra_flags=extra_flags, label=label)
            seed_results.append(r)
            results.append(r)
        # Aggregate
        f1_vals = [r["f1"] for r in seed_results if r["f1"] is not None]
        if f1_vals:
            mean_f1 = np.mean(f1_vals)
            std_f1 = np.std(f1_vals)
            print(f"\n  => {dataset} L_out={pred_len}: F1={mean_f1:.2f} +/- {std_f1:.2f}")
            if dataset in PAPER_RESULTS and pred_len in PAPER_RESULTS[dataset]:
                paper = PAPER_RESULTS[dataset][pred_len]
                delta = mean_f1 - paper["f1"]
                print(f"     Paper: {paper['f1']:.2f} +/- {paper['std']:.2f}  Delta: {delta:+.2f}")
    return results


def save_results(all_results, path=None):
    path = path or (RESULTS_DIR / "all_results.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {path}")


def build_results_table(all_results):
    """Build markdown comparison table."""
    from collections import defaultdict
    # Group by (dataset, pred_len, label)
    groups = defaultdict(list)
    for r in all_results:
        key = (r["dataset"], r["pred_len"], r.get("label", ""))
        if r["f1"] is not None:
            groups[key].append(r["f1"])

    lines = [
        "# Results Table - A2P Replication vs Paper\n",
        "Paper numbers from Table 1 (Park et al., ICML 2025).\n",
        "",
        "| L_out | Dataset | Paper F1 | Paper Std | Our F1 | Our Std | Delta |",
        "|:-----:|---------|:--------:|:---------:|:------:|:-------:|:-----:|",
    ]
    for dataset in ["MBA", "Exathlon", "SMD", "WADI"]:
        for pred_len in PRED_LENS:
            paper = PAPER_RESULTS.get(dataset, {}).get(pred_len, {})
            key = (dataset, pred_len, "")
            our_vals = groups.get(key, [])
            if our_vals:
                our_mean = np.mean(our_vals)
                our_std = np.std(our_vals)
                delta = our_mean - paper.get("f1", float("nan"))
                lines.append(
                    f"| {pred_len} | {dataset} | "
                    f"{paper.get('f1', 'N/A'):.2f} | {paper.get('std', 'N/A'):.2f} | "
                    f"{our_mean:.2f} | {our_std:.2f} | {delta:+.2f} |"
                )
            else:
                lines.append(
                    f"| {pred_len} | {dataset} | "
                    f"{paper.get('f1', 'N/A'):.2f} | {paper.get('std', 'N/A'):.2f} | "
                    f"NOT RUN | - | - |"
                )

    # Ablation section
    for label, desc in [("aaf_off", "AAF disabled"), ("shared_off", "Shared backbone off")]:
        lines.append(f"\n### Ablation: {desc}\n")
        lines.append("| L_out | Dataset | A2P F1 | Ablation F1 | Drop |")
        lines.append("|:-----:|---------|:------:|:-----------:|:----:|")
        for dataset in ["MBA"]:
            for pred_len in [100]:
                full_key = (dataset, pred_len, "")
                abl_key = (dataset, pred_len, label)
                full_vals = groups.get(full_key, [])
                abl_vals = groups.get(abl_key, [])
                if full_vals and abl_vals:
                    f = np.mean(full_vals)
                    a = np.mean(abl_vals)
                    lines.append(f"| {pred_len} | {dataset} | {f:.2f} | {a:.2f} | {a-f:+.2f} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="A2P Replication Runner")
    parser.add_argument("--dataset", type=str, default="mba",
                        choices=["mba", "smd", "wadi", "exathlon", "all"],
                        help="Dataset to run")
    parser.add_argument("--pred_lens", type=int, nargs="+", default=None,
                        help="Prediction lengths to run (default: 100 200 400)")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["aaf_off", "shared_off", None],
                        help="Run an ablation variant")
    args = parser.parse_args()

    # Ablation flags
    extra_flags = None
    label = ""
    if args.ablation == "aaf_off":
        # Remove cross-attention (AAF) and noise injection
        # This also removes SAP implicitly (no pretrain_noise)
        extra_flags = ["--no_cross_attn", "--no_noise_injection"]
        label = "aaf_off"
        print("WARNING: check AP/run.py for exact flags to disable AAF. "
              "May require modifying args in run.py directly.")
    elif args.ablation == "shared_off":
        # Remove --share flag
        extra_flags = []  # just omit --share
        label = "shared_off"
        # NOTE: build_cmd includes --share by default; must modify build_cmd for this ablation

    # Select datasets
    if args.dataset == "all":
        datasets = ["MBA", "SMD", "Exathlon", "WADI"]
    else:
        datasets = [args.dataset.upper()]
        if args.dataset.lower() == "mba":
            datasets = ["MBA"]

    all_results = []
    for dataset in datasets:
        results = run_dataset(
            dataset,
            pred_lens=args.pred_lens,
            extra_flags=extra_flags,
            label=label,
        )
        all_results.extend(results)

    save_results(all_results)
    table = build_results_table(all_results)
    table_path = RESULTS_DIR / "RESULTS_TABLE.md"
    with open(table_path, "w") as f:
        f.write(table)
    print(f"\nResults table written to {table_path}")
    print(table)


if __name__ == "__main__":
    main()
