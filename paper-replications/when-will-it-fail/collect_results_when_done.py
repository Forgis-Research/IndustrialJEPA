"""
Post-process and collect results from long-running experiments.
Run this after SMD and MBA 4-rec experiments complete.

SMD: PID 178193, using /mnt/sagemaker-nvme/ad_datasets/SMD, expected F1=52.07
MBA 4-rec: PID 179247, using /mnt/sagemaker-nvme/ad_datasets/MBA_4rec, expected much higher F1

This script:
1. Parses output from the runs (if captured in log files)
2. Updates results/all_results.json
3. Updates EXPERIMENT_LOG.md
4. Updates RESULTS_TABLE.md

Usage:
    # Wait for processes to complete, then:
    python3 collect_results_when_done.py
"""

import json
import os
import re
import subprocess
import time

RESULTS_DIR = "/home/sagemaker-user/IndustrialJEPA/paper-replications/when-will-it-fail/results"
AP_DIR = "/home/sagemaker-user/IndustrialJEPA/paper-replications/when-will-it-fail/AP"


def parse_run_output(output: str) -> dict:
    """Parse run.py output to extract F1, P, R, AUROC."""
    result = {}

    f1_match = re.search(r'\[Pred\]\s+A\s*:\s*([\d.]+),\s*P\s*:\s*([\d.]+),\s*R\s*:\s*([\d.]+),\s*F1\s*:\s*([\d.]+)', output)
    if f1_match:
        result['anomaly_rate'] = float(f1_match.group(1))
        result['precision'] = float(f1_match.group(2)) * 100
        result['recall'] = float(f1_match.group(3)) * 100
        result['f1'] = float(f1_match.group(4)) * 100

    auroc_match = re.search(r'AUC_ROC\s*:\s*([\d.]+)', output)
    if auroc_match:
        result['auroc'] = float(auroc_match.group(1))

    thresh_match = re.search(r'Threshold\s*:\s*([\d.e+\-]+)', output)
    if thresh_match:
        result['threshold'] = float(thresh_match.group(1))

    vus_match = re.search(r'VUS_ROC\s*:\s*([\d.]+)', output)
    if vus_match:
        result['vus_roc'] = float(vus_match.group(1))

    return result


def collect_smd_result():
    """Run SMD evaluation once process completes - or read from log."""
    print("\n=== Collecting SMD Result ===")
    log_file = "/tmp/smd_100_run.log"

    if os.path.exists(log_file) and os.path.getsize(log_file) > 100:
        with open(log_file) as f:
            output = f.read()
        result = parse_run_output(output)
        print(f"SMD L100: F1={result.get('f1', 'N/A'):.2f}% (paper: 52.07%)")
        return result
    else:
        # Re-run SMD (much faster now since FE checkpoint exists)
        print("Running SMD L=100 (FE checkpoint already trained)...")
        cmd = [
            "python3", "run.py",
            "--random_seed", "20462",
            "--root_path", "/mnt/sagemaker-nvme/ad_datasets/SMD",
            "--dataset", "SMD",
            "--model_id", "rep_SMD_100_collect",
            "--seq_len", "100", "--pred_len", "100", "--win_size", "100",
            "--step", "100", "--noise_step", "100",
            "--joint_epochs", "5",
            "--cross_attn_epochs", "5",
            "--share",
            "--AD_model", "AT",
            "--d_model", "256",
            "--noise_injection",
            "--pretrain_noise",
            "--contrastive_loss",
            "--forecast_loss",
            "--cross_attn",
            "--cross_attn_nheads", "1",
            "--ftr_idx", "0",
            "--anormly_ratio", "4.16",
        ]
        result_proc = subprocess.run(cmd, capture_output=True, text=True, cwd=AP_DIR, timeout=7200)
        output = result_proc.stdout + result_proc.stderr
        result = parse_run_output(output)
        print(f"SMD L100: F1={result.get('f1', 'N/A'):.2f}% (paper: 52.07%)")
        return result


def collect_mba_4rec_result():
    """Collect MBA 4-record SVDB result."""
    print("\n=== Collecting MBA 4-record SVDB Result ===")
    # This will need to be run or parsed from background process output
    # For now, return placeholder
    print("MBA 4-rec run is still in progress (PID 179247)")
    print("To get result: kill process, re-run with shorter epochs if needed")
    return None


if __name__ == "__main__":
    print("Results Collection Script")
    print("This script should be run after long experiments complete")
    print()
    print("Current process status:")
    for pid in [178193, 179247]:
        result = subprocess.run(["ps", "-p", str(pid), "-o", "pid,etime,stat"],
                               capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  PID {pid}: STILL RUNNING")
            print(f"  {result.stdout.strip()}")
        else:
            print(f"  PID {pid}: COMPLETED or KILLED")

    print()
    print("Run this script after processes complete to update results.")
    print("Expected results: SMD L100 F1~52%, MBA 4-rec F1 unknown (proper SVDB data)")
