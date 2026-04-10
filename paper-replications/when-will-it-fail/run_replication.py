"""
A2P replication runner.
Runs experiments, saves results in dcssl-replication schema.
"""

import subprocess
import json
import os
import sys
import time
import re
import datetime

RESULTS_DIR = "/home/sagemaker-user/IndustrialJEPA/paper-replications/when-will-it-fail/results"
AP_DIR = "/home/sagemaker-user/IndustrialJEPA/paper-replications/when-will-it-fail/AP"
DATA_ROOT = "/mnt/sagemaker-nvme/ad_datasets"

DATASET_PATHS = {
    "MBA": f"{DATA_ROOT}/MBA",
    "SMD": f"{DATA_ROOT}/SMD",
}

DATASET_ANOMALY_RATIO = {
    "MBA": 1.0,   # paper uses 1.0 (default)
    "SMD": 4.16,
}


def run_a2p(dataset, pred_len, seed, joint_epochs=5, cross_attn_epochs=5, extra_args=None):
    """Run A2P for one dataset/pred_len/seed combination. Returns dict with results."""
    root_path = DATASET_PATHS[dataset]
    anormly_ratio = DATASET_ANOMALY_RATIO.get(dataset, 1.0)
    model_id = f"rep_{dataset}_{pred_len}_{seed}"

    cmd = [
        "python3", "run.py",
        "--random_seed", str(seed),
        "--root_path", root_path,
        "--dataset", dataset,
        "--model_id", model_id,
        "--seq_len", str(pred_len),
        "--pred_len", str(pred_len),
        "--win_size", str(pred_len),
        "--step", str(pred_len),
        "--noise_step", str(min(pred_len, 100)),
        "--joint_epochs", str(joint_epochs),
        "--cross_attn_epochs", str(cross_attn_epochs),
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
        "--anormly_ratio", str(anormly_ratio),
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Running: {dataset} L={pred_len} seed={seed}")
    print(f"  CMD: {' '.join(cmd[-10:])}")  # last 10 args

    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd=AP_DIR, capture_output=True, text=True, timeout=3600
    )
    elapsed = time.time() - t0
    output = proc.stdout + proc.stderr

    # Parse results from output
    result = {
        "dataset": dataset,
        "pred_len": pred_len,
        "seed": seed,
        "elapsed_sec": elapsed,
        "returncode": proc.returncode,
        "f1": None,
        "precision": None,
        "recall": None,
        "mse": None,
        "raw_output_tail": output[-2000:],  # last 2000 chars
    }

    # Parse F1 line: "  [Pred]      A : 0.966133, P : 0.000000, R : 0.000000, F1 : 0.000000"
    f1_match = re.search(r'\[Pred\]\s+A\s*:\s*([\d.]+),\s*P\s*:\s*([\d.]+),\s*R\s*:\s*([\d.]+),\s*F1\s*:\s*([\d.]+)', output)
    if f1_match:
        result["precision"] = float(f1_match.group(2)) * 100
        result["recall"] = float(f1_match.group(3)) * 100
        result["f1"] = float(f1_match.group(4)) * 100

    # Parse MSE from output
    mse_match = re.search(r'mse:\s*([\d.]+)\s*\(', output)
    if mse_match:
        result["mse"] = float(mse_match.group(1))

    print(f"  => F1={result['f1']:.2f}%, P={result['precision']:.2f}%, R={result['recall']:.2f}%, t={elapsed:.0f}s" if result['f1'] is not None else f"  => FAILED (rc={proc.returncode})")
    if proc.returncode != 0:
        print(f"  STDERR tail: {proc.stderr[-500:]}")

    return result


def run_seeds(dataset, pred_len, seeds, **kwargs):
    """Run multiple seeds, return aggregated result."""
    results = []
    for seed in seeds:
        r = run_a2p(dataset, pred_len, seed, **kwargs)
        results.append(r)

    f1_vals = [r["f1"] for r in results if r["f1"] is not None]
    import numpy as np
    agg = {
        "dataset": dataset,
        "pred_len": pred_len,
        "seeds": seeds,
        "f1_mean": float(np.mean(f1_vals)) if f1_vals else None,
        "f1_std": float(np.std(f1_vals)) if len(f1_vals) > 1 else None,
        "f1_per_seed": {str(r["seed"]): r["f1"] for r in results},
        "per_seed_results": results,
    }
    f1_std_val = agg['f1_std'] if agg['f1_std'] is not None else 0.0
    print(f"\n  AGGREGATED {dataset} L={pred_len}: F1={agg['f1_mean']:.2f} +/- {f1_std_val:.2f}")
    return agg


def save_results(all_results, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {filepath}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MBA", choices=["MBA", "SMD"])
    parser.add_argument("--pred_len", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--joint_epochs", type=int, default=5)
    parser.add_argument("--cross_attn_epochs", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = run_seeds(
        dataset=args.dataset,
        pred_len=args.pred_len,
        seeds=args.seeds,
        joint_epochs=args.joint_epochs,
        cross_attn_epochs=args.cross_attn_epochs,
    )

    outfile = args.output or os.path.join(
        RESULTS_DIR,
        f"{args.dataset.lower()}_L{args.pred_len}_seeds{'_'.join(map(str, args.seeds))}.json"
    )
    save_results(result, outfile)
