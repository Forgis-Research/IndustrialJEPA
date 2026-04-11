#!/usr/bin/env python3
"""
Master script to run ALL DCSSL replication experiments in sequence.
Updates code: elapsed_time features, BatchNorm TCN, improved FPT detection.

Runs: simclr, supcon, dcssl on conditions 1, 2, 3 (9 total experiments)
Then: jepa_hc on all conditions

Estimated total time: ~4-5 hours
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT = "/home/sagemaker-user/IndustrialJEPA/dcssl-replication/run_experiments.py"
JEPA_SCRIPT = "/home/sagemaker-user/IndustrialJEPA/dcssl-replication/run_jepa_hc.py"
DATA_ROOT = "/mnt/sagemaker-nvme/femto_data/10. FEMTO Bearing"
OUTPUT_DIR = "/home/sagemaker-user/IndustrialJEPA/dcssl-replication/results"
LOG_DIR = "/home/sagemaker-user/IndustrialJEPA/dcssl-replication/logs"

Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

COMMON_ARGS = [
    "--data_root", DATA_ROOT,
    "--output_dir", OUTPUT_DIR,
    "--pretrain_epochs", "300",
    "--finetune_epochs", "150",
    "--pretrain_lr", "1e-3",
    "--finetune_lr", "5e-4",
    "--batch_size", "64",
    "--crop_length", "1024",
]

EXPERIMENTS = [
    (SCRIPT, ["--model", "simclr", "--condition", "1"]),
    (SCRIPT, ["--model", "simclr", "--condition", "2"]),
    (SCRIPT, ["--model", "simclr", "--condition", "3"]),
    (SCRIPT, ["--model", "supcon", "--condition", "1"]),
    (SCRIPT, ["--model", "supcon", "--condition", "2"]),
    (SCRIPT, ["--model", "supcon", "--condition", "3"]),
    (SCRIPT, ["--model", "dcssl", "--condition", "1"]),
    (SCRIPT, ["--model", "dcssl", "--condition", "2"]),
    (SCRIPT, ["--model", "dcssl", "--condition", "3"]),
    (JEPA_SCRIPT, ["--condition", "all"]),
]

start_time = time.time()

for i, (script, extra_args) in enumerate(EXPERIMENTS):
    exp_name = "_".join(extra_args).replace("--model_", "").replace("--condition_", "cond").replace("--", "").replace(" ", "_")
    log_path = Path(LOG_DIR) / f"exp_{i+1:02d}_{extra_args[-1]}.log"
    
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Exp {i+1}/{len(EXPERIMENTS)}: {extra_args}")
    print(f"  Log: {log_path}")
    print(f"  Time elapsed: {(time.time()-start_time)/60:.1f} min")
    print(f"{'='*60}")
    
    cmd = [sys.executable, "-u", script] + extra_args + COMMON_ARGS  # -u = unbuffered

    with open(log_path, "w") as log_file:
        proc = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    
    if proc.returncode != 0:
        print(f"  WARNING: Experiment returned code {proc.returncode}")
    else:
        print(f"  DONE. Time: {(time.time()-start_time)/60:.1f} min total")
    
    # Print last few lines of log
    try:
        with open(log_path) as f:
            lines = f.readlines()
        print("  Last output:")
        for line in lines[-8:]:
            print(f"    {line.rstrip()}")
    except:
        pass

print(f"\n{'='*60}")
print(f"ALL EXPERIMENTS COMPLETE")
print(f"Total time: {(time.time()-start_time)/60:.1f} minutes")
print(f"{'='*60}")

# Print final comparison table
import subprocess
result = subprocess.run(
    [sys.executable, 
     "/home/sagemaker-user/IndustrialJEPA/dcssl-replication/compile_results.py",
     "--results_dir", OUTPUT_DIR],
    capture_output=True, text=True,
)
print(result.stdout)
