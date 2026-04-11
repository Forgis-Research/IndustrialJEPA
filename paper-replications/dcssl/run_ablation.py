"""
Ablation study for DCSSL hyperparameters.

Focuses on:
1. Temperature τ
2. Temporal/instance window sizes
3. Lambda weights for dual losses
4. Crop length (input window size)

All ablations run on Condition 1 only (fastest feedback).
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from data_utils import load_condition_data, FEMTOPretrainDataset, FEMTORULDataset
from models import DCSSSLModel
from train_utils import run_full_pipeline

DATA_ROOT = Path("/mnt/sagemaker-nvme/femto_data/10. FEMTO Bearing")
OUTPUT_DIR = Path("/home/sagemaker-user/IndustrialJEPA/dcssl-replication/results/ablation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Fixed training config for ablations
BASE_CONFIG = {
    "pretrain_epochs": 200,
    "finetune_epochs": 100,
    "pretrain_lr": 1e-3,
    "finetune_lr": 5e-4,
    "batch_size": 64,
    "crop_length": 1024,
}

# Ablation configurations
ABLATIONS = [
    # Temperature ablation
    {"name": "tau_005", "temperature": 0.05, "lambda_temporal": 1.0, "lambda_instance": 1.0, "temporal_window": 0.1, "rul_window": 0.1, "crop_length": 1024},
    {"name": "tau_007", "temperature": 0.07, "lambda_temporal": 1.0, "lambda_instance": 1.0, "temporal_window": 0.1, "rul_window": 0.1, "crop_length": 1024},
    {"name": "tau_010", "temperature": 0.10, "lambda_temporal": 1.0, "lambda_instance": 1.0, "temporal_window": 0.1, "rul_window": 0.1, "crop_length": 1024},
    {"name": "tau_020", "temperature": 0.20, "lambda_temporal": 1.0, "lambda_instance": 1.0, "temporal_window": 0.1, "rul_window": 0.1, "crop_length": 1024},
    # Window ablation
    {"name": "win_005", "temperature": 0.1, "lambda_temporal": 1.0, "lambda_instance": 1.0, "temporal_window": 0.05, "rul_window": 0.05, "crop_length": 1024},
    {"name": "win_015", "temperature": 0.1, "lambda_temporal": 1.0, "lambda_instance": 1.0, "temporal_window": 0.15, "rul_window": 0.15, "crop_length": 1024},
    # Lambda ablation
    {"name": "lambda_t05i05", "temperature": 0.1, "lambda_temporal": 0.5, "lambda_instance": 0.5, "temporal_window": 0.1, "rul_window": 0.1, "crop_length": 1024},
    {"name": "lambda_t2i2", "temperature": 0.1, "lambda_temporal": 2.0, "lambda_instance": 2.0, "temporal_window": 0.1, "rul_window": 0.1, "crop_length": 1024},
    {"name": "lambda_t0i0", "temperature": 0.1, "lambda_temporal": 0.0, "lambda_instance": 0.0, "temporal_window": 0.1, "rul_window": 0.1, "crop_length": 1024},
    # Crop length
    {"name": "crop_512",  "temperature": 0.1, "lambda_temporal": 1.0, "lambda_instance": 1.0, "temporal_window": 0.1, "rul_window": 0.1, "crop_length": 512},
    {"name": "crop_2048", "temperature": 0.1, "lambda_temporal": 1.0, "lambda_instance": 1.0, "temporal_window": 0.1, "rul_window": 0.1, "crop_length": 2048},
]


def run_ablation(ablation_config: dict, train_data: list, test_data: list) -> dict:
    """Run one ablation experiment."""
    name = ablation_config["name"]
    print(f"\n  Ablation: {name}")

    model = DCSSSLModel(
        in_channels=2,
        encoder_hidden=64,
        encoder_out=128,
        n_tcn_blocks=8,
        kernel_size=3,
        dropout=0.1,
        temperature=ablation_config["temperature"],
        lambda_temporal=ablation_config["lambda_temporal"],
        lambda_instance=ablation_config["lambda_instance"],
        temporal_window=ablation_config["temporal_window"],
        rul_window=ablation_config["rul_window"],
    ).to(DEVICE)

    crop_length = ablation_config.get("crop_length", 1024)

    results = run_full_pipeline(
        model, train_data, test_data,
        output_dir=OUTPUT_DIR / name,
        model_name=name,
        pretrain_epochs=BASE_CONFIG["pretrain_epochs"],
        finetune_epochs=BASE_CONFIG["finetune_epochs"],
        pretrain_lr=BASE_CONFIG["pretrain_lr"],
        finetune_lr=BASE_CONFIG["finetune_lr"],
        batch_size=BASE_CONFIG["batch_size"],
        crop_length=crop_length,
        device=DEVICE,
        verbose=False,  # Less verbose for ablations
    )

    print(f"    {name}: avg_mse = {results['avg_mse']:.4f}")
    return results


def main():
    print("Loading condition 1 data...")
    train_data, test_data = load_condition_data(DATA_ROOT, 1, verbose=False)
    print(f"Train: {len(train_data)} bearings | Test: {len(test_data)} bearings")

    all_ablation_results = {}

    for ablation in ABLATIONS:
        try:
            results = run_ablation(ablation, train_data, test_data)
            all_ablation_results[ablation["name"]] = {
                "config": ablation,
                "avg_mse": results["avg_mse"],
                "per_bearing": {k: v["mse"] for k, v in results["per_bearing"].items()},
            }
        except Exception as e:
            print(f"    ERROR: {e}")
            all_ablation_results[ablation["name"]] = {"error": str(e)}

        # Save after each ablation
        save_path = OUTPUT_DIR / "ablation_results.json"
        with open(save_path, "w") as f:
            json.dump(all_ablation_results, f, indent=2)

    # Print summary table
    print("\n" + "="*80)
    print("ABLATION RESULTS SUMMARY (Condition 1 only)")
    print("="*80)
    print(f"{'Config':<25} {'Avg MSE':>10} | {'1_3':>8} {'1_4':>8} {'1_5':>8} {'1_6':>8} {'1_7':>8}")
    print("-"*80)

    sorted_results = sorted(
        [(k, v) for k, v in all_ablation_results.items() if "error" not in v],
        key=lambda x: x[1]["avg_mse"]
    )

    for name, res in sorted_results:
        pb = res.get("per_bearing", {})
        row = f"  {name:<23} {res['avg_mse']:>10.4f} |"
        for bearing in ["Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7"]:
            v = pb.get(bearing, None)
            row += f" {v:>8.4f}" if v is not None else f" {'—':>8}"
        print(row)

    print(f"\n  Paper DCSSL (cond 1 avg): 0.0280")
    print(f"  Paper SimCLR (cond 1 avg): 0.0638")


if __name__ == "__main__":
    main()
