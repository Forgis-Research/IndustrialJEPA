"""
Main experiment runner for DCSSL replication.

Runs all methods (SimCLR, SupCon, DCSSL) on all 3 FEMTO conditions
and produces a results table matching Table 3 from the paper.

Usage:
    python run_experiments.py --data_root /mnt/sagemaker-nvme/femto_data
    python run_experiments.py --data_root /mnt/sagemaker-nvme/femto_data --condition 1
    python run_experiments.py --data_root /mnt/sagemaker-nvme/femto_data --model dcssl --condition 1
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import sys

# Add dcssl-replication to path
sys.path.insert(0, str(Path(__file__).parent))

from data_utils import load_condition_data, get_data_stats
from models import DCSSSLModel, SimCLRModel, SupConModel, count_parameters
from train_utils import run_full_pipeline


# =====================================================================
# Paper target results for comparison
# =====================================================================
PAPER_RESULTS = {
    "DCSSL": {
        "Bearing1_3": 0.0011, "Bearing1_4": 0.0476, "Bearing1_5": 0.0005,
        "Bearing1_6": 0.0892, "Bearing1_7": 0.0009,
        "Bearing2_3": 0.0027, "Bearing2_4": 0.0014, "Bearing2_5": 0.2538,
        "Bearing2_6": 0.0012, "Bearing2_7": 0.0075,
        "Bearing3_3": 0.0068,
        "avg": 0.0375,
    },
    "SimCLR": {
        "Bearing1_3": 0.0029, "Bearing1_4": 0.2565, "Bearing1_5": 0.0030,
        "Bearing1_6": 0.0560, "Bearing1_7": 0.0006,
        "Bearing2_3": 0.0904, "Bearing2_4": 0.0021, "Bearing2_5": 0.1849,
        "Bearing2_6": 0.0024, "Bearing2_7": 0.2577,
        "Bearing3_3": 0.0013,
        "avg": 0.0583,
    },
    "SupCon": {
        "Bearing1_3": 0.0028, "Bearing1_4": 0.0080, "Bearing1_5": 0.0097,
        "Bearing1_6": 0.0473, "Bearing1_7": 0.0040,
        "Bearing2_3": 0.0569, "Bearing2_4": 0.0046, "Bearing2_5": 0.0735,
        "Bearing2_6": 0.0038, "Bearing2_7": 0.0150,
        "Bearing3_3": 0.0017,
        "avg": 0.0480,
    },
}

# Test bearings per condition
CONDITION_TEST_BEARINGS = {
    1: ["Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7"],
    2: ["Bearing2_3", "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7"],
    3: ["Bearing3_3"],
}


# =====================================================================
# Model factory
# =====================================================================

def create_model(model_name: str, device: torch.device) -> torch.nn.Module:
    """Create a fresh model instance."""
    kwargs = dict(
        in_channels=2,
        encoder_hidden=64,
        encoder_out=128,
        n_tcn_blocks=8,
        kernel_size=3,
        proj_hidden=128,
        proj_out=64,
        rul_hidden=64,
        dropout=0.1,
        temperature=0.1,
    )
    if model_name == "dcssl":
        model = DCSSSLModel(
            **kwargs,
            lambda_temporal=1.0,
            lambda_instance=1.0,
            temporal_window=0.1,
            rul_window=0.1,
        )
    elif model_name == "simclr":
        model = SimCLRModel(**kwargs)
    elif model_name == "supcon":
        model = SupConModel(**{**kwargs, "rul_window": 0.1})
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"  Model: {model_name} | Parameters: {count_parameters(model):,}")
    return model.to(device)


# =====================================================================
# Single experiment
# =====================================================================

def run_single_experiment(
    model_name: str,
    condition: int,
    data_root: Path,
    output_dir: Path,
    pretrain_epochs: int = 200,
    finetune_epochs: int = 100,
    pretrain_lr: float = 1e-3,
    finetune_lr: float = 5e-4,
    batch_size: int = 64,
    crop_length: int = 1024,
    device: torch.device = None,
    verbose: bool = True,
    fpt_threshold: float = 3.0,
) -> Dict:
    """
    Run a single (model, condition) experiment.

    Returns results dict.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name = f"{model_name}_cond{condition}"
    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name}")
        print(f"  Model: {model_name}")
        print(f"  Condition: {condition}")
        print(f"  Device: {device}")
        print(f"{'='*60}")

    # Load data
    if verbose:
        print(f"\nLoading condition {condition} data from {data_root}...")
    train_data, test_data = load_condition_data(
        data_root, condition, verbose=verbose, fpt_threshold=fpt_threshold
    )

    if not train_data:
        raise ValueError(f"No training data found for condition {condition}")
    if not test_data:
        raise ValueError(f"No test data found for condition {condition}")

    train_stats = get_data_stats(train_data)
    test_stats = get_data_stats(test_data)
    if verbose:
        print(f"\nTrain: {train_stats['n_bearings']} bearings, "
              f"{train_stats['total_snapshots']} snapshots")
        print(f"Test: {test_stats['n_bearings']} bearings, "
              f"{test_stats['total_snapshots']} snapshots")

    # Create model
    model = create_model(model_name, device)

    # Run pipeline
    exp_output_dir = output_dir / exp_name
    results = run_full_pipeline(
        model, train_data, test_data,
        output_dir=exp_output_dir,
        model_name=model_name,
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
        pretrain_lr=pretrain_lr,
        finetune_lr=finetune_lr,
        batch_size=batch_size,
        crop_length=crop_length,
        device=device,
        verbose=verbose,
    )

    # Compare to paper
    if verbose:
        print(f"\n  Comparison to paper (DCSSL baseline):")
        for bearing_name, res in results["per_bearing"].items():
            paper_dcssl = PAPER_RESULTS["DCSSL"].get(bearing_name, None)
            paper_this = PAPER_RESULTS.get(model_name.upper(), {}).get(bearing_name, None)
            our_mse = res["mse"]
            s = f"    {bearing_name}: ours={our_mse:.4f}"
            if paper_this:
                s += f" | paper_{model_name}={paper_this:.4f}"
            if paper_dcssl:
                s += f" | paper_dcssl={paper_dcssl:.4f}"
            print(s)

    return results


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="DCSSL Replication Experiments")
    parser.add_argument("--data_root", type=str,
                        default="/mnt/sagemaker-nvme/femto_data",
                        help="Path to FEMTO data root")
    parser.add_argument("--output_dir", type=str,
                        default="/home/sagemaker-user/IndustrialJEPA/dcssl-replication/results",
                        help="Output directory for results")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "simclr", "supcon", "dcssl"],
                        help="Which model to run")
    parser.add_argument("--condition", type=int, default=0,
                        help="Condition to run (0=all, 1-3=specific)")
    parser.add_argument("--pretrain_epochs", type=int, default=200)
    parser.add_argument("--finetune_epochs", type=int, default=100)
    parser.add_argument("--pretrain_lr", type=float, default=1e-3)
    parser.add_argument("--finetune_lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--crop_length", type=int, default=1024)
    parser.add_argument("--fpt_threshold", type=float, default=3.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Which models and conditions to run
    models = ["simclr", "supcon", "dcssl"] if args.model == "all" else [args.model]
    conditions = [1, 2, 3] if args.condition == 0 else [args.condition]

    all_results = {}
    start_time = time.time()

    for condition in conditions:
        for model_name in models:
            exp_key = f"{model_name}_cond{condition}"
            print(f"\n{'#'*60}")
            print(f"# Starting: {exp_key}")
            print(f"# Time elapsed: {(time.time()-start_time)/60:.1f} min")
            print(f"{'#'*60}")

            try:
                results = run_single_experiment(
                    model_name=model_name,
                    condition=condition,
                    data_root=data_root,
                    output_dir=output_dir,
                    pretrain_epochs=args.pretrain_epochs,
                    finetune_epochs=args.finetune_epochs,
                    pretrain_lr=args.pretrain_lr,
                    finetune_lr=args.finetune_lr,
                    batch_size=args.batch_size,
                    crop_length=args.crop_length,
                    device=device,
                    verbose=not args.quiet,
                    fpt_threshold=args.fpt_threshold,
                )
                all_results[exp_key] = results
            except Exception as e:
                print(f"  ERROR in {exp_key}: {e}")
                import traceback
                traceback.print_exc()
                all_results[exp_key] = {"error": str(e)}

            # Save running results after each experiment
            with open(output_dir / "all_results.json", "w") as f:
                json.dump(all_results, f, indent=2)

    # Print final summary table
    print_results_table(all_results)

    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = output_dir / f"final_results_{timestamp}.json"
    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {final_path}")
    print(f"Total time: {(time.time()-start_time)/60:.1f} minutes")


def print_results_table(all_results: Dict):
    """Print a comparison table of all results."""
    all_test_bearings = [
        "Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7",
        "Bearing2_3", "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7",
        "Bearing3_3",
    ]

    methods_in_results = {}
    for exp_key, results in all_results.items():
        if "error" in results:
            continue
        model_name = results.get("model", exp_key.split("_")[0])
        if model_name not in methods_in_results:
            methods_in_results[model_name] = {}
        per_bearing = results.get("per_bearing", {})
        methods_in_results[model_name].update(per_bearing)

    print("\n" + "="*100)
    print("RESULTS TABLE (MSE)")
    print("="*100)

    # Header
    methods = list(methods_in_results.keys())
    header = f"{'Bearing':<15}"
    for m in methods:
        header += f" {m:>12}"
    # Add paper results
    for m in ["SimCLR", "SupCon", "DCSSL"]:
        if m in PAPER_RESULTS:
            header += f" {'Paper_'+m:>14}"
    print(header)
    print("-"*100)

    # Per-bearing rows
    ours_by_method = {m: [] for m in methods}
    for bearing in all_test_bearings:
        row = f"{bearing:<15}"
        for m in methods:
            if bearing in methods_in_results[m]:
                v = methods_in_results[m][bearing]["mse"]
                row += f" {v:>12.4f}"
                ours_by_method[m].append(v)
            else:
                row += f" {'—':>12}"
        for m in ["SimCLR", "SupCon", "DCSSL"]:
            if m in PAPER_RESULTS:
                v = PAPER_RESULTS[m].get(bearing, None)
                if v is not None:
                    row += f" {v:>14.4f}"
                else:
                    row += f" {'—':>14}"
        print(row)

    print("-"*100)

    # Average row
    avg_row = f"{'Average':<15}"
    for m in methods:
        if ours_by_method[m]:
            avg = np.mean(ours_by_method[m])
            avg_row += f" {avg:>12.4f}"
        else:
            avg_row += f" {'—':>12}"
    for m in ["SimCLR", "SupCon", "DCSSL"]:
        if m in PAPER_RESULTS:
            avg_row += f" {PAPER_RESULTS[m]['avg']:>14.4f}"
    print(avg_row)
    print("="*100)


if __name__ == "__main__":
    main()
