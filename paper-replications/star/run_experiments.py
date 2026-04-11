"""
Main runner for STAR replication: all 4 C-MAPSS subsets x 5 seeds.

Usage:
  python run_experiments.py              # run all subsets
  python run_experiments.py FD001        # run specific subset
  python run_experiments.py FD001 FD002  # run multiple subsets

Results are saved to results/FDXXX_results.json after each subset.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_utils import prepare_data
from models import build_model, count_parameters
from train_utils import train, evaluate

RESULTS_DIR = Path(__file__).parent / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
EXPERIMENT_LOG = Path(__file__).parent / "EXPERIMENT_LOG.md"

SEEDS = [42, 123, 456, 789, 1024]
RUL_CAP = 125.0
MAX_EPOCHS = 200
PATIENCE = 20
VAL_FRACTION = 0.15

CONFIGS = {
    "FD001": dict(lr=0.0002, batch_size=32, window_length=32, n_scales=3, d_model=128, n_heads=1),
    "FD002": dict(lr=0.0002, batch_size=64, window_length=64, n_scales=4, d_model=64, n_heads=4),
    "FD003": dict(lr=0.0002, batch_size=32, window_length=48, n_scales=1, d_model=128, n_heads=1),
    "FD004": dict(lr=0.0002, batch_size=64, window_length=64, n_scales=4, d_model=256, n_heads=4),
}

PAPER_TARGETS = {
    "FD001": dict(rmse=10.61, score=169),
    "FD002": dict(rmse=13.47, score=784),
    "FD003": dict(rmse=10.71, score=202),
    "FD004": dict(rmse=15.87, score=1449),
}


def log_experiment(subset, seed, cfg, n_params, train_info, test_rmse, test_score, wall_time, notes=""):
    """Append an experiment entry to EXPERIMENT_LOG.md."""
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    cfg_hash = f"lr={cfg['lr']},bs={cfg['batch_size']},w={cfg['window_length']},scales={cfg['n_scales']},dm={cfg['d_model']},nh={cfg['n_heads']}"
    entry = f"""
### {ts} | {subset} seed={seed}

- **Hyperparams**: {cfg_hash}
- **Parameters**: {n_params:,}
- **Epochs run**: {train_info['epochs_run']} (best epoch {train_info['best_epoch']})
- **Best val RMSE**: {train_info['best_val_rmse']:.3f}
- **Test RMSE**: {test_rmse:.3f} (paper target: {PAPER_TARGETS[subset]['rmse']})
- **Test Score**: {test_score:.1f} (paper target: {PAPER_TARGETS[subset]['score']})
- **Wall time**: {wall_time:.0f}s
- **Notes**: {notes if notes else 'none'}

"""
    with open(EXPERIMENT_LOG, "a") as f:
        f.write(entry)


def run_subset(subset: str, device: torch.device) -> dict:
    """Run all 5 seeds for a single subset. Returns aggregated results."""
    cfg = CONFIGS[subset]
    print(f"\n{'='*60}")
    print(f"RUNNING {subset} | Config: {cfg}")
    print(f"Paper targets: RMSE={PAPER_TARGETS[subset]['rmse']}, Score={PAPER_TARGETS[subset]['score']}")
    print(f"{'='*60}")

    checkpoint_dir = CHECKPOINT_DIR / subset
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    all_rmse = []
    all_score = []
    all_preds = []
    all_trues = []
    seed_results = []

    for seed in SEEDS:
        print(f"\n--- {subset} seed={seed} ---")
        t0 = time.time()

        # Set global seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Prepare data
        data = prepare_data(
            subset,
            window_length=cfg["window_length"],
            batch_size=cfg["batch_size"],
            val_fraction=VAL_FRACTION,
            seed=seed,
            rul_cap=int(RUL_CAP),
        )

        print(f"  Data: {data['n_train_engines']} train engines, {data['n_val_engines']} val, "
              f"{data['n_test_engines']} test | {data['n_train_windows']} train windows")

        # Build model
        model = build_model(subset, device)
        n_params = count_parameters(model)
        print(f"  Model: {n_params:,} parameters")

        checkpoint_path = str(checkpoint_dir / f"seed{seed}.pt")

        # Train
        train_info = train(
            model=model,
            train_loader=data["train_loader"],
            val_loader=data["val_loader"],
            lr=cfg["lr"],
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            device=device,
            checkpoint_path=checkpoint_path,
            verbose=True,
        )

        # Evaluate
        test_rmse, test_score, preds, trues = evaluate(model, data["test_loader"], device)
        wall_time = time.time() - t0

        print(f"  RESULT: RMSE={test_rmse:.3f}, Score={test_score:.1f} | "
              f"Paper: RMSE={PAPER_TARGETS[subset]['rmse']}, Score={PAPER_TARGETS[subset]['score']}")
        print(f"  Training: {train_info['epochs_run']} epochs, best ep={train_info['best_epoch']}, "
              f"best_val_rmse={train_info['best_val_rmse']:.3f}, time={wall_time:.0f}s")

        all_rmse.append(test_rmse)
        all_score.append(test_score)
        all_preds.append(preds)
        all_trues.append(trues)

        seed_result = {
            "subset": subset,
            "seed": seed,
            "test_rmse": test_rmse,
            "test_score": test_score,
            "best_val_rmse": train_info["best_val_rmse"],
            "best_epoch": train_info["best_epoch"],
            "epochs_run": train_info["epochs_run"],
            "n_params": n_params,
            "wall_time_s": wall_time,
            "preds": preds.tolist(),
            "trues": trues.tolist(),
        }
        seed_results.append(seed_result)

        log_experiment(subset, seed, cfg, n_params, train_info, test_rmse, test_score, wall_time)

    # Aggregate
    rmse_mean = float(np.mean(all_rmse))
    rmse_std = float(np.std(all_rmse))
    score_mean = float(np.mean(all_score))
    score_std = float(np.std(all_score))

    print(f"\n{'='*60}")
    print(f"{subset} SUMMARY (5 seeds):")
    print(f"  RMSE: {rmse_mean:.3f} +/- {rmse_std:.3f} | Paper: {PAPER_TARGETS[subset]['rmse']}")
    print(f"  Score: {score_mean:.1f} +/- {score_std:.1f} | Paper: {PAPER_TARGETS[subset]['score']}")
    rmse_gap_pct = 100 * (rmse_mean - PAPER_TARGETS[subset]["rmse"]) / PAPER_TARGETS[subset]["rmse"]
    print(f"  RMSE gap vs paper: {rmse_gap_pct:+.1f}%")
    print(f"{'='*60}")

    result = {
        "subset": subset,
        "config": cfg,
        "seeds": SEEDS,
        "rmse_mean": rmse_mean,
        "rmse_std": rmse_std,
        "score_mean": score_mean,
        "score_std": score_std,
        "paper_rmse": PAPER_TARGETS[subset]["rmse"],
        "paper_score": PAPER_TARGETS[subset]["score"],
        "rmse_gap_pct": rmse_gap_pct,
        "per_seed": seed_results,
        "all_rmse": all_rmse,
        "all_score": all_score,
    }

    # Save JSON
    out_path = RESULTS_DIR / f"{subset}_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved to {out_path}")

    return result


def generate_rul_plot(subset: str, results: dict):
    """Generate predicted vs true RUL plot sorted by true RUL."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plots_dir = RESULTS_DIR / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Use median seed result (closest to mean RMSE)
        all_rmse = results["all_rmse"]
        median_idx = int(np.argsort(all_rmse)[len(all_rmse) // 2])
        seed_res = results["per_seed"][median_idx]
        preds = np.array(seed_res["preds"])
        trues = np.array(seed_res["trues"])

        # Sort by true RUL
        sort_idx = np.argsort(trues)
        preds_sorted = preds[sort_idx]
        trues_sorted = trues[sort_idx]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(trues_sorted, label="True RUL", color="black", linewidth=1.5)
        ax.plot(preds_sorted, label="Predicted RUL", color="blue", alpha=0.7, linewidth=1)
        ax.set_xlabel("Test engine (sorted by true RUL)")
        ax.set_ylabel("RUL (cycles)")
        ax.set_title(f"{subset}: Predicted vs True RUL | RMSE={results['rmse_mean']:.2f} +/- {results['rmse_std']:.2f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        out_path = plots_dir / f"rul_{subset}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  RUL plot saved to {out_path}")
    except Exception as e:
        print(f"  Warning: could not generate plot: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run STAR experiments on C-MAPSS subsets")
    parser.add_argument("subsets", nargs="*", default=["FD001", "FD002", "FD003", "FD004"],
                        help="Subsets to run (default: all 4)")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Subsets to run: {args.subsets}")
    print(f"Seeds: {SEEDS}")
    print(f"Max epochs: {MAX_EPOCHS}, Patience: {PATIENCE}")

    RESULTS_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    all_results = {}
    for subset in args.subsets:
        if subset not in CONFIGS:
            print(f"Unknown subset: {subset}, skipping")
            continue
        results = run_subset(subset, device)
        all_results[subset] = results
        generate_rul_plot(subset, results)

    # Print summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"{'Subset':<8} {'RMSE_mean':>10} {'RMSE_std':>9} {'Score_mean':>11} {'Paper_RMSE':>11} {'Gap%':>7}")
    for subset, res in all_results.items():
        print(f"{subset:<8} {res['rmse_mean']:>10.3f} {res['rmse_std']:>9.3f} "
              f"{res['score_mean']:>11.1f} {res['paper_rmse']:>11.2f} {res['rmse_gap_pct']:>+7.1f}%")

    return all_results


if __name__ == "__main__":
    main()
