#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Train all baseline models for comparison with IndustrialJEPA.

Usage:
    python scripts/train_all_baselines.py --epochs 10 --data-source aursad
    python scripts/train_all_baselines.py --models effort_ae s2e --epochs 20
"""

import argparse
import subprocess
import sys
from pathlib import Path


BASELINE_MODELS = [
    # Core baselines for comparison
    "effort_ae",     # Effort-only autoencoder (tests if causal structure matters)
    "s2e",           # Setpoint→Effort (direct prediction baseline)
    "temporal",      # Temporal self-prediction (JEPA-style, closest baseline)
    # Additional baselines
    "mae",           # Masked AutoEncoder
    "contrastive",   # SimCLR-style contrastive
]


def main():
    parser = argparse.ArgumentParser(description="Train all baseline models")
    parser.add_argument(
        "--models", nargs="+", default=BASELINE_MODELS,
        choices=BASELINE_MODELS,
        help="Which models to train (default: all)",
    )
    parser.add_argument(
        "--data-source", type=str, default="aursad",
        choices=["aursad", "voraus", "cnc", "hackathon"],
        help="Data source to use",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of epochs per model",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/baselines",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable WandB logging",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None,
        help="Max episodes to load (memory optimization). None = use all.",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training {len(args.models)} baseline models on {args.data_source}")
    print(f"Output: {output_dir}")
    print("-" * 60)

    for model in args.models:
        print(f"\n{'='*60}")
        print(f"Training: {model}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, "-m", "industrialjepa.baselines.train",
            "--model", model,
            "--data-source", args.data_source,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--output-dir", str(output_dir),
        ]

        if args.wandb:
            cmd.append("--wandb")

        if args.max_episodes is not None:
            cmd.extend(["--max-episodes", str(args.max_episodes)])

        # Model-specific flags
        if model == "temporal":
            cmd.extend(["--temporal-mode", "jepa"])  # Use JEPA-style temporal

        print(f"Command: {' '.join(cmd)}")

        if args.dry_run:
            print("(dry run - not executing)")
            continue

        try:
            result = subprocess.run(cmd, check=True)
            print(f"\n[OK] {model} training complete")
        except subprocess.CalledProcessError as e:
            print(f"\n[FAIL] {model} training failed with exit code {e.returncode}")
            continue
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            sys.exit(1)

    print(f"\n{'='*60}")
    print("All baselines complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
