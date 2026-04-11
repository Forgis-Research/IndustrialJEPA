#!/bin/bash
# SageMaker Setup Script for IndustrialJEPA
# Run once after cloning: bash scripts/sagemaker_setup.sh
# After restart, just run: bash scripts/sagemaker_run.sh

set -e

echo "=== Installing uv ==="
pip install uv

echo "=== Creating virtual environment ==="
uv venv .venv
source .venv/bin/activate

echo "=== Installing dependencies ==="
uv pip install -e .
uv pip install wandb

echo "=== Setup complete! ==="
echo ""
echo "To run training:"
echo "  source .venv/bin/activate"
echo "  export WANDB_API_KEY='your_key'"
echo "  python scripts/train_world_model.py --epochs 10 --wandb"
