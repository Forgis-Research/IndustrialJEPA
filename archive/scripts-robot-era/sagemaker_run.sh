#!/bin/bash
# Quick run script for SageMaker (after setup)
# Usage: bash scripts/sagemaker_run.sh [epochs]

EPOCHS=${1:-10}

cd ~/IndustrialJEPA
source .venv/bin/activate

# Load from .env if it exists
if [ -f ~/.env ]; then
    export $(cat ~/.env | xargs)
fi

# Or set manually: export WANDB_API_KEY="your_key"

echo "=== Starting training for $EPOCHS epochs ==="
python scripts/train_world_model.py --epochs $EPOCHS --wandb
