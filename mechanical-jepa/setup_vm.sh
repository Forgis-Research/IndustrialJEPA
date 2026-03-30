#!/bin/bash
# VM Setup Script for Mechanical-JEPA
# Usage: bash setup_vm.sh [--full]
#
# Quick setup (~2 min): Downloads CWRU only (134MB)
# Full setup (~15 min): Downloads CWRU + IMS (6.5GB total)

set -e

FULL_SETUP=false
if [[ "$1" == "--full" ]]; then
    FULL_SETUP=true
fi

echo "=============================================="
echo "Mechanical-JEPA VM Setup"
echo "=============================================="

# Check if we're in the right directory
if [[ ! -f "train.py" ]]; then
    echo "Error: Run this from the mechanical-jepa directory"
    exit 1
fi

# Install dependencies
echo ""
echo "[1/4] Installing Python dependencies..."
pip install -q -r requirements.txt

# Install Kaggle CLI if doing full setup
if $FULL_SETUP; then
    pip install -q kaggle
fi

# Create data directories
echo ""
echo "[2/4] Creating data directories..."
mkdir -p data/bearings/raw/cwru
mkdir -p data/bearings/raw/ims

# Download CWRU (always - it's quick)
echo ""
echo "[3/4] Downloading CWRU dataset (~134MB)..."
python data/bearings/prepare_bearing_dataset.py --download --dataset cwru

# Download IMS if full setup
if $FULL_SETUP; then
    echo ""
    echo "[3b/4] Downloading IMS dataset (~6GB via Kaggle)..."
    echo "Note: Requires KAGGLE_USERNAME and KAGGLE_KEY env vars"

    # Check for Kaggle credentials
    if [[ -z "$KAGGLE_USERNAME" ]] && [[ ! -f ~/.kaggle/kaggle.json ]]; then
        echo "Warning: Kaggle credentials not found."
        echo "Set KAGGLE_USERNAME and KAGGLE_KEY, or create ~/.kaggle/kaggle.json"
        echo "Skipping IMS download."
    else
        python data/bearings/prepare_bearing_dataset.py --download --dataset ims
    fi
fi

# Process datasets
echo ""
echo "[4/4] Processing datasets into unified format..."
python data/bearings/prepare_bearing_dataset.py --process

# Verify
echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
python data/bearings/prepare_bearing_dataset.py --verify

# Quick test
echo ""
echo "Running quick model test..."
python -c "
from src.models import MechanicalJEPA
import torch
model = MechanicalJEPA()
x = torch.randn(2, 3, 4096)
loss, _, _ = model(x)
print(f'Model test passed! Loss: {loss.item():.4f}')
"

echo ""
echo "Ready to train! Run:"
echo "  python train.py --epochs 30"
echo ""
if ! $FULL_SETUP; then
    echo "For full setup with IMS dataset, run:"
    echo "  bash setup_vm.sh --full"
fi
