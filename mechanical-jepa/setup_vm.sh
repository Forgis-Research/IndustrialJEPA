#!/bin/bash
# VM Setup Script for Mechanical-JEPA
# Usage: bash setup_vm.sh [OPTIONS]
#
# Options:
#   --full       Downloads CWRU + IMS (6.5GB total)
#   --paderborn  Downloads and extracts Paderborn dataset (~5GB)
#   --all        Downloads all datasets (CWRU + IMS + Paderborn)
#
# Quick setup (default): Downloads CWRU only (134MB)

set -e

FULL_SETUP=false
PADERBORN=false
ALL_DATASETS=false

for arg in "$@"; do
    case $arg in
        --full)
            FULL_SETUP=true
            ;;
        --paderborn)
            PADERBORN=true
            ;;
        --all)
            ALL_DATASETS=true
            FULL_SETUP=true
            PADERBORN=true
            ;;
    esac
done

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
echo "[1/5] Installing Python dependencies..."
pip install -q -r requirements.txt

# Install Kaggle CLI if doing full setup
if $FULL_SETUP; then
    pip install -q kaggle
fi

# Install RAR extraction tools if needed for Paderborn
if $PADERBORN; then
    echo ""
    echo "[1b/5] Checking RAR extraction tools..."
    if ! command -v unrar &> /dev/null && ! command -v 7z &> /dev/null; then
        echo "  Installing unrar..."
        # Try different package managers
        if command -v apt-get &> /dev/null; then
            sudo apt-get update -qq && sudo apt-get install -y -qq unrar || \
            sudo apt-get install -y -qq p7zip-full || \
            echo "  WARNING: Could not install unrar. Install manually."
        elif command -v yum &> /dev/null; then
            sudo yum install -y -q unrar || sudo yum install -y -q p7zip || \
            echo "  WARNING: Could not install unrar. Install manually."
        elif command -v brew &> /dev/null; then
            brew install unar || brew install p7zip || \
            echo "  WARNING: Could not install unrar. Install manually."
        else
            echo "  WARNING: Unknown package manager. Install unrar or 7z manually."
        fi
    else
        echo "  RAR extraction tool already installed."
    fi
fi

# Create data directories
echo ""
echo "[2/5] Creating data directories..."
mkdir -p data/bearings/raw/cwru
mkdir -p data/bearings/raw/ims
mkdir -p data/bearings/raw/paderborn

# Download CWRU (always - it's quick)
echo ""
echo "[3/5] Downloading CWRU dataset (~134MB)..."
python data/bearings/prepare_bearing_dataset.py --download --dataset cwru

# Download IMS if full setup
if $FULL_SETUP; then
    echo ""
    echo "[3b/5] Downloading IMS dataset (~6GB via Kaggle)..."
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

# Download and extract Paderborn if requested
if $PADERBORN; then
    echo ""
    echo "[3c/5] Downloading Paderborn dataset (~5GB)..."
    python data/bearings/prepare_bearing_dataset.py --download --dataset paderborn

    echo ""
    echo "[3d/5] Extracting Paderborn RAR files..."
    python data/bearings/prepare_bearing_dataset.py --extract
fi

# Process datasets
echo ""
echo "[4/5] Processing datasets into unified format..."
python data/bearings/prepare_bearing_dataset.py --process

# Verify
echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
python data/bearings/prepare_bearing_dataset.py --verify

# Quick test
echo ""
echo "[5/5] Running quick model test..."
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

# Print additional options
if ! $FULL_SETUP && ! $PADERBORN; then
    echo "Additional datasets available:"
    echo "  bash setup_vm.sh --full       # Add IMS dataset"
    echo "  bash setup_vm.sh --paderborn  # Add Paderborn dataset"
    echo "  bash setup_vm.sh --all        # All datasets"
fi
