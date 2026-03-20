#!/bin/bash
# Start overnight cross-machine transfer research
# Run this before starting Claude agent

set -e

echo "=============================================="
echo "Cross-Machine Transfer Overnight Research"
echo "=============================================="

cd ~/IndustrialJEPA

# Pull latest
git pull

# Check GPU
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Check memory
echo ""
echo "Memory Status:"
free -h

# Quick sanity check
echo ""
echo "Quick Data Check:"
python -c "
from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig

# Test AURSAD
config = FactoryNetConfig(data_source='aursad', max_episodes=10)
ds = FactoryNetDataset(config, split='train')
print(f'AURSAD: {len(ds)} windows')

# Test Voraus
config = FactoryNetConfig(data_source='voraus', max_episodes=10)
ds = FactoryNetDataset(config, split='train')
print(f'Voraus: {len(ds)} windows')

print('Data loading: OK')
"

# Show current status
echo ""
echo "=============================================="
echo "Current Objectives Status:"
echo "=============================================="
cat autoresearch/OBJECTIVES_STATUS.md | head -30

echo ""
echo "=============================================="
echo "Ready for overnight research!"
echo "=============================================="
echo ""
echo "Next: Start Claude agent with:"
echo "  claude --dangerously-skip-permissions"
echo ""
echo "Then paste prompt from:"
echo "  autoresearch/CROSS_MACHINE_PROMPT.md"
echo ""
