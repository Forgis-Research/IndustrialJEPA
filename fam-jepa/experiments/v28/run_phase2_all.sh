#!/bin/bash
# Sequential runner for V28 Phase 2 — all 3 tries × 2 datasets × 3 seeds.
# Pretrain-FT-eval per (try, dataset, seed) takes ~5-15 min on A10G.
# Total: ~3-4 hours.

set -e
cd "$(dirname "$0")/../.."   # cd to fam-jepa root
LOG=experiments/v28/logs/phase2_all.log
mkdir -p experiments/v28/logs

echo "=== Phase 2 starting at $(date -u) ===" | tee -a "$LOG"

cd experiments/v28

# Try A: lag features under RevIN
echo "" | tee -a "../../$LOG"
echo "=== Phase 2A: lag features (FD001, MBA) ===" | tee -a "../../$LOG"
python phase2a_lag.py 2>&1 | tee -a "../../$LOG"

# Try B: aux stat-prediction loss under RevIN
echo "" | tee -a "../../$LOG"
echo "=== Phase 2B: aux stat loss (FD001, MBA) ===" | tee -a "../../$LOG"
python phase2b_stat.py 2>&1 | tee -a "../../$LOG"

# Try C: dense horizon FT
echo "" | tee -a "../../$LOG"
echo "=== Phase 2C: dense horizon FT (FD001, MBA) ===" | tee -a "../../$LOG"
python phase2c_dense_ft.py 2>&1 | tee -a "../../$LOG"

echo "=== Phase 2 complete at $(date -u) ===" | tee -a "../../$LOG"
