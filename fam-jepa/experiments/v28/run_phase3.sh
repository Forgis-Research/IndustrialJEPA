#!/bin/bash
# Phase 3 — comprehensive benchmark.
#  1. Try A* (lag + none) on FD001 — additional test missing from Phase 2.
#  2. v27 baseline on the two NEW datasets (GECCO, BATADAL), 3 seeds.
#  3. Try C (dense FT) on every dataset — head-to-head against the baseline.
#
# Total: ~30 min on A10G.

set -e
cd "$(dirname "$0")/../.."
LOG=experiments/v28/logs/phase3_all.log

echo "=== Phase 3 starting at $(date -u) ===" | tee -a "$LOG"
cd experiments/v28

echo "--- Phase 2D follow-up: lag + none on FD001 ---" | tee -a "../../$LOG"
python phase2d_lag_none.py 2>&1 | tee -a "../../$LOG"

echo "--- Phase 3 baseline on NEW datasets (GECCO, BATADAL) ---" | tee -a "../../$LOG"
python phase3_benchmark.py --variant baseline --datasets GECCO BATADAL 2>&1 | tee -a "../../$LOG"

echo "--- Phase 3 dense-FT (Try C) on ALL datasets ---" | tee -a "../../$LOG"
python phase3_benchmark.py --variant dense \
  --datasets FD001 FD002 FD003 SMAP MSL PSM SMD MBA GECCO BATADAL 2>&1 \
  | tee -a "../../$LOG"

echo "=== Phase 3 done at $(date -u) ===" | tee -a "../../$LOG"
