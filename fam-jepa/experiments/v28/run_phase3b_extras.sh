#!/bin/bash
# Phase 3B follow-up — queue these once Phase 3 dense finishes.
#
# Phase 2D showed lag+none beats v27 baseline by +0.019 mean h-AUROC on FD001.
# We extend that to the rest of the C-MAPSS family (FD002, FD003) to test
# if the lag-feature trick generalises across degradation regimes.
#
# We also add lag+revin on the larger anomaly datasets (SMD, GECCO, BATADAL)
# since Phase 2A showed lag features help on MBA under RevIN.

set -e
cd "$(dirname "$0")/../.."
LOG=experiments/v28/logs/phase3b_extras.log

echo "=== Phase 3B extras starting at $(date -u) ===" | tee -a "$LOG"
cd experiments/v28

# Lag + 'none' on FD002 + FD003 (extend the C-MAPSS Try A* winner)
echo "--- lag + none on FD002 + FD003 ---" | tee -a "../../$LOG"
python phase2d_lag_none.py --datasets FD002 FD003 2>&1 | tee -a "../../$LOG"

# Lag + 'revin' on SMD + GECCO + BATADAL (extend the MBA Try A winner)
echo "--- lag + revin on SMD, GECCO, BATADAL ---" | tee -a "../../$LOG"
python phase2a_lag.py --datasets SMD GECCO BATADAL 2>&1 | tee -a "../../$LOG"

echo "=== Phase 3B extras done at $(date -u) ===" | tee -a "../../$LOG"
