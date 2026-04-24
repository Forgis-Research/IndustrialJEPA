#!/bin/bash
# Phase 4-7 finalization: compute dense surfaces, re-render PNGs and notebook,
# update RESULTS.md, commit & push. Runs after Phase 3B finishes.

set -e
cd "$(dirname "$0")/../.."
LOG=experiments/v28/logs/finalize.log

echo "=== finalize starting at $(date -u) ===" | tee -a "$LOG"

# 1. Compute dense surfaces on the v28 best ckpt per dataset
echo "--- compute dense surfaces ---" | tee -a "$LOG"
cd experiments/v28
python compute_dense_surfaces.py 2>&1 | tee -a "../../$LOG"

# 2. Re-render triplet PNGs (now uses v28 dense surfaces if present)
cd ../..
echo "--- re-render PNGs ---" | tee -a "$LOG"
python experiments/v28/render_surface_pngs.py 2>&1 | tee -a "$LOG"

# 3. Re-render notebook so master table picks up the new dense numbers
echo "--- re-render Quarto notebook ---" | tee -a "$LOG"
quarto render notebooks/28_v28_analysis.qmd 2>&1 | tee -a "$LOG"
python -c "t=open('notebooks/28_v28_analysis.html').read(); print('PNGs:',t.count('data:image/png;base64,'),'<img:',t.count('<img '))" | tee -a "$LOG"

echo "=== finalize done at $(date -u) ===" | tee -a "$LOG"
