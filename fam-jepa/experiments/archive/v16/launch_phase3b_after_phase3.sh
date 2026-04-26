#!/bin/bash
# Launch Phase 3b (MSL) after Phase 3 (SMAP) completes.

PHASE3_PID=146951
PHASE3B_LOG="/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v16/phase3b_stdout.log"
PHASE3B_SCRIPT="/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v16/phase3b_msl_100epochs.py"

echo "Waiting for Phase 3/SMAP (PID $PHASE3_PID) to complete..."
echo "Checking every 60 seconds."

while kill -0 $PHASE3_PID 2>/dev/null; do
    sleep 60
    echo "$(date): Phase 3/SMAP still running..."
done

echo "$(date): Phase 3/SMAP completed! Launching Phase 3b/MSL..."
sleep 30  # Brief pause to let GPU memory free up

cd /home/sagemaker-user/IndustrialJEPA
python -u "$PHASE3B_SCRIPT" > "$PHASE3B_LOG" 2>&1

echo "$(date): Phase 3b complete. Results in $PHASE3B_LOG"
