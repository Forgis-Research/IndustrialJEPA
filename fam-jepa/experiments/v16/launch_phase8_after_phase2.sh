#!/bin/bash
# Launch Phase 8 after Phase 2 completes.
# Checks every 5 minutes for Phase 2 completion.

PHASE2_PID=94008
PHASE8_LOG="/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v16/phase8_stdout.log"
PHASE8_SCRIPT="/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v16/phase8_label_efficiency.py"

echo "Waiting for Phase 2 (PID $PHASE2_PID) to complete..."
echo "Checking every 60 seconds."

while kill -0 $PHASE2_PID 2>/dev/null; do
    sleep 60
    echo "$(date): Phase 2 still running..."
done

echo "$(date): Phase 2 completed! Launching Phase 8 label efficiency..."
sleep 30  # Brief pause to let GPU memory free up

cd /home/sagemaker-user/IndustrialJEPA
python -u "$PHASE8_SCRIPT" > "$PHASE8_LOG" 2>&1

echo "$(date): Phase 8 complete. Results in $PHASE8_LOG"
