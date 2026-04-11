#!/bin/bash
# Run all DCSSL replication experiments
# Logs output to /home/sagemaker-user/IndustrialJEPA/dcssl-replication/logs/

set -e
LOGDIR="/home/sagemaker-user/IndustrialJEPA/dcssl-replication/logs"
mkdir -p "$LOGDIR"

DATA_ROOT="/mnt/sagemaker-nvme/femto_data/10. FEMTO Bearing"
RESULTS_DIR="/home/sagemaker-user/IndustrialJEPA/dcssl-replication/results"
SCRIPT="/home/sagemaker-user/IndustrialJEPA/dcssl-replication/run_experiments.py"

echo "Starting DCSSL replication at $(date)"
echo "Logs in: $LOGDIR"

for MODEL in simclr supcon dcssl; do
    for COND in 1 2 3; do
        EXP="${MODEL}_cond${COND}"
        LOG="${LOGDIR}/${EXP}.log"
        echo ""
        echo "=== Running $EXP at $(date) ==="
        python3 "$SCRIPT" \
            --data_root "$DATA_ROOT" \
            --output_dir "$RESULTS_DIR" \
            --model "$MODEL" \
            --condition "$COND" \
            --pretrain_epochs 300 \
            --finetune_epochs 150 \
            --pretrain_lr 1e-3 \
            --finetune_lr 5e-4 \
            --batch_size 64 \
            --crop_length 1024 \
            2>&1 | tee "$LOG"
        echo "=== Finished $EXP at $(date) ==="
    done
done

echo ""
echo "All experiments complete at $(date)"
