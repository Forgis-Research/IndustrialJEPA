#!/bin/bash
# Run Chronos-2 baseline on all remaining v24 datasets, 3 seeds each.
# Order: small first (fail fast), then slow last.
#
# Usage:  bash experiments/v24/run_chronos_sweep.sh
# Logs:   experiments/v24/logs/chronos2_<DATASET>.log

set -e
cd /home/sagemaker-user/IndustrialJEPA/fam-jepa

# Already done: FD001 (3 seeds), SMAP (s42 running -> will redo s123, s456 later)
# Skip: sepsis (too slow - 40K patients)
ORDER=(MBA FD003 FD002 GECCO BATADAL MSL PSM SMD)
# Note: Sepsis and PhysioNet 2012 skipped (too many test obs for Chronos-2
# inference at ~0.8 s/obs; would take 10+ hours).

for DS in "${ORDER[@]}"; do
    for SEED in 42 123 456; do
        OUT="experiments/v24/results/baseline_chronos2_${DS}_s${SEED}.json"
        if [ -f "$OUT" ]; then
            echo "skip ${DS}_s${SEED} (exists)"
            continue
        fi
        echo "=== ${DS} seed ${SEED} ==="
        python experiments/v24/baseline_chronos2.py \
            --dataset "${DS}" --seed "${SEED}" --cache-features \
            2>&1 | tee -a "experiments/v24/logs/chronos2_${DS}.log" \
            | grep -E "AUPRC:|AUROC:|F1:|wrote |Error|Traceback" || true
    done
done

python experiments/v24/aggregate_chronos.py
echo "DONE chronos sweep"
