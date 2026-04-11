#!/bin/bash
# Overnight runner: waits for FD001 to finish, then runs FD002, FD003, FD004, ablations
# Run this in background: nohup bash run_overnight.sh > logs/overnight.log 2>&1 &

set -e
cd /home/sagemaker-user/IndustrialJEPA/paper-replications/star
LOGDIR="logs"
mkdir -p "$LOGDIR"

log() {
    echo "$(date -u '+%Y-%m-%dT%H:%M:%S') $*"
}

commit_and_push() {
    local subset="$1"
    cd /home/sagemaker-user/IndustrialJEPA
    git add paper-replications/star/results/${subset}_results.json \
            paper-replications/star/results/plots/rul_${subset}.png \
            paper-replications/star/EXPERIMENT_LOG.md \
            2>/dev/null || true
    git commit -m "STAR: ${subset} results (5 seeds) - see EXPERIMENT_LOG" 2>/dev/null || true
    git push origin main 2>/dev/null || true
    cd /home/sagemaker-user/IndustrialJEPA/paper-replications/star
    log "Committed and pushed ${subset} results"
}

# Wait for FD001 to complete (JSON written after all 5 seeds)
log "Waiting for FD001 to complete (results JSON)..."
while true; do
    if [ -f results/FD001_results.json ]; then
        log "FD001 complete (JSON found)"
        break
    fi
    COUNT=$(grep -c "FD001 seed=" EXPERIMENT_LOG.md 2>/dev/null || echo 0)
    log "FD001 progress: $COUNT/5 seeds done, waiting for JSON..."
    sleep 60
done

# Commit FD001 results
if [ -f results/FD001_results.json ]; then
    commit_and_push "FD001"
else
    log "WARNING: FD001_results.json not found after completion"
fi

# Run FD002
log "Starting FD002..."
python run_experiments.py FD002 2>&1 | tee "$LOGDIR/FD002.log"
if [ -f results/FD002_results.json ]; then
    commit_and_push "FD002"
fi

# Run FD003
log "Starting FD003..."
python run_experiments.py FD003 2>&1 | tee "$LOGDIR/FD003.log"
if [ -f results/FD003_results.json ]; then
    commit_and_push "FD003"
fi

# Run FD004
log "Starting FD004..."
python run_experiments.py FD004 2>&1 | tee "$LOGDIR/FD004.log"
if [ -f results/FD004_results.json ]; then
    commit_and_push "FD004"
fi

log "All main experiments complete!"

# Run ablations
log "Starting ablations..."
mkdir -p results/ablations

log "Ablation 1: condition normalization (FD002, FD004)..."
python ablations.py cond_norm 2>&1 | tee "$LOGDIR/ablation_cond_norm.log"

log "Ablation 2: RUL cap sweep (FD001)..."
python ablations.py rul_cap 2>&1 | tee "$LOGDIR/ablation_rul_cap.log"

log "Ablation 3: patch length sweep (FD001)..."
python ablations.py patch_length 2>&1 | tee "$LOGDIR/ablation_patch_length.log"

log "Ablation 4: n_heads sweep (FD001, FD003)..."
python ablations.py nheads 2>&1 | tee "$LOGDIR/ablation_nheads.log"

# Generate final RESULTS.md
log "Generating RESULTS.md..."
python summarize_results.py 2>&1 | tee "$LOGDIR/summarize.log"

# Final commit
cd /home/sagemaker-user/IndustrialJEPA
git add paper-replications/star/results/ablations/ \
        paper-replications/star/RESULTS.md \
        paper-replications/star/EXPERIMENT_LOG.md \
        paper-replications/star/logs/ \
        2>/dev/null || true
git commit -m "STAR: ablations complete, RESULTS.md generated" 2>/dev/null || true
git push origin main 2>/dev/null || true

log "OVERNIGHT RUN COMPLETE"
