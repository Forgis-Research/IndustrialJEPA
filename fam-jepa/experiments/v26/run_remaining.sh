#!/bin/bash
# V26 remaining runs: FD003 -> SMAP -> MSL -> PSM -> SMD -> MBA -> PhysioNet
# Chain to keep GPU busy overnight. Each phase logs separately.

set -e
V26=/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v26

cd /home/sagemaker-user/IndustrialJEPA/fam-jepa

# Wait for any active phase2_cmapss on FD002 to finish (no-op if already done)
while pgrep -f "phase2_cmapss.py --subset FD002" > /dev/null; do
    sleep 10
done

echo "=== FD003 starting at $(date) ===" >> $V26/logs/chain.log
python $V26/phase2_cmapss.py --subset FD003 > $V26/logs/phase2_FD003.log 2>&1
echo "=== FD003 done at $(date) ===" >> $V26/logs/chain.log

for DS in SMAP MSL PSM SMD MBA; do
    echo "=== $DS starting at $(date) ===" >> $V26/logs/chain.log
    python $V26/phase3_anomaly.py --dataset $DS > $V26/logs/phase3_${DS}.log 2>&1
    echo "=== $DS done at $(date) ===" >> $V26/logs/chain.log
done

echo "=== physionet starting at $(date) ===" >> $V26/logs/chain.log
python $V26/phase4_physionet.py > $V26/logs/phase4_physionet.log 2>&1
echo "=== physionet done at $(date) ===" >> $V26/logs/chain.log

echo "=== dense eval starting at $(date) ===" >> $V26/logs/chain.log
python $V26/phase5_dense.py > $V26/logs/phase5_dense.log 2>&1
echo "=== dense eval done at $(date) ===" >> $V26/logs/chain.log

echo "=== chain complete at $(date) ===" >> $V26/logs/chain.log
