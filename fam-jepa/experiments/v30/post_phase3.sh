#!/bin/bash
# Run after Phase 3 completes (master_table.json populated for all 11 datasets).
set -e
cd /home/sagemaker-user/IndustrialJEPA/fam-jepa

echo "=== Phase 4b legacy metrics ==="
python experiments/v30/phase4_legacy_metrics.py 2>&1 | tail -20

echo "=== Phase 5 figures ==="
python experiments/v30/phase5_figures.py 2>&1 | tail -10

echo "=== Phase 7 SESSION_SUMMARY.md ==="
python experiments/v30/finalize_session_summary.py 2>&1 | tail -5

echo "=== Phase 3 summary table ==="
python experiments/v30/phase3_summary.py

echo "=== Quarto notebook render ==="
quarto render notebooks/30_v30_analysis.qmd 2>&1 | tail -10 || echo "(quarto failed; non-fatal)"

echo ""
echo "=== Final master_table.json ==="
cat experiments/v30/results/master_table.json | python -c "
import json, sys
d = json.load(sys.stdin)
ds = d.get('datasets', {})
print(f'datasets: {len(ds)}, total time: {d.get(\"time_total_s\", 0):.0f}s')
for n, r in ds.items():
    for lf, v in r.items():
        m, s, k = v.get('mean_h_auroc'), v.get('std_h_auroc'), v.get('n_seeds')
        if m: print(f'  {n}/{lf}: {m:.4f}' + (f' +/- {s:.4f}' if s else '') + f' ({k}s)')
"
