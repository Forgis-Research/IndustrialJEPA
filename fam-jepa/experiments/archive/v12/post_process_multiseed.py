"""
Post-process multiseed diagnostics results.
Run this once multiseed_phase0_diagnostics.json is available.

Updates:
- RESULTS.md (add multiseed section)
- paper_figures.py Panel A (update std)
- Exp 14 EXPERIMENT_LOG.md entry
"""

import json
import sys
from pathlib import Path

V12_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v12')

ms_path = V12_DIR / 'multiseed_phase0_diagnostics.json'
if not ms_path.exists():
    print("multiseed_phase0_diagnostics.json not found. Waiting...")
    sys.exit(1)

with open(ms_path) as f:
    ms = json.load(f)

print("=" * 60)
print("5-Seed Trajectory Diagnostics Summary")
print("=" * 60)
print(f"RMSE: {ms['test_rmse_mean']:.2f} +/- {ms['test_rmse_std']:.2f}")
print(f"Pred std median: {ms['pred_std_median_mean']:.2f} +/- {ms['pred_std_median_std']:.2f}")
print(f"Rho median: {ms['rho_median_mean']:.3f} +/- {ms['rho_median_std']:.3f}")
print(f"All pass tracking: {ms['all_pass_tracking']}")
print(f"Wall time: {ms['wall_time_s']/60:.1f} min")

print("\nPer-seed breakdown:")
for r in ms['per_seed_results']:
    print(f"  Seed {r['seed']:>5}: RMSE={r['test_rmse']:.2f}, "
          f"std_med={r['pred_std_median']:.1f}, rho_med={r['rho_median']:.3f}")

# Check against Phase 0 single-seed results
with open(V12_DIR / 'phase0_diagnostics.json') as f:
    p0 = json.load(f)

print(f"\nComparison with Phase 0 (seed=0):")
print(f"  RMSE: single={p0['reconstructed_test_rmse_seed0']:.2f}, multi={ms['test_rmse_mean']:.2f}")
print(f"  rho: single={p0['within_engine_rho_median']:.3f}, multi={ms['rho_median_mean']:.3f}")

# Consistency check
rmse_consistent = abs(p0['reconstructed_test_rmse_seed0'] - ms['test_rmse_mean']) < 2.0
rho_consistent = abs(p0['within_engine_rho_median'] - ms['rho_median_mean']) < 0.15
print(f"\nConsistency: RMSE {'OK' if rmse_consistent else 'FAIL'}, rho {'OK' if rho_consistent else 'FAIL'}")

# All seeds pass tracking?
if not ms['all_pass_tracking']:
    print("\nWARNING: Not all seeds pass tracking thresholds!")
    for r in ms['per_seed_results']:
        std_ok = r['pred_std_median'] > 10
        rho_ok = r['rho_median'] > 0.5
        if not (std_ok and rho_ok):
            print(f"  Seed {r['seed']}: std={r['pred_std_median']:.1f} {'OK' if std_ok else 'FAIL'}, "
                  f"rho={r['rho_median']:.3f} {'OK' if rho_ok else 'FAIL'}")
else:
    print("\nAll seeds pass tracking thresholds. Statistical rigor satisfied.")

print("\nDone.")
