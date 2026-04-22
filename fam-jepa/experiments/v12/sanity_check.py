"""
Sanity check all V12 results for internal consistency.
Run this before writing the paper to catch any anomalies.
"""

import json
from pathlib import Path

V12_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v12')

# Load all confirmed results
with open(V12_DIR / 'engine_summary_regressor.json') as f: reg = json.load(f)
with open(V12_DIR / 'phase0_diagnostics.json') as f: p0 = json.load(f)
with open(V12_DIR / 'shuffle_test.json') as f: shuf = json.load(f)
with open(V12_DIR / 'health_index_recovery.json') as f: hi = json.load(f)
with open(V12_DIR / 'sliding_eval.json') as f: slid = json.load(f)
with open(V12_DIR / 'val_test_gap.json') as f: vtg = json.load(f)
with open(V12_DIR / 'frozen_vs_e2e_tracking.json') as f: fte = json.load(f)
with open(V12_DIR / 'pca_analysis.json') as f: pca = json.load(f)
with open(V12_DIR / 'hi_alternative_params.json') as f: hialt = json.load(f)
with open(V12_DIR / 'extra_fd003_fd004_diagnostics.json') as f: fd34 = json.load(f)

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

checks = []

def check(name, condition, value, expected_str, severity="FAIL"):
    status = PASS if condition else severity
    checks.append((name, status, str(value), expected_str))
    return condition

print("=" * 72)
print("V12 Sanity Checks")
print("=" * 72)

# ============================================================
# 1. Magnitude checks - all values in reasonable range
# ============================================================

# RMSE should be 10-50 for C-MAPSS FD001
check("FD001 E2E RMSE in range [10, 50]",
      10 <= p0['v11_reported_rmse'] <= 50,
      p0['v11_reported_rmse'], "[10, 50]")

check("FD001 frozen RMSE in range [10, 50]",
      10 <= fte['frozen']['test_rmse'] <= 50,
      fte['frozen']['test_rmse'], "[10, 50]")

check("Engine-summary regressor RMSE in range [10, 50]",
      10 <= reg['mean_rmse'] <= 50,
      reg['mean_rmse'], "[10, 50]")

check("Constant predictor RMSE in range [30, 60]",
      30 <= p0['constant_predictor_rmse'] <= 60,
      p0['constant_predictor_rmse'], "[30, 60]")

# ============================================================
# 2. Direction checks
# ============================================================

# Constant predictor should be worst
check("Constant predictor > regressor RMSE",
      p0['constant_predictor_rmse'] > reg['mean_rmse'],
      f"{p0['constant_predictor_rmse']:.1f} > {reg['mean_rmse']:.1f}",
      "constant > regressor")

check("Regressor > frozen RMSE",
      reg['mean_rmse'] > fte['frozen']['test_rmse'],
      f"{reg['mean_rmse']:.1f} > {fte['frozen']['test_rmse']:.1f}",
      "regressor > frozen")

check("Frozen > E2E RMSE",
      fte['frozen']['test_rmse'] > fte['e2e']['test_rmse'],
      f"{fte['frozen']['test_rmse']:.1f} > {fte['e2e']['test_rmse']:.1f}",
      "frozen > E2E")

# Sliding RMSE should be <= last-window RMSE (more data = better)
check("Sliding RMSE <= last-window RMSE",
      slid['sliding_rmse_overall'] <= slid['last_window_rmse'],
      f"{slid['sliding_rmse_overall']:.2f} <= {slid['last_window_rmse']:.2f}",
      "sliding <= last-window")

# Val R2 should be < train R2 (generalization gap)
check("Val R2 < train R2 (no train leakage)",
      hi['r2_val'] < hi['r2_train'],
      f"{hi['r2_val']:.3f} < {hi['r2_train']:.3f}",
      "val < train")

# ============================================================
# 3. Target criterion checks
# ============================================================

check("Phase 0: pred_std_median > 10 (tracking threshold)",
      p0['per_engine_pred_std_median'] > 10,
      p0['per_engine_pred_std_median'], "> 10")

check("Phase 0: rho_median > 0.5 (tracking threshold)",
      p0['within_engine_rho_median'] > 0.5,
      p0['within_engine_rho_median'], "> 0.5")

check("Phase 0: V11 beats regressor by > 1 RMSE",
      reg['delta_vs_v11_e2e'] > 1.0,
      reg['delta_vs_v11_e2e'], "> 1")

check("Phase 3: val R2 > 0.7 (H.I. recovery threshold)",
      hi['r2_val'] > 0.7,
      hi['r2_val'], "> 0.7")

check("Phase 3: all H.I. parameterizations > 0.7",
      all(hialt[k]['r2_val'] > 0.7 for k in hialt),
      {k: round(hialt[k]['r2_val'], 3) for k in hialt}, "all > 0.7")

# ============================================================
# 4. Shuffle test check
# ============================================================

check("Shuffle test: shuffled RMSE >> normal RMSE (h_past carries info)",
      shuf['shuffled_rmse_mean'] > shuf['normal_rmse'] * 2,
      f"{shuf['shuffled_rmse_mean']:.1f} >> {shuf['normal_rmse']:.1f}",
      "shuffled > 2x normal")

check("Normal RMSE in shuffle test consistent with reconstructed E2E",
      abs(shuf['normal_rmse'] - fte['e2e']['test_rmse']) < 0.5,
      f"|{shuf['normal_rmse']:.2f} - {fte['e2e']['test_rmse']:.2f}| < 0.5",
      "consistent", "WARN")

# ============================================================
# 5. PCA checks
# ============================================================

check("PC1 explained variance > 0.3",
      pca['pc1_explained_var'] > 0.3,
      pca['pc1_explained_var'], "> 0.3")

check("PC1 |rho| with H.I. > 0.5",
      abs(pca['pc1_rho_hi']) > 0.5,
      abs(pca['pc1_rho_hi']), "> 0.5")

check("PC1 dominates (|rho| >> PC2 |rho|)",
      abs(pca['pc_rho_with_hi'][0]) > 3 * abs(pca['pc_rho_with_hi'][1]),
      f"PC1={abs(pca['pc_rho_with_hi'][0]):.3f} >> PC2={abs(pca['pc_rho_with_hi'][1]):.3f}",
      "PC1 dominant")

# ============================================================
# 6. FD002 distribution shift check
# ============================================================

check("FD002 val/test gap is large (> 8 RMSE)",
      vtg['FD002']['val_test_gap'] > 8,
      vtg['FD002']['val_test_gap'], "> 8")

check("FD001 val/test gap is small (< 5 RMSE in absolute value)",
      abs(vtg['FD001']['val_test_gap']) < 5,
      vtg['FD001']['val_test_gap'], "abs < 5")

check("FD002 val probe RMSE < FD001 val probe RMSE (FD002 encoder learned well)",
      vtg['FD002']['val_probe_rmse'] < vtg['FD001']['val_probe_rmse'],
      f"{vtg['FD002']['val_probe_rmse']:.2f} < {vtg['FD001']['val_probe_rmse']:.2f}",
      "FD002 val < FD001 val", "WARN")

# ============================================================
# 7. Multi-subset checks
# ============================================================

for subset, data in [('FD003', fd34['FD003']), ('FD004', fd34['FD004'])]:
    check(f"{subset}: rho_median > 0.5",
          data['rho_median'] > 0.5,
          data['rho_median'], "> 0.5")
    check(f"{subset}: JEPA beats regressor",
          data['delta_jepa_vs_regressor'] < 0,  # delta is negative = JEPA beats
          data['delta_jepa_vs_regressor'], "< 0 (JEPA wins)")

# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 72)
print(f"{'Check':<55} {'Status':>6} {'Value':>12} {'Expected':>12}")
print("-" * 72)
n_pass = n_fail = n_warn = 0
for name, status, value, expected in checks:
    icon = "OK" if status == PASS else ("!!" if status == FAIL else "??")
    print(f"  {name:<53} [{icon}] {value:>12} {expected:>12}")
    if status == PASS: n_pass += 1
    elif status == FAIL: n_fail += 1
    elif status == WARN: n_warn += 1

print("=" * 72)
print(f"\nSummary: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL")

if n_fail > 0:
    print("\nFAILED CHECKS:")
    for name, status, value, expected in checks:
        if status == FAIL:
            print(f"  ** FAIL: {name}")
            print(f"     Got: {value}, Expected: {expected}")
elif n_warn > 0:
    print("\nWARNINGS (investigate before publication):")
    for name, status, value, expected in checks:
        if status == WARN:
            print(f"  ?? WARN: {name}")
            print(f"     Got: {value}, Expected: {expected}")
else:
    print("\nAll checks passed! Results are internally consistent.")
    print("Safe to write up for NeurIPS submission.")
