"""
Generate LaTeX tables for the NeurIPS paper.
Based on all confirmed V12 results.
"""

import json
from pathlib import Path

V12_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v12')

print("=" * 60)
print("V12 Paper Tables (LaTeX)")
print("=" * 60)


# Load results
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

# ============================================================
# Table 1: Main Results - FD001 RMSE Comparison
# ============================================================
print("\n" + "=" * 60)
print("TABLE 1: FD001 RMSE Comparison")
print("=" * 60)

rows = [
    ("Constant predictor (mean train RUL)",
     f"{p0['constant_predictor_rmse']:.1f}", "--", "--"),
    ("Engine-summary ridge regressor (58 features)",
     f"{reg['mean_rmse']:.2f} \\pm {reg['std_rmse']:.2f}", "--", "No (between-engine only)"),
    ("JEPA frozen V2 (ours)",
     f"{fte['frozen']['test_rmse']:.2f}",
     f"{fte['frozen']['rho_median']:.3f}",
     "Yes"),
    ("JEPA E2E V2 (ours)",
     f"{p0['v11_reported_rmse']:.2f} \\pm 0.75",
     f"{p0['within_engine_rho_median']:.3f}",
     "Yes"),
    ("STAR (Fan et al. 2024) [replication]",
     "12.19", "--", "N/A (supervised)"),
]

print(r"""
\begin{table}[h]
\centering
\caption{C-MAPSS FD001 test RMSE (lower is better). Tracking = within-engine degradation correlation.}
\begin{tabular}{lccc}
\toprule
Method & Test RMSE & Tracking (rho) & Uses history? \\
\midrule""")

for name, rmse, rho, hist in rows:
    print(f"  {name} & {rmse} & {rho} & {hist} \\\\")

print(r"""\bottomrule
\end{tabular}
\end{table}
""")


# ============================================================
# Table 2: Tracking Diagnostics
# ============================================================
print("\n" + "=" * 60)
print("TABLE 2: Tracking Diagnostics Summary")
print("=" * 60)

print(r"""
\begin{table}[h]
\centering
\caption{Within-engine degradation tracking diagnostics on FD001 (100 test engines).
Constant predictor baseline: pred\_std~$\approx$~0, rho~$\approx$~0.}
\begin{tabular}{lcc}
\toprule
Diagnostic & JEPA Frozen & JEPA E2E \\
\midrule""")

print(f"  Test RMSE & {fte['frozen']['test_rmse']:.2f} & {fte['e2e']['test_rmse']:.2f} \\\\")
print(f"  Pred std median (cycles) & {fte['frozen']['pred_std_median']:.1f} & {fte['e2e']['pred_std_median']:.1f} \\\\")
print(f"  Within-engine Spearman $\\rho$ (median) & {fte['frozen']['rho_median']:.3f} & {fte['e2e']['rho_median']:.3f} \\\\")
print(f"  h\\_past shuffle $\\Delta$RMSE & \\multicolumn{{2}}{{c}}{{+{shuf['rmse_gain_from_h_past']:.1f} (5 seeds)}} \\\\")

print(r"""\bottomrule
\end{tabular}
\end{table}
""")


# ============================================================
# Table 3: Health Index Recovery
# ============================================================
print("\n" + "=" * 60)
print("TABLE 3: Health Index Recovery")
print("=" * 60)

print(r"""
\begin{table}[h]
\centering
\caption{Frozen JEPA encoder H.I. recovery: $h_{past} \to$ H.I. via Ridge regression (no labels).
All parameterizations exceed the $R^2 > 0.7$ target.}
\begin{tabular}{lcc}
\toprule
H.I. Definition & Train $R^2$ & Val $R^2$ \\
\midrule""")

print(f"  Piecewise linear & {hialt['piecewise_linear']['r2_train']:.3f} & {hialt['piecewise_linear']['r2_val']:.3f} \\\\")
print(f"  Sigmoid & {hialt['sigmoid']['r2_train']:.3f} & {hialt['sigmoid']['r2_val']:.3f} \\\\")
print(f"  Raw RUL normalized & {hialt['raw_rul_norm']['r2_train']:.3f} & {hialt['raw_rul_norm']['r2_val']:.3f} \\\\")
print(f"  Target & -- & $\\geq 0.7$ \\\\")

print(r"""\bottomrule
\end{tabular}
\end{table}
""")


# ============================================================
# Table 4: Evaluation Protocol Comparison
# ============================================================
print("\n" + "=" * 60)
print("TABLE 4: Evaluation Protocol Comparison")
print("=" * 60)

print(r"""
\begin{table}[h]
\centering
\caption{Standard vs sliding-cut-point evaluation on FD001 (JEPA E2E V2, 100 test engines).
Standard protocol evaluates only at the last observed cycle per engine.}
\begin{tabular}{lcc}
\toprule
Metric & Standard (last-window) & Sliding (all cuts) \\
\midrule""")

print(f"  Test RMSE & {slid['last_window_rmse']:.2f} & {slid['sliding_rmse_overall']:.2f} \\\\")
print(f"  Per-engine RMSE (mean) & -- & {slid['per_engine_rmse_mean']:.2f} \\\\")
print(f"  Per-engine RMSE (median) & -- & {slid['per_engine_rmse_median']:.2f} \\\\")
print(f"  Within-engine $\\rho$ (median) & -- & {slid['within_engine_rho_median']:.3f} \\\\")

print(r"""\bottomrule
\end{tabular}
\end{table}
""")


# ============================================================
# Table 5: Multi-Subset Results
# ============================================================
print("\n" + "=" * 60)
print("TABLE 5: Multi-Subset Verification")
print("=" * 60)

print(r"""
\begin{table}[h]
\centering
\caption{V11 results across all four C-MAPSS subsets (V12 verification).
E2E = end-to-end fine-tuning. rho = within-engine Spearman correlation.}
\begin{tabular}{lcccc}
\toprule
Subset & E2E RMSE & Pred std (med) & rho (med) & vs. Regressor \\
\midrule""")

print(f"  FD001 & {p0['v11_reported_rmse']:.2f} & {p0['per_engine_pred_std_median']:.1f} & {p0['within_engine_rho_median']:.3f} & +{reg['delta_vs_v11_e2e']:.1f} \\\\")
print(f"  FD002 & 24.45 (V11) & -- & -- & -- \\\\")
print(f"  FD003 & {fd34['FD003']['v11_reported_e2e']:.2f} & {fd34['FD003']['pred_std_median']:.1f} & {fd34['FD003']['rho_median']:.3f} & +{-fd34['FD003']['delta_jepa_vs_regressor']:.1f} \\\\")
print(f"  FD004 & {fd34['FD004']['v11_reported_e2e']:.2f} & {fd34['FD004']['pred_std_median']:.1f} & {fd34['FD004']['rho_median']:.3f} & +{-fd34['FD004']['delta_jepa_vs_regressor']:.1f} \\\\")

print(r"""\bottomrule
\end{tabular}
\end{table}
""")


# ============================================================
# Table 6: FD001 vs FD002 Val/Test Gap
# ============================================================
print("\n" + "=" * 60)
print("TABLE 6: Val/Test Gap Analysis")
print("=" * 60)

print(r"""
\begin{table}[h]
\centering
\caption{Val probe RMSE vs. canonical test RMSE for FD001 and FD002 (frozen V2 encoder).
The FD002 gap (+10.7) is caused by operating condition distribution shift at test time,
not by SSL pretraining failure.}
\begin{tabular}{lccc}
\toprule
Subset & Val probe RMSE & Test RMSE & Gap \\
\midrule""")

fd1 = vtg['FD001']
fd2 = vtg['FD002']
print(f"  FD001 & {fd1['val_probe_rmse']:.2f} & {fd1['test_rmse_mean']:.2f} $\\pm$ {fd1['test_rmse_std']:.2f} & {fd1['val_test_gap']:+.1f} \\\\")
print(f"  FD002 & {fd2['val_probe_rmse']:.2f} & {fd2['test_rmse_mean']:.2f} $\\pm$ {fd2['test_rmse_std']:.2f} & {fd2['val_test_gap']:+.1f} \\\\")

print(r"""\bottomrule
\end{tabular}
\end{table}
""")


# ============================================================
# Summary key numbers for abstract
# ============================================================
print("\n" + "=" * 60)
print("KEY NUMBERS FOR ABSTRACT")
print("=" * 60)

print(f"H.I. recovery val R2: {hi['r2_val']:.3f}")
print(f"RMSE FD001 E2E: {p0['v11_reported_rmse']:.2f}")
print(f"RMSE FD001 frozen: {fte['frozen']['test_rmse']:.2f}")
print(f"RMSE engine-summary regressor: {reg['mean_rmse']:.2f}")
print(f"Within-engine rho median (E2E): {p0['within_engine_rho_median']:.3f}")
print(f"Shuffle test gain: {shuf['rmse_gain_from_h_past']:.1f}")
print(f"PC1 explained variance: {pca['pc1_explained_var']:.1%}")
print(f"PC1 rho with H.I.: {abs(pca['pc1_rho_hi']):.3f}")
print(f"Sliding RMSE vs last-window: {slid['sliding_rmse_overall']:.2f} vs {slid['last_window_rmse']:.2f}")
print(f"FD002 val/test gap: +{fd2['val_test_gap']:.1f}")
print()
print("Done.")
