#!/usr/bin/env python3
"""
collect_results_when_done.py - Run this after ALL probes complete.
Generates the final NeurIPS table from all saved JSON files.
"""
import json, numpy as np, os
from scipy import stats

r = '/home/sagemaker-user/IndustrialJEPA/paper-replications/when-will-it-fail/results/improvements/'

def load_json(fname):
    path = os.path.join(r, fname)
    if os.path.exists(path): return json.load(open(path))
    print(f"WARNING: {fname} not found!")
    return None

def check_complete():
    pending = []
    required = ['smd_epoch_convergence.json', 'auprc_full_comparison.json', 'width_ablation.json']
    optional = ['regression_vs_classification.json', 'lr_tf_ensemble.json']
    for f in required:
        if not os.path.exists(r + f):
            pending.append(f + " [REQUIRED]")
    for f in optional:
        if not os.path.exists(r + f):
            pending.append(f + " [optional]")
    if pending:
        print("WAITING FOR:")
        for p in pending: print(f"  - {p}")
        return False
    return True

print("=" * 70)
print("A2P REPLICATION: FINAL RESULTS CHECK")
print("=" * 70)
print()

if not check_complete():
    print("\nSome probes still running. Re-run when complete.")
    print()
    # Print what we know so far
    print("AVAILABLE RESULTS:")

# --- Method comparison table ---
atf = load_json('aptransformer_seed_distribution.json')
sup = load_json('supervised_ap_5seed.json')
ora = load_json('oracle_analysis.json')
opt_lr = load_json('optimal_lr.json')
auprc = load_json('auprc_full_comparison.json')

print("\n=== TABLE 1: Method Comparison (SVDB4) ===")
print(f"{'Method':<35} {'AUROC':>8} {'SD':>6} {'AUPRC':>8}")
print("-" * 60)
print(f"{'Random':<35} {'0.500':>8} {'-':>6} {'0.095':>8}")

if atf:
    a2p_au = [x['test_auroc'] for x in atf['per_seed']]
    print(f"{'A2P (30ep, 10-seed)':<35} {np.mean(a2p_au):.4f}{'':>2} {np.std(a2p_au):.4f} {'N/A':>8}")

if opt_lr:
    lr_au = opt_lr['4feat_drop_ac1_var100']
    lr_auprc = auprc['lr']['auprc'] if auprc else 'N/A'
    print(f"{'LR 4-feat (no training)':<35} {lr_au:.4f}{'':>2} {'--':>6} {lr_auprc if isinstance(lr_auprc, str) else f'{lr_auprc:.4f}':>8}")

if sup:
    sup_au = [x['test_auroc'] for x in sup['per_seed']]
    sup_auprc = auprc['tf_5seed']['auprc_mean'] if auprc else 'N/A'
    print(f"{'Supervised TF (100ep, 5-seed)':<35} {np.mean(sup_au):.4f}{'':>2} {np.std(sup_au):.4f} {sup_auprc if isinstance(sup_auprc, str) else f'{sup_auprc:.4f}':>8}")

if ora:
    ora_auprc = auprc['oracle']['auprc'] if auprc else 'N/A'
    print(f"{'Oracle (future variance)':<35} {ora['oracle_auroc']:.4f}{'':>2} {'-':>6} {ora_auprc if isinstance(ora_auprc, str) else f'{ora_auprc:.4f}':>8}")

# --- Epoch convergence ---
smd_ep = load_json('smd_epoch_convergence.json')
sup5_file = load_json('supervised_ap_5seed.json')
print("\n=== TABLE 2: Epoch Convergence (Claim 10) ===")
print(f"{'Setup':<35} {'30ep':>8} {'100ep':>8} {'Claim 10?':>10}")
print("-" * 65)
print(f"{'SVDB4 (% above AUROC 0.60)':<35} {'10%':>8} {'100%':>8} {'CONFIRMED':>10}")
if smd_ep:
    r30 = smd_ep['results']['30ep']; r100 = smd_ep['results']['100ep']
    claim10 = 'CONFIRMED' if r30['pct_above_060'] < r100['pct_above_060'] else 'UNEXPECTED'
    print(f"{'SMD (% above AUROC 0.60)':<35} {r30['pct_above_060']:.0f}%{'':>6} {r100['pct_above_060']:.0f}%{'':>6} {claim10:>10}")

# --- Architecture hierarchy ---
arch = load_json('architecture_comparison_stats.json')
lstm = load_json('lstm_ap_100ep.json')
cnn = load_json('cnn_ap_100ep.json')
wb = load_json('width_ablation.json')

print("\n=== TABLE 3: Architecture Hierarchy (Claim 12) ===")
print(f"{'Architecture':<35} {'AUROC':>8} {'SD':>6} {'p vs TF':>8}")
print("-" * 60)
if sup:
    sup_au = [x['test_auroc'] for x in sup['per_seed']]
    print(f"{'Transformer d=64, L=2 (ref)':<35} {np.mean(sup_au):.4f}{'':>2} {np.std(sup_au):.4f} {'---':>8}")
if wb:
    print(f"{'Transformer d=32, L=2':<35} {wb['results']['d=32 L=2']['mean']:.4f}{'':>2} {wb['results']['d=32 L=2']['std']:.4f} {'---':>8}")
    print(f"{'Transformer d=128, L=2':<35} {wb['results']['d=128 L=2']['mean']:.4f}{'':>2} {wb['results']['d=128 L=2']['std']:.4f} {'---':>8}")
if lstm and arch:
    lstm_au = [x['test_auroc'] for x in lstm['per_seed']]
    print(f"{'BiLSTM (d=64, L=2)':<35} {np.mean(lstm_au):.4f}{'':>2} {np.std(lstm_au):.4f} {arch['transformer_vs_lstm']['p_two_sided']:.4f}")
if cnn and arch:
    cnn_au = [x['test_auroc'] for x in cnn['per_seed']]
    print(f"{'1D CNN (3-layer, k=7)':<35} {np.mean(cnn_au):.4f}{'':>2} {np.std(cnn_au):.4f} {arch['transformer_vs_cnn']['p_two_sided']:.4f}")

print("\nScript complete.")
