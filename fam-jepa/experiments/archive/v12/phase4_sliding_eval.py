"""
Phase 4: Sliding-cut-point diagnostic

The canonical C-MAPSS last-window-only test allowed the constant-prediction concern.
Add secondary evaluation sweeping all cut points per test engine, stride 1.

Output:
  experiments/v12/sliding_eval.json
  analysis/plots/v12/sliding_trajectories.png
"""

import json
import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V12_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v12')
PLOTS_V12 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/analysis/plots/v12')

sys.path.insert(0, str(V11_DIR))
from data_utils import load_cmapss_subset, load_raw, N_SENSORS, RUL_CAP
from models import TrajectoryJEPA, RULProbe
from train_utils import DEVICE

PRETRAIN_CKPT = V11_DIR / 'best_pretrain_L1_v2.pt'
E2E_CKPT = V12_DIR / 'v2_e2e_seed0_reconstructed.pt'

print("=" * 60)
print("Phase 4: Sliding-cut-point diagnostic")
print("=" * 60)
print(f"Device: {DEVICE}")
t0 = time.time()

# Load data
data = load_cmapss_subset('FD001')
test_engines = data['test_engines']
test_rul = data['test_rul']

print(f"Test engines: {len(test_engines)}")

# Load reconstructed E2E model
print(f"\nLoading V2 E2E reconstructed model from {E2E_CKPT}...")
model = TrajectoryJEPA(
    n_sensors=N_SENSORS, patch_length=1,
    d_model=256, n_heads=4, n_layers=2, d_ff=512,
    dropout=0.1, ema_momentum=0.996, predictor_hidden=256
).to(DEVICE)
probe = RULProbe(256).to(DEVICE)

ckpt = torch.load(str(E2E_CKPT), map_location=DEVICE)
model.context_encoder.load_state_dict(ckpt['encoder_state'])
probe.load_state_dict(ckpt['probe_state'])
model.eval(); probe.eval()

print(f"Loaded. Test RMSE at load time: {ckpt['test_rmse']:.2f}, val RMSE: {ckpt['val_rmse']:.2f}")


@torch.no_grad()
def infer_at_cycle(model, probe_net, seq, cycle_idx):
    """Infer RUL at cycle_idx (0-indexed) using prefix seq[:cycle_idx+1]."""
    prefix = seq[:cycle_idx+1]
    x = torch.tensor(prefix, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    h = model.encode_past(x, None)
    pred_norm = probe_net(h)
    return float(pred_norm.cpu().numpy()[0]) * RUL_CAP


# For each test engine, run inference at every cycle from 30 to last
print("\nRunning sliding-cut inference on all test engines...")
test_engine_ids = sorted(test_engines.keys())

per_engine_results = []
all_trajectories = {}

for idx, eid in enumerate(test_engine_ids):
    seq = test_engines[eid]
    T = seq.shape[0]
    oracle_rul = float(test_rul[idx])

    min_cycle = min(30, T)
    cycles = list(range(min_cycle, T + 1))

    preds = []
    true_ruls = []

    for c in cycles:
        true_rul_at_c = oracle_rul + (T - c)  # uncapped true RUL
        pred = infer_at_cycle(model, probe, seq, c-1)
        preds.append(pred)
        true_ruls.append(float(true_rul_at_c))

    preds = np.array(preds)
    true_ruls = np.array(true_ruls)
    true_ruls_capped = np.minimum(true_ruls, RUL_CAP)

    # Compute per-engine metrics
    pred_std = float(np.std(preds))
    per_engine_rmse = float(np.sqrt(np.mean((preds - true_ruls_capped) ** 2)))
    last_window_rmse = float((preds[-1] - true_ruls_capped[-1]) ** 2) ** 0.5

    if len(preds) >= 3:
        rho, _ = spearmanr(preds, true_ruls_capped)
        if np.isnan(rho):
            rho = 0.0
    else:
        rho = 0.0

    per_engine_results.append({
        "engine_id": int(eid),
        "T": T,
        "oracle_rul": oracle_rul,
        "pred_std": float(pred_std),
        "per_engine_rmse": float(per_engine_rmse),
        "last_window_rmse": float(last_window_rmse),
        "rho": float(rho),
        "last_pred": float(preds[-1]),
    })
    all_trajectories[int(eid)] = {
        "cycles": cycles,
        "preds": preds.tolist(),
        "true_ruls_capped": true_ruls_capped.tolist(),
        "pred_std": float(pred_std),
        "rho": float(rho),
        "per_engine_rmse": float(per_engine_rmse),
    }

    if idx < 5:
        print(f"  Engine {eid}: T={T}, oracle_rul={oracle_rul:.0f}, "
              f"per_eng_rmse={per_engine_rmse:.2f}, pred_std={pred_std:.1f}, rho={rho:.3f}")

# Aggregate statistics
all_pred_stds = np.array([r["pred_std"] for r in per_engine_results])
all_rhos = np.array([r["rho"] for r in per_engine_results])
all_per_eng_rmses = np.array([r["per_engine_rmse"] for r in per_engine_results])
all_last_window_rmses = np.array([r["last_window_rmse"] for r in per_engine_results])

# Overall sliding-cut RMSE (aggregate all predictions across all engines and cuts)
all_preds_flat = []
all_targets_flat = []
for eid, traj in all_trajectories.items():
    all_preds_flat.extend(traj["preds"])
    all_targets_flat.extend(traj["true_ruls_capped"])
all_preds_flat = np.array(all_preds_flat)
all_targets_flat = np.array(all_targets_flat)
sliding_rmse_overall = float(np.sqrt(np.mean((all_preds_flat - all_targets_flat) ** 2)))

# Last-window RMSE (standard protocol)
last_preds = np.array([r["last_pred"] for r in per_engine_results])
last_targets = test_rul
last_window_rmse_overall = float(np.sqrt(np.mean((last_preds - last_targets) ** 2)))

print(f"\n--- Aggregated Results ---")
print(f"Per-engine sliding RMSE: mean={all_per_eng_rmses.mean():.2f}, median={np.median(all_per_eng_rmses):.2f}")
print(f"Overall sliding-cut RMSE (all cuts): {sliding_rmse_overall:.2f}")
print(f"Last-window RMSE (standard): {last_window_rmse_overall:.2f}")
print(f"Per-engine pred std: mean={all_pred_stds.mean():.2f}, median={np.median(all_pred_stds):.2f}")
print(f"Within-engine rho: mean={all_rhos.mean():.3f}, median={np.median(all_rhos):.3f}")
print(f"Rho>0.7: {np.sum(all_rhos > 0.7)}/100 engines")

# Plot trajectories for 10 engines
print("\nPlotting sliding trajectories for 10 engines...")
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle(f'Phase 4: Sliding-Cut Evaluation - V2 E2E\n'
             f'Sliding RMSE={sliding_rmse_overall:.2f}, Last-window={last_window_rmse_overall:.2f}',
             fontsize=12, fontweight='bold')

plot_ids = test_engine_ids[:10]
for i, eid in enumerate(plot_ids):
    ax = axes[i // 5, i % 5]
    traj = all_trajectories[int(eid)]
    cycles = np.array(traj["cycles"])
    preds = np.array(traj["preds"])
    trues = np.array(traj["true_ruls_capped"])

    ax.fill_between(cycles, trues, alpha=0.15, color='black', label='True RUL area')
    ax.plot(cycles, trues, 'k-', linewidth=2, label='True RUL (capped)', alpha=0.8)
    ax.plot(cycles, preds, 'r-', linewidth=1.5, label='Predicted', alpha=0.8)

    # Mark last window
    ax.scatter(cycles[-1], preds[-1], color='red', s=50, zorder=5)
    ax.scatter(cycles[-1], trues[-1], color='black', s=50, zorder=5)

    ax.set_title(f'Engine {eid}\nRMSE={traj["per_engine_rmse"]:.1f}, rho={traj["rho"]:.2f}',
                 fontsize=8)
    ax.set_xlabel('Cycle', fontsize=7)
    ax.set_ylabel('RUL (cycles)', fontsize=7)
    ax.set_ylim(-5, RUL_CAP + 10)
    if i == 0:
        ax.legend(fontsize=6)

plt.tight_layout()
sliding_plot_path = PLOTS_V12 / 'sliding_trajectories.png'
plt.savefig(str(sliding_plot_path), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved to {sliding_plot_path}")

sliding_results = {
    "per_engine_rmse_mean": float(all_per_eng_rmses.mean()),
    "per_engine_rmse_median": float(np.median(all_per_eng_rmses)),
    "sliding_rmse_overall": sliding_rmse_overall,
    "last_window_rmse": last_window_rmse_overall,
    "per_engine_pred_std_mean": float(all_pred_stds.mean()),
    "per_engine_pred_std_median": float(np.median(all_pred_stds)),
    "within_engine_rho_mean": float(all_rhos.mean()),
    "within_engine_rho_median": float(np.median(all_rhos)),
    "n_engines_rho_gt_07": int(np.sum(all_rhos > 0.7)),
    "n_total_engines": len(test_engine_ids),
    "per_engine_results": per_engine_results,
    "wall_time_s": float(time.time() - t0),
}

with open(V12_DIR / 'sliding_eval.json', 'w') as f:
    json.dump(sliding_results, f, indent=2)
print(f"Saved to {V12_DIR / 'sliding_eval.json'}")
print(f"Wall time: {time.time() - t0:.1f}s")
