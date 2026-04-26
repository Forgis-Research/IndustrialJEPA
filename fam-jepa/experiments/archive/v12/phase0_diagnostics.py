"""
Phase 0.1 + 0.2: V2 E2E model reconstruction + prediction trajectory diagnostics

0.1: Load V2 pretrained checkpoint, re-run fine-tune at seed=0, plot prediction
     trajectories for 10 FD001 test engines at every cycle from 30 to last.

0.2: For ALL FD001 test engines, compute:
     - Per-engine prediction std (constant predictor ~ 0)
     - Within-engine Spearman rho (target > 0.7)
     - Constant predictor baseline RMSE (mean(train_rul) on test)

Output:
  analysis/plots/v12/phase0_prediction_trajectories.png
  experiments/v12/phase0_diagnostics.json
"""

import json
import sys
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V12_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v12')
PLOTS_V12 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/analysis/plots/v12')
V12_DIR.mkdir(exist_ok=True)
PLOTS_V12.mkdir(exist_ok=True)

sys.path.insert(0, str(V11_DIR))
from data_utils import (
    load_cmapss_subset, load_raw, get_sensor_cols, compute_rul_labels,
    CMAPSSFinetuneDataset, CMAPSSTestDataset, collate_finetune, collate_test,
    RUL_CAP, N_SENSORS
)
from models import TrajectoryJEPA, RULProbe
from train_utils import DEVICE, finetune, _eval_test_rmse

PRETRAIN_CKPT = V11_DIR / 'best_pretrain_L1_v2.pt'

print("=" * 60)
print("Phase 0.1 + 0.2: V2 E2E reconstruction + trajectory diagnostics")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"Pretrain checkpoint: {PRETRAIN_CKPT}")

t0 = time.time()

# Load data
print("\nLoading FD001 data...")
data = load_cmapss_subset('FD001')
train_engines = data['train_engines']
val_engines = data['val_engines']
test_engines = data['test_engines']
test_rul = data['test_rul']

print(f"Train engines: {len(train_engines)}")
print(f"Val engines: {len(val_engines)}")
print(f"Test engines: {len(test_engines)}")

# Compute mean train RUL (at last cycle) for constant predictor baseline
# Training engines: RUL at last window = 1 (all run to failure, so last cycle RUL=1)
# But the "constant predictor" we compare against is mean(test_rul)
# and mean of training RUL labels at last cut
# Actually: constant predictor = predict mean(train_rul_at_last_window) for every test engine
# train_rul_at_last_window = 1 for all training engines (they go to failure)
# That doesn't make sense as a baseline.
# Better: constant predictor = predict mean(test_rul) for every test engine
# Or: predict mean of ALL training RUL labels
# The spec says: "compute the RMSE you would get by predicting mean(train_rul) for every test engine"
# train_rul = the actual RUL values at ALL training time steps (all cuts)
# Let's compute both.

print("\nComputing training RUL distribution...")
all_train_ruls = []
for eid, seq in train_engines.items():
    T = seq.shape[0]
    ruls = np.minimum(np.arange(T, 0, -1, dtype=np.float32), RUL_CAP)
    all_train_ruls.extend(ruls.tolist())
all_train_ruls = np.array(all_train_ruls)
mean_train_rul_all = float(np.mean(all_train_ruls))

# Also: mean of last-window RUL for training engines (used in fine-tuning)
train_last_ruls = []
for eid, seq in train_engines.items():
    T = seq.shape[0]
    rul_last = float(min(T, RUL_CAP))  # RUL at cycle 1 = T, capped; but at cycle T it's 1
    # Actually compute_rul_labels(T)[-1] = 1 always (runs to failure)
    train_last_ruls.append(1.0)  # all training engines reach failure
mean_train_last_rul = float(np.mean(train_last_ruls))

print(f"Mean train RUL (all cycles): {mean_train_rul_all:.1f}")
print(f"Mean train RUL (last cycle): {mean_train_last_rul:.1f}")
print(f"Test RUL: mean={test_rul.mean():.1f}, std={test_rul.std():.1f}")

# Constant predictor: predict mean(test_rul) for every test engine
# Using training mean as proxy (fair - don't use test stats)
constant_pred = mean_train_rul_all
constant_rmse = float(np.sqrt(np.mean((constant_pred - test_rul) ** 2)))
print(f"\nConstant predictor RMSE (using mean_train_all={constant_pred:.1f}): {constant_rmse:.2f}")

# Build V2 architecture
print("\nBuilding V2 architecture (d_model=256, n_layers=2, n_heads=4)...")
jepa_model = TrajectoryJEPA(
    n_sensors=N_SENSORS,
    patch_length=1,
    d_model=256,
    n_heads=4,
    n_layers=2,
    d_ff=512,
    dropout=0.1,
    ema_momentum=0.996,
    predictor_hidden=256
)

print(f"Loading pretrain checkpoint: {PRETRAIN_CKPT}")
state = torch.load(PRETRAIN_CKPT, map_location=DEVICE)
jepa_model.load_state_dict(state)
jepa_model = jepa_model.to(DEVICE)

n_params = sum(p.numel() for p in jepa_model.parameters())
print(f"Model parameters: {n_params:,}")

# Run E2E fine-tuning (seed=0)
print("\nRunning E2E fine-tuning (seed=0, 100 epochs, patience=20)...")
t_finetune = time.time()
ft_result = finetune(
    model=jepa_model,
    train_engines=train_engines,
    val_engines=val_engines,
    test_engines=test_engines,
    test_rul=test_rul,
    n_epochs=100,
    lr_probe=1e-3,
    lr_e2e=1e-4,
    batch_size=16,
    early_stop_patience=20,
    mode='e2e',
    seed=0,
    verbose=True,
)
print(f"Fine-tune done in {time.time()-t_finetune:.1f}s")
print(f"Val RMSE: {ft_result['val_rmse']:.2f}, Test RMSE: {ft_result['test_rmse']:.2f}")
print(f"(V11 reported E2E @ seed=0: ~13.80 mean over 5 seeds)")

# We need to keep the fine-tuned model and probe.
# Since finetune() returns metrics but we need the actual probe,
# let's redo fine-tuning with access to the probe object.

# Re-run fine-tune to get probe for inference
print("\nRe-running fine-tune to retain probe for trajectory inference...")
jepa_model2 = TrajectoryJEPA(
    n_sensors=N_SENSORS,
    patch_length=1,
    d_model=256,
    n_heads=4,
    n_layers=2,
    d_ff=512,
    dropout=0.1,
    ema_momentum=0.996,
    predictor_hidden=256
)
jepa_model2.load_state_dict(torch.load(PRETRAIN_CKPT, map_location=DEVICE))
jepa_model2 = jepa_model2.to(DEVICE)

probe = RULProbe(256).to(DEVICE)
from torch.utils.data import DataLoader

# Set up data
torch.manual_seed(0)
np.random.seed(0)

train_ds = CMAPSSFinetuneDataset(train_engines, n_cuts_per_engine=5, seed=0)
val_ds = CMAPSSFinetuneDataset(val_engines, use_last_only=True)
test_ds = CMAPSSTestDataset(test_engines, test_rul)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_finetune)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_finetune)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_test)

# E2E optimizer
for p in jepa_model2.context_encoder.parameters():
    p.requires_grad = True
optimizer = torch.optim.Adam(
    list(jepa_model2.context_encoder.parameters()) +
    list(jepa_model2.predictor.parameters()) +
    list(probe.parameters()),
    lr=1e-4
)

best_val = float('inf')
best_enc_state = None
best_probe_state = None
patience_count = 0

for epoch in range(1, 101):
    jepa_model2.train(); probe.train()
    for past, mask, rul in train_loader:
        past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
        optimizer.zero_grad()
        h = jepa_model2.encode_past(past, mask)
        pred = probe(h)
        loss = F.mse_loss(pred, rul)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        optimizer.step()

    # Val RMSE
    jepa_model2.eval(); probe.eval()
    preds_v, tgts_v = [], []
    with torch.no_grad():
        for past, mask, rul in val_loader:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = jepa_model2.encode_past(past, mask)
            pred = probe(h)
            preds_v.append(pred.cpu().numpy())
            tgts_v.append(rul.numpy())
    val_rmse = float(np.sqrt(np.mean((np.concatenate(preds_v)*RUL_CAP - np.concatenate(tgts_v)*RUL_CAP)**2)))

    if val_rmse < best_val:
        best_val = val_rmse
        best_enc_state = copy.deepcopy(jepa_model2.context_encoder.state_dict())
        best_probe_state = copy.deepcopy(probe.state_dict())
        patience_count = 0
    else:
        patience_count += 1
        if patience_count >= 20:
            print(f"  Early stop at epoch {epoch}, best val={best_val:.2f}")
            break

    if epoch % 20 == 0:
        print(f"  Epoch {epoch}: val_rmse={val_rmse:.2f} (best={best_val:.2f})")

# Load best states
jepa_model2.context_encoder.load_state_dict(best_enc_state)
probe.load_state_dict(best_probe_state)

# Test RMSE
jepa_model2.eval(); probe.eval()
preds_t, tgts_t = [], []
with torch.no_grad():
    for past, mask, rul_gt in test_loader:
        past, mask = past.to(DEVICE), mask.to(DEVICE)
        h = jepa_model2.encode_past(past, mask)
        pred = probe(h)
        preds_t.append(pred.cpu().numpy() * RUL_CAP)
        tgts_t.append(rul_gt.numpy())
preds_t = np.concatenate(preds_t)
tgts_t = np.concatenate(tgts_t)
reconstructed_test_rmse = float(np.sqrt(np.mean((preds_t - tgts_t)**2)))
print(f"\nReconstructed model test RMSE (seed=0): {reconstructed_test_rmse:.2f}")

# Save probe and encoder
torch.save({
    'encoder_state': jepa_model2.context_encoder.state_dict(),
    'probe_state': probe.state_dict(),
    'test_rmse': reconstructed_test_rmse,
    'val_rmse': best_val,
}, str(V12_DIR / 'v2_e2e_seed0_reconstructed.pt'))
print(f"Saved reconstructed checkpoint to {V12_DIR / 'v2_e2e_seed0_reconstructed.pt'}")


# ============================================================
# Phase 0.2: Trajectory inference at every cycle
# ============================================================
print("\n" + "="*60)
print("Phase 0.2: Trajectory inference at every cycle")
print("="*60)

# Load raw test data to get true oracle RUL + engine lengths
train_df, test_df, test_rul_arr = load_raw('FD001')

# Get stats for normalization (reuse data['stats'])
stats = data['stats']

# Rebuild test sequences (already done in test_engines)
test_engine_ids = sorted(test_engines.keys())

@torch.no_grad()
def infer_at_cycle(model, probe_net, seq, cycle_idx):
    """
    Infer RUL at cycle_idx (0-indexed) using prefix seq[:cycle_idx+1].
    Returns raw RUL prediction.
    """
    prefix = seq[:cycle_idx+1]  # (t, 14)
    x = torch.tensor(prefix, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, t, 14)
    model.eval(); probe_net.eval()
    h = model.encode_past(x, None)
    pred_norm = probe_net(h)
    return float(pred_norm.cpu().numpy()[0]) * RUL_CAP


# For each test engine, run inference at every cycle from 30 to last
print(f"\nRunning trajectory inference on {len(test_engine_ids)} test engines...")

per_engine_pred_stds = []
per_engine_rhos = []
all_trajectories = {}

for idx, eid in enumerate(test_engine_ids):
    seq = test_engines[eid]
    T = seq.shape[0]
    oracle_rul = float(test_rul[idx])  # RUL at last observed cycle

    min_cycle = min(30, T)
    cycles = list(range(min_cycle, T + 1))  # 1-indexed: cycle 30 to T

    preds = []
    true_ruls = []

    for c in cycles:
        # c = number of cycles observed (1-indexed)
        # true RUL at cycle c = oracle_rul + (T - c)
        true_rul_at_c = oracle_rul + (T - c)
        true_rul_capped = min(true_rul_at_c, RUL_CAP)

        pred = infer_at_cycle(jepa_model2, probe, seq, c-1)  # c-1 is 0-indexed
        preds.append(pred)
        true_ruls.append(true_rul_capped)

    preds = np.array(preds)
    true_ruls = np.array(true_ruls)

    pred_std = float(np.std(preds))
    if len(preds) >= 3:
        rho, _ = spearmanr(preds, true_ruls)
        if np.isnan(rho):
            rho = 0.0
    else:
        rho = 0.0

    per_engine_pred_stds.append(pred_std)
    per_engine_rhos.append(float(rho))

    all_trajectories[int(eid)] = {
        'cycles': cycles,
        'preds': preds.tolist(),
        'true_ruls': true_ruls.tolist(),
        'T': T,
        'oracle_rul': oracle_rul,
        'pred_std': pred_std,
        'rho': float(rho),
    }

    if idx < 5:
        print(f"  Engine {eid}: T={T}, oracle_rul={oracle_rul:.1f}, "
              f"pred range=[{preds.min():.1f},{preds.max():.1f}], "
              f"pred_std={pred_std:.2f}, rho={rho:.3f}")

per_engine_pred_stds = np.array(per_engine_pred_stds)
per_engine_rhos = np.array(per_engine_rhos)

print(f"\nPer-engine pred std: mean={per_engine_pred_stds.mean():.2f}, median={np.median(per_engine_pred_stds):.2f}")
print(f"Per-engine rho: mean={per_engine_rhos.mean():.3f}, median={np.median(per_engine_rhos):.3f}")
print(f"Rho distribution: <0={np.sum(per_engine_rhos<0)}, 0-0.3={np.sum((per_engine_rhos>=0)&(per_engine_rhos<0.3))}, 0.3-0.7={np.sum((per_engine_rhos>=0.3)&(per_engine_rhos<0.7))}, >0.7={np.sum(per_engine_rhos>=0.7)}")


# ============================================================
# Phase 0.1: Plot prediction trajectories for 10 engines
# ============================================================
print("\nPlotting prediction trajectories for 10 engines...")

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('V2 E2E: Prediction Trajectories (V12 Diagnostic)\nEvery cycle from 30 to last',
             fontsize=13, fontweight='bold')

# Pick 10 engines: first 10 by engine_id
plot_ids = test_engine_ids[:10]

for i, eid in enumerate(plot_ids):
    ax = axes[i // 5, i % 5]
    traj = all_trajectories[int(eid)]
    cycles = np.array(traj['cycles'])
    preds = np.array(traj['preds'])
    true_ruls = np.array(traj['true_ruls'])

    ax.plot(cycles, true_ruls, 'k-', linewidth=2, label='True RUL', alpha=0.8)
    ax.plot(cycles, preds, 'r-', linewidth=1.5, label='Predicted', alpha=0.8)
    ax.axhline(np.mean(preds), color='orange', linestyle='--', linewidth=1, alpha=0.7,
               label=f'Mean pred={np.mean(preds):.0f}')

    ax.set_title(f'Engine {eid}\nstd={traj["pred_std"]:.1f}, rho={traj["rho"]:.2f}',
                 fontsize=9)
    ax.set_xlabel('Cycle', fontsize=8)
    ax.set_ylabel('RUL (cycles)', fontsize=8)
    ax.set_ylim(-5, RUL_CAP + 10)
    if i == 0:
        ax.legend(fontsize=7)

plt.tight_layout()
traj_plot_path = PLOTS_V12 / 'phase0_prediction_trajectories.png'
plt.savefig(str(traj_plot_path), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved trajectory plot to {traj_plot_path}")


# Also make a summary overview
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle('Phase 0 Diagnostics: Is V11 a constant predictor?', fontsize=13)

# Histogram of per-engine pred std
axes2[0].hist(per_engine_pred_stds, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
axes2[0].axvline(3, color='red', linestyle='--', label='Threshold=3')
axes2[0].axvline(10, color='green', linestyle='--', label='Target=10')
axes2[0].axvline(np.median(per_engine_pred_stds), color='orange', linestyle='-',
                  label=f'Median={np.median(per_engine_pred_stds):.1f}')
axes2[0].set_xlabel('Per-engine prediction std (cycles)')
axes2[0].set_title('Prediction Variability Within Engine\n(>3 = tracking, <3 = constant)')
axes2[0].legend(fontsize=8)

# Histogram of per-engine rho
axes2[1].hist(per_engine_rhos, bins=20, color='darkorange', edgecolor='white', alpha=0.8)
axes2[1].axvline(0.3, color='red', linestyle='--', label='Threshold=0.3')
axes2[1].axvline(0.7, color='green', linestyle='--', label='Target=0.7')
axes2[1].axvline(np.median(per_engine_rhos), color='blue', linestyle='-',
                  label=f'Median={np.median(per_engine_rhos):.2f}')
axes2[1].set_xlabel('Within-engine Spearman rho')
axes2[1].set_title('Within-engine Tracking Quality\n(>0.5 = real tracker)')
axes2[1].legend(fontsize=8)

# Scatter: pred at last cycle vs true RUL
last_preds = []
for eid in test_engine_ids:
    traj = all_trajectories[int(eid)]
    last_preds.append(traj['preds'][-1])
last_preds = np.array(last_preds)

axes2[2].scatter(test_rul, last_preds, alpha=0.7, s=30, color='purple')
axes2[2].plot([0, 150], [0, 150], 'k--', linewidth=1, label='Perfect')
axes2[2].axhline(np.mean(last_preds), color='red', linestyle='--',
                  label=f'Mean pred={np.mean(last_preds):.0f}')
axes2[2].set_xlabel('True RUL (cycles)')
axes2[2].set_ylabel('Predicted RUL at last cycle')
axes2[2].set_title(f'Last-window Predictions\nRMSE={reconstructed_test_rmse:.2f}')
axes2[2].legend(fontsize=8)
axes2[2].set_xlim(-5, 160)
axes2[2].set_ylim(-5, 160)

plt.tight_layout()
summary_plot_path = PLOTS_V12 / 'phase0_diagnostics_summary.png'
plt.savefig(str(summary_plot_path), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved summary plot to {summary_plot_path}")


# ============================================================
# Decision rule
# ============================================================
pred_std_median = float(np.median(per_engine_pred_stds))
rho_median = float(np.median(per_engine_rhos))
v11_reported = 13.80

print("\n" + "="*60)
print("PHASE 0 DECISION RULE")
print("="*60)
print(f"per_engine_pred_std_median = {pred_std_median:.2f}  (threshold: >10 for real tracker, <3 for constant)")
print(f"within_engine_rho_median   = {rho_median:.3f}  (threshold: >0.5 for real tracker, <0.3 for constant)")
print(f"constant_predictor_rmse    = {constant_rmse:.2f}")
print(f"v11_reported_rmse          = {v11_reported}")
print(f"reconstructed_rmse (seed0) = {reconstructed_test_rmse:.2f}")

# Note: we don't have the regressor result here (run separately)
# but we'll report the partial verdict based on trajectory diagnostics

if pred_std_median > 10 and rho_median > 0.5:
    traj_verdict = "TRACKING: model shows within-engine degradation tracking"
elif pred_std_median < 3 or rho_median < 0.3:
    traj_verdict = "CONSTANT: model is NOT tracking degradation within engine (constant-like output)"
else:
    traj_verdict = "AMBIGUOUS: partial tracking (std>3 but <10, or rho>0.3 but <0.5)"

print(f"\nTrajectory verdict: {traj_verdict}")

diagnostics = {
    "per_engine_pred_std_mean": float(per_engine_pred_stds.mean()),
    "per_engine_pred_std_median": float(np.median(per_engine_pred_stds)),
    "per_engine_pred_std_all": per_engine_pred_stds.tolist(),
    "within_engine_rho_mean": float(per_engine_rhos.mean()),
    "within_engine_rho_median": float(np.median(per_engine_rhos)),
    "within_engine_rho_all": per_engine_rhos.tolist(),
    "constant_predictor_rmse": constant_rmse,
    "constant_predictor_value": float(constant_pred),
    "v11_reported_rmse": v11_reported,
    "reconstructed_test_rmse_seed0": reconstructed_test_rmse,
    "n_test_engines": len(test_engine_ids),
    "trajectory_verdict": traj_verdict,
    "note": "Complete verdict requires engine_summary_regressor.json (Phase 0.2b)",
    "wall_time_s": float(time.time() - t0),
}

with open(V12_DIR / 'phase0_diagnostics.json', 'w') as f:
    json.dump(diagnostics, f, indent=2)

print(f"\nSaved diagnostics to {V12_DIR / 'phase0_diagnostics.json'}")
print(f"Total wall time: {time.time() - t0:.1f}s")
