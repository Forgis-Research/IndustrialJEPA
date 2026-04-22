"""
Phase 1: Diagnose FD002 val/test gap

1.1: Measure val/test gap explicitly for FD001 and FD002
1.2: Check FD002 test-time condition assignment
1.3: Op-settings as input channels ablation for FD002

Output:
  experiments/v12/val_test_gap.json
  analysis/plots/v12/fd002_condition_assignment.png
  experiments/v12/fd002_condition_input_results.json
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
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V12_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v12')
PLOTS_V12 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/analysis/plots/v12')
V12_DIR.mkdir(exist_ok=True)
PLOTS_V12.mkdir(exist_ok=True)

sys.path.insert(0, str(V11_DIR))
from data_utils import (
    load_cmapss_subset, load_raw, get_sensor_cols, get_op_cols, fit_normalizer,
    build_engine_sequences, CMAPSSFinetuneDataset, CMAPSSTestDataset,
    collate_finetune, collate_test, RUL_CAP, N_SENSORS
)
from models import TrajectoryJEPA, RULProbe
from train_utils import DEVICE, finetune, _eval_test_rmse, linear_probe_rmse

SEEDS = [42, 123, 456, 789, 1024]

print("=" * 60)
print("Phase 1: FD002 Val/Test Gap Diagnosis")
print("=" * 60)
print(f"Device: {DEVICE}")


# ============================================================
# 1.1 Val/test gap for FD001 and FD002
# ============================================================
print("\n--- Phase 1.1: Val/Test Gap ---")

val_test_gap_results = {}

for subset, ckpt_name in [('FD001', 'best_pretrain_L1_v2.pt'), ('FD002', 'best_pretrain_fd002.pt')]:
    print(f"\nSubset: {subset}")
    ckpt_path = V11_DIR / ckpt_name
    print(f"Checkpoint: {ckpt_path}")

    data = load_cmapss_subset(subset)
    train_engines = data['train_engines']
    val_engines = data['val_engines']
    test_engines = data['test_engines']
    test_rul = data['test_rul']

    print(f"  Train: {len(train_engines)}, Val: {len(val_engines)}, Test: {len(test_engines)}")

    # Build V2 model and load pretrained
    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1,
        d_model=256, n_heads=4, n_layers=2, d_ff=512,
        dropout=0.1, ema_momentum=0.996, predictor_hidden=256
    ).to(DEVICE)
    state = torch.load(str(ckpt_path), map_location=DEVICE)
    model.load_state_dict(state)

    # Compute val probe RMSE (use the linear_probe_rmse function from v11)
    t0 = time.time()
    val_probe_rmse = linear_probe_rmse(model, train_engines, val_engines, n_epochs=100)
    print(f"  Val probe RMSE: {val_probe_rmse:.2f}  ({time.time()-t0:.1f}s)")

    # Fine-tune frozen on full training data, evaluate on test
    # Run 3 seeds for stability
    test_rmses = []
    for seed in SEEDS[:3]:
        result = finetune(
            model=model,
            train_engines=train_engines,
            val_engines=val_engines,
            test_engines=test_engines,
            test_rul=test_rul,
            n_epochs=100,
            mode='frozen',
            seed=seed,
            verbose=False,
        )
        test_rmses.append(result['test_rmse'])
        print(f"    seed={seed}: test_rmse={result['test_rmse']:.2f}")

    test_rmse_mean = float(np.mean(test_rmses))
    test_rmse_std = float(np.std(test_rmses))
    gap = test_rmse_mean - val_probe_rmse

    print(f"  Test RMSE: {test_rmse_mean:.2f} +/- {test_rmse_std:.2f}")
    print(f"  Val/Test gap: {gap:+.2f}")

    val_test_gap_results[subset] = {
        "val_probe_rmse": float(val_probe_rmse),
        "test_rmse_mean": float(test_rmse_mean),
        "test_rmse_std": float(test_rmse_std),
        "val_test_gap": float(gap),
        "per_seed_test_rmse": test_rmses,
    }

print("\n--- Val/Test Gap Summary ---")
for subset, r in val_test_gap_results.items():
    print(f"  {subset}: val={r['val_probe_rmse']:.2f}, test={r['test_rmse_mean']:.2f}, gap={r['val_test_gap']:+.2f}")

with open(V12_DIR / 'val_test_gap.json', 'w') as f:
    json.dump(val_test_gap_results, f, indent=2)
print(f"Saved to {V12_DIR / 'val_test_gap.json'}")


# ============================================================
# 1.2 FD002 test-time condition assignment
# ============================================================
print("\n--- Phase 1.2: FD002 Test-time Condition Assignment ---")

train_df_002, test_df_002, _ = load_raw('FD002')
op_cols = get_op_cols()

# Fit KMeans on training data
print("Fitting KMeans(k=6) on FD002 training op settings...")
op_train = train_df_002[op_cols].values
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
kmeans.fit(op_train)
train_cond_labels = kmeans.predict(op_train)

# Train condition distribution (all cycles)
train_cond_counts = np.bincount(train_cond_labels, minlength=6)
train_cond_frac = train_cond_counts / train_cond_counts.sum()
print(f"Training condition distribution: {dict(enumerate(train_cond_frac.round(3)))}")

# Test engines: condition at LAST window (last 64 cycles or just last cycle)
# The key is: what condition are the test engine LAST WINDOWS predominantly in?
test_engine_ids_002 = sorted(test_df_002['engine_id'].unique())
test_last_window_conds = []

for eid in test_engine_ids_002:
    eng = test_df_002[test_df_002['engine_id'] == eid].sort_values('cycle')
    # Last 30 cycles (conservative) or last cycle
    last_rows = eng.tail(30)
    last_op = last_rows[op_cols].values
    last_conds = kmeans.predict(last_op)
    # Most common condition in last window
    most_common = int(np.bincount(last_conds).argmax())
    test_last_window_conds.append(most_common)

test_cond_counts = np.bincount(test_last_window_conds, minlength=6)
test_cond_frac = test_cond_counts / max(1, test_cond_counts.sum())
print(f"Test last-window condition distribution: {dict(enumerate(test_cond_frac.round(3)))}")

# Check overrepresentation
print("\nCondition overrepresentation (test / train):")
for c in range(6):
    ratio = (test_cond_frac[c] + 1e-9) / (train_cond_frac[c] + 1e-9)
    flag = " <-- OVERREPRESENTED (>1.5x)" if ratio > 1.5 else ""
    print(f"  Cond {c}: train={train_cond_frac[c]:.3f}, test={test_cond_frac[c]:.3f}, ratio={ratio:.2f}{flag}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('FD002: Operating Condition Distribution - Train vs Test Last Windows', fontsize=12)

x = np.arange(6)
width = 0.35
axes[0].bar(x - width/2, train_cond_frac, width, label='Training (all cycles)', color='steelblue', alpha=0.8)
axes[0].bar(x + width/2, test_cond_frac, width, label='Test (last 30 cycles)', color='darkorange', alpha=0.8)
axes[0].set_xlabel('Operating Condition (KMeans cluster)')
axes[0].set_ylabel('Fraction')
axes[0].set_title('FD002: Condition Distribution')
axes[0].legend()
axes[0].set_xticks(x)
axes[0].set_xticklabels([f'Cond {i}' for i in range(6)])

# Overrepresentation ratio
ratios = [(test_cond_frac[c] + 1e-9) / (train_cond_frac[c] + 1e-9) for c in range(6)]
bar_colors = ['red' if r > 1.5 else 'steelblue' for r in ratios]
axes[1].bar(x, ratios, color=bar_colors, alpha=0.8, edgecolor='black')
axes[1].axhline(1.5, color='red', linestyle='--', label='1.5x threshold')
axes[1].axhline(1.0, color='black', linestyle='-', linewidth=0.5)
axes[1].set_xlabel('Operating Condition')
axes[1].set_ylabel('Test/Train ratio')
axes[1].set_title('FD002: Test Overrepresentation Ratio')
axes[1].legend()
axes[1].set_xticks(x)
axes[1].set_xticklabels([f'Cond {i}' for i in range(6)])

plt.tight_layout()
cond_plot_path = PLOTS_V12 / 'fd002_condition_assignment.png'
plt.savefig(str(cond_plot_path), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved condition plot to {cond_plot_path}")

condition_diagnosis = {
    "train_condition_fractions": train_cond_frac.tolist(),
    "test_last_window_fractions": test_cond_frac.tolist(),
    "overrepresentation_ratios": ratios,
    "max_overrepresentation": float(max(ratios)),
    "any_overrepresented": any(r > 1.5 for r in ratios),
    "overrepresented_conditions": [c for c, r in enumerate(ratios) if r > 1.5],
}


# ============================================================
# 1.3 Op-settings as input channels ablation for FD002
# ============================================================
print("\n--- Phase 1.3: Op-settings as input channels (FD002 ablation) ---")

# Build FD002 data with 17 channels (14 sensors + 3 op settings)
print("Building FD002 data with 17 channels (sensors + op settings)...")

# Load raw data
train_df_002, test_df_002, test_rul_002 = load_raw('FD002')
op_cols_002 = get_op_cols()
sensor_cols_002 = get_sensor_cols()

# Fit global (not per-condition) normalizer for sensors + op settings
# Op settings: use global min-max over training data
all_train_engine_ids = sorted(train_df_002['engine_id'].unique())
rng = np.random.default_rng(42)
n_val = max(1, int(0.15 * len(all_train_engine_ids)))
val_ids_002 = set(rng.choice(all_train_engine_ids, size=n_val, replace=False).tolist())
train_ids_002 = [i for i in all_train_engine_ids if i not in val_ids_002]

train_df_002_split = train_df_002[train_df_002['engine_id'].isin(train_ids_002)]

# Sensor normalization (global, not per-condition)
sensor_stats = {}
for col in sensor_cols_002:
    sensor_stats[col] = (
        float(train_df_002_split[col].min()),
        float(train_df_002_split[col].max())
    )

# Op setting normalization
op_stats = {}
for col in op_cols_002:
    op_stats[col] = (
        float(train_df_002_split[col].min()),
        float(train_df_002_split[col].max())
    )


def build_17channel_sequences(df, sensor_stats, op_stats, engine_ids=None):
    """Build 17-channel sequences (14 sensors + 3 op settings) with global normalization."""
    sequences = {}
    if engine_ids is None:
        engine_ids = sorted(df['engine_id'].unique())

    for eid in engine_ids:
        grp = df[df['engine_id'] == eid].sort_values('cycle')
        sensor_vals = grp[sensor_cols_002].values.astype(np.float32)
        op_vals = grp[op_cols_002].values.astype(np.float32)

        # Normalize sensors
        norm_sensors = np.zeros_like(sensor_vals)
        for i, col in enumerate(sensor_cols_002):
            mn, mx = sensor_stats[col]
            if mx > mn:
                norm_sensors[:, i] = (sensor_vals[:, i] - mn) / (mx - mn)
            else:
                norm_sensors[:, i] = 0.0

        # Normalize op settings
        norm_op = np.zeros_like(op_vals)
        for i, col in enumerate(op_cols_002):
            mn, mx = op_stats[col]
            if mx > mn:
                norm_op[:, i] = (op_vals[:, i] - mn) / (mx - mn)
            else:
                norm_op[:, i] = 0.0

        # Concatenate: 14 + 3 = 17 channels
        combined = np.concatenate([norm_sensors, norm_op], axis=1)
        sequences[int(eid)] = combined

    return sequences


print("Building 17-channel sequences for FD002...")
train_seqs_17ch = build_17channel_sequences(
    train_df_002[train_df_002['engine_id'].isin(train_ids_002)],
    sensor_stats, op_stats, train_ids_002
)
val_seqs_17ch = build_17channel_sequences(
    train_df_002[train_df_002['engine_id'].isin(val_ids_002)],
    sensor_stats, op_stats, list(val_ids_002)
)
test_seqs_17ch = build_17channel_sequences(
    test_df_002, sensor_stats, op_stats
)

print(f"  Train: {len(train_seqs_17ch)}, Val: {len(val_seqs_17ch)}, Test: {len(test_seqs_17ch)}")
sample_seq = list(train_seqs_17ch.values())[0]
print(f"  Sample sequence shape: {sample_seq.shape} (should be (T, 17))")

# Build V2 model with 17 input channels
print("\nPretraining V2 model with 17 channels on FD002...")

from data_utils import (CMAPSSPretrainDataset, collate_pretrain)
from train_utils import pretrain

t_pretrain = time.time()
model_17ch = TrajectoryJEPA(
    n_sensors=17,  # 14 + 3 op settings
    patch_length=1,
    d_model=256,
    n_heads=4,
    n_layers=2,
    d_ff=512,
    dropout=0.1,
    ema_momentum=0.996,
    predictor_hidden=256
).to(DEVICE)

n_params_17ch = sum(p.numel() for p in model_17ch.parameters())
print(f"17-ch model parameters: {n_params_17ch:,}")

# Pretrain for 100 epochs (abbreviated - full 200 is too slow, 100 captures trend)
history_17ch, best_probe_17ch = pretrain(
    model_17ch,
    train_seqs_17ch,
    val_seqs_17ch,
    n_epochs=100,
    batch_size=8,
    lr=3e-4,
    weight_decay=0.01,
    n_cuts_per_epoch=20,
    min_past=10,
    min_horizon=5,
    max_horizon=30,
    lambda_var=0.01,
    probe_every=20,
    checkpoint_path=str(V12_DIR / 'best_pretrain_fd002_17ch.pt'),
    verbose=True,
)
print(f"Pretrain done in {time.time()-t_pretrain:.1f}s, best probe RMSE: {best_probe_17ch:.2f}")

# Fine-tune: frozen + E2E, 5 seeds
print("\nFine-tuning 17ch FD002 model (frozen + E2E), 5 seeds...")

# Need to override the CMAPSSFinetuneDataset/TestDataset to use our 17ch sequences
# The dataset uses (T, 14) sequences - we need (T, 17)
# CMAPSSFinetuneDataset works with any (T, S) sequences so it should work directly

fd002_condition_results = {
    "n_sensors_baseline": 14,
    "n_sensors_ablation": 17,
    "pretrain_best_probe_rmse": float(best_probe_17ch),
    "baseline_frozen_rmse": 26.33,  # from V11 results
    "baseline_e2e_rmse": 24.45,
    "seeds": SEEDS,
}

# Build test dataset for evaluation
# test_rul_002 contains ground truth RUL for test engines
frozen_rmses = []
e2e_rmses = []

for seed in SEEDS:
    # Reload model from pretrain checkpoint
    model_17ch_run = TrajectoryJEPA(
        n_sensors=17, patch_length=1, d_model=256, n_heads=4, n_layers=2, d_ff=512,
        dropout=0.1, ema_momentum=0.996, predictor_hidden=256
    ).to(DEVICE)
    model_17ch_run.load_state_dict(torch.load(str(V12_DIR / 'best_pretrain_fd002_17ch.pt'), map_location=DEVICE))

    # Frozen
    result_frozen = finetune(
        model=model_17ch_run,
        train_engines=train_seqs_17ch,
        val_engines=val_seqs_17ch,
        test_engines=test_seqs_17ch,
        test_rul=test_rul_002,
        n_epochs=100,
        mode='frozen',
        seed=seed,
        verbose=False,
    )
    frozen_rmses.append(result_frozen['test_rmse'])
    print(f"  seed={seed} frozen: {result_frozen['test_rmse']:.2f}")

    # E2E
    model_17ch_e2e = TrajectoryJEPA(
        n_sensors=17, patch_length=1, d_model=256, n_heads=4, n_layers=2, d_ff=512,
        dropout=0.1, ema_momentum=0.996, predictor_hidden=256
    ).to(DEVICE)
    model_17ch_e2e.load_state_dict(torch.load(str(V12_DIR / 'best_pretrain_fd002_17ch.pt'), map_location=DEVICE))

    result_e2e = finetune(
        model=model_17ch_e2e,
        train_engines=train_seqs_17ch,
        val_engines=val_seqs_17ch,
        test_engines=test_seqs_17ch,
        test_rul=test_rul_002,
        n_epochs=100,
        mode='e2e',
        seed=seed,
        verbose=False,
    )
    e2e_rmses.append(result_e2e['test_rmse'])
    print(f"  seed={seed} e2e: {result_e2e['test_rmse']:.2f}")

frozen_mean = float(np.mean(frozen_rmses))
frozen_std = float(np.std(frozen_rmses))
e2e_mean = float(np.mean(e2e_rmses))
e2e_std = float(np.std(e2e_rmses))

print(f"\n17ch FD002 frozen: {frozen_mean:.2f} +/- {frozen_std:.2f}  (baseline: 26.33)")
print(f"17ch FD002 e2e:    {e2e_mean:.2f} +/- {e2e_std:.2f}  (baseline: 24.45)")
print(f"Improvement frozen: {26.33 - frozen_mean:+.2f}")
print(f"Improvement e2e:    {24.45 - e2e_mean:+.2f}")

if frozen_mean < 20:
    verdict_1_3 = f"CONFIRMED: 17ch FD002 frozen RMSE={frozen_mean:.2f} < 20. Condition-awareness hypothesis confirmed."
elif frozen_mean < 24:
    verdict_1_3 = f"PARTIAL: 17ch FD002 frozen RMSE={frozen_mean:.2f} improved from 26.33 but not <20. Partial support."
else:
    verdict_1_3 = f"NOT CONFIRMED: 17ch FD002 frozen RMSE={frozen_mean:.2f} not better than baseline 26.33. Bug is elsewhere."

print(f"\nVerdict 1.3: {verdict_1_3}")

fd002_condition_results.update({
    "frozen_mean_rmse": frozen_mean,
    "frozen_std_rmse": frozen_std,
    "e2e_mean_rmse": e2e_mean,
    "e2e_std_rmse": e2e_std,
    "frozen_per_seed": frozen_rmses,
    "e2e_per_seed": e2e_rmses,
    "improvement_frozen": float(26.33 - frozen_mean),
    "improvement_e2e": float(24.45 - e2e_mean),
    "verdict": verdict_1_3,
    "condition_diagnosis": condition_diagnosis,
})

with open(V12_DIR / 'fd002_condition_input_results.json', 'w') as f:
    json.dump(fd002_condition_results, f, indent=2)
print(f"Saved to {V12_DIR / 'fd002_condition_input_results.json'}")

print("\nPhase 1 complete.")
