"""
Phase 0.2b: Engine-summary ridge regressor lower bound
The most important single experiment in V12.

Fits a 60-feature ridge regressor on per-engine summary statistics.
If this matches V11's 13.80, the benchmark doesn't require degradation tracking.

Output: experiments/v12/engine_summary_regressor.json
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V12_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v12')
V12_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(V11_DIR))
from data_utils import load_raw, get_sensor_cols, fit_normalizer, build_engine_sequences, RUL_CAP

SEEDS = [42, 123, 456, 789, 1024]
V11_E2E_RMSE = 13.80
V11_FROZEN_RMSE = 17.81

print("=" * 60)
print("Phase 0.2b: Engine-summary ridge regressor")
print("=" * 60)
t0 = time.time()

# Load FD001 data
train_df, test_df, test_rul = load_raw('FD001')
sensor_cols = get_sensor_cols()
n_sensors = len(sensor_cols)
print(f"Sensors: {sensor_cols}")
print(f"N sensors: {n_sensors}")

# Fit normalizer on training data
stats = fit_normalizer(train_df, per_condition=False)

# Build normalized engine sequences
all_train_seqs = build_engine_sequences(train_df, stats, per_condition=False)
all_test_seqs = build_engine_sequences(test_df, stats, per_condition=False)

all_ids = sorted(all_train_seqs.keys())
print(f"Total train engines: {len(all_ids)}")
print(f"Total test engines: {len(all_test_seqs)}")

# Mean train lifecycle for relative length feature
mean_train_T = np.mean([all_train_seqs[i].shape[0] for i in all_ids])
print(f"Mean train engine length: {mean_train_T:.1f}")


def compute_engine_features(seq: np.ndarray, mean_T: float) -> np.ndarray:
    """
    Compute 60-feature summary for one engine.
    seq: (T, 14) normalized sensor array
    Features:
    - T_obs, T_obs / mean_T (2 features)
    - Last-30-cycle per-sensor mean (14)
    - Last-30-cycle per-sensor std (14)
    - Last-30-cycle per-sensor linear slope (14)
    - Global delta: mean(last_30) - mean(first_30) per sensor (14)
    Total: 2 + 14*4 = 58 features (note: spec says ~60, this gives 58)
    """
    T = seq.shape[0]
    features = []

    # Length features (2)
    features.append(float(T))
    features.append(float(T) / mean_T)

    # Last 30 cycles (or all if T < 30)
    last_n = min(30, T)
    last_seg = seq[-last_n:]  # (last_n, 14)

    # First 30 cycles
    first_n = min(30, T)
    first_seg = seq[:first_n]  # (first_n, 14)

    # Per-sensor last-30 mean (14)
    last_mean = last_seg.mean(axis=0)
    features.extend(last_mean.tolist())

    # Per-sensor last-30 std (14)
    last_std = last_seg.std(axis=0)
    features.extend(last_std.tolist())

    # Per-sensor last-30 linear slope (14)
    if last_n >= 2:
        x = np.arange(last_n, dtype=np.float32)
        x_mean = x.mean()
        x_var = np.sum((x - x_mean) ** 2)
        if x_var > 0:
            slopes = np.sum((x[:, None] - x_mean) * (last_seg - last_seg.mean(axis=0)), axis=0) / x_var
        else:
            slopes = np.zeros(last_seg.shape[1])
    else:
        slopes = np.zeros(last_seg.shape[1])
    features.extend(slopes.tolist())

    # Global delta: mean(last_30) - mean(first_30) (14)
    first_mean = first_seg.mean(axis=0)
    delta = last_mean - first_mean
    features.extend(delta.tolist())

    return np.array(features, dtype=np.float32)


# Compute features for ALL training engines
print("\nComputing training engine features...")
all_features = {}
for eid in all_ids:
    seq = all_train_seqs[eid]
    all_features[eid] = compute_engine_features(seq, mean_train_T)

n_features = all_features[all_ids[0]].shape[0]
print(f"Feature dimension: {n_features}")

# Compute test engine features
print("Computing test engine features...")
test_features = {}
test_engine_ids = sorted(all_test_seqs.keys())
for eid in test_engine_ids:
    seq = all_test_seqs[eid]
    test_features[eid] = compute_engine_features(seq, mean_train_T)

# Build test X matrix (ordered by engine_id to match test_rul)
X_test = np.stack([test_features[eid] for eid in test_engine_ids])
y_test = test_rul  # ground truth RUL (raw cycles, not capped)

print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"Test RUL stats: mean={y_test.mean():.1f}, std={y_test.std():.1f}, min={y_test.min():.1f}, max={y_test.max():.1f}")

# Run 5-seed cross-validation
# Each seed picks a different 85/15 train/val split (matching V11 protocol for finding val)
# But for training the regressor, we use ALL 85% train engines
# We always evaluate on the CANONICAL test set

per_seed_rmse = []
feature_weights_all = []

for seed in SEEDS:
    rng = np.random.default_rng(seed)
    n_val = max(1, int(0.15 * len(all_ids)))
    val_ids = set(rng.choice(all_ids, size=n_val, replace=False).tolist())
    train_ids = [i for i in all_ids if i not in val_ids]

    # Training data: use train_ids, RUL labels from last cycle (capped)
    X_train_seed = np.stack([all_features[eid] for eid in train_ids])
    # RUL label for each training engine = min(T_obs, 125) since train sequences end at failure
    # Actually: RUL at LAST cycle = 1, but we want to predict RUL at last window, which is 1
    # But for the last-window eval protocol, we want RUL at the cut point of the test engine
    # For training, we use each engine's LAST window RUL = 1 cycle is too easy
    # Better: use the engine's TRUE lifetime for regression label
    # Per V11 protocol: training label = RUL at last window = 1 (not useful for fitting)
    # Actually we should fit on multiple cuts - use the last window for consistency
    # Let's use: label = min(T_obs, RUL_CAP) = "how long the engine ran, capped at 125"
    # This is what the regressor would naturally learn from T_obs features

    # Actually the V11 fine-tune label is: normalized RUL at any cut point / RUL_CAP
    # For a per-engine regressor, the most useful label is: what is the RUL at the test cut?
    # Since we evaluate at last window, we want to predict test_rul[engine_id]
    # For training engines: RUL at last cycle = 1 (they ran to failure)
    # So label for training = 1 for all engines in last-window setting
    # This is a degenerate case! Let's use multi-cut training:

    # Use the CAPPED RUL at the last observed cycle for training engines
    # = 1 for all (they all fail). That's degenerate.
    # Better: use random cut points as in V11 fine-tuning

    # Most natural for a summary regressor: predict lifetime from features
    # Lifetime = T_obs (engine length). Then at test time, test engine length = T_obs_test
    # and test RUL = oracle_RUL (from file), but regressor output is T_obs - something...

    # THE CORRECT FORMULATION:
    # For test engines: we observe up to the cut point. The regressor should predict
    # how many cycles remain. The test cut is the LAST OBSERVED CYCLE.
    # For training with per-engine features computed at last cycle:
    # The "natural" label is the engine's total life T (if we subtract current cycle)
    # But we don't know T for test engines.
    # What V12 spec says: evaluate on "last-window protocol" = same as V11
    # For training: use RUL labels at the training engine's last cycle.
    # Last cycle RUL = 1 (all training engines run to failure).
    # This is degenerate.
    #
    # RESOLUTION: The spirit of Phase 0.2b is to build a "dumb" regressor that varies
    # across engines but is flat within each engine. The key question is whether it
    # can match 13.80 RMSE on test set.
    #
    # Best approach: train on MULTIPLE CUT POINTS per training engine.
    # For each training engine, sample cuts at [25%, 50%, 75%, 100%] of T_obs.
    # At each cut point: compute features from the PREFIX up to cut, label = capped RUL.

    # Recompute features at multiple cut points
    X_multi = []
    y_multi = []
    for eid in train_ids:
        seq = all_train_seqs[eid]
        T = seq.shape[0]
        # Cut points at 30, 50%, 75%, 100% of T (at least 30 cycles)
        cut_points = sorted(set([
            30, max(30, T // 2), max(30, 3*T//4), T
        ]))
        for cp in cut_points:
            if cp > T:
                cp = T
            prefix = seq[:cp]
            feats = compute_engine_features(prefix, mean_train_T)
            # True RUL at cut point: T_obs - cp (uncapped) then cap
            rul_at_cut = min(T - cp, RUL_CAP)
            # For last window (cp=T): rul = min(0, cap) = 0? No: T cycles total, at cycle T, RUL=1
            # Actually: at cycle cp (1-indexed), RUL = T - cp (0-indexed remaining)
            # At last cycle (cp=T): RUL = 0? But V11 uses RUL at last cycle = 1
            # C-MAPSS: RUL = remaining cycles until failure. At cycle T: RUL = T - T = 0? No.
            # Standard formula: RUL at cycle t = (T_max - t), where T_max = total cycles
            # At t=T_max: RUL = 0. But V11 RESULTS say engines have RUL~mean~92 at test time.
            # For TRAINING engines that run to failure: RUL labels = arange(T,0,-1)
            # So at cycle t (1-indexed): RUL = T - t + 1 (counting remaining INCLUDING current)
            # Wait: compute_rul_labels(n_cycles) = arange(n_cycles, 0, -1)
            # So for T cycles: [T, T-1, ..., 2, 1]. At last cycle: RUL = 1.
            # At cut point cp (1-indexed): RUL = T - cp + 1
            rul_at_cut = float(min(T - cp + 1, RUL_CAP))
            X_multi.append(feats)
            y_multi.append(rul_at_cut)

    X_train_m = np.stack(X_multi)
    y_train_m = np.array(y_multi, dtype=np.float32)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_m)
    X_test_scaled = scaler.transform(X_test)

    # Fit ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train_m)

    # Predict on test set
    preds = ridge.predict(X_test_scaled)
    # Clip to [0, 125]
    preds = np.clip(preds, 0, RUL_CAP)

    rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
    per_seed_rmse.append(rmse)
    feature_weights_all.append(ridge.coef_.copy())
    print(f"  seed={seed}: RMSE={rmse:.3f}")

mean_rmse = float(np.mean(per_seed_rmse))
std_rmse = float(np.std(per_seed_rmse))
print(f"\nEngine-summary regressor: {mean_rmse:.3f} +/- {std_rmse:.3f}")
print(f"V11 E2E:  {V11_E2E_RMSE}")
print(f"V11 frozen: {V11_FROZEN_RMSE}")
print(f"Delta vs E2E: {mean_rmse - V11_E2E_RMSE:+.2f}")
print(f"Delta vs frozen: {mean_rmse - V11_FROZEN_RMSE:+.2f}")

# Determine verdict
delta_vs_e2e = mean_rmse - V11_E2E_RMSE
if delta_vs_e2e > 1.0:
    verdict = f"V11 E2E beats regressor by {delta_vs_e2e:.2f} RMSE - benchmark requires something beyond simple summary statistics"
elif delta_vs_e2e > -1.0:
    verdict = f"V11 E2E within {abs(delta_vs_e2e):.2f} RMSE of regressor - benchmark cannot distinguish V11 from summary features"
else:
    verdict = f"V11 E2E WORSE than regressor by {abs(delta_vs_e2e):.2f} RMSE - ALARM: regressor dominates"

print(f"\nVERDICT: {verdict}")

# Top 10 feature weights (averaged over seeds)
feature_names = (
    ['T_obs', 'T_obs_rel'] +
    [f'last30_mean_s{i+1}' for i in range(n_sensors)] +
    [f'last30_std_s{i+1}' for i in range(n_sensors)] +
    [f'last30_slope_s{i+1}' for i in range(n_sensors)] +
    [f'delta_s{i+1}' for i in range(n_sensors)]
)

mean_weights = np.mean(np.stack(feature_weights_all), axis=0)
abs_weights = np.abs(mean_weights)
top10_idx = np.argsort(abs_weights)[::-1][:10]
top10 = [{"feature": feature_names[i], "weight": float(mean_weights[i])} for i in top10_idx]

print("\nTop-10 feature weights (by |weight|):")
for f in top10:
    print(f"  {f['feature']}: {f['weight']:+.4f}")

result = {
    "mean_rmse": mean_rmse,
    "std_rmse": std_rmse,
    "per_seed_rmse": per_seed_rmse,
    "v11_e2e_rmse": V11_E2E_RMSE,
    "v11_frozen_rmse": V11_FROZEN_RMSE,
    "delta_vs_v11_e2e": float(delta_vs_e2e),
    "delta_vs_v11_frozen": float(mean_rmse - V11_FROZEN_RMSE),
    "top_10_feature_weights": top10,
    "n_features": n_features,
    "verdict": verdict,
    "wall_time_s": float(time.time() - t0),
}

with open(V12_DIR / 'engine_summary_regressor.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"\nResults saved to {V12_DIR / 'engine_summary_regressor.json'}")
print(f"Wall time: {time.time() - t0:.1f}s")
