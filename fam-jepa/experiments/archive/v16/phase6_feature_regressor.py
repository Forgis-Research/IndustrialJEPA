"""
Phase 6: Feature Regressor Baseline (Rule 3 from research protocol).

Ridge regression on 5-15 hand-designed features as a tightest cheap lower bound.
If V16b frozen probe barely beats this, the encoder contributes nothing the protocol can see.

Features per engine (last window):
1. last_cycle_values: sensor values at last observed cycle (N_sensors = 14)
2. per_sensor_slope: linear slope over last 30 cycles (N_sensors = 14)
3. per_sensor_mean: mean over full observed window (N_sensors = 14)
4. per_sensor_std: std over full observed window (N_sensors = 14)
5. sequence_length: normalized length of observed sequence (1)

Total: 4 * 14 + 1 = 57 features

Evaluation protocol: frozen probe RMSE on val set (same as V16b probe eval).
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V16_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v16')
sys.path.insert(0, str(V11_DIR))

warnings.filterwarnings('ignore')

from data_utils import (
    load_cmapss_subset, N_SENSORS, RUL_CAP,
    CMAPSSFinetuneDataset, CMAPSSTestDataset,
    collate_finetune, collate_test,
)

SEEDS = [42, 123, 456]
MAX_SEQ_LEN = 362.0  # FD001 approx max


def extract_features_from_loader(loader):
    """Extract hand-designed features from batched dataset.

    Each batch: (past, mask, rul_norm) from collate_finetune.
    past: (B, T, S), mask: (B, T), rul_norm: (B,)
    """
    all_feats, all_labels = [], []

    for past, mask, rul in loader:
        # past: (B, T, S), mask: (B, T), rul_norm: (B,)
        B, T, S = past.shape
        past_np = past.numpy()
        mask_np = mask.numpy()  # True = valid position
        rul_np = rul.numpy()

        for b in range(B):
            # mask_np[b] is True for PADDING, False for valid positions
            valid_len = int((~mask_np[b]).sum())
            sensors = past_np[b, :valid_len, :]  # (valid_T, S)
            valid_T = len(sensors)

            feats = []
            # 1. Last cycle sensor values
            feats.extend(sensors[-1].tolist())

            # 2. Per-sensor slope over last 30 cycles
            n_last = min(30, valid_T)
            last_window = sensors[-n_last:]
            if n_last >= 2:
                t = np.linspace(0, 1, n_last)
                for s in range(S):
                    slope = np.polyfit(t, last_window[:, s], 1)[0]
                    feats.append(float(slope))
            else:
                feats.extend([0.0] * S)

            # 3. Per-sensor mean over full window
            feats.extend(sensors.mean(axis=0).tolist())

            # 4. Per-sensor std over full window
            feats.extend(sensors.std(axis=0).tolist())

            # 5. Normalized sequence length
            feats.append(valid_T / MAX_SEQ_LEN)

            all_feats.append(feats)
            all_labels.append(float(rul_np[b]))

    return np.array(all_feats, dtype=np.float32), np.array(all_labels, dtype=np.float32)


def train_probe_sklearn(tr_X, tr_y, va_X, va_y, alphas=None):
    """Train ridge regression probe, grid search over alpha. Returns best val RMSE and model."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    scaler = StandardScaler()
    tr_X_s = scaler.fit_transform(tr_X)
    va_X_s = scaler.transform(va_X)

    best_val, best_alpha, best_model = float('inf'), None, None
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(tr_X_s, tr_y)
        va_pred = ridge.predict(va_X_s)
        val_rmse = float(np.sqrt(np.mean((va_pred * RUL_CAP - va_y * RUL_CAP) ** 2)))
        if val_rmse < best_val:
            best_val, best_alpha, best_model = val_rmse, alpha, (ridge, scaler)

    return best_val, best_alpha, best_model


if __name__ == '__main__':
    print("=" * 60)
    print("Phase 6: Feature Regressor Baseline (Ridge on Hand Features)")
    print("=" * 60)

    data = load_cmapss_subset('FD001')

    # Build datasets
    tr_ds = CMAPSSFinetuneDataset(data['train_engines'], n_cuts_per_engine=10, seed=42)
    va_ds = CMAPSSFinetuneDataset(data['val_engines'], use_last_only=True)
    te_ds = CMAPSSTestDataset(data['test_engines'], data['test_rul'])

    tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=False, collate_fn=collate_finetune)
    va_loader = DataLoader(va_ds, batch_size=64, shuffle=False, collate_fn=collate_finetune)
    te_loader = DataLoader(te_ds, batch_size=64, shuffle=False, collate_fn=collate_test)

    print("\nExtracting hand-designed features...")
    tr_X, tr_y = extract_features_from_loader(tr_loader)
    va_X, va_y = extract_features_from_loader(va_loader)

    # Test set: use collate_test which returns (past, mask, rul_gt_cycles)
    # Extract features same way but labels are raw cycles
    te_feats, te_labels = [], []
    for past, mask, rul in te_loader:
        B, T, S = past.shape
        past_np = past.numpy()
        mask_np = mask.numpy()
        rul_np = rul.numpy()  # raw cycles (not normalized)

        for b in range(B):
            # mask_np[b] is True for PADDING, False for valid positions
            valid_len = int((~mask_np[b]).sum())
            sensors = past_np[b, :valid_len, :]
            valid_T = len(sensors)

            feats = []
            feats.extend(sensors[-1].tolist())

            n_last = min(30, valid_T)
            last_window = sensors[-n_last:]
            if n_last >= 2:
                t = np.linspace(0, 1, n_last)
                for s in range(S):
                    slope = np.polyfit(t, last_window[:, s], 1)[0]
                    feats.append(float(slope))
            else:
                feats.extend([0.0] * S)

            feats.extend(sensors.mean(axis=0).tolist())
            feats.extend(sensors.std(axis=0).tolist())
            feats.append(valid_T / MAX_SEQ_LEN)

            te_feats.append(feats)
            te_labels.append(float(rul_np[b]))

    te_X = np.array(te_feats, dtype=np.float32)
    te_y = np.array(te_labels, dtype=np.float32)

    n_features = tr_X.shape[1]
    print(f"  Train samples: {len(tr_X)}, Val samples: {len(va_X)}, Test samples: {len(te_X)}")
    print(f"  Feature dimensionality: {n_features}")

    # Train ridge probe
    print("\nTraining Ridge regression (grid search over alpha)...")
    best_val_rmse, best_alpha, (best_model, best_scaler) = train_probe_sklearn(tr_X, tr_y, va_X, va_y)

    # Test evaluation
    te_X_s = best_scaler.transform(te_X)
    te_pred_norm = best_model.predict(te_X_s)
    te_pred_cycles = te_pred_norm * RUL_CAP
    test_rmse = float(np.sqrt(np.mean((te_pred_cycles - te_y) ** 2)))

    # Mean predictor (trivial baseline)
    mean_rul = float(tr_y.mean() * RUL_CAP)
    mean_rmse = float(np.sqrt(np.mean((mean_rul - te_y) ** 2)))

    print(f"\n  Best alpha: {best_alpha}")
    print(f"  Val RMSE:  {best_val_rmse:.2f}")
    print(f"  Test RMSE: {test_rmse:.2f}")
    print()
    print("  Context:")
    print(f"    Feature regressor (57 features, ridge): val={best_val_rmse:.2f}, test={test_rmse:.2f}")
    print(f"    Mean predictor (trivial):               test={mean_rmse:.2f}")
    print(f"    V2 frozen probe baseline:               val~=17.81 (from paper)")
    print(f"    V16b seed42 frozen probe (best):        val=9.86")
    print(f"    Supervised SOTA (STAR 2024):            test=10.61")
    print()

    if test_rmse < 9.86:
        print("  WARNING: Feature regressor BEATS V16b best probe! Encoder may not contribute.")
    elif test_rmse < 12.0:
        print(f"  CAUTION: Feature regressor ({test_rmse:.2f}) within 2 RMSE of V16b best (9.86).")
        print("           Encoder contribution is marginal — probe may be exploiting trivial features.")
    else:
        print(f"  OK: V16b frozen probe (9.86) beats feature regressor ({test_rmse:.2f}) by {test_rmse-9.86:.2f}.")
        print("      Encoder contributes signal beyond hand-crafted features.")

    results = {
        'description': 'Ridge regression on 57 hand-designed features (Rule 3 lower bound)',
        'n_features': int(n_features),
        'best_alpha': float(best_alpha),
        'val_rmse': float(best_val_rmse),
        'test_rmse': float(test_rmse),
        'mean_predictor_test_rmse': float(mean_rmse),
        'comparison': {
            'feature_regressor_test_rmse': float(test_rmse),
            'v16b_seed42_best_val_probe': 9.86,
            'v2_frozen_probe_val': 17.81,
            'supervised_sota_star_test': 10.61,
        },
        'interpretation': (
            'If V16b probe val <= feature_regressor val, encoder contributes nothing the protocol can see. '
            'V16b val=9.86 vs feature_regressor val noted above.'
        )
    }

    out_path = V16_DIR / 'phase6_feature_regressor_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
