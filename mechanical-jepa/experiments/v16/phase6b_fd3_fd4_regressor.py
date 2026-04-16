"""
Phase 6b: Feature Regressor Baseline for FD003 and FD004 (Rule 3 verification).

Verifies the 'vs. Regressor' margins in the paper appendix table.
Currently showing \todo{verify} for FD003 (+2.9) and FD004 (+5.9).

These were computed in V12 using engine-summary VAL regressor vs E2E TEST RMSE (mixed split).
This script computes correctly: TRAIN-set regressor, TEST-set evaluation (same split as E2E).

Fix history:
- v1: Used CMAPSSFinetuneDataset(n_cuts_per_engine=10) for training -> 32-cycle windows
       -> sequence_length feature = 32/526=0.061 for ALL train, but variable for test -> catastrophic shift
- v2: Used CMAPSSFinetuneDataset(use_last_only=True) for training -> still broken
       -> CMAPSSFinetuneDataset labels RUL=1.0 for all training engines' last cycle (target RUL is 0)
- v3 (this): Extract features DIRECTLY from engine arrays to control exactly what we compute.
       - For training: extract features + RUL from each multiple windows per engine
         using raw_train_df (has cycle numbers, so we can compute RUL = min(max_cycle-cycle, RUL_CAP))
       - For test: use CMAPSSTestDataset (correct protocol)
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V16_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v16')
sys.path.insert(0, str(V11_DIR))

warnings.filterwarnings('ignore')

from data_utils import (
    load_cmapss_subset, N_SENSORS, RUL_CAP,
    CMAPSSTestDataset,
    collate_test,
)

SENSOR_COLS = [f's{i}' for i in range(1, 22)]

def extract_window_features(cycles_normalized, seq_len_normalized):
    """
    Extract hand features from a (T, N_SENSORS) normalized cycle array.

    Features (57 total):
    - 14 last-cycle sensor values
    - 14 per-sensor slope over last 30 cycles
    - 14 per-sensor mean
    - 14 per-sensor std
    - 1 sequence length (normalized)
    """
    T, S = cycles_normalized.shape
    feats = []
    # last cycle (14 features)
    feats.extend(cycles_normalized[-1].tolist())
    # slope over last 30 cycles (14 features)
    n_last = min(30, T)
    last_window = cycles_normalized[-n_last:]
    if n_last >= 2:
        t = np.linspace(0, 1, n_last)
        for s in range(S):
            feats.append(float(np.polyfit(t, last_window[:, s], 1)[0]))
    else:
        feats.extend([0.0] * S)
    # mean (14)
    feats.extend(cycles_normalized.mean(axis=0).tolist())
    # std (14)
    feats.extend(cycles_normalized.std(axis=0).tolist())
    # sequence length (1)
    feats.append(seq_len_normalized)
    return feats


def extract_training_features(data, max_seq_len, n_cuts=10, seed=42):
    """
    Extract features from training engines using multiple windows per engine.

    For each engine, samples n_cuts random cutpoints and extracts:
    - Features from the window up to that cutpoint
    - RUL = min(max_cycle - cutpoint_cycle, RUL_CAP) in normalized form [0, 1]

    Uses raw_train_df to get actual cycle numbers for RUL computation.
    """
    rng = np.random.RandomState(seed)
    raw_df = data['raw_train_df']
    train_engines = data['train_engines']  # dict {engine_id: (T, 14) normalized sensors}
    train_ids = data['train_ids']

    all_feats, all_labels = [], []

    for eng_id in train_ids:
        cycles_norm = train_engines[eng_id]  # (T, 14) normalized
        T = len(cycles_norm)

        # Get cycle numbers for this engine from raw_df
        eng_df = raw_df[raw_df['engine_id'] == eng_id]
        max_cycle = eng_df['cycle'].max()

        # Sample n_cuts cutpoints (each at least 2 cycles)
        if T <= 2:
            cutpoints = [T]
        else:
            cutpoints = sorted(rng.choice(range(2, T + 1), size=min(n_cuts, T - 1), replace=False))
            if T not in cutpoints:
                cutpoints[-1] = T  # always include the last point

        for cut in cutpoints:
            window = cycles_norm[:cut]  # (cut, 14)
            seq_len_norm = cut / max_seq_len

            # RUL: max_cycle corresponds to T, so cut -> max_cycle - (T - cut)
            cycle_at_cut = max_cycle - (T - cut)
            rul_raw = max_cycle - cycle_at_cut  # = T - cut (remaining cycles until end of training obs)
            rul_capped = min(rul_raw, RUL_CAP)
            rul_normalized = rul_capped / RUL_CAP

            feats = extract_window_features(window, seq_len_norm)
            all_feats.append(feats)
            all_labels.append(rul_normalized)

    return np.array(all_feats, dtype=np.float32), np.array(all_labels, dtype=np.float32)


def extract_test_features(loader, max_seq_len):
    """Extract features from CMAPSSTestDataset loader (correct test protocol)."""
    all_feats, all_labels = [], []
    for past, mask, rul in loader:
        B, T, S = past.shape
        past_np = past.numpy()
        mask_np = mask.numpy()
        rul_np = rul.numpy()  # raw cycles (ground truth)
        for b in range(B):
            valid_len = int((~mask_np[b]).sum())
            sensors = past_np[b, :valid_len, :]
            seq_len_norm = valid_len / max_seq_len
            feats = extract_window_features(sensors, seq_len_norm)
            all_feats.append(feats)
            all_labels.append(float(rul_np[b]))
    return np.array(all_feats, dtype=np.float32), np.array(all_labels, dtype=np.float32)


def run_feature_regressor(subset):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    print(f"\n{'='*50}")
    print(f"Running feature regressor on {subset}")
    print(f"{'='*50}")

    max_seq_len = {'FD001': 362.0, 'FD002': 378.0, 'FD003': 526.0, 'FD004': 543.0}[subset]
    data = load_cmapss_subset(subset)

    te_ds = CMAPSSTestDataset(data['test_engines'], data['test_rul'])
    te_loader = DataLoader(te_ds, batch_size=256, shuffle=False, collate_fn=collate_test)

    print(f"  Extracting training features (direct from engine arrays)...")
    tr_X, tr_y = extract_training_features(data, max_seq_len, n_cuts=10, seed=42)
    te_X, te_y = extract_test_features(te_loader, max_seq_len)

    print(f"  Features: {tr_X.shape[1]}")
    print(f"  Train samples: {len(tr_X)} (from {len(data['train_ids'])} engines x ~10 cuts)")
    print(f"  Test samples: {len(te_X)}")
    print(f"  Train y range (normalized): {tr_y.min():.4f} to {tr_y.max():.4f}")
    print(f"  Test y range (raw cycles): {te_y.min():.1f} to {te_y.max():.1f}")

    scaler = StandardScaler()
    tr_X_s = scaler.fit_transform(tr_X)
    te_X_s = scaler.transform(te_X)

    # RidgeCV with 5-fold CV on training data
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    ridge = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_squared_error')
    ridge.fit(tr_X_s, tr_y)
    best_alpha = ridge.alpha_
    print(f"  Best alpha (5-fold CV on train): {best_alpha}")

    te_pred = ridge.predict(te_X_s)
    te_pred_cycles = te_pred * RUL_CAP  # convert normalized -> cycles
    test_rmse = float(np.sqrt(np.mean((te_pred_cycles - te_y) ** 2)))

    # Mean predictor baseline
    mean_rul_cycles = tr_y.mean() * RUL_CAP
    mean_test_rmse = float(np.sqrt(np.mean((mean_rul_cycles - te_y) ** 2)))

    print(f"  Test RMSE (feature regressor): {test_rmse:.2f}")
    print(f"  Test RMSE (mean predictor): {mean_test_rmse:.2f}")
    print(f"  Pred range (cycles): {te_pred_cycles.min():.1f} to {te_pred_cycles.max():.1f}")

    return {
        'subset': subset,
        'n_features': tr_X.shape[1],
        'n_train_samples': len(tr_X),
        'n_test_samples': len(te_X),
        'best_alpha': best_alpha,
        'test_rmse': test_rmse,
        'mean_predictor_test_rmse': mean_test_rmse,
    }


if __name__ == '__main__':
    print("Phase 6b: Feature Regressor for FD001, FD003, FD004")
    print("Purpose: Verify paper appendix 'vs. Regressor' margins")
    print("Method: Direct extraction from engine arrays (no DataLoader windowing artifacts)")
    print()

    results = {}
    for subset in ['FD001', 'FD003', 'FD004']:
        results[subset] = run_feature_regressor(subset)

    # Known JEPA E2E results (5-seed mean from V12)
    JEPA_E2E = {'FD001': 14.23, 'FD003': 15.37, 'FD004': 25.62}
    # V12 mixed-split values (from engine-summary VAL regressor, incorrect comparison)
    PAPER_MARGINS = {'FD001': 5.0, 'FD003': 2.9, 'FD004': 5.9}
    # FD001 was corrected to +3.5 (17.72 regressor vs 14.23 JEPA)
    PAPER_CORRECTED = {'FD001': 3.5}

    print("\n" + "=" * 60)
    print("SUMMARY: Regressor vs JEPA E2E (correct test-to-test comparison)")
    print("=" * 60)
    for subset in ['FD001', 'FD003', 'FD004']:
        r = results[subset]
        jepa = JEPA_E2E[subset]
        margin = r['test_rmse'] - jepa
        paper_val = PAPER_MARGINS[subset]
        print(f"\n{subset}:")
        print(f"  Feature regressor test RMSE: {r['test_rmse']:.2f}")
        print(f"  Mean predictor test RMSE: {r['mean_predictor_test_rmse']:.2f}")
        print(f"  JEPA E2E test RMSE: {jepa:.2f}")
        print(f"  JEPA beats regressor by: {margin:.2f} cycles")
        print(f"  Paper (V12 mixed-split): +{paper_val}")
        if abs(margin - paper_val) < 1.5:
            print(f"  VERDICT: CONSISTENT (within 1.5 cycles of V12 value)")
        else:
            print(f"  VERDICT: DIFFERENT ({margin:.2f} vs V12 {paper_val:.1f}) - paper needs update")

    out_path = V16_DIR / 'phase6b_fd3_fd4_regressor_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
