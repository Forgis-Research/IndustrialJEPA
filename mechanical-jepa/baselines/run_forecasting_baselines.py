"""
Phase 4: Health Indicator Forecasting and RUL Estimation Baselines

Task 3: Given HI trajectory [h(t-k)..h(t)], predict h(t+1)..h(t+H)
Task 4: Given current features, predict rul_percent

HI types: RMS, kurtosis, band energy
Horizons: 1, 5, 10 steps ahead
Context: 20 past measurements

Only FEMTO (run-to-failure, ~3500 samples) is used for HI forecasting.
FEMTO + XJTU-SY for RUL.

Outputs:
- results/forecasting_baselines.json
"""

import numpy as np
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from data_utils import load_rul_data, TARGET_SR, proc_native, get_sr, get_ch0
from features import (extract_features, extract_features_batch, compute_rms,
                       compute_kurtosis, compute_crest_factor, N_FEATURES)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
import xgboost as xgb
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

SEEDS = [42, 123, 456]
TOKEN = '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'
HF_BASE = 'hf://datasets/Forgis/Mechanical-Components'
STORAGE_OPTS = {'token': TOKEN}
CACHE_DIR = '/tmp/hf_cache/bearings'


def load_parquet(filename: str) -> pd.DataFrame:
    """Load from local cache or HF."""
    import os
    basename = os.path.basename(filename)
    local = os.path.join(CACHE_DIR, basename)
    if os.path.exists(local):
        return pd.read_parquet(local)
    return pd.read_parquet(f'{HF_BASE}/{filename}', storage_options=STORAGE_OPTS)
RESULTS_PATH = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/baselines/results/forecasting_baselines.json'
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

CONTEXT_LEN = 20    # past windows for forecasting
HORIZONS = [1, 5, 10]  # predict this many steps ahead


# ============================================================
# HI COMPUTATION FROM FEMTO SIGNALS
# ============================================================

def compute_femto_hi_series(hi_type: str = 'rms', verbose: bool = True) -> dict:
    """
    Load all FEMTO shards and compute HI time series per episode.
    Returns: {episode_id: {'hi': array, 'rul': array}}
    """
    episodes = {}

    for shard_idx in range(4):
        try:
            df = load_parquet(f'bearings/train-{shard_idx:05d}-of-00005.parquet')
        except Exception as e:
            print(f"Could not load shard {shard_idx}: {e}")
            continue

        df = df[df['source_id'] == 'femto'].sort_values(['episode_id', 'episode_position'])

        for ep_id, ep_df in df.groupby('episode_id'):
            hi_vals, rul_vals, pos_vals = [], [], []
            for _, row in ep_df.iterrows():
                try:
                    sig = np.array(row['signal'])[0]  # channel 0 (horizontal)
                    if hi_type == 'rms':
                        hi = compute_rms(sig)
                    elif hi_type == 'kurtosis':
                        hi = compute_kurtosis(sig)
                    elif hi_type == 'crest_factor':
                        hi = compute_crest_factor(sig)
                    else:
                        raise ValueError(f"Unknown hi_type: {hi_type}")

                    hi_vals.append(hi)
                    rul_vals.append(float(row['rul_percent']) if row['rul_percent'] is not None else np.nan)
                    pos_vals.append(float(row['episode_position']))
                except Exception:
                    pass

            if hi_vals:
                if ep_id not in episodes:
                    episodes[ep_id] = {'hi': [], 'rul': [], 'pos': []}
                episodes[ep_id]['hi'].extend(hi_vals)
                episodes[ep_id]['rul'].extend(rul_vals)
                episodes[ep_id]['pos'].extend(pos_vals)

    # Convert to numpy
    for ep_id in episodes:
        episodes[ep_id]['hi'] = np.array(episodes[ep_id]['hi'])
        episodes[ep_id]['rul'] = np.array(episodes[ep_id]['rul'])
        episodes[ep_id]['pos'] = np.array(episodes[ep_id]['pos'])

    if verbose:
        print(f"FEMTO episodes loaded: {list(episodes.keys())}")
        for k, v in episodes.items():
            print(f"  {k}: {len(v['hi'])} steps, HI range [{v['hi'].min():.4f}, {v['hi'].max():.4f}]")

    return episodes


def build_forecast_dataset(episodes: dict, context_len: int = CONTEXT_LEN,
                             horizon: int = 1, test_episodes: list = None):
    """
    Build (X, y) pairs for HI forecasting.
    X: context_len past HI values
    y: HI at horizon steps ahead

    test_episodes: list of episode_ids for test set (rest are train)
    """
    if test_episodes is None:
        all_ep = list(episodes.keys())
        n_test = max(1, len(all_ep) // 4)
        test_episodes = all_ep[-n_test:]

    X_train, y_train = [], []
    X_test, y_test = [], []

    for ep_id, ep_data in episodes.items():
        hi = ep_data['hi']
        # Normalize HI per episode (zero mean, unit std)
        mu, sigma = hi.mean(), hi.std()
        if sigma < 1e-10:
            continue
        hi_norm = (hi - mu) / sigma

        is_test = ep_id in test_episodes

        for t in range(context_len, len(hi_norm) - horizon):
            x = hi_norm[t - context_len:t]
            y = hi_norm[t + horizon - 1]  # target: value at t+horizon

            if is_test:
                X_test.append(x)
                y_test.append(y)
            else:
                X_train.append(x)
                y_train.append(y)

    return (np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32),
            np.array(X_test, dtype=np.float32), np.array(y_test, dtype=np.float32))


def eval_forecast(y_true, y_pred):
    """Compute forecasting metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
    spearman_r, spearman_p = spearmanr(y_true, y_pred)
    return {
        'rmse': rmse, 'mae': mae, 'r2': r2,
        'spearman_r': float(spearman_r), 'spearman_p': float(spearman_p),
    }


# ============================================================
# FORECASTING BASELINES
# ============================================================

def run_forecasting_baselines(seeds=SEEDS):
    results = {}
    device = 'cpu'

    for hi_type in ['rms', 'kurtosis']:
        print(f"\n{'='*60}")
        print(f"=== HI Type: {hi_type.upper()} ===")
        print('='*60)

        episodes = compute_femto_hi_series(hi_type=hi_type, verbose=True)

        if len(episodes) < 2:
            print("Not enough episodes for train/test split")
            continue

        all_ep = sorted(episodes.keys())
        # Use last 2 episodes as test (bearing life trajectories)
        test_episodes = all_ep[-2:] if len(all_ep) >= 4 else all_ep[-1:]
        print(f"Train episodes: {[e for e in all_ep if e not in test_episodes]}")
        print(f"Test episodes: {test_episodes}")

        hi_results = {}

        for horizon in HORIZONS:
            print(f"\n-- Horizon H={horizon} --")

            X_tr, y_tr, X_te, y_te = build_forecast_dataset(
                episodes, context_len=CONTEXT_LEN, horizon=horizon, test_episodes=test_episodes
            )
            print(f"  Train: {X_tr.shape[0]}, Test: {X_te.shape[0]}")

            if X_tr.shape[0] < 10 or X_te.shape[0] < 5:
                print("  Insufficient data, skipping")
                continue

            h_results = {}

            # Trivial: last value
            y_pred_lv = X_te[:, -1]
            h_results['trivial_last_value'] = eval_forecast(y_te, y_pred_lv)
            print(f"  Last value: RMSE={h_results['trivial_last_value']['rmse']:.4f}")

            # Trivial: moving average
            y_pred_ma = X_te.mean(axis=1)
            h_results['trivial_moving_avg'] = eval_forecast(y_te, y_pred_ma)
            print(f"  Moving avg: RMSE={h_results['trivial_moving_avg']['rmse']:.4f}")

            # Trivial: linear extrapolation
            def linear_extrap(X, h):
                # Fit line to last k=10 points and extrapolate h steps
                k = min(10, X.shape[1])
                preds = []
                for i in range(len(X)):
                    x = X[i, -k:]
                    t = np.arange(k)
                    A = np.column_stack([t, np.ones(k)])
                    try:
                        coeffs, _, _, _ = np.linalg.lstsq(A, x, rcond=None)
                        pred = coeffs[0] * (k - 1 + h) + coeffs[1]
                    except Exception:
                        pred = x[-1]
                    preds.append(pred)
                return np.array(preds)

            y_pred_le = linear_extrap(X_te, horizon)
            h_results['trivial_linear_extrap'] = eval_forecast(y_te, y_pred_le)
            print(f"  Linear extrap: RMSE={h_results['trivial_linear_extrap']['rmse']:.4f}")

            # Trivial: constant (training mean)
            y_pred_const = np.full(len(y_te), y_tr.mean())
            h_results['trivial_constant_mean'] = eval_forecast(y_te, y_pred_const)
            print(f"  Constant mean: RMSE={h_results['trivial_constant_mean']['rmse']:.4f}")

            # Ridge regression
            ridge_rmses = []
            for seed in seeds:
                scaler = StandardScaler()
                X_tr_sc = scaler.fit_transform(X_tr)
                X_te_sc = scaler.transform(X_te)
                ridge = Ridge(alpha=1.0)
                ridge.fit(X_tr_sc, y_tr)
                y_pred = ridge.predict(X_te_sc)
                ridge_rmses.append(eval_forecast(y_te, y_pred)['rmse'])
            h_results['ridge'] = {
                'rmse_mean': float(np.mean(ridge_rmses)), 'rmse_std': float(np.std(ridge_rmses)), 'seeds': seeds
            }
            print(f"  Ridge: RMSE={np.mean(ridge_rmses):.4f}±{np.std(ridge_rmses):.4f}")

            # Random Forest
            rf_rmses = []
            for seed in seeds:
                rf = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
                rf.fit(X_tr, y_tr)
                y_pred = rf.predict(X_te)
                rf_rmses.append(eval_forecast(y_te, y_pred)['rmse'])
            h_results['random_forest'] = {
                'rmse_mean': float(np.mean(rf_rmses)), 'rmse_std': float(np.std(rf_rmses)), 'seeds': seeds
            }
            print(f"  RandomForest: RMSE={np.mean(rf_rmses):.4f}±{np.std(rf_rmses):.4f}")

            # XGBoost
            xgb_rmses = []
            for seed in seeds:
                reg = xgb.XGBRegressor(n_estimators=200, random_state=seed, n_jobs=-1, verbosity=0)
                reg.fit(X_tr, y_tr)
                y_pred = reg.predict(X_te)
                xgb_rmses.append(eval_forecast(y_te, y_pred)['rmse'])
            h_results['xgboost'] = {
                'rmse_mean': float(np.mean(xgb_rmses)), 'rmse_std': float(np.std(xgb_rmses)), 'seeds': seeds
            }
            print(f"  XGBoost: RMSE={np.mean(xgb_rmses):.4f}±{np.std(xgb_rmses):.4f}")

            # ARIMA on test episodes (one-by-one)
            try:
                from statsmodels.tsa.arima.model import ARIMA
                arima_rmses = []
                for ep_id in test_episodes:
                    hi_ep = episodes[ep_id]['hi']
                    mu, sigma = hi_ep.mean(), hi_ep.std()
                    if sigma < 1e-10:
                        continue
                    hi_norm = (hi_ep - mu) / sigma
                    # Simple ARIMA(2,1,2) on this episode
                    if len(hi_norm) < CONTEXT_LEN + horizon + 10:
                        continue
                    # Use first 80% for fitting, last 20% for eval
                    n_fit = int(0.8 * len(hi_norm))
                    train_hi = hi_norm[:n_fit]
                    test_hi = hi_norm[n_fit:]
                    try:
                        arima = ARIMA(train_hi, order=(2, 1, 2))
                        fit = arima.fit()
                        n_pred = len(test_hi)
                        pred = fit.forecast(steps=n_pred)
                        # Evaluate just first-step predictions
                        arima_rmses.append(float(np.sqrt(mean_squared_error(test_hi, pred))))
                    except Exception as e:
                        pass
                if arima_rmses:
                    h_results['arima'] = {
                        'rmse_mean': float(np.mean(arima_rmses)),
                        'rmse_std': float(np.std(arima_rmses)),
                        'note': 'ARIMA(2,1,2) per-episode, episode-level RMSE'
                    }
                    print(f"  ARIMA: RMSE={np.mean(arima_rmses):.4f}±{np.std(arima_rmses):.4f}")
            except ImportError:
                print("  ARIMA: statsmodels not available, skipping")
            except Exception as e:
                print(f"  ARIMA failed: {e}")

            hi_results[f'horizon_{horizon}'] = h_results

        results[f'hi_forecasting_{hi_type}'] = hi_results

        # Incremental save
        with open(RESULTS_PATH, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nPartial results saved.")

    # ============================================================
    # TASK 4: RUL ESTIMATION
    # ============================================================
    print(f"\n{'='*60}")
    print("=== Task 4: RUL Estimation ===")
    print('='*60)

    try:
        rul_rows = load_rul_data(sources=['femto', 'xjtu_sy'], verbose=True)

        # Extract features from raw signals
        print("Extracting features for RUL estimation...")
        X_rul_list, y_rul_list, src_list, ep_list = [], [], [], []

        for row_dict in rul_rows:
            try:
                sig = row_dict['signal']  # already ch0, float32
                feats = extract_features(sig)
                X_rul_list.append(feats)
                y_rul_list.append(float(row_dict['rul_percent']))
                src_list.append(str(row_dict['source']))
                ep_list.append(str(row_dict['episode_id']))
            except Exception:
                pass

        X_rul = np.array(X_rul_list, dtype=np.float32)
        y_rul = np.array(y_rul_list, dtype=np.float32)
        src_arr = np.array(src_list)
        ep_arr = np.array(ep_list)

        print(f"RUL dataset: {X_rul.shape[0]} samples, {len(np.unique(ep_arr))} episodes")

        # Episode-based split
        unique_eps = np.unique(ep_arr)
        rng = np.random.default_rng(42)
        rng.shuffle(unique_eps)
        n_test = max(1, len(unique_eps) // 4)
        test_eps = set(unique_eps[-n_test:])
        train_mask = np.array([ep not in test_eps for ep in ep_arr])
        test_mask = ~train_mask

        X_rul_tr = X_rul[train_mask]
        y_rul_tr = y_rul[train_mask]
        X_rul_te = X_rul[test_mask]
        y_rul_te = y_rul[test_mask]
        print(f"  Train: {X_rul_tr.shape[0]}, Test: {X_rul_te.shape[0]}")

        rul_results = {}

        # Trivial: constant mean
        y_rul_const = np.full(len(y_rul_te), y_rul_tr.mean())
        rul_results['trivial_constant_mean'] = eval_forecast(y_rul_te, y_rul_const)
        print(f"  Constant mean: RMSE={rul_results['trivial_constant_mean']['rmse']:.4f}")

        # Trivial: linear progression (1 - episode_position is a strong oracle-like baseline)
        pos_arr = np.array([r['episode_position'] for r in rul_rows], dtype=np.float32)
        y_rul_lin = 1.0 - pos_arr[test_mask]
        rul_results['trivial_linear_position'] = eval_forecast(y_rul_te, y_rul_lin)
        print(f"  Linear position: RMSE={rul_results['trivial_linear_position']['rmse']:.4f}")

        # Ridge
        ridge_rmses = []
        for seed in seeds:
            scaler = StandardScaler()
            Xtr_sc = scaler.fit_transform(X_rul_tr)
            Xte_sc = scaler.transform(X_rul_te)
            ridge = Ridge(alpha=1.0)
            ridge.fit(Xtr_sc, y_rul_tr)
            y_pred = np.clip(ridge.predict(Xte_sc), 0, 1)
            ridge_rmses.append(eval_forecast(y_rul_te, y_pred)['rmse'])
        rul_results['ridge'] = {
            'rmse_mean': float(np.mean(ridge_rmses)), 'rmse_std': float(np.std(ridge_rmses)), 'seeds': seeds
        }
        print(f"  Ridge: RMSE={np.mean(ridge_rmses):.4f}±{np.std(ridge_rmses):.4f}")

        # Random Forest
        rf_rmses = []
        for seed in seeds:
            rf = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
            rf.fit(X_rul_tr, y_rul_tr)
            y_pred = np.clip(rf.predict(X_rul_te), 0, 1)
            rf_rmses.append(eval_forecast(y_rul_te, y_pred)['rmse'])
        rul_results['random_forest'] = {
            'rmse_mean': float(np.mean(rf_rmses)), 'rmse_std': float(np.std(rf_rmses)), 'seeds': seeds
        }
        print(f"  RandomForest: RMSE={np.mean(rf_rmses):.4f}±{np.std(rf_rmses):.4f}")

        # XGBoost
        xgb_rmses = []
        for seed in seeds:
            reg = xgb.XGBRegressor(n_estimators=200, random_state=seed, n_jobs=-1, verbosity=0)
            reg.fit(X_rul_tr, y_rul_tr)
            y_pred = np.clip(reg.predict(X_rul_te), 0, 1)
            xgb_rmses.append(eval_forecast(y_rul_te, y_pred)['rmse'])
        rul_results['xgboost'] = {
            'rmse_mean': float(np.mean(xgb_rmses)), 'rmse_std': float(np.std(xgb_rmses)), 'seeds': seeds
        }
        print(f"  XGBoost: RMSE={np.mean(xgb_rmses):.4f}±{np.std(xgb_rmses):.4f}")

        results['rul_estimation'] = rul_results

    except Exception as e:
        print(f"RUL estimation failed: {e}")
        import traceback; traceback.print_exc()
        results['rul_estimation'] = {'error': str(e)}

    # Final save
    results['_meta'] = {
        'timestamp': datetime.now().isoformat(),
        'seeds': seeds,
        'hi_types': ['rms', 'kurtosis'],
        'horizons': HORIZONS,
        'context_len': CONTEXT_LEN,
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFinal results saved to {RESULTS_PATH}")

    return results


if __name__ == '__main__':
    run_forecasting_baselines()
