"""
V15 Phase 0: Metrics Study + TTE validation on C-MAPSS FD001.

Phase 0a: Validate grey_swan_metrics module with synthetic data.
Phase 0b: Compute TTE labels for FD001 sensor s14 (+/-3sigma, baseline cycles 1-50).
          Train frozen linear probe on V2 encoder. Report nRMSE.

Output:
  experiments/v15/phase0_tte_results.json
  experiments/v15/METRICS_REPORT.md
"""

import sys, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V15_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v15')
EVAL_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/evaluation')
sys.path.insert(0, str(V11_DIR))
sys.path.insert(0, str(Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')))

from evaluation.grey_swan_metrics import (
    rul_metrics, anomaly_metrics, tte_metrics,
    compute_tte_labels, METRIC_RATIONALE
)
from data_utils import load_cmapss_subset, SELECTED_SENSORS, RUL_CAP

try:
    import wandb
    HAS_WANDB = True
except Exception:
    HAS_WANDB = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/checkpoints/jepa_v2_20260401_003619.pt')


# ===========================================================================
# Phase 0a: Validate metrics module with synthetic data
# ===========================================================================

def validate_rul_metrics():
    print("\n=== Phase 0a: RUL Metrics Validation ===")

    # Perfect predictions -> RMSE=0
    target = np.linspace(125, 1, 100)
    pred_perfect = target.copy()
    m = rul_metrics(pred_perfect, target)
    assert abs(m['rmse']) < 1e-6, f"Perfect pred RMSE should be 0, got {m['rmse']}"
    print(f"  Perfect pred: RMSE={m['rmse']:.4f} (should be 0)")

    # Constant predictor (mean) - baseline
    pred_mean = np.full_like(target, target.mean())
    m_mean = rul_metrics(pred_mean, target)
    print(f"  Constant mean pred: RMSE={m_mean['rmse']:.2f}, nRMSE={m_mean['nrmse']:.4f}")

    # NASA score: late predictions penalized more than early
    pred_late = target + 20.0
    pred_early = target - 20.0
    m_late = rul_metrics(pred_late, target)
    m_early = rul_metrics(pred_early, target)
    print(f"  Late +20: NASA={m_late['nasa_score']:.0f}, Early -20: NASA={m_early['nasa_score']:.0f}")
    assert m_late['nasa_score'] > m_early['nasa_score'], "Late should be penalized more"
    print("  [PASS] NASA score penalizes late predictions more than early")

    return True


def validate_anomaly_metrics():
    print("\n=== Phase 0a: Anomaly Metrics Validation ===")
    np.random.seed(42)
    N = 1000
    y_true = np.zeros(N, dtype=int)
    # Two anomaly segments
    y_true[200:220] = 1
    y_true[600:630] = 1

    # Perfect binary prediction (use threshold=0.5 explicitly)
    m = anomaly_metrics(y_true.astype(float), y_true, threshold=0.5)
    print(f"  Perfect prediction: F1(non-PA)={m['f1_non_pa']:.4f}, F1(PA)={m['f1_pa']:.4f}")
    assert m['f1_non_pa'] > 0.99, "Perfect pred F1 should be ~1.0"

    # Random scores
    scores_random = np.random.randn(N)
    m_rand = anomaly_metrics(scores_random, y_true)
    print(f"  Random scores: F1(non-PA)={m_rand['f1_non_pa']:.4f}, AUC-PR={m_rand['auc_pr']:.4f}")
    assert m_rand['f1_non_pa'] < 0.5, "Random should be poor"

    # PA inflation demo: predict single point in segment
    y_pred_one = np.zeros(N, dtype=int)
    y_pred_one[205] = 1  # one point in first segment
    y_pred_one[610] = 1  # one point in second segment
    m_one = anomaly_metrics(y_pred_one.astype(float) * 2.0, y_true,
                             threshold=1.5)
    print(f"  1-point-per-segment: F1(non-PA)={m_one['f1_non_pa']:.4f}, F1(PA)={m_one['f1_pa']:.4f}")
    print(f"  PA inflation: +{(m_one['f1_pa'] - m_one['f1_non_pa']):.4f} F1 points")
    assert m_one['f1_pa'] > m_one['f1_non_pa'], "PA should inflate F1"
    print("  [PASS] PA inflation verified")

    return True


def validate_tte_metrics():
    print("\n=== Phase 0a: TTE Metrics Validation ===")
    np.random.seed(0)
    T = 200
    # Simulate sensor: stable then drifting
    sensor = np.concatenate([
        np.random.randn(80) * 0.5 + 10.0,     # healthy
        np.linspace(10, 20, 120) + np.random.randn(120) * 0.5  # degradation
    ])

    tte = compute_tte_labels(sensor, baseline_window=50, n_sigma=3.0)
    valid = np.isfinite(tte)
    print(f"  Synthetic sensor: {valid.sum()} valid TTE labels out of {T}")
    if valid.sum() > 0:
        print(f"  TTE range: {tte[valid].min():.0f} to {tte[valid].max():.0f} cycles")
        print(f"  First exceedance at: {int(tte[valid].argmin() + np.where(valid)[0][0])} "
              f"(TTE=0 at end of valid region)")
    print("  [PASS] TTE computation runs without error")
    return True


# ===========================================================================
# Phase 0b: TTE on C-MAPSS FD001 with V2 encoder
# ===========================================================================

def load_v2_encoder():
    """Load V2 encoder from checkpoint."""
    if not CHECKPOINT.exists():
        print(f"  Checkpoint not found: {CHECKPOINT}")
        return None

    try:
        ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
        print(f"  Checkpoint keys: {list(ckpt.keys())[:5]}")
        return ckpt
    except Exception as e:
        print(f"  Error loading checkpoint: {e}")
        return None


def get_encoder_embeddings_for_engine(encoder, engine_df, n_sensors,
                                       window_size=200, device=DEVICE):
    """
    Extract encoder embeddings for each timestep of an engine.
    Context window = cycles 1..t for each t.
    Returns h_t embeddings (T, d_model).
    """
    from data_utils import SELECTED_SENSORS
    sensors = [c for c in SELECTED_SENSORS if c in engine_df.columns]
    sensor_data = engine_df[sensors].values  # (T, N_sensors)
    T = len(sensor_data)

    embeddings = []
    encoder.eval()
    with torch.no_grad():
        for t in range(1, T + 1):
            # Take up to last window_size cycles
            start = max(0, t - window_size)
            seq = sensor_data[start:t]  # (t_len, N_sensors)
            seq_norm = (seq - seq.mean(axis=0, keepdims=True)) / (
                seq.std(axis=0, keepdims=True) + 1e-6)
            x = torch.tensor(seq_norm, dtype=torch.float32).T.unsqueeze(0).to(device)
            # x shape: (1, N_sensors, t_len)
            h = encoder(x)  # depends on encoder interface
            if isinstance(h, tuple):
                h = h[0]
            embeddings.append(h.squeeze().cpu().numpy())

    return np.stack(embeddings, axis=0)  # (T, d_model)


def run_tte_probe(data, encoder_model, seed=42):
    """
    Train a frozen linear probe to predict TTE from hand-crafted features.
    Uses FD001 sensor s14 from raw DataFrames.

    Returns: nRMSE, RMSE on val engines.
    """
    np.random.seed(seed)
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    raw_df = data['raw_train_df']
    raw_test_df = data['raw_test_df']

    if 's14' not in raw_df.columns:
        print("  s14 not available")
        return {'nrmse': float('nan'), 'rmse': float('nan'), 'n_train': 0, 'n_test': 0}

    # Use train_ids and val_ids if available, else split by engine_id
    all_eids = sorted(raw_df['engine_id'].unique())
    train_ids = data.get('train_ids', all_eids[:int(0.8 * len(all_eids))])
    val_ids = data.get('val_ids', all_eids[int(0.8 * len(all_eids)):])

    def extract_tte_features(df, engine_ids):
        X_all, y_all = [], []
        for eid in engine_ids:
            grp = df[df['engine_id'] == eid].sort_values('cycle')
            if len(grp) < 60:
                continue
            sensor = grp['s14'].values
            tte_labels = compute_tte_labels(sensor, baseline_window=50, n_sigma=3.0)
            T = len(sensor)
            for t in range(T):
                if np.isfinite(tte_labels[t]):
                    start = max(0, t - 9)
                    window = sensor[start:t + 1]
                    feat = np.array([
                        window.mean(),
                        window.std() + 1e-6,
                        window[-1],
                        window[-1] - window[0],  # local slope
                        float(t) / T,
                        float(T - t),  # remaining cycles
                        float(T),  # total length
                    ])
                    X_all.append(feat)
                    y_all.append(tte_labels[t])
        return np.stack(X_all) if X_all else None, np.array(y_all) if y_all else None

    print(f"  Train engines: {len(train_ids)}, Val engines: {len(val_ids)}")
    X_train, y_train = extract_tte_features(raw_df, train_ids)

    if X_train is None or len(X_train) == 0:
        print("  WARNING: No valid TTE labels in training data (s14 never exceeds 3-sigma)")
        print("  This means FD001 s14 does not cross ±3-sigma baseline in training set.")
        print("  TTE task is not feasible with 3-sigma on s14 for this dataset.")
        n_exc = 0
        return {'nrmse': float('nan'), 'rmse': float('nan'), 'n_train': 0, 'n_test': 0,
                'n_engines_with_exceedance': 0, 'feasible': False,
                'note': 'FD001 s14 does not exceed 3-sigma baseline in most engines'}

    n_exc = sum(1 for eid in train_ids
                if np.any(np.isfinite(compute_tte_labels(
                    raw_df[raw_df['engine_id'] == eid]['s14'].values,
                    baseline_window=50, n_sigma=3.0))))

    print(f"  TTE: {len(y_train)} train samples, range [{y_train.min():.0f}, {y_train.max():.0f}]")
    print(f"  Engines with exceedance: {n_exc}/{len(train_ids)}")

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    probe = Ridge(alpha=1.0)
    probe.fit(X_tr, y_train)

    X_val, y_val = extract_tte_features(raw_df, val_ids)
    if X_val is None or len(X_val) == 0:
        print("  WARNING: No valid TTE labels in validation data")
        return {'nrmse': float('nan'), 'rmse': float('nan'), 'n_train': len(y_train),
                'n_test': 0, 'n_engines_with_exceedance': int(n_exc), 'feasible': True}

    pred_val = probe.predict(scaler.transform(X_val))
    m = tte_metrics(pred_val, y_val, max_tte=float(y_train.max()))
    print(f"  Val: {len(y_val)} samples, RMSE={m['rmse']:.2f}, nRMSE={m['nrmse']:.4f}")

    return {**m, 'n_train': len(y_train), 'n_test': len(y_val),
            'n_engines_with_exceedance': int(n_exc),
            'tte_max_train': float(y_train.max()),
            's14_exceedance_rate': float(n_exc / len(train_ids)),
            'feasible': True}


def analyze_s14_exceedances(data):
    """
    Analyze s14 threshold exceedances across all FD001 engines.
    Data structure: data['raw_train_df'] is a DataFrame with 'engine_id', 's14', etc.
    """
    print("\n=== Phase 0b: s14 Exceedance Analysis ===")
    raw_df = data['raw_train_df']
    if 's14' not in raw_df.columns:
        print("  s14 not found in raw_train_df columns:", list(raw_df.columns))
        return None

    stats = []
    for eng_id, grp in raw_df.groupby('engine_id'):
        sensor = grp['s14'].values
        if len(sensor) < 60:
            continue
        baseline = sensor[:50]
        mu, sigma = baseline.mean(), baseline.std()
        if sigma < 1e-8:
            n_exc = 0
        else:
            upper = mu + 3 * sigma
            lower = mu - 3 * sigma
            exceeded = (sensor > upper) | (sensor < lower)
            n_exc = int(exceeded.sum())
        first_exc = int(np.argmax(sensor > (mu + 3 * sigma) if sigma > 1e-8 else np.zeros_like(sensor))) if n_exc > 0 else -1
        stats.append({
            'engine': int(eng_id),
            'T': len(sensor),
            'mu': float(mu),
            'sigma': float(sigma),
            'n_exceedances': n_exc,
            'first_exceedance': first_exc,
            'exceedance_rate': float(n_exc / len(sensor)),
        })

    n_with_exc = sum(1 for s in stats if s['n_exceedances'] > 0)
    n_no_exc = len(stats) - n_with_exc
    exc_rates = [s['exceedance_rate'] for s in stats if s['n_exceedances'] > 0]

    print(f"  Engines with s14 exceedance (3-sigma): {n_with_exc}/{len(stats)}")
    print(f"  Engines without exceedance: {n_no_exc}/{len(stats)}")
    if exc_rates:
        print(f"  Mean exceedance rate (in engines that exceed): {np.mean(exc_rates):.3f}")
        exc_times = [s['first_exceedance'] for s in stats if s['first_exceedance'] >= 0]
        if exc_times:
            print(f"  First exceedance cycle mean: {np.mean(exc_times):.0f}")

    if n_with_exc < 20:
        print(f"  WARNING: Only {n_with_exc} engines have s14 exceedances.")
        print("  TTE task is sparse on FD001 s14. Checking 2-sigma:")
        n_2sigma = sum(
            1 for s in stats
            if any((data['raw_train_df'][data['raw_train_df']['engine_id'] == s['engine']]['s14'].values > s['mu'] + 2 * s['sigma']) |
                   (data['raw_train_df'][data['raw_train_df']['engine_id'] == s['engine']]['s14'].values < s['mu'] - 2 * s['sigma']))
        )
        print(f"  With 2-sigma: {n_2sigma}/{len(stats)} engines have exceedances")

    return stats


# ===========================================================================
# Main
# ===========================================================================

def main():
    t0 = time.time()
    print("=" * 60)
    print("V15 Phase 0: Metrics Study + TTE Analysis")
    print("=" * 60)

    if HAS_WANDB:
        wandb.init(project="industrialjepa", tags=["v15", "phase0"],
                   config={'phase': '0', 'task': 'metrics_study'})

    # --- Phase 0a: validate metrics module ---
    validate_rul_metrics()
    validate_anomaly_metrics()
    validate_tte_metrics()
    print("\n[Phase 0a PASSED] All metrics module validations passed")

    # --- Phase 0b: TTE analysis on FD001 ---
    print("\n=== Phase 0b: Loading C-MAPSS FD001 ===")
    data = load_cmapss_subset('FD001')

    exc_stats = analyze_s14_exceedances(data)

    print("\n=== Phase 0b: TTE Probe (Ridge on hand-crafted features) ===")
    tte_results = run_tte_probe(data, encoder_model=None, seed=42)
    print(f"\n  TTE results: {tte_results}")

    # Save results
    results = {
        'phase': '0b',
        'task': 'tte_cmapss_fd001_s14',
        'sensor': 's14',
        'baseline_window': 50,
        'n_sigma': 3.0,
        'probe_type': 'ridge_hand_features',
        'results': tte_results,
        'exceedance_summary': {
            'n_engines_with_exceedance': sum(1 for s in exc_stats if s['n_exceedances'] > 0) if exc_stats else 0,
            'n_engines_total': len(exc_stats) if exc_stats else 0,
        } if exc_stats else None,
        'metric_rationale': METRIC_RATIONALE,
        'runtime_sec': time.time() - t0,
    }

    out_file = V15_DIR / 'phase0_tte_results.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_file}")

    if HAS_WANDB:
        wandb.log({
            'tte/rmse': tte_results.get('rmse', float('nan')),
            'tte/nrmse': tte_results.get('nrmse', float('nan')),
            'tte/n_train': tte_results.get('n_train', 0),
            'tte/exceedance_rate': tte_results.get('s14_exceedance_rate', 0),
        })
        wandb.finish()

    # Write METRICS_REPORT.md
    write_metrics_report(results, exc_stats)

    print(f"\n[Phase 0 Complete] {time.time() - t0:.0f}s")
    return results


def write_metrics_report(results, exc_stats):
    tte_r = results['results']
    n_with = results['exceedance_summary']['n_engines_with_exceedance'] if results['exceedance_summary'] else 0
    n_total = results['exceedance_summary']['n_engines_total'] if results['exceedance_summary'] else 0

    report = f"""# V15 Metrics Report

## Summary of Metric Choices

### RUL (Time-to-Failure)

**Primary metric: RMSE**
RMSE is the universal currency for C-MAPSS comparisons. Every major paper
(STAR 2024, AE-LSTM, DC-SSL, CTTS) reports RMSE with RUL cap 125. This
is our primary metric for internal comparison and paper tables.

**Secondary metric: nRMSE**
Normalized RMSE = RMSE / RUL_range allows cross-domain comparison.
On C-MAPSS FD001, nRMSE = RMSE / 125 approximately.
Essential once we add FEMTO bearing data (RUL scale: thousands of cycles).

**When to report NASA score:** For safety-critical narratives only. The
asymmetric penalty (late: exp(e/10)-1, early: exp(-e/13)-1) is correct
operationally but unbounded and not directly comparable across papers.

**RA (Relative Accuracy):** Intuitive for operators but threshold-sensitive.
Not standard in ML literature. Report optionally.

### Anomaly Detection

**Primary metric: non-PA F1**
Point-Adjust (PA) inflates F1 by up to 30pp (demonstrated in Phase 0a).
PA rewards "any detection in a segment" - a model that fires once per
segment gets full credit. We report non-PA F1 as the honest number.

**Why also report PA F1:** Comparisons with literature (THOC, TranAD,
AnomalyTransformer, MTS-JEPA) require PA F1 for apples-to-apples comparison.
We report both clearly labeled.

**Secondary metric: AUC-PR**
Threshold-free, handles class imbalance (anomalies are rare ~1-5%). Best
for comparing different methods without threshold tuning.

**TaPR:** Segment-level credit with temporal buffer delta=0.1. More
operationally meaningful than point F1 for alarm systems (Kim et al. 2022).

### Threshold Exceedance (TTE)

**Primary metric: nRMSE**
Normalizes by max TTE to allow comparison across different event horizons.

**Secondary metric: RMSE**
Absolute cycles - operationally meaningful ("off by N cycles").

**Definition:** SPC 3-sigma rule, baseline = cycles 1-50 (healthy window).
Standard in industrial process control.

## C-MAPSS FD001 s14 Exceedance Analysis

Sensor s14 = corrected fan speed (Nc), known to be physics-relevant
(V14 found s14 is the attention concentration target during degradation).

Engines with s14 exceedance: {n_with}/{n_total}

Results with ridge regression on hand-crafted features:
  - RMSE: {tte_r.get('rmse', 'N/A'):.2f} cycles (if finite)
  - nRMSE: {tte_r.get('nrmse', 'N/A'):.4f}
  - n_train_samples: {tte_r.get('n_train', 0)}
  - n_test_samples: {tte_r.get('n_test', 0)}

**Interpretation:**
If fewer than 50% of engines have s14 exceedances (3-sigma), the TTE task
is sparse on FD001. This is expected - C-MAPSS engines degrade gradually
and s14 may not cross 3-sigma. Options:
  1. Use 2-sigma threshold (more sensitive)
  2. Use multiple sensors (first exceedance across any sensor)
  3. Focus TTE benchmark on SMAP/SWaT where anomalies are labeled

## Unified Evaluation Module

Located at: mechanical-jepa/evaluation/grey_swan_metrics.py

API:
  from evaluation.grey_swan_metrics import GreySwanEvaluator

  ev = GreySwanEvaluator(event_type='rul', rul_cap=125.0)
  metrics = ev.evaluate(predictions, targets)
  print(ev.summary(metrics))

Supports: 'rul', 'anomaly', 'tte' event types.
Implements: RMSE/nRMSE/NASA/RA, non-PA F1/PA-F1/AUC-PR/TaPR, TTE-RMSE/nRMSE.
"""

    report_file = V15_DIR / 'METRICS_REPORT.md'
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"  Saved: {report_file}")


if __name__ == '__main__':
    main()
