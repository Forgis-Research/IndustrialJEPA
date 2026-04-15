"""
Grey Swan Unified Evaluation Module (V15).

Supports three event types:
  - 'rul'      : Time-to-failure (RUL) regression
  - 'anomaly'  : Point/sequence anomaly detection
  - 'tte'      : Threshold exceedance (SPC 3-sigma rule)

References:
  - NASA Scoring Function: Saxena et al. (2008), IJRPC
  - TaPR: Kim et al. (2022), KDD
  - TSAD-Eval: Schmidl et al. (2022), VLDB
  - nRMSE definition: Heimes (2008)
"""

import numpy as np
from typing import Optional, Tuple, Union
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 1. RUL METRICS
# ---------------------------------------------------------------------------

def rul_metrics(pred: np.ndarray, target: np.ndarray,
                rul_cap: float = 125.0) -> dict:
    """
    Compute RUL regression metrics.

    Args:
        pred:   (N,) predicted RUL values (uncapped, any scale)
        target: (N,) ground-truth RUL values (uncapped, any scale)
        rul_cap: maximum RUL cap (default 125 for C-MAPSS)

    Returns dict with:
        rmse        : Root mean squared error (primary metric - most reported)
        nrmse       : Normalized RMSE = RMSE / max(target)  (cross-domain comparison)
        nasa_score  : NASA asymmetric scoring function (penalizes late more)
        ra_percent  : Relative accuracy (percent of predictions within ±10% of target)
        mae         : Mean absolute error
        bias        : Mean error (positive = over-predicts)

    Primary metric recommendation: RMSE (most comparable across papers)
    Secondary metric recommendation: nRMSE (cross-domain)

    Note on NASA score: exponential asymmetry penalizes late predictions heavily
    (predicting engine survived when it failed). Use as secondary for safety context.
    RA is intuitive for operators ("X% of predictions within 10% tolerance") but
    depends on threshold choice and is rare in ML papers.
    """
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"

    err = pred - target

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))

    # nRMSE: normalize by range of target (not max, to handle near-zero targets)
    target_range = float(np.max(target) - np.min(target))
    nrmse = rmse / target_range if target_range > 1e-6 else float('nan')

    # NASA Scoring Function (Saxena et al. 2008)
    # s_i = exp(-err_i/13) - 1  if err_i < 0 (early prediction)
    # s_i = exp(err_i/10) - 1   if err_i >= 0 (late prediction)
    # Score = sum(s_i)  [lower is better]
    scores = np.where(err < 0,
                      np.exp(-err / 13.0) - 1.0,
                      np.exp(err / 10.0) - 1.0)
    nasa_score = float(np.sum(scores))

    # Relative accuracy: fraction within ±10% of target
    # Only meaningful when target > 0; mask zero targets
    valid = target > 1e-6
    if valid.sum() > 0:
        rel_err = np.abs(err[valid]) / target[valid]
        ra_percent = float(100.0 * np.mean(rel_err <= 0.10))
    else:
        ra_percent = float('nan')

    return {
        'rmse': rmse,
        'nrmse': nrmse,
        'nasa_score': nasa_score,
        'ra_percent': ra_percent,
        'mae': mae,
        'bias': bias,
        'n_samples': int(len(pred)),
    }


# ---------------------------------------------------------------------------
# 2. ANOMALY DETECTION METRICS
# ---------------------------------------------------------------------------

def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (TP, FP, TN, FN)."""
    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))
    return TP, FP, TN, FN


def _f1(tp, fp, fn) -> float:
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom > 0 else 0.0


def _precision_recall(tp, fp, fn) -> Tuple[float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return prec, rec


def pa_adjustment(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Point-Adjust (PA) protocol: if ANY point in an anomaly segment is predicted
    as anomaly, mark the entire segment as correctly predicted.

    This is the standard PA used by THOC, TranAD, etc. but is controversial
    (inflates F1 significantly). We compute both PA and non-PA F1.

    Reference: Xu et al. (2021), "Anomaly Transformer"
    """
    y_pred_pa = y_pred.copy()
    # Find anomaly segments in y_true
    in_segment = False
    seg_start = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_segment:
            in_segment = True
            seg_start = i
        elif y_true[i] == 0 and in_segment:
            in_segment = False
            seg = slice(seg_start, i)
            if y_pred[seg].any():
                y_pred_pa[seg] = 1
    # Handle segment at end
    if in_segment:
        seg = slice(seg_start, len(y_true))
        if y_pred[seg].any():
            y_pred_pa[seg] = 1
    return y_pred_pa


def tapr(y_true: np.ndarray, y_pred: np.ndarray,
         delta: float = 0.1) -> Tuple[float, float, float]:
    """
    Time-series Aware Precision and Recall (TaPR).

    Kim et al., "Towards a Rigorous Evaluation of Time-Series Anomaly Detection",
    AAAI 2022.

    TaPR relaxes the strict point alignment by using temporal overlap with a
    buffer delta (fraction of anomaly segment length).

    Returns: (TaPR_precision, TaPR_recall, TaPR_F1)
    """
    def find_segments(arr: np.ndarray) -> list:
        segs = []
        start = None
        for i, v in enumerate(arr):
            if v == 1 and start is None:
                start = i
            elif v == 0 and start is not None:
                segs.append((start, i))
                start = None
        if start is not None:
            segs.append((start, len(arr)))
        return segs

    pred_segs = find_segments(y_pred)
    true_segs = find_segments(y_true)

    if not pred_segs and not true_segs:
        return 1.0, 1.0, 1.0
    if not pred_segs:
        return 0.0, 0.0, 0.0
    if not true_segs:
        return 0.0, 0.0, 0.0

    # TaPR precision: for each predicted segment, is it close to a true segment?
    tp_pred = 0
    for ps, pe in pred_segs:
        for ts, te in true_segs:
            overlap = max(0, min(pe, te) - max(ps, ts))
            buf = delta * (te - ts)
            if overlap >= buf or (max(ps, ts) <= min(pe, te) and overlap > 0):
                tp_pred += 1
                break

    # TaPR recall: for each true segment, is it detected?
    tp_true = 0
    for ts, te in true_segs:
        for ps, pe in pred_segs:
            overlap = max(0, min(pe, te) - max(ps, ts))
            buf = delta * (te - ts)
            if overlap >= buf or (max(ps, ts) <= min(pe, te) and overlap > 0):
                tp_true += 1
                break

    tapr_prec = tp_pred / len(pred_segs) if pred_segs else 0.0
    tapr_rec = tp_true / len(true_segs) if true_segs else 0.0
    denom = tapr_prec + tapr_rec
    tapr_f1 = 2 * tapr_prec * tapr_rec / denom if denom > 0 else 0.0

    return tapr_prec, tapr_rec, tapr_f1


def auc_pr(scores: np.ndarray, y_true: np.ndarray,
           n_thresholds: int = 200) -> float:
    """
    Area under the precision-recall curve.
    Uses trapezoidal rule. Threshold swept from max to min score.
    """
    thresholds = np.linspace(scores.max(), scores.min(), n_thresholds)
    precisions, recalls = [1.0], [0.0]
    for thresh in thresholds:
        y_pred = (scores >= thresh).astype(int)
        tp, fp, _, fn = _confusion(y_true, y_pred)
        prec, rec = _precision_recall(tp, fp, fn)
        precisions.append(prec)
        recalls.append(rec)
    precisions.append(0.0)
    recalls.append(1.0)
    # Sort by recall for trapezoidal integration
    sort_idx = np.argsort(recalls)
    rec_sorted = np.array(recalls)[sort_idx]
    prec_sorted = np.array(precisions)[sort_idx]
    return float(np.trapz(prec_sorted, rec_sorted))


def anomaly_metrics(scores: np.ndarray, y_true: np.ndarray,
                    threshold: Optional[float] = None,
                    threshold_percentile: float = 95.0) -> dict:
    """
    Compute anomaly detection metrics.

    Args:
        scores:  (N,) anomaly scores (higher = more anomalous)
        y_true:  (N,) binary ground truth (1 = anomaly)
        threshold: if None, use threshold_percentile of scores on normal points
        threshold_percentile: percentile for auto-thresholding

    Returns dict with:
        f1_non_pa       : F1 without point-adjust (strict, recommended primary)
        f1_pa           : F1 with point-adjust (lenient, match literature)
        precision_non_pa: precision without PA
        recall_non_pa   : recall without PA
        auc_pr          : area under PR curve
        tapr_f1         : TaPR F1 with delta=0.1
        tapr_prec       : TaPR precision
        tapr_rec        : TaPR recall
        threshold_used  : threshold value applied

    Primary metric: non-PA F1 (honest)
    Secondary metric: AUC-PR (threshold-free)
    """
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    # Auto-threshold on anomaly-free training data if not given
    if threshold is None:
        normal_scores = scores[y_true == 0]
        if len(normal_scores) > 0:
            threshold = float(np.percentile(normal_scores, threshold_percentile))
        else:
            threshold = float(np.percentile(scores, threshold_percentile))

    # Use strict > to avoid flagging all normal points at threshold=0
    y_pred = (scores > threshold).astype(int)

    tp, fp, tn, fn = _confusion(y_true, y_pred)
    prec, rec = _precision_recall(tp, fp, fn)
    f1_np = _f1(tp, fp, fn)

    # PA variant
    y_pred_pa = pa_adjustment(y_true, y_pred)
    tp_pa, fp_pa, _, fn_pa = _confusion(y_true, y_pred_pa)
    f1_pa = _f1(tp_pa, fp_pa, fn_pa)

    # AUC-PR
    aucpr = auc_pr(scores, y_true)

    # TaPR
    tapr_p, tapr_r, tapr_f = tapr(y_true, y_pred)

    return {
        'f1_non_pa': f1_np,
        'f1_pa': f1_pa,
        'precision_non_pa': prec,
        'recall_non_pa': rec,
        'auc_pr': aucpr,
        'tapr_f1': tapr_f,
        'tapr_prec': tapr_p,
        'tapr_rec': tapr_r,
        'threshold_used': float(threshold),
        'n_anomaly': int(y_true.sum()),
        'n_total': int(len(y_true)),
        'prevalence': float(y_true.mean()),
    }


# ---------------------------------------------------------------------------
# 3. THRESHOLD EXCEEDANCE (TTE) METRICS
# ---------------------------------------------------------------------------

def compute_tte_labels(sensor_data: np.ndarray,
                       baseline_window: int = 50,
                       n_sigma: float = 3.0,
                       method: str = 'first') -> np.ndarray:
    """
    Compute time-to-threshold-exceedance (TTE) labels for each timestep.

    Standard SPC (Statistical Process Control) 3-sigma rule:
    An exceedance occurs when the sensor crosses mu +/- n_sigma * sigma
    where mu, sigma are estimated from the baseline_window.

    Args:
        sensor_data: (T,) 1D sensor time series (single engine, single sensor)
        baseline_window: number of initial cycles to estimate mu/sigma
        n_sigma: threshold multiplier (3.0 = standard SPC)
        method: 'first' = TTE to first exceedance; 'next' = TTE to next exceedance

    Returns:
        tte_labels: (T,) array where tte_labels[t] = cycles until first exceedance
                    after time t. NaN if no exceedance occurs after t.

    Usage: compute_tte_labels(engine_df['s14'].values)
    """
    T = len(sensor_data)
    assert baseline_window < T, "baseline_window must be < series length"

    baseline = sensor_data[:baseline_window]
    mu = float(np.mean(baseline))
    sigma = float(np.std(baseline, ddof=1))

    if sigma < 1e-10:
        # Sensor is constant - no meaningful exceedance
        return np.full(T, np.nan)

    upper = mu + n_sigma * sigma
    lower = mu - n_sigma * sigma

    # Find all exceedance timesteps
    exceeded = np.where((sensor_data > upper) | (sensor_data < lower))[0]

    tte_labels = np.full(T, np.nan)
    if len(exceeded) == 0:
        return tte_labels

    if method == 'first':
        # TTE[t] = first_exceedance_time - t (if first_exceedance_time > t)
        first_exc = exceeded[0]
        for t in range(T):
            if t <= first_exc:
                tte_labels[t] = first_exc - t
            # After exceedance: NaN (event has occurred)
    elif method == 'next':
        # TTE[t] = next exceedance after t
        for t in range(T):
            future_exc = exceeded[exceeded > t]
            if len(future_exc) > 0:
                tte_labels[t] = future_exc[0] - t

    return tte_labels


def tte_metrics(pred_tte: np.ndarray, true_tte: np.ndarray,
                max_tte: Optional[float] = None) -> dict:
    """
    Metrics for time-to-threshold-exceedance predictions.

    Mirrors rul_metrics but for threshold exceedance events.
    Only evaluates on timesteps where ground truth TTE is finite.

    Args:
        pred_tte:  (N,) predicted TTE at each timestep
        true_tte:  (N,) ground-truth TTE (NaN for timesteps past exceedance)
        max_tte:   cap for normalization (if None, use max of true_tte)
    """
    pred_tte = np.asarray(pred_tte, dtype=float)
    true_tte = np.asarray(true_tte, dtype=float)

    # Only evaluate on valid (non-NaN, non-inf) labels
    valid = np.isfinite(true_tte) & np.isfinite(pred_tte)
    if valid.sum() == 0:
        return {'rmse': float('nan'), 'nrmse': float('nan'),
                'mae': float('nan'), 'n_valid': 0}

    p, t = pred_tte[valid], true_tte[valid]
    err = p - t

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))

    cap = max_tte if max_tte is not None else float(np.max(t))
    nrmse = rmse / cap if cap > 1e-6 else float('nan')

    return {
        'rmse': rmse,
        'nrmse': nrmse,
        'mae': mae,
        'bias': bias,
        'n_valid': int(valid.sum()),
        'n_total': int(len(pred_tte)),
    }


# ---------------------------------------------------------------------------
# 4. UNIFIED EVALUATOR
# ---------------------------------------------------------------------------

@dataclass
class GreySwanEvaluator:
    """
    Unified evaluator for grey swan prediction.

    Usage:
        ev = GreySwanEvaluator(event_type='rul', rul_cap=125.0)
        metrics = ev.evaluate(predictions, ground_truth)
    """
    event_type: str  # 'rul', 'anomaly', 'tte'
    rul_cap: float = 125.0
    anomaly_threshold: Optional[float] = None
    anomaly_threshold_percentile: float = 95.0
    tte_baseline_window: int = 50
    tte_n_sigma: float = 3.0

    def evaluate(self, predictions: np.ndarray, ground_truth: np.ndarray) -> dict:
        if self.event_type == 'rul':
            return rul_metrics(predictions, ground_truth, rul_cap=self.rul_cap)
        elif self.event_type == 'anomaly':
            return anomaly_metrics(
                predictions, ground_truth,
                threshold=self.anomaly_threshold,
                threshold_percentile=self.anomaly_threshold_percentile,
            )
        elif self.event_type == 'tte':
            return tte_metrics(predictions, ground_truth)
        else:
            raise ValueError(f"Unknown event_type: {self.event_type}")

    def compute_tte_from_sensor(self, sensor_data: np.ndarray) -> np.ndarray:
        """Helper to compute TTE labels from raw sensor data."""
        return compute_tte_labels(
            sensor_data,
            baseline_window=self.tte_baseline_window,
            n_sigma=self.tte_n_sigma,
        )

    def summary(self, metrics: dict) -> str:
        """Return a one-line summary string."""
        if self.event_type == 'rul':
            return (f"RMSE={metrics['rmse']:.2f} | nRMSE={metrics['nrmse']:.4f} | "
                    f"NASA={metrics['nasa_score']:.0f} | RA={metrics['ra_percent']:.1f}%")
        elif self.event_type == 'anomaly':
            return (f"F1(non-PA)={metrics['f1_non_pa']:.4f} | "
                    f"F1(PA)={metrics['f1_pa']:.4f} | "
                    f"AUC-PR={metrics['auc_pr']:.4f} | "
                    f"TaPR-F1={metrics['tapr_f1']:.4f}")
        elif self.event_type == 'tte':
            return (f"RMSE={metrics['rmse']:.2f} | nRMSE={metrics['nrmse']:.4f} | "
                    f"n_valid={metrics['n_valid']}")
        return str(metrics)


# ---------------------------------------------------------------------------
# 5. METRIC RATIONALE (programmatic reference)
# ---------------------------------------------------------------------------

METRIC_RATIONALE = {
    'rul': {
        'primary': 'rmse',
        'secondary': 'nrmse',
        'rationale': (
            "RMSE is the universal currency for C-MAPSS comparisons (STAR, AE-LSTM, "
            "DC-SSL, CTTS, all use it). nRMSE allows cross-domain comparison where "
            "RUL scales differ (e.g., C-MAPSS ~125 cycles vs FEMTO ~thousands). "
            "NASA score is useful for safety-critical reporting (penalizes late "
            "predictions 2.5x more than early ones) but is unbounded and hard to "
            "compare across papers. RA is operationally interpretable but "
            "threshold-sensitive and rarely reported in ML papers."
        ),
    },
    'anomaly': {
        'primary': 'f1_non_pa',
        'secondary': 'auc_pr',
        'rationale': (
            "Non-PA F1 is the honest metric - no credit for detecting any point in "
            "an anomaly segment. PA-F1 (standard in THOC, TranAD, AnomalyTransformer) "
            "inflates results by up to 30pp and masks model failures. We report both "
            "for compatibility. AUC-PR is threshold-free and better for imbalanced "
            "data (anomalies are rare). TaPR provides segment-level temporal credit "
            "with buffer delta=0.1 (Kim et al. AAAI 2022)."
        ),
    },
    'tte': {
        'primary': 'nrmse',
        'secondary': 'rmse',
        'rationale': (
            "TTE-RMSE in cycles is useful for reporting to operators (e.g., 'off by "
            "N cycles'). nRMSE is better for cross-domain comparison where event "
            "horizons differ. We define TTE using SPC 3-sigma rule on baseline "
            "cycles 1-50 (healthy baseline); this is operationally standard."
        ),
    },
}
