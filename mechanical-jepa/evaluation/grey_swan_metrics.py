"""
Grey Swan Unified Evaluation Module (V20).

FIRST-EVENT PREDICTION: all tasks use the same label definition.

  y(t) = 1 if the first event has occurred by time t, else 0.

This applies to failure (C-MAPSS), anomaly onset (SMAP/MSL/PSM),
arrhythmia (MBA), fault onset (Paderborn). Subsequent events are ignored.

Two-stage evaluation:

  Stage 1 — DETECTION: "Will the first event occur within horizon H?"
    Binary F1, Precision, Recall, AUROC.

  Stage 2 — TIMING: "In which of W windows will it occur?"
    Ordinal classification → per-window F1, macro-F1.

The metric is always F1. No point-adjustment. No segment credit.
Legacy metrics (RMSE, PA-F1) are computed for literature comparability only.

References:
  - NASA Scoring Function: Saxena et al. (2008), IJRPC
  - TSAD-Eval / PA-F1 critique: Schmidl et al. (2022), VLDB
"""

import numpy as np
from typing import Optional, Tuple, Union
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 0. FIRST-EVENT LABEL CONVERSION
# ---------------------------------------------------------------------------

def labels_to_first_event_tte(labels: np.ndarray) -> np.ndarray:
    """
    Convert a binary label array (0=normal, 1=event) to time-to-first-event.

    For each timestep t, returns the number of steps until the first event onset.
    After the first event onset, returns 0 (event has occurred).
    If no event exists in the sequence, returns inf for all timesteps.

    This is the canonical label conversion for anomaly/fault datasets.
    For RUL datasets (C-MAPSS), ground-truth RUL is already time-to-event.

    Args:
        labels: (T,) binary array, 1 = event active

    Returns:
        tte: (T,) float array, time-to-first-event from each timestep
    """
    labels = np.asarray(labels, dtype=int)
    T = len(labels)
    tte = np.full(T, np.inf, dtype=float)

    # Find first event onset
    event_indices = np.where(labels == 1)[0]
    if len(event_indices) == 0:
        return tte  # no event in this sequence

    first_event = int(event_indices[0])

    for t in range(T):
        if t < first_event:
            tte[t] = float(first_event - t)
        else:
            tte[t] = 0.0  # event has occurred

    return tte


# ---------------------------------------------------------------------------
# 1. RUL METRICS (legacy)
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


def auroc(scores: np.ndarray, y_true: np.ndarray) -> float:
    """
    Area under the ROC curve via sklearn (exact, uses all unique thresholds).
    Falls back to manual computation if sklearn unavailable.
    """
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_true)) < 2:
            return float('nan')
        return float(roc_auc_score(y_true, scores))
    except ImportError:
        # Fallback: manual trapezoidal
        n_thresholds = 500
        thresholds = np.linspace(scores.max(), scores.min(), n_thresholds)
        tprs, fprs = [0.0], [0.0]
        for thresh in thresholds:
            y_pred = (scores >= thresh).astype(int)
            tp, fp, tn, fn = _confusion(y_true, y_pred)
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            tprs.append(tpr)
            fprs.append(fpr)
        tprs.append(1.0)
        fprs.append(1.0)
        sort_idx = np.argsort(fprs)
        return float(np.trapz(np.array(tprs)[sort_idx], np.array(fprs)[sort_idx]))


def f1_at_horizon(pred_rul: np.ndarray, true_rul: np.ndarray,
                  k: int, rul_cap: float = 125.0) -> dict:
    """
    Binary F1 for 'will fail within k cycles'.

    Converts RUL regression to binary classification:
      y_true_bin = (true_rul <= k)
      y_pred_bin = (pred_rul <= k)

    Args:
        pred_rul: (N,) predicted RUL (uncapped scale)
        true_rul: (N,) ground-truth RUL (uncapped scale)
        k: horizon in cycles
        rul_cap: RUL cap (for reference only)

    Returns dict with f1, precision, recall, auc_pr.
    """
    pred_rul = np.asarray(pred_rul, dtype=float)
    true_rul = np.asarray(true_rul, dtype=float)

    y_true_bin = (true_rul <= k).astype(int)
    y_pred_bin = (pred_rul <= k).astype(int)

    tp, fp, tn, fn = _confusion(y_true_bin, y_pred_bin)
    prec, rec = _precision_recall(tp, fp, fn)
    f1 = _f1(tp, fp, fn)

    # Use negative predicted RUL as "score" for AUC-PR (lower RUL = more likely to fail)
    aucpr = auc_pr(-pred_rul, y_true_bin)

    return {
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'auc_pr': aucpr,
        'k': k,
        'n_positive': int(y_true_bin.sum()),
        'n_total': int(len(y_true_bin)),
    }


def auc_pr(scores: np.ndarray, y_true: np.ndarray) -> float:
    """
    Area under the precision-recall curve via sklearn (exact).
    Falls back to manual computation if sklearn unavailable.
    """
    try:
        from sklearn.metrics import average_precision_score
        if len(np.unique(y_true)) < 2:
            return float('nan')
        return float(average_precision_score(y_true, scores))
    except ImportError:
        # Fallback: manual trapezoidal
        n_thresholds = 500
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

    # PA precision and recall
    prec_pa, rec_pa = _precision_recall(tp_pa, fp_pa, fn_pa)

    # AUC-PR and AUROC
    aucpr = auc_pr(scores, y_true)
    auc_roc = auroc(scores, y_true)

    # TaPR
    tapr_p, tapr_r, tapr_f = tapr(y_true, y_pred)

    return {
        'f1_non_pa': f1_np,
        'f1_pa': f1_pa,
        'precision_non_pa': prec,
        'recall_non_pa': rec,
        'precision_pa': prec_pa,
        'recall_pa': rec_pa,
        'auc_pr': aucpr,
        'auroc': auc_roc,
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
# 5. UNIFIED EVALUATION ENTRY POINTS
# ---------------------------------------------------------------------------

def evaluate_rul_run(pred: np.ndarray, target: np.ndarray,
                     rul_cap: float = 125.0,
                     horizons: tuple = (10, 20, 30, 50)) -> dict:
    """
    Single entry point for RUL evaluation. Returns standardized dict.

    Every RUL experiment script should call this instead of rul_metrics directly.
    Returns RMSE, NASA-S, and F1@k for all horizons in one dict.
    """
    base = rul_metrics(pred, target, rul_cap=rul_cap)
    f1_by_k = {}
    for k in horizons:
        f1_by_k[k] = f1_at_horizon(pred, target, k=k, rul_cap=rul_cap)
    base['f1_by_k'] = f1_by_k
    return base


def evaluate_anomaly_run(scores: np.ndarray, y_true: np.ndarray,
                         threshold: Optional[float] = None,
                         threshold_percentile: float = 95.0) -> dict:
    """
    Single entry point for anomaly evaluation. Returns standardized dict.

    Every anomaly experiment script should call this instead of anomaly_metrics directly.
    Returns all metrics: F1 (PA + non-PA), P, R, AUROC, AUPRC, TaPR.
    """
    return anomaly_metrics(scores, y_true, threshold=threshold,
                           threshold_percentile=threshold_percentile)


def aggregate_seeds(per_seed_metrics: list, key_metrics: Optional[list] = None) -> dict:
    """
    Aggregate metrics across seeds into standardized reporting format.

    Args:
        per_seed_metrics: list of dicts, one per seed (from evaluate_*_run)
        key_metrics: which keys to aggregate. If None, aggregates all float keys.

    Returns dict with:
        {metric}_mean, {metric}_std, {metric}_ci95_lo, {metric}_ci95_hi,
        n_seeds, per_seed (the raw list)

    95% CI uses t-distribution: mean ± t_{n-1, 0.025} * std / sqrt(n)
    """
    from scipy.stats import t as t_dist

    n = len(per_seed_metrics)
    if n == 0:
        return {'n_seeds': 0, 'per_seed': []}

    if key_metrics is None:
        # Auto-detect: all float/int keys in first dict (skip nested dicts)
        key_metrics = [k for k, v in per_seed_metrics[0].items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)]

    result = {'n_seeds': n, 'per_seed': per_seed_metrics}

    for key in key_metrics:
        values = [m[key] for m in per_seed_metrics if key in m
                  and np.isfinite(m[key])]
        if not values:
            result[f'{key}_mean'] = float('nan')
            result[f'{key}_std'] = float('nan')
            result[f'{key}_ci95_lo'] = float('nan')
            result[f'{key}_ci95_hi'] = float('nan')
            continue

        arr = np.array(values, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

        if len(arr) > 1:
            t_crit = float(t_dist.ppf(0.975, df=len(arr) - 1))
            margin = t_crit * std / np.sqrt(len(arr))
        else:
            margin = float('nan')

        result[f'{key}_mean'] = mean
        result[f'{key}_std'] = std
        result[f'{key}_ci95_lo'] = mean - margin
        result[f'{key}_ci95_hi'] = mean + margin

    return result


def format_result(agg: dict, metric: str) -> str:
    """
    Format an aggregated metric as a standardized string.

    Example: format_result(agg, 'rmse') -> '15.53 ± 1.68 (3s, 95% CI [11.35, 19.71])'
    """
    mean = agg.get(f'{metric}_mean', float('nan'))
    std = agg.get(f'{metric}_std', float('nan'))
    lo = agg.get(f'{metric}_ci95_lo', float('nan'))
    hi = agg.get(f'{metric}_ci95_hi', float('nan'))
    n = agg.get('n_seeds', 0)

    if np.isnan(mean):
        return 'N/A'
    if n <= 1 or np.isnan(std):
        return f'{mean:.4f} (1s, no CI)'
    return f'{mean:.4f} ± {std:.4f} ({n}s, 95% CI [{lo:.4f}, {hi:.4f}])'


# ---------------------------------------------------------------------------
# 6. UNIFIED TWO-STAGE EVENT PREDICTION EVALUATION
# ---------------------------------------------------------------------------

def event_detection(time_to_event: np.ndarray, pred_time_to_event: np.ndarray,
                    horizon: float) -> dict:
    """
    Stage 1: Event detection — "Will an event occur within horizon H?"

    Converts continuous time-to-event predictions into binary classification.
    Works for ANY event type: RUL, anomaly onset, threshold exceedance.

    Args:
        time_to_event:      (N,) ground-truth time-to-event (cycles, steps, seconds...)
                            NaN/inf = no event observed in this sample's future.
        pred_time_to_event: (N,) predicted time-to-event (same units)
        horizon:            detection horizon H (same units). "Will event occur within H?"

    Returns dict with:
        f1, precision, recall, auroc, auprc, n_positive, n_total
    """
    tte = np.asarray(time_to_event, dtype=float)
    pred = np.asarray(pred_time_to_event, dtype=float)

    # Ground truth: event within horizon?
    y_true = (np.isfinite(tte) & (tte <= horizon)).astype(int)
    y_pred = (np.isfinite(pred) & (pred <= horizon)).astype(int)

    # Continuous score for AUROC/AUPRC: lower predicted TTE = more likely event
    # Clamp inf/nan to horizon+1 so they score as "no event"
    score = np.where(np.isfinite(pred), -pred, -(horizon + 1))

    tp, fp, tn, fn = _confusion(y_true, y_pred)
    prec, rec = _precision_recall(tp, fp, fn)
    f1 = _f1(tp, fp, fn)
    auc_roc = auroc(score, y_true)
    aucpr = auc_pr(score, y_true)

    return {
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'auroc': auc_roc,
        'auprc': aucpr,
        'horizon': horizon,
        'n_positive': int(y_true.sum()),
        'n_total': int(len(y_true)),
    }


def event_timing(time_to_event: np.ndarray, pred_time_to_event: np.ndarray,
                 window_size: float, n_windows: int) -> dict:
    """
    Stage 2: Event timing — "In which window will the event occur?"

    Quantizes the time axis into n_windows bins of width window_size:
      Window 0: [0, Δ), Window 1: [Δ, 2Δ), ..., Window n-1: [(n-1)Δ, nΔ)
    Plus a "no event within horizon" bin (class = n_windows).

    Both ground-truth and predicted TTE are mapped to window indices.
    Evaluated as classification over (n_windows + 1) classes.

    Args:
        time_to_event:      (N,) ground-truth time-to-event
        pred_time_to_event: (N,) predicted time-to-event
        window_size:        width Δ of each window (same units as TTE)
        n_windows:          number of windows (total horizon = n_windows * window_size)

    Returns dict with:
        macro_f1, per_window_f1, per_window_precision, per_window_recall,
        accuracy, confusion_matrix, window_size, n_windows
    """
    tte = np.asarray(time_to_event, dtype=float)
    pred = np.asarray(pred_time_to_event, dtype=float)
    total_horizon = window_size * n_windows

    def _to_window(arr):
        w = np.full(len(arr), n_windows, dtype=int)  # default = "no event"
        finite = np.isfinite(arr) & (arr >= 0)
        w[finite] = np.clip(
            (arr[finite] / window_size).astype(int),
            0, n_windows - 1
        )
        # If finite but beyond total_horizon, mark as "no event"
        w[finite & (arr >= total_horizon)] = n_windows
        return w

    y_true_w = _to_window(tte)
    y_pred_w = _to_window(pred)

    n_classes = n_windows + 1  # +1 for "no event"
    per_window = {}
    f1s = []

    for c in range(n_classes):
        label = f'w{c}' if c < n_windows else 'no_event'
        yt = (y_true_w == c).astype(int)
        yp = (y_pred_w == c).astype(int)
        tp, fp, _, fn = _confusion(yt, yp)
        p, r = _precision_recall(tp, fp, fn)
        f = _f1(tp, fp, fn)
        per_window[label] = {'f1': f, 'precision': p, 'recall': r,
                             'n_true': int(yt.sum())}
        f1s.append(f)

    # Macro-F1: average over all classes with support > 0
    supported = [f1s[c] for c in range(n_classes)
                 if (y_true_w == c).sum() > 0]
    macro_f1 = float(np.mean(supported)) if supported else 0.0

    return {
        'macro_f1': macro_f1,
        'accuracy': float((y_true_w == y_pred_w).mean()),
        'per_window': per_window,
        'window_size': window_size,
        'n_windows': n_windows,
        'total_horizon': total_horizon,
        'n_total': int(len(tte)),
    }


def evaluate_event_prediction(time_to_event: np.ndarray,
                              pred_time_to_event: np.ndarray,
                              window_size: float,
                              n_windows: int,
                              legacy_rul_cap: Optional[float] = None) -> dict:
    """
    UNIFIED ENTRY POINT: Run both stages + optional legacy metrics.

    This is the ONE function every experiment script should call.

    Args:
        time_to_event:      (N,) ground-truth time-to-event
        pred_time_to_event: (N,) predicted time-to-event
        window_size:        width of each quantization window
        n_windows:          number of windows
        legacy_rul_cap:     if set, also compute RMSE/NASA-S for C-MAPSS compat

    Returns dict with:
        stage1: {detection at each window boundary as horizon}
        stage2: {window-quantized timing F1}
        legacy: {RMSE, NASA-S if applicable}
    """
    total_horizon = window_size * n_windows

    # Stage 1: detection at multiple horizons (each window boundary)
    stage1 = {}
    for w in range(1, n_windows + 1):
        h = w * window_size
        stage1[f'h{int(h)}'] = event_detection(
            time_to_event, pred_time_to_event, horizon=h)
    # Also overall detection at total horizon
    stage1['overall'] = event_detection(
        time_to_event, pred_time_to_event, horizon=total_horizon)

    # Stage 2: timing
    stage2 = event_timing(time_to_event, pred_time_to_event,
                          window_size=window_size, n_windows=n_windows)

    result = {
        'detection': stage1,
        'timing': stage2,
    }

    # Legacy metrics for backward compatibility
    if legacy_rul_cap is not None:
        tte = np.asarray(time_to_event, dtype=float)
        pred = np.asarray(pred_time_to_event, dtype=float)
        valid = np.isfinite(tte) & np.isfinite(pred)
        if valid.sum() > 0:
            result['legacy'] = rul_metrics(
                pred[valid], tte[valid], rul_cap=legacy_rul_cap)

    return result
