"""V21 Surface -> Legacy Metrics.

Convert the stored probability surface p(t, Δt) to legacy metrics for
literature comparability:

- RMSE (C-MAPSS): derive predicted RUL from surface, compare to true RUL
- PA-F1 (anomaly): derive per-timestep score, apply point-adjustment, F1
- non-PA F1, Precision, Recall at best threshold on the chosen score

All conversions are deterministic and read from stored .npz files so we
never rerun inference to recompute a metric.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# C-MAPSS: surface -> predicted RUL
# ---------------------------------------------------------------------------

def surface_to_rul(p_surface: np.ndarray, horizons: np.ndarray) -> np.ndarray:
    """Expected time-to-event from surface.

    RUL_hat(t) = (Δt_max * (1 - p(Δt_max))) + sum_k g_k * Δt_k
    where g_k encodes incremental event probability. We use the simpler
    threshold-crossing estimator:

      RUL_hat(t) = smallest Δt_k such that p(t, Δt_k) >= 0.5
                    (else Δt_max — "no failure within horizon")

    This matches the semantics of the surface: p(t, Δt) is P(event in [t, t+Δt]).
    A calibrated model crosses 0.5 exactly at the expected event time.

    Args:
        p_surface: (N, K) probability surface
        horizons:  (K,)   horizon values (steps/cycles), monotonic increasing

    Returns:
        rul_hat:   (N,)   scalar RUL prediction per observation
    """
    p = np.asarray(p_surface, dtype=np.float64)
    h = np.asarray(horizons, dtype=np.float64)
    N, K = p.shape
    # For each row find first column where p >= 0.5
    crossed = p >= 0.5
    # First index where crossed is True; if none, use K (-> h[K-1])
    first = np.where(crossed.any(axis=1),
                     crossed.argmax(axis=1),
                     K - 1)
    # But argmax returns 0 for all-False — handle explicitly
    any_cross = crossed.any(axis=1)
    rul = np.where(any_cross, h[first], h[-1])
    return rul.astype(np.float32)


def surface_to_rul_expected(p_surface: np.ndarray,
                            horizons: np.ndarray) -> np.ndarray:
    """Alternative estimator: expected value of time-to-event from cdf.

    If p(t, Δt) is the cdf P(tte <= Δt), then E[tte] ≈ integral of (1-p)dΔt.
    Approximated by the trapezoidal rule over the horizon grid, plus a tail.
    """
    p = np.asarray(p_surface, dtype=np.float64)
    h = np.asarray(horizons, dtype=np.float64)
    # Augment: p(Δt=0) = 0
    h_aug = np.concatenate([[0.0], h])
    p_aug = np.concatenate([np.zeros((p.shape[0], 1)), p], axis=1)
    surv = 1.0 - p_aug  # P(tte > Δt)
    # Trapezoidal integral of survival over [0, h_max]
    e_tte = np.trapz(surv, h_aug, axis=1)
    # Add tail past max horizon — assume 0 (capped at h_max)
    return e_tte.astype(np.float32)


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def nasa_score(pred: np.ndarray, true: np.ndarray) -> float:
    err = np.asarray(pred) - np.asarray(true)
    s = np.where(err < 0, np.exp(-err / 13.0) - 1.0, np.exp(err / 10.0) - 1.0)
    return float(np.sum(s))


# ---------------------------------------------------------------------------
# Anomaly: surface -> per-timestep score
# ---------------------------------------------------------------------------

def surface_to_anomaly_scores(p_surface: np.ndarray,
                              t_index: np.ndarray,
                              T: int,
                              horizons: np.ndarray,
                              horizon_for_score: Optional[int] = None,
                              mode: str = 'max') -> np.ndarray:
    """Map surface to per-timestep anomaly score on [0, T).

    Each observation at time t covers timesteps [t-W, t] via the encoder and
    predicts into [t+1, t+max_horizon]. For a per-timestep score on the test
    array, we aggregate predictions whose horizon lands on a given timestep.

    Simple version: for each observation at t, place p(t, Δt_k) at t+Δt_k
    and average / max across contributors.

    Args:
        p_surface:         (N, K) surface
        t_index:           (N,)   observation times
        T:                 total timesteps in test series
        horizons:          (K,)   horizon values in steps
        horizon_for_score: if not None, use p(t, Δt=this) placed at t+this.
                            If None, use `mode` across horizons.
        mode:              'max' (default) or 'mean' — aggregation at each
                            target timestep across multiple contributors.

    Returns:
        scores: (T,) float in [0, 1]. Timesteps never targeted: 0.
    """
    p = np.asarray(p_surface, dtype=np.float32)
    horizons = np.asarray(horizons, dtype=np.int64)
    t_index = np.asarray(t_index, dtype=np.int64)
    N, K = p.shape

    if horizon_for_score is not None:
        k = int(np.where(horizons == horizon_for_score)[0][0])
        scores = np.zeros(T, dtype=np.float32)
        counts = np.zeros(T, dtype=np.float32)
        tgt = t_index + int(horizon_for_score)
        valid = (tgt >= 0) & (tgt < T)
        np.add.at(scores, tgt[valid], p[valid, k])
        np.add.at(counts, tgt[valid], 1.0)
        scores[counts > 0] /= counts[counts > 0]
        return scores

    # Aggregate over all horizons
    scores = np.zeros(T, dtype=np.float32)
    if mode == 'max':
        for k in range(K):
            tgt = t_index + int(horizons[k])
            valid = (tgt >= 0) & (tgt < T)
            cur = np.zeros(T, dtype=np.float32)
            # Use maximum where multiple (t, k) land on same target
            for i in np.where(valid)[0]:
                cur[tgt[i]] = max(cur[tgt[i]], p[i, k])
            scores = np.maximum(scores, cur)
    else:
        counts = np.zeros(T, dtype=np.float32)
        for k in range(K):
            tgt = t_index + int(horizons[k])
            valid = (tgt >= 0) & (tgt < T)
            np.add.at(scores, tgt[valid], p[valid, k])
            np.add.at(counts, tgt[valid], 1.0)
        mask = counts > 0
        scores[mask] /= counts[mask]

    return scores


def surface_to_observation_score(p_surface: np.ndarray,
                                 horizons: np.ndarray,
                                 horizon_for_score: Optional[int] = None
                                 ) -> np.ndarray:
    """Per-observation anomaly score (one score per window).

    If horizon_for_score is None, returns max over horizons.
    """
    p = np.asarray(p_surface, dtype=np.float32)
    if horizon_for_score is None:
        return p.max(axis=1)
    k = int(np.where(np.asarray(horizons) == horizon_for_score)[0][0])
    return p[:, k]


# ---------------------------------------------------------------------------
# Point-adjustment F1 (anomaly datasets)
# ---------------------------------------------------------------------------

def _adjust_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Standard point-adjustment: if any prediction in an anomaly segment
    is positive, mark the entire segment as correctly predicted.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32).copy()
    in_segment = False
    start = 0
    for t in range(len(y_true)):
        if y_true[t] == 1 and not in_segment:
            in_segment = True
            start = t
        elif y_true[t] == 0 and in_segment:
            # end of segment at t-1
            if y_pred[start:t].any():
                y_pred[start:t] = 1
            in_segment = False
    if in_segment:
        if y_pred[start:].any():
            y_pred[start:] = 1
    return y_pred


def binary_prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {'precision': float(p), 'recall': float(r), 'f1': float(f1),
            'tp': tp, 'fp': fp, 'fn': fn}


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray,
                      n_thresholds: int = 200,
                      pa: bool = False) -> dict:
    """Sweep thresholds, return best F1 and associated P/R/threshold.

    If pa=True, apply point-adjustment at each threshold before F1.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32)
    if len(s) == 0 or y.sum() == 0 or y.sum() == len(y):
        return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0,
                'threshold': 0.5, 'pa': pa}
    # Candidate thresholds: quantiles to avoid sweeping identical values
    qs = np.linspace(0.0, 1.0, n_thresholds + 1)[1:-1]
    thrs = np.unique(np.quantile(s, qs))
    best = {'f1': -1.0}
    for thr in thrs:
        yp = (s >= thr).astype(np.int32)
        if pa:
            yp = _adjust_predictions(y, yp)
        m = binary_prf(y, yp)
        m['threshold'] = float(thr)
        if m['f1'] > best.get('f1', -1.0):
            best = m
    best['pa'] = pa
    return best


def anomaly_legacy_metrics(p_surface: np.ndarray, t_index: np.ndarray,
                           labels: np.ndarray, horizons: np.ndarray,
                           horizon_for_score: Optional[int] = 100) -> dict:
    """Compute full legacy anomaly metrics from a stored surface.

    horizon_for_score: use p(t, Δt=100) placed at t+100 as the per-timestep
        score, matching MTS-JEPA's 100-step window semantics.
    """
    T = len(labels)
    scores = surface_to_anomaly_scores(
        p_surface, t_index, T, horizons,
        horizon_for_score=horizon_for_score)
    # Eval only on timesteps that were covered by at least one observation
    # (ie where score != 0 after aggregation OR a labeled anomaly at that time)
    # For simplicity evaluate on all timesteps; uncovered timesteps get score 0.
    best_pa = best_f1_threshold(scores, labels, pa=True)
    best_npa = best_f1_threshold(scores, labels, pa=False)
    return {
        'pa_f1': best_pa['f1'], 'pa_precision': best_pa['precision'],
        'pa_recall': best_pa['recall'], 'pa_threshold': best_pa['threshold'],
        'non_pa_f1': best_npa['f1'], 'non_pa_precision': best_npa['precision'],
        'non_pa_recall': best_npa['recall'], 'non_pa_threshold': best_npa['threshold'],
        'horizon_for_score': horizon_for_score,
    }
