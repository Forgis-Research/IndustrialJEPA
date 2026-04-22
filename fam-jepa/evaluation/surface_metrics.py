"""
Probability Surface Evaluation for FAM.

Primary metric: AUPRC pooled over all (t, Δt) cells — one number per dataset.
Secondary metric: AUROC pooled over all (t, Δt) cells.
Breakdown: AUPRC(Δt) per horizon (for appendix).
Supplementary: F1 at best threshold, reliability diagram data.

All metrics expect:
  p_surface: (N, K) or (T, K) predicted probabilities
  y_surface: (N, K) or (T, K) binary ground truth

Where N = number of observation times (pooled across samples),
      K = number of horizons.

Higher scores in p_surface = more likely event (natural direction for
sklearn's average_precision_score and roc_auc_score).
"""

import numpy as np
from typing import Optional, Tuple


def evaluate_probability_surface(
    p_surface: np.ndarray,
    y_surface: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> dict:
    """
    Primary evaluation: pool all (t, Δt) cells, compute AUPRC + AUROC.

    This is the ONE function every v21+ experiment should call.

    Args:
        p_surface: (N, K) predicted probabilities in [0, 1].
                   N = observation times, K = horizons.
        y_surface: (N, K) binary labels (1 = event within horizon).
        mask:      (N, K) bool, True = valid cell. If None, all cells valid.
                   Use to exclude padding or cells beyond sequence end.

    Returns dict with:
        auprc:           pooled AUPRC (primary metric)
        auroc:           pooled AUROC (secondary metric)
        f1_best:         F1 at best threshold on this data
        precision_best:  precision at best-F1 threshold
        recall_best:     recall at best-F1 threshold
        threshold_best:  the threshold that maximizes F1
        prevalence:      fraction of positive cells
        n_cells:         total valid cells
        n_positive:      number of positive cells
    """
    from sklearn.metrics import (
        average_precision_score, roc_auc_score,
        precision_recall_curve,
    )

    p = np.asarray(p_surface, dtype=np.float64)
    y = np.asarray(y_surface, dtype=np.int32)

    # Flatten and mask
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        valid = mask.ravel()
    else:
        valid = np.ones(p.size, dtype=bool)

    p_flat = p.ravel()[valid]
    y_flat = y.ravel()[valid]

    n_cells = int(valid.sum())
    n_pos = int(y_flat.sum())
    prevalence = n_pos / n_cells if n_cells > 0 else 0.0

    # Guard: need both classes for AUPRC/AUROC
    if n_pos == 0 or n_pos == n_cells:
        return {
            'auprc': float('nan'),
            'auroc': float('nan'),
            'f1_best': 0.0,
            'precision_best': 0.0,
            'recall_best': 0.0,
            'threshold_best': 0.5,
            'prevalence': prevalence,
            'n_cells': n_cells,
            'n_positive': n_pos,
        }

    auprc = float(average_precision_score(y_flat, p_flat))
    auroc = float(roc_auc_score(y_flat, p_flat))

    # Best F1 from PR curve
    prec_arr, rec_arr, thresholds = precision_recall_curve(y_flat, p_flat)
    f1_arr = np.where(
        (prec_arr + rec_arr) > 0,
        2 * prec_arr * rec_arr / (prec_arr + rec_arr),
        0.0,
    )
    # precision_recall_curve returns len(thresholds) = len(prec) - 1
    best_idx = int(np.argmax(f1_arr[:-1]))
    f1_best = float(f1_arr[best_idx])
    threshold_best = float(thresholds[best_idx])
    precision_best = float(prec_arr[best_idx])
    recall_best = float(rec_arr[best_idx])

    return {
        'auprc': auprc,
        'auroc': auroc,
        'f1_best': f1_best,
        'precision_best': precision_best,
        'recall_best': recall_best,
        'threshold_best': threshold_best,
        'prevalence': prevalence,
        'n_cells': n_cells,
        'n_positive': n_pos,
    }


def auprc_per_horizon(
    p_surface: np.ndarray,
    y_surface: np.ndarray,
    horizon_labels: Optional[list] = None,
    mask: Optional[np.ndarray] = None,
) -> dict:
    """
    Per-horizon AUPRC breakdown (for appendix figures).

    For each horizon Δt_k (column k), pool across all observation times t,
    compute AUPRC.

    Args:
        p_surface: (N, K) predicted probabilities
        y_surface: (N, K) binary labels
        horizon_labels: optional list of K horizon names/values
        mask: (N, K) bool, True = valid

    Returns dict with:
        auprc_per_k:  list of K AUPRC values
        auroc_per_k:  list of K AUROC values
        horizon_labels: list of K labels
        prevalence_per_k: list of K prevalences
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    p = np.asarray(p_surface, dtype=np.float64)
    y = np.asarray(y_surface, dtype=np.int32)
    K = p.shape[1]

    if horizon_labels is None:
        horizon_labels = [f'k{i}' for i in range(K)]

    auprc_list = []
    auroc_list = []
    prev_list = []

    for k in range(K):
        p_k = p[:, k]
        y_k = y[:, k]

        if mask is not None:
            valid_k = mask[:, k]
            p_k = p_k[valid_k]
            y_k = y_k[valid_k]

        n_pos = int(y_k.sum())
        n_total = len(y_k)
        prev_list.append(n_pos / n_total if n_total > 0 else 0.0)

        if n_pos == 0 or n_pos == n_total:
            auprc_list.append(float('nan'))
            auroc_list.append(float('nan'))
        else:
            auprc_list.append(float(average_precision_score(y_k, p_k)))
            auroc_list.append(float(roc_auc_score(y_k, p_k)))

    return {
        'auprc_per_k': auprc_list,
        'auroc_per_k': auroc_list,
        'horizon_labels': horizon_labels,
        'prevalence_per_k': prev_list,
    }


def reliability_diagram(
    p_surface: np.ndarray,
    y_surface: np.ndarray,
    n_bins: int = 10,
    mask: Optional[np.ndarray] = None,
) -> dict:
    """
    Calibration reliability diagram data.

    Bins predicted probabilities and computes mean predicted vs observed
    frequency in each bin.

    Args:
        p_surface: (N, K) predicted probabilities
        y_surface: (N, K) binary labels
        n_bins: number of equal-width bins in [0, 1]
        mask: (N, K) bool, True = valid

    Returns dict with:
        bin_centers:   (n_bins,) center of each bin
        bin_means:     (n_bins,) mean predicted probability in bin
        bin_freqs:     (n_bins,) observed event frequency in bin
        bin_counts:    (n_bins,) number of cells in bin
        ece:           Expected Calibration Error (weighted by bin count)
    """
    p = np.asarray(p_surface, dtype=np.float64).ravel()
    y = np.asarray(y_surface, dtype=np.float64).ravel()

    if mask is not None:
        valid = np.asarray(mask, dtype=bool).ravel()
        p = p[valid]
        y = y[valid]

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_means = np.zeros(n_bins)
    bin_freqs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (p >= lo) & (p <= hi)  # inclusive on last bin
        else:
            in_bin = (p >= lo) & (p < hi)

        bin_counts[i] = int(in_bin.sum())
        if bin_counts[i] > 0:
            bin_means[i] = float(p[in_bin].mean())
            bin_freqs[i] = float(y[in_bin].mean())

    # Expected Calibration Error
    total = max(int(bin_counts.sum()), 1)
    ece = float(np.sum(bin_counts * np.abs(bin_means - bin_freqs)) / total)

    return {
        'bin_centers': bin_centers.tolist(),
        'bin_means': bin_means.tolist(),
        'bin_freqs': bin_freqs.tolist(),
        'bin_counts': bin_counts.tolist(),
        'ece': ece,
        'n_bins': n_bins,
    }
