# SPDX-FileCopyrightText: 2025 Industrial World Model Authors
# SPDX-License-Identifier: MIT

"""
Evaluation Metrics for Industrial World Model.

Covers all major industrial AI tasks:
1. Forecasting: MAE, MSE, RMSE, MAPE, sMAPE
2. RUL Prediction: NASA scoring, Percentage Error, Timeliness
3. Anomaly Detection: Precision, Recall, F1, AUC-ROC, AUC-PR
4. Classification: Accuracy, F1-macro, Confusion Matrix
5. Calibration: ECE, MCE, Brier Score
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple, Union, List
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Container for metric results."""
    name: str
    value: float
    std: Optional[float] = None
    details: Optional[Dict] = None


def compute_forecasting_metrics(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    horizon: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute forecasting metrics.

    Args:
        predictions: [B, H, D] or [N, D] predicted values
        targets: [B, H, D] or [N, D] target values
        horizon: If provided, compute per-horizon metrics

    Returns:
        Dict of metric names to values
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # Flatten if needed
    if predictions.ndim == 3:
        B, H, D = predictions.shape
        predictions = predictions.reshape(-1, D)
        targets = targets.reshape(-1, D)

    # Handle NaN/Inf
    mask = np.isfinite(predictions) & np.isfinite(targets)
    predictions = np.where(mask, predictions, 0)
    targets = np.where(mask, targets, 0)

    metrics = {}

    # Mean Absolute Error
    mae = np.abs(predictions - targets).mean()
    metrics["mae"] = float(mae)

    # Mean Squared Error
    mse = ((predictions - targets) ** 2).mean()
    metrics["mse"] = float(mse)

    # Root Mean Squared Error
    metrics["rmse"] = float(np.sqrt(mse))

    # Mean Absolute Percentage Error (avoid division by zero)
    denom = np.abs(targets) + 1e-8
    mape = np.abs((predictions - targets) / denom).mean() * 100
    metrics["mape"] = float(mape)

    # Symmetric MAPE
    smape = 2 * np.abs(predictions - targets) / (np.abs(predictions) + np.abs(targets) + 1e-8)
    metrics["smape"] = float(smape.mean() * 100)

    # Normalized RMSE
    target_std = np.std(targets) + 1e-8
    metrics["nrmse"] = float(np.sqrt(mse) / target_std)

    # R² Score
    ss_res = ((targets - predictions) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    metrics["r2"] = float(r2)

    return metrics


def compute_rul_metrics(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    late_penalty: float = 13.0,
    early_penalty: float = 10.0,
) -> Dict[str, float]:
    """
    Compute Remaining Useful Life (RUL) prediction metrics.

    Uses NASA's asymmetric scoring function for turbofan engines.

    Args:
        predictions: [N] predicted RUL values
        targets: [N] true RUL values
        late_penalty: Exponent for late predictions (dangerous)
        early_penalty: Exponent for early predictions (conservative)

    Returns:
        Dict of RUL metrics
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    predictions = predictions.flatten()
    targets = targets.flatten()

    metrics = {}

    # Basic metrics
    errors = predictions - targets  # Positive = late (overestimate RUL)

    metrics["mae"] = float(np.abs(errors).mean())
    metrics["rmse"] = float(np.sqrt((errors ** 2).mean()))

    # Percentage errors
    pct_errors = np.abs(errors) / (targets + 1e-8) * 100
    metrics["mape"] = float(pct_errors.mean())

    # NASA Scoring Function (asymmetric)
    scores = np.zeros_like(errors)
    late_mask = errors > 0
    early_mask = ~late_mask

    # Late predictions (more dangerous): exp(error/13) - 1
    scores[late_mask] = np.exp(errors[late_mask] / late_penalty) - 1

    # Early predictions (conservative): exp(-error/10) - 1
    scores[early_mask] = np.exp(-errors[early_mask] / early_penalty) - 1

    metrics["nasa_score"] = float(scores.sum())
    metrics["nasa_score_mean"] = float(scores.mean())

    # Timeliness metrics
    within_10 = np.abs(errors) <= 10
    within_20 = np.abs(errors) <= 20
    metrics["accuracy_10"] = float(within_10.mean() * 100)
    metrics["accuracy_20"] = float(within_20.mean() * 100)

    # Late/Early breakdown
    metrics["late_ratio"] = float(late_mask.mean() * 100)
    metrics["early_ratio"] = float(early_mask.mean() * 100)

    return metrics


def compute_anomaly_metrics(
    scores: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    threshold: Optional[float] = None,
    point_adjust: bool = True,
) -> Dict[str, float]:
    """
    Compute anomaly detection metrics.

    Args:
        scores: [N] or [B, T] anomaly scores
        labels: [N] or [B, T] binary labels (1 = anomaly)
        threshold: Decision threshold (None = optimal)
        point_adjust: Apply point-adjust F1 (common in time series)

    Returns:
        Dict of anomaly detection metrics
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    scores = scores.flatten()
    labels = labels.flatten()

    metrics = {}

    # AUC-ROC
    try:
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
        metrics["auc_roc"] = float(roc_auc_score(labels, scores))

        # AUC-PR (more informative for imbalanced)
        precision, recall, _ = precision_recall_curve(labels, scores)
        metrics["auc_pr"] = float(auc(recall, precision))
    except ImportError:
        # Manual AUC computation if sklearn not available
        metrics["auc_roc"] = _manual_auc_roc(scores, labels)

    # Find optimal threshold if not provided
    if threshold is None:
        threshold = _find_optimal_threshold(scores, labels)

    metrics["threshold"] = float(threshold)

    # Binary predictions
    predictions = (scores >= threshold).astype(int)

    # Basic metrics
    tp = ((predictions == 1) & (labels == 1)).sum()
    fp = ((predictions == 1) & (labels == 0)).sum()
    fn = ((predictions == 0) & (labels == 1)).sum()
    tn = ((predictions == 0) & (labels == 0)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    metrics["precision"] = float(precision)
    metrics["recall"] = float(recall)
    metrics["f1"] = float(f1)
    metrics["accuracy"] = float((tp + tn) / (tp + fp + fn + tn))

    # Point-adjust F1 (standard for time series anomaly detection)
    if point_adjust:
        pa_f1 = _point_adjust_f1(predictions, labels)
        metrics["f1_pa"] = float(pa_f1)

    # False positive rate
    metrics["fpr"] = float(fp / (fp + tn + 1e-8))

    return metrics


def _point_adjust_f1(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute point-adjusted F1 score.

    If any point in an anomaly segment is detected, the whole segment counts.
    """
    # Find anomaly segments in labels
    segments = _find_segments(labels)

    if len(segments) == 0:
        return 0.0

    # Check if each segment has at least one detection
    detected_segments = 0
    for start, end in segments:
        if predictions[start:end].any():
            detected_segments += 1

    # Precision: what fraction of predicted positives are in true segments
    pred_positives = predictions.sum()
    true_positives = 0
    for i, p in enumerate(predictions):
        if p and labels[i]:
            true_positives += 1

    precision = true_positives / (pred_positives + 1e-8)
    recall = detected_segments / len(segments)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return f1


def _find_segments(labels: np.ndarray) -> List[Tuple[int, int]]:
    """Find contiguous anomaly segments."""
    segments = []
    in_segment = False
    start = 0

    for i, label in enumerate(labels):
        if label == 1 and not in_segment:
            in_segment = True
            start = i
        elif label == 0 and in_segment:
            in_segment = False
            segments.append((start, i))

    if in_segment:
        segments.append((start, len(labels)))

    return segments


def _find_optimal_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Find threshold that maximizes F1."""
    thresholds = np.percentile(scores, np.linspace(1, 99, 99))
    best_f1 = 0
    best_threshold = 0.5

    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold


def _manual_auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Manual AUC-ROC computation."""
    # Sort by scores
    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_indices]

    # Compute TPR and FPR at each threshold
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    tpr = np.cumsum(sorted_labels) / (n_pos + 1e-8)
    fpr = np.cumsum(1 - sorted_labels) / (n_neg + 1e-8)

    # AUC via trapezoidal rule
    auc = np.trapz(tpr, fpr)
    return float(auc)


def compute_classification_metrics(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    num_classes: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute multi-class classification metrics.

    Args:
        predictions: [N] predicted class indices or [N, C] logits
        targets: [N] true class indices
        num_classes: Number of classes (inferred if not provided)

    Returns:
        Dict of classification metrics
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # Convert logits to predictions if needed
    if predictions.ndim == 2:
        predictions = predictions.argmax(axis=-1)

    predictions = predictions.flatten().astype(int)
    targets = targets.flatten().astype(int)

    if num_classes is None:
        num_classes = max(predictions.max(), targets.max()) + 1

    metrics = {}

    # Accuracy
    metrics["accuracy"] = float((predictions == targets).mean() * 100)

    # Per-class metrics
    precisions = []
    recalls = []
    f1s = []

    for c in range(num_classes):
        tp = ((predictions == c) & (targets == c)).sum()
        fp = ((predictions == c) & (targets != c)).sum()
        fn = ((predictions != c) & (targets == c)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    # Macro averages
    metrics["precision_macro"] = float(np.mean(precisions) * 100)
    metrics["recall_macro"] = float(np.mean(recalls) * 100)
    metrics["f1_macro"] = float(np.mean(f1s) * 100)

    # Weighted F1
    class_counts = np.bincount(targets, minlength=num_classes)
    weights = class_counts / class_counts.sum()
    metrics["f1_weighted"] = float((np.array(f1s) * weights).sum() * 100)

    return metrics


def compute_calibration_metrics(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute calibration metrics for uncertainty quantification.

    Args:
        predictions: [N, C] predicted probabilities or [N, Q] quantile predictions
        targets: [N] true class indices or [N] regression targets
        n_bins: Number of bins for ECE computation

    Returns:
        Dict of calibration metrics
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    metrics = {}

    # For classification (probabilities)
    if predictions.ndim == 2 and predictions.shape[1] > 1:
        max_probs = predictions.max(axis=-1)
        predicted_classes = predictions.argmax(axis=-1)
        correct = (predicted_classes == targets).astype(float)

        # Expected Calibration Error (ECE)
        ece = _compute_ece(max_probs, correct, n_bins)
        metrics["ece"] = float(ece * 100)

        # Maximum Calibration Error (MCE)
        mce = _compute_mce(max_probs, correct, n_bins)
        metrics["mce"] = float(mce * 100)

        # Brier Score
        one_hot = np.eye(predictions.shape[1])[targets]
        brier = ((predictions - one_hot) ** 2).sum(axis=-1).mean()
        metrics["brier"] = float(brier)

    # For regression (quantile predictions)
    else:
        # Check quantile coverage
        if predictions.ndim == 2:
            # Assume predictions are [N, Q] quantile predictions
            quantiles = np.linspace(0.1, 0.9, predictions.shape[1])
            coverage = _compute_quantile_coverage(predictions, targets, quantiles)
            metrics["quantile_coverage"] = float(coverage * 100)

    return metrics


def _compute_ece(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int) -> float:
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin

    return ece


def _compute_mce(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int) -> float:
    """Compute Maximum Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    max_gap = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])

        if in_bin.sum() > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            gap = np.abs(avg_accuracy - avg_confidence)
            max_gap = max(max_gap, gap)

    return max_gap


def _compute_quantile_coverage(
    predictions: np.ndarray,
    targets: np.ndarray,
    quantiles: np.ndarray,
) -> float:
    """Compute average quantile coverage."""
    coverages = []

    for i, q in enumerate(quantiles):
        below = (targets <= predictions[:, i]).mean()
        expected = q
        coverages.append(np.abs(below - expected))

    return 1 - np.mean(coverages)
