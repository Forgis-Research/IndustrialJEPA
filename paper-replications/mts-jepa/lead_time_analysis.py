"""
Lead-time analysis for MTS-JEPA (NeurIPS Critical Fix #1).

For every correctly-predicted anomalous window:
1. Check if the context window X_t is itself anomalous
2. Compute the lead time
3. Categorize: TRUE_PREDICTION / CONTINUATION / BOUNDARY
"""
import numpy as np
from data_utils import make_non_overlapping_windows, make_window_labels


def lead_time_analysis(test_data, test_labels, predictions, threshold,
                       window_length=100):
    """
    Analyze whether predictions are genuine early warnings or near-detection.

    Args:
        test_data: (T, V) test time series
        test_labels: (T,) point-level labels
        predictions: (N,) predicted anomaly probabilities for target windows
        threshold: classification threshold

    Returns:
        dict with categorized results
    """
    windows = make_non_overlapping_windows(
        test_data[:len(test_labels)].reshape(-1, 1) if test_data.ndim == 1 else test_data,
        window_length
    )
    window_labels = make_window_labels(test_labels, window_length)
    point_labels_reshaped = test_labels[:len(window_labels) * window_length].reshape(
        len(window_labels), window_length
    )

    # Context-target pairs: context X_t, target label y_{t+1}
    n_pairs = len(window_labels) - 1
    context_labels = window_labels[:-1]  # Whether context window has anomaly
    target_labels = window_labels[1:]   # Whether target window has anomaly

    # Point-level labels for context windows
    context_point_labels = point_labels_reshaped[:-1]  # (N, T_w)
    target_point_labels = point_labels_reshaped[1:]

    # Predictions for target windows
    pred_binary = (predictions[:n_pairs] > threshold).astype(int)

    results = {
        'TRUE_PREDICTION': [],    # Context fully normal, target anomalous
        'CONTINUATION': [],        # Context already anomalous
        'BOUNDARY': [],           # Anomaly starts in last 20% of context
        'CORRECT_NORMAL': [],     # Correctly predicted normal
        'FALSE_POSITIVE': [],     # Normal predicted as anomalous
        'FALSE_NEGATIVE': [],     # Anomalous missed
    }

    for i in range(min(n_pairs, len(predictions))):
        target_is_anomalous = target_labels[i] == 1
        predicted_anomalous = pred_binary[i] == 1
        context_has_anomaly = context_labels[i] == 1

        if target_is_anomalous and predicted_anomalous:
            # Correctly predicted anomaly — classify by type
            if not context_has_anomaly:
                # Check if anomaly starts in last 20% of context
                boundary_start = int(0.8 * window_length)
                ctx_tail_anomalous = context_point_labels[i][boundary_start:].any()

                if ctx_tail_anomalous:
                    results['BOUNDARY'].append(i)
                else:
                    # Genuine prediction
                    # Compute lead time: first anomalous point in target
                    anomaly_starts = np.argmax(target_point_labels[i] > 0)
                    lead_time = anomaly_starts  # timesteps into target before anomaly
                    results['TRUE_PREDICTION'].append({
                        'index': i,
                        'lead_time': int(lead_time),
                        'prob': float(predictions[i]),
                    })
            else:
                results['CONTINUATION'].append(i)
        elif not target_is_anomalous and not predicted_anomalous:
            results['CORRECT_NORMAL'].append(i)
        elif not target_is_anomalous and predicted_anomalous:
            results['FALSE_POSITIVE'].append(i)
        elif target_is_anomalous and not predicted_anomalous:
            results['FALSE_NEGATIVE'].append(i)

    # Summary statistics
    n_true_pred = len(results['TRUE_PREDICTION'])
    n_continuation = len(results['CONTINUATION'])
    n_boundary = len(results['BOUNDARY'])
    n_total_correct = n_true_pred + n_continuation + n_boundary

    summary = {
        'n_total_pairs': n_pairs,
        'n_true_prediction': n_true_pred,
        'n_continuation': n_continuation,
        'n_boundary': n_boundary,
        'n_total_correct_anomaly': n_total_correct,
        'n_false_positive': len(results['FALSE_POSITIVE']),
        'n_false_negative': len(results['FALSE_NEGATIVE']),
        'n_correct_normal': len(results['CORRECT_NORMAL']),
    }

    if n_total_correct > 0:
        summary['frac_true_prediction'] = n_true_pred / n_total_correct
        summary['frac_continuation'] = n_continuation / n_total_correct
        summary['frac_boundary'] = n_boundary / n_total_correct
    else:
        summary['frac_true_prediction'] = 0
        summary['frac_continuation'] = 0
        summary['frac_boundary'] = 0

    if n_true_pred > 0:
        lead_times = [r['lead_time'] for r in results['TRUE_PREDICTION']]
        summary['mean_lead_time'] = np.mean(lead_times)
        summary['median_lead_time'] = np.median(lead_times)
        summary['std_lead_time'] = np.std(lead_times)
    else:
        summary['mean_lead_time'] = 0
        summary['median_lead_time'] = 0
        summary['std_lead_time'] = 0

    return summary, results


def compute_true_prediction_auc(test_labels, predictions, window_length=100):
    """
    Compute AUC only on the TRUE_PREDICTION subset.
    This filters out continuation/boundary cases for a fairer metric.
    """
    from sklearn.metrics import roc_auc_score

    window_labels = make_window_labels(test_labels, window_length)
    n_pairs = len(window_labels) - 1

    context_labels = window_labels[:-1]
    target_labels = window_labels[1:]

    # TRUE_PREDICTION: context normal, target anomalous
    # Also include: context normal, target normal (true negatives)
    mask = context_labels[:min(n_pairs, len(predictions))] == 0  # Only context-normal pairs

    if mask.sum() == 0 or target_labels[mask].sum() == 0:
        return 0.5  # Degenerate case

    filtered_labels = target_labels[mask]
    filtered_preds = predictions[mask[:len(predictions)]]

    if len(np.unique(filtered_labels)) < 2:
        return 0.5

    return roc_auc_score(filtered_labels[:len(filtered_preds)], filtered_preds[:len(filtered_labels)])
