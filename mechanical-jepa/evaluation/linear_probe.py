"""
Linear probe utilities for frozen-encoder evaluation.

Two probes:
  1. RUL regression: LinearRegression / Ridge on frozen h_past -> RUL
  2. Anomaly classification: LogisticRegression on frozen h_past -> {0, 1}

Both use sklearn for simplicity and reproducibility.
The anomaly probe is the standard SSL evaluation protocol used by
SSD (Sehwag et al. 2021), matching MTS-JEPA's approach.
"""

import numpy as np
from typing import Optional


def fit_anomaly_probe(X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      C: float = 1.0, max_iter: int = 1000,
                      seed: int = 42) -> dict:
    """
    Fit logistic regression on frozen representations for anomaly classification.

    Args:
        X_train: (N_train, D) frozen encoder features for training windows
        y_train: (N_train,) binary labels (0=normal, 1=anomaly)
        X_test:  (N_test, D) frozen encoder features for test windows
        y_test:  (N_test,) binary labels
        C: inverse regularization strength
        max_iter: max iterations for solver
        seed: random state

    Returns dict with f1_non_pa, f1_pa, precision, recall, auroc, auc_pr, etc.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Standardize features (important for logistic regression)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(C=C, max_iter=max_iter, random_state=seed,
                             solver='lbfgs', class_weight='balanced')
    clf.fit(X_train_s, y_train)

    # Predict probabilities (use prob of class 1 as anomaly score)
    y_prob = clf.predict_proba(X_test_s)[:, 1]

    # Use evaluate_anomaly_run for consistent metrics
    from evaluation.grey_swan_metrics import evaluate_anomaly_run

    # evaluate_anomaly_run expects: scores (higher=more anomalous), y_true, threshold
    # For logistic regression, y_prob IS the score and threshold=0.5 matches clf.predict()
    metrics = evaluate_anomaly_run(y_prob, y_test, threshold=0.5)
    metrics['accuracy'] = float(clf.score(X_test_s, y_test))
    metrics['n_train'] = len(X_train)
    metrics['n_test'] = len(X_test)
    metrics['n_anomaly_train'] = int(y_train.sum())
    metrics['n_anomaly_test'] = int(y_test.sum())
    metrics['C'] = C
    return metrics


def mahalanobis_scores(X_train: np.ndarray, X_test: np.ndarray,
                       k: Optional[int] = None,
                       variance_retention: float = 0.99) -> np.ndarray:
    """
    Compute Mahalanobis distance scores using PCA-regularized covariance.

    Args:
        X_train: (N_train, D) training representations
        X_test:  (N_test, D) test representations
        k: PCA rank for covariance regularization. If None, choose k to
           retain >= variance_retention of training variance (label-free).
        variance_retention: cumulative variance threshold if k is None

    Returns:
        scores: (N_test,) Mahalanobis distances (higher = more anomalous)
    """
    from sklearn.decomposition import PCA

    D = X_train.shape[1]

    if k is None:
        # Label-free k selection: retain >= variance_retention of variance
        pca_full = PCA(n_components=min(D, X_train.shape[0]))
        pca_full.fit(X_train)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        k = int(np.searchsorted(cumvar, variance_retention) + 1)
        k = min(k, D)

    pca = PCA(n_components=k)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    mu = X_train_pca.mean(axis=0)
    cov = np.cov(X_train_pca, rowvar=False)

    # Regularize for numerical stability
    cov += np.eye(k) * 1e-6
    cov_inv = np.linalg.inv(cov)

    diff = X_test_pca - mu
    scores = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

    return scores
