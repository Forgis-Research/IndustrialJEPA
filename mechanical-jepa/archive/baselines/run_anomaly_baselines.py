"""
Phase 3: Anomaly Detection Baselines

One-class learning: train on healthy only, detect faulty/degrading.

Methods:
- Trivial: RMS threshold, kurtosis threshold, constant healthy
- Feature-based: IsolationForest, OCSVM, LOF, Mahalanobis, PCA reconstruction
- Deep: 1D CNN Autoencoder reconstruction error

Metrics: AUROC (primary), AUPRC, F1@optimal threshold

Outputs:
- results/anomaly_detection_baselines.json
"""

import numpy as np
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from data_utils import load_anomaly_data as load_anomaly_detection_data
from features import extract_features_batch, compute_rms, compute_kurtosis, N_FEATURES

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEEDS = [42, 123, 456]
RESULTS_PATH = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/baselines/results/anomaly_detection_baselines.json'
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)


def compute_metrics(y_true, scores):
    """
    Compute AUROC, AUPRC, F1 at optimal threshold.
    scores: higher = more anomalous.
    """
    if len(np.unique(y_true)) < 2:
        return {'auroc': np.nan, 'auprc': np.nan, 'f1_optimal': np.nan}
    try:
        auroc = roc_auc_score(y_true, scores)
        auprc = average_precision_score(y_true, scores)
        # Find optimal threshold by F1
        thresholds = np.percentile(scores, np.linspace(0, 100, 50))
        best_f1 = 0.0
        for thr in thresholds:
            y_pred = (scores >= thr).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            best_f1 = max(best_f1, f1)
        return {'auroc': float(auroc), 'auprc': float(auprc), 'f1_optimal': float(best_f1)}
    except Exception as e:
        return {'auroc': np.nan, 'auprc': np.nan, 'f1_optimal': np.nan, 'error': str(e)}


# ============================================================
# 1D CNN Autoencoder
# ============================================================

class ConvAutoencoder(nn.Module):
    """1D CNN Autoencoder for anomaly detection via reconstruction error."""
    def __init__(self, window_len: int = 16384):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=8, padding=28),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=32, stride=8, padding=12),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=16, stride=4, padding=6),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=8, stride=4, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 128, kernel_size=8, stride=4, padding=3, output_padding=3),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=16, stride=4, padding=6, output_padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=32, stride=8, padding=12, output_padding=7),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=64, stride=8, padding=28, output_padding=7),
        )

    def forward(self, x):
        # x: (B, L)
        z = self.encoder(x.unsqueeze(1))
        out = self.decoder(z).squeeze(1)
        # Match output length to input length
        if out.shape[-1] != x.shape[-1]:
            out = out[..., :x.shape[-1]]
        return out

    def anomaly_score(self, x):
        """Reconstruction error (MSE per sample)."""
        with torch.no_grad():
            recon = self(x)
            return ((recon - x) ** 2).mean(dim=-1)


def train_autoencoder(X_train, epochs=30, lr=1e-3, device='cuda', seed=42):
    torch.manual_seed(seed)
    model = ConvAutoencoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    X_t = torch.tensor(X_train, dtype=torch.float32)
    ds = TensorDataset(X_t)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            recon = model(xb)
            loss = nn.MSELoss()(recon, xb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"    AE epoch {epoch+1}/{epochs}, loss={total_loss/len(loader):.6f}")
    return model


def get_ae_scores(model, X, device='cuda'):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    ds = TensorDataset(X_t)
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    scores = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            s = model.anomaly_score(xb)
            scores.append(s.cpu().numpy())
    return np.concatenate(scores)


def run_anomaly_baselines(seeds=SEEDS):
    results = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    all_source_results = {}

    for source in ['femto', 'cwru', 'mafaulda']:
        print(f"\n{'='*60}")
        print(f"=== Source: {source.upper()} ===")
        print('='*60)

        try:
            X_train, X_test, y_test = load_anomaly_detection_data(source, verbose=True)
        except Exception as e:
            print(f"Failed to load {source}: {e}")
            continue

        if len(X_train) < 5 or len(np.unique(y_test)) < 2:
            print(f"Skipping {source}: insufficient data (train={len(X_train)}, test classes={np.unique(y_test)})")
            continue

        src_results = {}

        # Extract features
        print("Extracting features...")
        F_train = extract_features_batch(X_train)
        F_test = extract_features_batch(X_test)

        # -------------------------------------------------------
        # TRIVIAL BASELINES
        # -------------------------------------------------------
        print("\n-- Trivial Baselines --")

        # Constant healthy (always predict 0)
        scores_zero = np.zeros(len(y_test))
        src_results['trivial_constant_healthy'] = compute_metrics(y_test, scores_zero)
        print(f"  Constant healthy: AUROC={src_results['trivial_constant_healthy']['auroc']:.4f}")

        # RMS threshold: compute RMS per window
        rms_train = np.array([compute_rms(x) for x in X_train])
        rms_test = np.array([compute_rms(x) for x in X_test])
        mu_rms, sigma_rms = rms_train.mean(), rms_train.std()
        # Anomaly score = (rms - mu) / sigma (z-score)
        rms_scores = (rms_test - mu_rms) / (sigma_rms + 1e-10)
        src_results['trivial_rms_threshold'] = compute_metrics(y_test, rms_scores)
        print(f"  RMS threshold: AUROC={src_results['trivial_rms_threshold']['auroc']:.4f}")

        # Kurtosis threshold
        kurt_train = np.array([compute_kurtosis(x) for x in X_train])
        kurt_test = np.array([compute_kurtosis(x) for x in X_test])
        mu_k, sigma_k = kurt_train.mean(), kurt_train.std()
        kurt_scores = (kurt_test - mu_k) / (sigma_k + 1e-10)
        src_results['trivial_kurtosis_threshold'] = compute_metrics(y_test, kurt_scores)
        print(f"  Kurtosis threshold: AUROC={src_results['trivial_kurtosis_threshold']['auroc']:.4f}")

        # -------------------------------------------------------
        # FEATURE-BASED BASELINES
        # -------------------------------------------------------
        print("\n-- Feature-Based Baselines --")

        scaler = StandardScaler()
        F_tr_norm = scaler.fit_transform(F_train)
        F_te_norm = scaler.transform(F_test)

        # Isolation Forest (multiple seeds)
        if_aucs = []
        for seed in seeds:
            try:
                iso = IsolationForest(n_estimators=200, contamination='auto', random_state=seed, n_jobs=-1)
                iso.fit(F_tr_norm)
                scores = -iso.score_samples(F_te_norm)  # higher = more anomalous
                m = compute_metrics(y_test, scores)
                if_aucs.append(m['auroc'])
            except Exception as e:
                print(f"    IsoForest seed={seed} failed: {e}")
        if if_aucs:
            src_results['isolation_forest'] = {
                'auroc_mean': float(np.mean(if_aucs)),
                'auroc_std': float(np.std(if_aucs)),
                'seeds': seeds,
            }
            print(f"  IsolationForest: AUROC={np.mean(if_aucs):.4f}±{np.std(if_aucs):.4f}")

        # One-Class SVM (single seed, slow)
        try:
            ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
            ocsvm.fit(F_tr_norm)
            scores = -ocsvm.score_samples(F_te_norm)
            src_results['ocsvm'] = compute_metrics(y_test, scores)
            print(f"  OCSVM: AUROC={src_results['ocsvm']['auroc']:.4f}")
        except Exception as e:
            print(f"  OCSVM failed: {e}")
            src_results['ocsvm'] = {'error': str(e)}

        # LOF
        try:
            lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
            lof.fit(F_tr_norm)
            scores = -lof.score_samples(F_te_norm)
            src_results['lof'] = compute_metrics(y_test, scores)
            print(f"  LOF: AUROC={src_results['lof']['auroc']:.4f}")
        except Exception as e:
            print(f"  LOF failed: {e}")
            src_results['lof'] = {'error': str(e)}

        # Mahalanobis distance
        try:
            from sklearn.covariance import EmpiricalCovariance
            cov = EmpiricalCovariance(assume_centered=False)
            cov.fit(F_tr_norm)
            scores = cov.mahalanobis(F_te_norm)
            src_results['mahalanobis'] = compute_metrics(y_test, scores)
            print(f"  Mahalanobis: AUROC={src_results['mahalanobis']['auroc']:.4f}")
        except Exception as e:
            print(f"  Mahalanobis failed: {e}")
            src_results['mahalanobis'] = {'error': str(e)}

        # PCA reconstruction error
        try:
            n_comp = min(10, F_tr_norm.shape[1], F_tr_norm.shape[0] - 1)
            pca = PCA(n_components=n_comp)
            pca.fit(F_tr_norm)
            F_te_recon = pca.inverse_transform(pca.transform(F_te_norm))
            scores = np.mean((F_te_norm - F_te_recon) ** 2, axis=1)
            src_results['pca_recon'] = compute_metrics(y_test, scores)
            print(f"  PCA recon: AUROC={src_results['pca_recon']['auroc']:.4f}")
        except Exception as e:
            print(f"  PCA recon failed: {e}")
            src_results['pca_recon'] = {'error': str(e)}

        # -------------------------------------------------------
        # DEEP LEARNING: AUTOENCODER
        # -------------------------------------------------------
        print("\n-- Autoencoder (Deep) --")

        ae_aucs = []
        for seed in seeds:
            try:
                model = train_autoencoder(X_train, epochs=30, lr=1e-3, device=device, seed=seed)
                scores = get_ae_scores(model, X_test, device=device)
                m = compute_metrics(y_test, scores)
                ae_aucs.append(m['auroc'])
                print(f"  AE seed={seed}: AUROC={m['auroc']:.4f}")
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  AE seed={seed} failed: {e}")
                import traceback; traceback.print_exc()

        if ae_aucs:
            src_results['autoencoder'] = {
                'auroc_mean': float(np.mean(ae_aucs)),
                'auroc_std': float(np.std(ae_aucs)),
                'seeds': seeds,
            }
            print(f"  Autoencoder: AUROC={np.mean(ae_aucs):.4f}±{np.std(ae_aucs):.4f}")

        all_source_results[source] = src_results

        # Save incrementally (crash recovery)
        results['by_source'] = all_source_results
        results['_meta'] = {
            'timestamp': datetime.now().isoformat(),
            'seeds': seeds,
        }
        with open(RESULTS_PATH, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nPartial results saved.")

    # Final save
    results['by_source'] = all_source_results
    results['_meta'] = {
        'timestamp': datetime.now().isoformat(),
        'seeds': seeds,
        'sources': list(all_source_results.keys()),
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFinal results saved to {RESULTS_PATH}")

    # Summary
    print("\n=== ANOMALY DETECTION SUMMARY ===")
    for src, src_res in all_source_results.items():
        print(f"\n  Source: {src}")
        print(f"  {'Method':<30} {'AUROC':>8} {'AUPRC':>8}")
        print(f"  {'-'*50}")
        for method, metrics in src_res.items():
            auroc = metrics.get('auroc_mean', metrics.get('auroc', np.nan))
            auprc = metrics.get('auprc', np.nan)
            if isinstance(auroc, float):
                std = metrics.get('auroc_std', '')
                std_str = f"±{std:.3f}" if isinstance(std, float) else ''
                print(f"  {method:<30} {auroc:>8.4f}{std_str}  {auprc if isinstance(auprc, float) else '?':>8}")

    return results


if __name__ == '__main__':
    run_anomaly_baselines()
