"""
Phase 2: Anomaly Classification Baselines

Cross-domain setup: train on CWRU+MAFAULDA+SEU, test on Ottawa+Paderborn.
This tests how well feature-based methods generalize across machines/labs.

Methods:
- Trivial: majority class, random stratified, nearest centroid
- Feature-based (primary): LR, RF, XGBoost, SVM on handcrafted features
- Deep: 1D CNN, 1D ResNet (trained on CWRU+SEU only — longer signals)

Metrics: Macro F1 (primary), accuracy
Seeds: 42, 123, 456

Outputs:
- results/classification_baselines.json
"""

import numpy as np
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from data_utils import (load_classification_data, load_classification_data_deep,
                         DEEP_WINDOW_LEN)
from features import extract_features_batch, N_FEATURES

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import f1_score, accuracy_score
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEEDS = [42, 123, 456]
RESULTS_PATH = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/baselines/results/classification_baselines.json'
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)


# ============================================================
# DEEP MODELS
# ============================================================

class CNN1D(nn.Module):
    """1D CNN for fixed-window vibration classification."""
    def __init__(self, n_classes: int, window_len: int = DEEP_WINDOW_LEN):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=8, padding=28),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=32, stride=4, padding=14),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=16, stride=4, padding=6),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.head(self.conv(x.unsqueeze(1)))


class ResNet1D(nn.Module):
    """1D ResNet with 3 residual blocks."""
    def __init__(self, n_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        configs = [(64, 128, 4), (128, 256, 4), (256, 256, 4)]
        for in_ch, out_ch, stride in configs:
            self.blocks.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1),
                nn.BatchNorm1d(out_ch), nn.ReLU(),
                nn.Conv1d(out_ch, out_ch, 3, stride=1, padding=1),
                nn.BatchNorm1d(out_ch),
            ))
            self.shortcuts.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm1d(out_ch),
            ))
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4, 256),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.stem(x.unsqueeze(1))
        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = self.relu(block(x) + shortcut(x))
        return self.head(self.pool(x))


def train_deep_model(model, X_train, y_enc, n_classes, epochs=40, lr=1e-3, device='cuda', seed=42):
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_enc, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1}/{epochs}, loss={total_loss/len(loader):.4f}")
    return model


def eval_deep_model(model, X_test, device='cuda'):
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32)), batch_size=256)
    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(model(xb.to(device)).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


# ============================================================
# MAIN
# ============================================================

def run_classification_baselines():
    results = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ============================================================
    # FEATURE-BASED BASELINES
    # ============================================================
    print("\n=== Loading Feature Data ===")
    t0 = time.time()
    X_tr_sigs, y_train, X_te_sigs, y_test, src_train, src_test = load_classification_data(verbose=True)
    print(f"Load time: {time.time()-t0:.1f}s")

    # Encode labels (handle missing test classes)
    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_test]))
    y_tr_enc = le.transform(y_train)
    y_te_enc = le.transform(y_test)
    n_classes = len(le.classes_)
    print(f"Classes ({n_classes}): {le.classes_.tolist()}")

    print("\n=== Extracting Features ===")
    t0 = time.time()
    F_train = extract_features_batch(X_tr_sigs, verbose=True)
    F_test = extract_features_batch(X_te_sigs, verbose=True)
    print(f"Feature extraction: {time.time()-t0:.1f}s")
    print(f"F_train: {F_train.shape}, F_test: {F_test.shape}")

    scaler = StandardScaler()
    F_tr = scaler.fit_transform(F_train)
    F_te = scaler.transform(F_test)

    # Save metadata
    results['_meta'] = {
        'timestamp': datetime.now().isoformat(),
        'seeds': SEEDS,
        'n_train': len(X_tr_sigs),
        'n_test': len(X_te_sigs),
        'n_classes': n_classes,
        'class_names': le.classes_.tolist(),
        'train_sources': list(np.unique(src_train).tolist()),
        'test_sources': list(np.unique(src_test).tolist()),
        'n_features': N_FEATURES,
        'note': 'Cross-domain: train on CWRU+MAFAULDA+SEU, test on Ottawa+Paderborn',
    }

    # -------------------------------------------------------
    # TRIVIAL BASELINES
    # -------------------------------------------------------
    print("\n=== Trivial Baselines ===")

    # Majority class
    majority = np.bincount(y_tr_enc).argmax()
    y_maj = np.full(len(y_te_enc), majority)
    maj_f1 = f1_score(y_te_enc, y_maj, average='macro', zero_division=0)
    maj_acc = accuracy_score(y_te_enc, y_maj)
    print(f"Majority class: F1={maj_f1:.4f}, Acc={maj_acc:.4f}")
    results['trivial_majority'] = {'macro_f1': float(maj_f1), 'accuracy': float(maj_acc)}

    # Random stratified
    rand_f1s, rand_accs = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        probs = np.bincount(y_tr_enc, minlength=n_classes) / len(y_tr_enc)
        y_rand = rng.choice(n_classes, size=len(y_te_enc), p=probs)
        rand_f1s.append(f1_score(y_te_enc, y_rand, average='macro', zero_division=0))
        rand_accs.append(accuracy_score(y_te_enc, y_rand))
    print(f"Random stratified: F1={np.mean(rand_f1s):.4f}±{np.std(rand_f1s):.4f}")
    results['trivial_random'] = {
        'macro_f1_mean': float(np.mean(rand_f1s)), 'macro_f1_std': float(np.std(rand_f1s)),
        'accuracy_mean': float(np.mean(rand_accs)), 'accuracy_std': float(np.std(rand_accs)),
        'seeds': SEEDS,
    }

    # Nearest centroid
    try:
        nc = NearestCentroid()
        nc.fit(F_tr, y_tr_enc)
        y_nc = nc.predict(F_te)
        nc_f1 = f1_score(y_te_enc, y_nc, average='macro', zero_division=0)
        nc_acc = accuracy_score(y_te_enc, y_nc)
        print(f"Nearest Centroid: F1={nc_f1:.4f}, Acc={nc_acc:.4f}")
        results['trivial_nearest_centroid'] = {'macro_f1': float(nc_f1), 'accuracy': float(nc_acc)}
    except Exception as e:
        print(f"Nearest Centroid failed: {e}")
        results['trivial_nearest_centroid'] = {'error': str(e)}

    # -------------------------------------------------------
    # FEATURE-BASED BASELINES
    # -------------------------------------------------------
    print("\n=== Feature-Based Baselines ===")

    def run_with_seeds(name, make_clf, seeds=SEEDS):
        f1s, accs = [], []
        for seed in seeds:
            try:
                clf = make_clf(seed)
                clf.fit(F_tr, y_tr_enc)
                y_p = clf.predict(F_te)
                f1s.append(f1_score(y_te_enc, y_p, average='macro', zero_division=0))
                accs.append(accuracy_score(y_te_enc, y_p))
            except Exception as e:
                print(f"  {name} seed={seed} error: {e}")
        if f1s:
            print(f"{name}: F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}, Acc={np.mean(accs):.4f}±{np.std(accs):.4f}")
            return {
                'macro_f1_mean': float(np.mean(f1s)), 'macro_f1_std': float(np.std(f1s)),
                'accuracy_mean': float(np.mean(accs)), 'accuracy_std': float(np.std(accs)),
                'all_f1s': [float(x) for x in f1s],
                'seeds': seeds,
            }
        return {'error': 'all seeds failed'}

    results['lr'] = run_with_seeds('Logistic Regression',
        lambda s: LogisticRegression(max_iter=2000, random_state=s, C=1.0, multi_class='auto'))
    results['random_forest'] = run_with_seeds('Random Forest',
        lambda s: RandomForestClassifier(n_estimators=200, random_state=s, n_jobs=-1))
    results['xgboost'] = run_with_seeds('XGBoost',
        lambda s: xgb.XGBClassifier(n_estimators=200, random_state=s, n_jobs=-1,
                                      verbosity=0, eval_metric='mlogloss'))
    results['svm_rbf'] = run_with_seeds('SVM RBF',
        lambda s: SVC(kernel='rbf', C=10.0, gamma='scale'),
        seeds=SEEDS[:1])  # SVM is slow

    # Incremental save
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Feature results saved.")

    # -------------------------------------------------------
    # DEEP LEARNING BASELINES
    # -------------------------------------------------------
    print("\n=== Deep Learning Baselines ===")
    print("Loading fixed-length windows for deep models...")
    try:
        X_tr_deep, y_tr_deep, X_te_deep, y_te_deep, src_tr_d, src_te_d = load_classification_data_deep(verbose=True)

        if len(X_tr_deep) < 10 or len(X_te_deep) < 5:
            print("Insufficient deep data, skipping deep baselines")
            results['deep_note'] = 'Skipped: insufficient data after filtering for window_len'
        else:
            # Re-encode labels for deep subset
            le_deep = LabelEncoder()
            le_deep.fit(np.concatenate([y_tr_deep, y_te_deep]))
            y_tr_d = le_deep.transform(y_tr_deep)
            y_te_d = le_deep.transform(y_te_deep)
            n_cls_deep = len(le_deep.classes_)
            print(f"Deep classes ({n_cls_deep}): {le_deep.classes_.tolist()}")
            print(f"Deep train: {X_tr_deep.shape}, test: {X_te_deep.shape}")

            def run_deep(model_name, model_class):
                f1s, accs = [], []
                for seed in SEEDS:
                    try:
                        model = model_class(n_cls_deep)
                        t0 = time.time()
                        model = train_deep_model(model, X_tr_deep, y_tr_d, n_cls_deep,
                                                  epochs=40, device=device, seed=seed)
                        y_pred = eval_deep_model(model, X_te_deep, device)
                        f1 = f1_score(y_te_d, y_pred, average='macro', zero_division=0)
                        acc = accuracy_score(y_te_d, y_pred)
                        f1s.append(f1)
                        accs.append(acc)
                        print(f"  seed={seed}: F1={f1:.4f}, time={time.time()-t0:.0f}s")
                        del model
                        torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"  {model_name} seed={seed} failed: {e}")
                        import traceback; traceback.print_exc()
                if f1s:
                    print(f"{model_name}: F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")
                    return {
                        'macro_f1_mean': float(np.mean(f1s)), 'macro_f1_std': float(np.std(f1s)),
                        'accuracy_mean': float(np.mean(accs)), 'accuracy_std': float(np.std(accs)),
                        'seeds': SEEDS,
                        'n_train': len(X_tr_deep), 'n_test': len(X_te_deep),
                        'deep_classes': le_deep.classes_.tolist(),
                    }
                return {'error': 'all seeds failed'}

            print("\n-- CNN 1D --")
            results['cnn_1d'] = run_deep('CNN 1D', CNN1D)
            with open(RESULTS_PATH, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            print("\n-- ResNet 1D --")
            results['resnet_1d'] = run_deep('ResNet 1D', ResNet1D)

    except Exception as e:
        print(f"Deep baselines failed: {e}")
        import traceback; traceback.print_exc()
        results['deep_note'] = f'Error: {str(e)}'

    # -------------------------------------------------------
    # FINAL SAVE AND SUMMARY
    # -------------------------------------------------------
    results['_meta']['timestamp_end'] = datetime.now().isoformat()
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFinal results saved to {RESULTS_PATH}")

    print("\n=== CLASSIFICATION SUMMARY ===")
    print(f"{'Method':<30} {'Macro F1':>12} {'Accuracy':>10}")
    print("-" * 56)
    for k, v in results.items():
        if k.startswith('_') or k == 'deep_note':
            continue
        f1 = v.get('macro_f1_mean', v.get('macro_f1', None))
        acc = v.get('accuracy_mean', v.get('accuracy', None))
        if f1 is not None:
            std = v.get('macro_f1_std', '')
            std_str = f"±{std:.4f}" if isinstance(std, float) else ''
            print(f"{k:<30} {f1:>8.4f}{std_str:<8}  {acc if isinstance(acc, float) else '?':>8.4f}")

    return results


if __name__ == '__main__':
    run_classification_baselines()
