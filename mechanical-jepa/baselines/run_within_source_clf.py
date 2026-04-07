"""
Phase 2D Extension: Within-source (in-domain) classification baselines.

This is a sanity check — we should see high F1 in-domain to confirm our features work.
If in-domain F1 is also low, there's a bug.

Within-source setup: 80/20 stratified split within each source.
Run for CWRU (easy, saturated) and Paderborn (harder).

Outputs appended to results/classification_baselines.json as 'in_domain' section.
"""

import numpy as np
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from data_utils import load_parquet, proc_native, get_sr
from features import extract_features_batch

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

SEEDS = [42, 123, 456]
RESULTS_PATH = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/baselines/results/classification_baselines.json'


def load_source_data(source_name: str, filter_fn=None):
    """Load signals and labels for a single source."""
    file_map = {
        'cwru': 'bearings/extra_cwru_mfpt.parquet',
        'mfpt': 'bearings/extra_cwru_mfpt.parquet',
        'mafaulda': None,  # multiple shards
        'paderborn': 'bearings/train-00004-of-00005.parquet',
        'ottawa': 'bearings/ottawa_bearings.parquet',
        'femto_health': 'bearings/train-00000-of-00005.parquet',
    }

    if source_name == 'mafaulda':
        dfs = []
        import pandas as pd
        for i in range(8):
            try:
                dfs.append(load_parquet(f'bearings/mafaulda_{i:03d}.parquet'))
            except Exception:
                break
        import pandas as pd
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = load_parquet(file_map[source_name])
        if filter_fn:
            df = filter_fn(df)
        elif source_name in ['cwru', 'mfpt']:
            df = df[df['source_id'] == source_name]
        elif source_name == 'paderborn':
            df = df[df['source_id'] == 'paderborn']

    sigs, labels = [], []
    for _, row in df.iterrows():
        w = proc_native(row)
        if w is not None:
            sigs.append(w)
            labels.append(str(row['fault_type']))
    return sigs, np.array(labels)


def within_source_baseline(source_name: str, verbose=True):
    print(f"\n=== In-Domain: {source_name.upper()} ===")
    sigs, labels = load_source_data(source_name)
    print(f"  {len(sigs)} samples, {len(np.unique(labels))} classes: {np.unique(labels)}")

    if len(sigs) < 10 or len(np.unique(labels)) < 2:
        print(f"  Skipping: insufficient data")
        return None

    F = extract_features_batch(sigs)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    scaler = StandardScaler()

    f1s, accs = [], []
    for seed in SEEDS:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        for tr_idx, te_idx in sss.split(F, y):
            F_tr = scaler.fit_transform(F[tr_idx])
            F_te = scaler.transform(F[te_idx])
            rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
            rf.fit(F_tr, y[tr_idx])
            y_pred = rf.predict(F_te)
            f1s.append(f1_score(y[te_idx], y_pred, average='macro', zero_division=0))
            accs.append(accuracy_score(y[te_idx], y_pred))

    print(f"  RF in-domain: F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}, Acc={np.mean(accs):.4f}")
    return {
        'macro_f1_mean': float(np.mean(f1s)), 'macro_f1_std': float(np.std(f1s)),
        'accuracy_mean': float(np.mean(accs)), 'accuracy_std': float(np.std(accs)),
        'n_samples': len(sigs), 'n_classes': len(np.unique(labels)),
        'class_names': le.classes_.tolist(),
        'seeds': SEEDS, 'method': 'random_forest',
    }


if __name__ == '__main__':
    # Load existing results
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    in_domain_results = {}

    for source in ['cwru', 'paderborn', 'ottawa', 'mafaulda']:
        try:
            res = within_source_baseline(source)
            if res:
                in_domain_results[source] = res
        except Exception as e:
            print(f"  {source} failed: {e}")
            import traceback; traceback.print_exc()

    results['in_domain_rf'] = in_domain_results

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n=== IN-DOMAIN SUMMARY (Random Forest, 80/20 split) ===")
    for src, res in in_domain_results.items():
        print(f"  {src:<15}: F1={res['macro_f1_mean']:.4f}±{res['macro_f1_std']:.4f}, "
              f"Acc={res['accuracy_mean']:.4f}, classes={res['n_classes']}")

    print("\n=== CROSS-DOMAIN vs IN-DOMAIN COMPARISON ===")
    xd_f1 = results.get('random_forest', {}).get('macro_f1_mean', None)
    print(f"  Random Forest cross-domain F1: {xd_f1:.4f}" if xd_f1 else "  Cross-domain: N/A")
    for src, res in in_domain_results.items():
        print(f"  Random Forest in-domain ({src}): F1={res['macro_f1_mean']:.4f}")
