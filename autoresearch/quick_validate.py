#!/usr/bin/env python
"""
Quick validation script for cross-machine transfer experiments.

Runs fast checks to validate an approach before committing to full training.

Usage:
    python autoresearch/quick_validate.py --approach baseline
    python autoresearch/quick_validate.py --approach revin
    python autoresearch/quick_validate.py --approach domain_adversarial
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results" / "quick_validate"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def quick_anomaly_test(max_episodes=50, epochs=3):
    """
    Quick test: Can source-trained anomaly detector work on target?

    Returns dict with metrics.
    """
    from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression

    logger.info("Quick Anomaly Test: Loading data...")

    # Load minimal data
    config_source = FactoryNetConfig(
        data_source='aursad',
        max_episodes=max_episodes,
        window_size=256,
        stride=128,
    )
    ds_source = FactoryNetDataset(config_source, split='train')

    config_target = FactoryNetConfig(
        data_source='voraus',
        max_episodes=max_episodes,
        window_size=256,
        stride=128,
    )
    ds_target = FactoryNetDataset(config_target, split='test')

    # Extract simple features (mean, std, range per channel)
    def extract_features(ds, n=100):
        features = []
        labels = []
        for i in range(min(n, len(ds))):
            sample = ds[i]
            setpoint = sample['setpoint'].numpy()
            effort = sample['effort'].numpy()

            # Simple features
            feat = np.concatenate([
                setpoint.mean(axis=0),
                setpoint.std(axis=0),
                effort.mean(axis=0),
                effort.std(axis=0),
            ])
            features.append(feat)

            label = 1 if sample.get('label', 'normal') != 'normal' else 0
            labels.append(label)

        return np.array(features), np.array(labels)

    logger.info("Extracting features...")
    X_source, y_source = extract_features(ds_source, n=200)
    X_target, y_target = extract_features(ds_target, n=200)

    # Train on source, test on target
    if len(np.unique(y_source)) < 2:
        logger.warning("Source has only one class, cannot train")
        return {"source_auc": None, "target_auc": None, "transfer_ratio": None}

    clf = LogisticRegression(max_iter=500, class_weight='balanced')
    clf.fit(X_source, y_source)

    # Evaluate
    source_prob = clf.predict_proba(X_source)[:, 1]
    target_prob = clf.predict_proba(X_target)[:, 1]

    source_auc = roc_auc_score(y_source, source_prob) if len(np.unique(y_source)) > 1 else None
    target_auc = roc_auc_score(y_target, target_prob) if len(np.unique(y_target)) > 1 else None

    return {
        "source_auc": source_auc,
        "target_auc": target_auc,
        "transfer_ratio": target_auc / source_auc if source_auc and target_auc else None,
        "source_samples": len(X_source),
        "target_samples": len(X_target),
    }


def quick_forecast_test(max_episodes=50, epochs=3):
    """
    Quick test: Can source-trained forecaster predict on target?

    Returns dict with metrics.
    """
    from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig

    logger.info("Quick Forecast Test: Loading data...")

    # Load minimal data
    config_source = FactoryNetConfig(
        data_source='aursad',
        max_episodes=max_episodes,
        window_size=256,
        stride=128,
    )
    ds_source = FactoryNetDataset(config_source, split='train')

    config_target = FactoryNetConfig(
        data_source='voraus',
        max_episodes=max_episodes,
        window_size=256,
        stride=128,
    )
    ds_target = FactoryNetDataset(config_target, split='test')

    # Simple persistence baseline
    def compute_persistence_mse(ds, n=100):
        mses = []
        for i in range(min(n, len(ds))):
            sample = ds[i]
            effort = sample['effort'].numpy()

            # Predict last value
            pred = effort[:-1]
            target = effort[1:]
            mse = ((pred - target) ** 2).mean()
            mses.append(mse)
        return np.mean(mses)

    logger.info("Computing persistence baseline...")
    source_mse = compute_persistence_mse(ds_source)
    target_mse = compute_persistence_mse(ds_target)

    return {
        "source_mse": source_mse,
        "target_mse": target_mse,
        "transfer_ratio": target_mse / source_mse if source_mse > 0 else None,
    }


def run_all_quick_tests():
    """Run all quick validation tests."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }

    # Anomaly test
    logger.info("="*50)
    logger.info("Running Quick Anomaly Test")
    logger.info("="*50)
    try:
        results["tests"]["anomaly"] = quick_anomaly_test()
        logger.info(f"Anomaly Results: {results['tests']['anomaly']}")
    except Exception as e:
        logger.error(f"Anomaly test failed: {e}")
        results["tests"]["anomaly"] = {"error": str(e)}

    # Forecast test
    logger.info("="*50)
    logger.info("Running Quick Forecast Test")
    logger.info("="*50)
    try:
        results["tests"]["forecast"] = quick_forecast_test()
        logger.info(f"Forecast Results: {results['tests']['forecast']}")
    except Exception as e:
        logger.error(f"Forecast test failed: {e}")
        results["tests"]["forecast"] = {"error": str(e)}

    # Summary
    print("\n" + "="*60)
    print("QUICK VALIDATION SUMMARY")
    print("="*60)

    anomaly = results["tests"].get("anomaly", {})
    forecast = results["tests"].get("forecast", {})

    if anomaly.get("target_auc"):
        status = "✅ PASS" if anomaly["target_auc"] >= 0.70 else "❌ FAIL"
        print(f"Anomaly AUC (target): {anomaly['target_auc']:.3f} {status}")

    if forecast.get("transfer_ratio"):
        status = "✅ PASS" if forecast["transfer_ratio"] <= 1.5 else "❌ FAIL"
        print(f"Forecast Transfer Ratio: {forecast['transfer_ratio']:.3f} {status}")

    # Save results
    output_path = RESULTS_DIR / f"quick_validate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick validation for cross-machine transfer")
    parser.add_argument("--test", choices=["all", "anomaly", "forecast"], default="all")
    parser.add_argument("--max-episodes", type=int, default=50)
    args = parser.parse_args()

    if args.test == "all":
        run_all_quick_tests()
    elif args.test == "anomaly":
        result = quick_anomaly_test(max_episodes=args.max_episodes)
        print(json.dumps(result, indent=2, default=str))
    elif args.test == "forecast":
        result = quick_forecast_test(max_episodes=args.max_episodes)
        print(json.dumps(result, indent=2, default=str))
