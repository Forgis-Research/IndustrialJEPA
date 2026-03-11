# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Quick analysis of all available datasets for anomaly detection viability.
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig


def analyze_dataset(subset_name: str):
    print(f"\n{'='*60}")
    print(f"DATASET: {subset_name}")
    print("=" * 60)

    try:
        config = FactoryNetConfig(
            dataset_name="Forgis/factorynet-hackathon",
            subset=subset_name,
            window_size=256,
            stride=256,
            normalize=False,
            train_healthy_only=False,
        )

        ds = FactoryNetDataset(config, split="test")
        print(f"Windows: {len(ds)}")

        if len(ds) == 0:
            print("No data!")
            return None

        # Quick stats
        healthy_mags = []
        faulty_mags = []
        fault_types = set()

        for i in range(min(len(ds), 500)):  # Sample for speed
            _, effort, meta = ds[i]
            is_anomaly = meta["is_anomaly"]
            fault_type = meta["fault_type"]
            fault_types.add(fault_type)

            mag = np.abs(effort.numpy()).mean()
            if is_anomaly:
                faulty_mags.append(mag)
            else:
                healthy_mags.append(mag)

        print(f"Fault types: {fault_types}")
        print(f"Healthy: {len(healthy_mags)}, Faulty: {len(faulty_mags)}")

        if len(faulty_mags) == 0:
            print("No anomalies in test set!")
            return None

        healthy_mags = np.array(healthy_mags)
        faulty_mags = np.array(faulty_mags)

        effect = (faulty_mags.mean() - healthy_mags.mean()) / healthy_mags.std()
        print(f"Effect size: {effect:+.2f} std")

        # AUC
        labels = [0] * len(healthy_mags) + [1] * len(faulty_mags)
        scores = list(healthy_mags) + list(faulty_mags)
        auc = roc_auc_score(labels, scores)
        print(f"AUC-ROC (magnitude): {auc:.4f}")

        return {"effect": effect, "auc": auc, "n_healthy": len(healthy_mags), "n_faulty": len(faulty_mags)}

    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    datasets = ["aursad", "voraus-ad", "nasa-milling", "rh20t", "reassemble"]

    results = {}
    for ds in datasets:
        result = analyze_dataset(ds)
        if result:
            results[ds] = result

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Dataset':<15} {'Effect Size':<15} {'AUC-ROC':<10} {'Detectable?':<12}")
    print("-" * 52)

    for ds, r in results.items():
        detectable = "YES" if r["auc"] > 0.7 else ("MAYBE" if r["auc"] > 0.6 else "NO")
        print(f"{ds:<15} {r['effect']:+.2f} std       {r['auc']:.4f}     {detectable}")


if __name__ == "__main__":
    main()
