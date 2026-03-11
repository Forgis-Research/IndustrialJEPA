# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Analyze setpoint/effort schema for all FactoryNet datasets.
"""

from datasets import load_dataset
from huggingface_hub import hf_hub_download
import json
import numpy as np

DATASETS = {
    "aursad": "aursad",
    "voraus": "voraus",
    "nasa_milling": "nasa_milling",
    "rh20t": "rh20t",
    "reassemble": "reassemble",
}

def analyze_dataset(name, data_dir):
    print(f"\n{'='*70}")
    print(f"DATASET: {name}")
    print("="*70)

    try:
        ds = load_dataset("Forgis/factorynet-hackathon", data_dir=data_dir, split="train")
        df = ds.to_pandas()

        print(f"Total rows: {len(df):,}")
        print(f"Episodes: {df['episode_id'].nunique()}")

        # Categorize columns
        setpoint_cols = [c for c in df.columns if 'setpoint' in c.lower()]
        effort_cols = [c for c in df.columns if 'effort' in c.lower()]
        feedback_cols = [c for c in df.columns if 'feedback' in c.lower()]
        ctx_cols = [c for c in df.columns if 'ctx' in c.lower()]

        print(f"\nSetpoint columns ({len(setpoint_cols)}):")
        for c in sorted(setpoint_cols):
            sample = df[c].dropna()
            if len(sample) > 0:
                print(f"  {c}: range=[{sample.min():.2f}, {sample.max():.2f}], std={sample.std():.3f}")

        print(f"\nEffort columns ({len(effort_cols)}):")
        for c in sorted(effort_cols):
            sample = df[c].dropna()
            if len(sample) > 0:
                print(f"  {c}: range=[{sample.min():.2f}, {sample.max():.2f}], std={sample.std():.3f}")

        print(f"\nFeedback columns: {len(feedback_cols)}")
        print(f"Context columns: {len(ctx_cols)}")

        # Check for anomaly labels
        if 'ctx_anomaly_label' in df.columns:
            labels = df['ctx_anomaly_label'].value_counts()
            print(f"\nAnomaly labels: {dict(labels)}")

        # Load metadata if available
        try:
            meta_file = f"metadata/{data_dir}_metadata.json"
            if data_dir in ['nasa_milling', 'rh20t', 'reassemble']:
                meta_file = f"{data_dir}/{data_dir}_metadata.json"

            meta_path = hf_hub_download(
                repo_id="Forgis/factorynet-hackathon",
                filename=meta_file,
                repo_type="dataset"
            )
            with open(meta_path) as f:
                metadata = json.load(f)

            if isinstance(metadata, list) and len(metadata) > 0:
                m = metadata[0]
                print(f"\nMachine: {m.get('machine_model', 'unknown')} ({m.get('machine_family', 'unknown')})")
                print(f"Task: {m.get('task_type', 'unknown')}")
                print(f"DOF: {m.get('num_axes', 'unknown')}")
        except Exception as e:
            print(f"\nMetadata: Could not load ({e})")

        return {
            "name": name,
            "rows": len(df),
            "episodes": df['episode_id'].nunique(),
            "setpoint_cols": setpoint_cols,
            "effort_cols": effort_cols,
        }

    except Exception as e:
        print(f"ERROR: {e}")
        return None


def main():
    results = {}
    for name, data_dir in DATASETS.items():
        result = analyze_dataset(name, data_dir)
        if result:
            results[name] = result

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Dataset':<15} {'Rows':>10} {'Episodes':>10} {'Setpoint':>10} {'Effort':>10}")
    print("-"*55)
    for name, r in results.items():
        print(f"{name:<15} {r['rows']:>10,} {r['episodes']:>10} {len(r['setpoint_cols']):>10} {len(r['effort_cols']):>10}")


if __name__ == "__main__":
    main()
