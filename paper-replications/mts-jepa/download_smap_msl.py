"""
Download SMAP and MSL datasets.
Uses the standard pre-processed NPY files from ServerMachineDataset / OmniAnomaly repos.
"""
import os
import subprocess
import numpy as np
import ast
import csv

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def download_smap_msl():
    """Download SMAP and MSL using the OmniAnomaly preprocessed data."""
    repo_dir = os.path.join(DATA_DIR, "omnianomaly_repo")

    if not os.path.exists(os.path.join(repo_dir, "README.md")):
        print("Cloning OmniAnomaly repository (has preprocessed SMAP/MSL)...")
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/NetManAIOps/OmniAnomaly.git",
            repo_dir
        ], check=True, capture_output=True)

    # Check for ServerMachineDataset structure
    smd_dir = os.path.join(repo_dir, "ServerMachineDataset")
    if os.path.exists(smd_dir):
        print(f"Found ServerMachineDataset at {smd_dir}")

    # The standard approach is to use the preprocessed .pkl or .npy files
    # OmniAnomaly provides get_data.py that downloads the processed arrays
    # Let's check what preprocessing scripts exist
    processed_dir = os.path.join(repo_dir, "processed")
    if os.path.exists(processed_dir):
        print(f"Found processed directory: {os.listdir(processed_dir)}")

    # Alternative: use direct download of the standard benchmark files
    # These are commonly hosted on Google Drive or academic servers
    # Let's build from the telemanom channel files instead

    labels_path = os.path.join(DATA_DIR, "labeled_anomalies.csv")

    for dataset_name in ["SMAP", "MSL"]:
        ds_dir = os.path.join(DATA_DIR, dataset_name)
        os.makedirs(ds_dir, exist_ok=True)

        # Check if already processed
        if os.path.exists(os.path.join(ds_dir, "train.npy")):
            arr = np.load(os.path.join(ds_dir, "train.npy"))
            if arr.ndim == 2:
                print(f"{dataset_name} already processed: {arr.shape}")
                continue

        # Parse channel info from labeled_anomalies.csv
        with open(labels_path, 'r') as f:
            reader = csv.DictReader(f)
            channels = [row for row in reader if row['spacecraft'] == dataset_name]

        print(f"\n{dataset_name}: {len(channels)} channels")

        # For the standard SMAP/MSL benchmark, each entity is a channel
        # The standard approach concatenates all channels along time for pre-training
        # but evaluates per-channel for anomaly detection

        # We need to get the actual data files. The telemanom repo requires
        # running the model to download data. Let's use an alternative.

        # Try downloading from the standard academic mirror
        try:
            import urllib.request
            base_url = "https://s3-us-west-2.amazonaws.com/telemanom/data"

            all_train = []
            all_test = []
            all_labels_arr = []

            for ch_info in channels:
                chan_id = ch_info['chan_id']

                train_url = f"{base_url}/train/{chan_id}.npy"
                test_url = f"{base_url}/test/{chan_id}.npy"

                train_path = os.path.join(ds_dir, f"train_{chan_id}.npy")
                test_path = os.path.join(ds_dir, f"test_{chan_id}.npy")

                if not os.path.exists(train_path):
                    urllib.request.urlretrieve(train_url, train_path)
                if not os.path.exists(test_path):
                    urllib.request.urlretrieve(test_url, test_path)

                train_ch = np.load(train_path).astype(np.float32)
                test_ch = np.load(test_path).astype(np.float32)

                # Parse anomaly labels for this channel
                anomaly_sequences = ch_info.get('anomaly_sequences', '[]')
                try:
                    seqs = ast.literal_eval(anomaly_sequences)
                except:
                    seqs = []

                labels = np.zeros(len(test_ch), dtype=np.int32)
                num_values = int(ch_info.get('num_values', len(test_ch)))
                for seq in seqs:
                    start, end = seq[0], seq[1]
                    labels[start:min(end+1, len(labels))] = 1

                all_train.append(train_ch)
                all_test.append(test_ch)
                all_labels_arr.append(labels)

                print(f"  {chan_id}: train {train_ch.shape}, test {test_ch.shape}")

            # Concatenate all channels along time axis
            # Standard entity-level benchmark: each channel is separate
            # For MTS-JEPA: they concatenate along time for pre-training,
            # but this creates a single long multivariate series

            # Actually, SMAP/MSL channels have the SAME dimensionality
            # (25 for SMAP, 55 for MSL) — they're different sensor entities
            # The standard approach treats the ENTIRE dataset as one entity
            combined_train = np.concatenate(all_train, axis=0)
            combined_test = np.concatenate(all_test, axis=0)
            combined_labels = np.concatenate(all_labels_arr, axis=0)

            np.save(os.path.join(ds_dir, "train.npy"), combined_train)
            np.save(os.path.join(ds_dir, "test.npy"), combined_test)
            np.save(os.path.join(ds_dir, "test_labels.npy"), combined_labels)

            print(f"\n  {dataset_name} combined: train {combined_train.shape}, "
                  f"test {combined_test.shape}, anomaly rate {combined_labels.mean():.4f}")

        except Exception as e:
            print(f"  Failed to download {dataset_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    download_smap_msl()
