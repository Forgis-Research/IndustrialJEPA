"""
Download MSL, SMAP, and PSM datasets for MTS-JEPA replication.
SWaT requires registration — skipped unless already available.
"""
import os
import urllib.request
import zipfile
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def download_file(url, dest):
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return
    print(f"  Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)

def download_smap_msl():
    """Download SMAP and MSL from the telemanom repo."""
    base = "https://s3-us-west-2.amazonaws.com/telemanom/data"
    for dataset in ["SMAP", "MSL"]:
        ds_dir = os.path.join(DATA_DIR, dataset)
        os.makedirs(ds_dir, exist_ok=True)

        for split in ["train", "test"]:
            url = f"{base}/{split}/{dataset.lower()}_{split}.npy" if False else None

    # Use the standard processed NPY files from the telemanom repo
    # These are hosted as individual channel files; we use the combined versions
    import subprocess

    repo_dir = os.path.join(DATA_DIR, "telemanom_repo")
    if not os.path.exists(repo_dir):
        print("Cloning telemanom repository...")
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/khundman/telemanom.git",
            repo_dir
        ], check=True)

    # The telemanom repo has individual channel NPY files
    # We need to also get the labels
    labels_url = "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"
    labels_path = os.path.join(DATA_DIR, "labeled_anomalies.csv")
    download_file(labels_url, labels_path)

    # Process SMAP and MSL from telemanom repo
    for dataset in ["SMAP", "MSL"]:
        ds_dir = os.path.join(DATA_DIR, dataset)
        os.makedirs(ds_dir, exist_ok=True)

        repo_data = os.path.join(repo_dir, "data")

        # Concatenate individual channel files into combined arrays
        import csv
        with open(labels_path, 'r') as f:
            reader = csv.DictReader(f)
            channels = [row for row in reader if row['spacecraft'] == dataset]

        train_arrays = []
        test_arrays = []
        test_labels_list = []
        channel_names = []

        for ch_info in channels:
            chan = ch_info['chan_id']
            train_file = os.path.join(repo_data, "train", f"{chan}.npy")
            test_file = os.path.join(repo_data, "test", f"{chan}.npy")

            if os.path.exists(train_file) and os.path.exists(test_file):
                train_data = np.load(train_file)
                test_data = np.load(test_file)

                # Each channel file is (T, n_features) — typically (T, 25) for SMAP or (T, 55) for MSL
                # But these are per-channel files where column 0 is the telemetry value
                # and remaining columns are commanding data
                train_arrays.append(train_data)
                test_arrays.append(test_data)
                channel_names.append(chan)

                # Parse anomaly labels
                anomaly_sequences = ch_info.get('anomaly_sequences', '[]')
                # anomaly_sequences is like "[[start, end], [start, end]]"
                import ast
                try:
                    seqs = ast.literal_eval(anomaly_sequences)
                except:
                    seqs = []

                labels = np.zeros(len(test_data), dtype=np.int32)
                for seq in seqs:
                    labels[seq[0]:seq[1]+1] = 1
                test_labels_list.append(labels)

        if train_arrays:
            # For SMAP/MSL, the standard benchmark uses individual channels
            # Each channel has its own train/test split
            # We save per-channel data for the standard evaluation
            print(f"  {dataset}: Found {len(train_arrays)} channels")

            # Save channel list
            np.save(os.path.join(ds_dir, "channel_names.npy"), np.array(channel_names))

            # Save per-channel data
            for i, chan in enumerate(channel_names):
                chan_dir = os.path.join(ds_dir, "channels", chan)
                os.makedirs(chan_dir, exist_ok=True)
                np.save(os.path.join(chan_dir, "train.npy"), train_arrays[i])
                np.save(os.path.join(chan_dir, "test.npy"), test_arrays[i])
                np.save(os.path.join(chan_dir, "test_labels.npy"), test_labels_list[i])

            # Also save concatenated versions (standard for entity-level evaluation)
            # For the standard benchmark: concatenate all channels' data along time axis
            # Each channel becomes a separate "entity"

            # Save combined train/test (concatenate along time, keeping all features)
            # Standard approach: use first channel's dimensionality (they all have the same)
            n_features = train_arrays[0].shape[1]

            # Concatenated across channels for entity-level training
            all_train = np.concatenate(train_arrays, axis=0)
            all_test = np.concatenate(test_arrays, axis=0)
            all_labels = np.concatenate(test_labels_list, axis=0)

            np.save(os.path.join(ds_dir, "train.npy"), all_train)
            np.save(os.path.join(ds_dir, "test.npy"), all_test)
            np.save(os.path.join(ds_dir, "test_labels.npy"), all_labels)

            print(f"  {dataset}: train {all_train.shape}, test {all_test.shape}, "
                  f"anomaly rate {all_labels.mean():.4f}")


def download_psm():
    """Download PSM (Pooled Server Metrics) from the RANSynCoders repo."""
    ds_dir = os.path.join(DATA_DIR, "PSM")
    os.makedirs(ds_dir, exist_ok=True)

    # PSM is available from the RANSynCoders GitHub
    base = "https://raw.githubusercontent.com/eBay/RANSynCoders/main/data"

    for fname in ["train.csv", "test.csv", "test_label.csv"]:
        dest = os.path.join(ds_dir, fname)
        if os.path.exists(dest):
            print(f"  Already exists: {dest}")
            continue

        url = f"{base}/{fname}"
        try:
            download_file(url, dest)
        except Exception as e:
            print(f"  Failed to download {fname} from RANSynCoders: {e}")
            # Try alternative source
            alt_base = "https://raw.githubusercontent.com/salesforce/DeepTime/main/datasets/anomaly_detection/PSM"
            try:
                download_file(f"{alt_base}/{fname}", dest)
            except Exception as e2:
                print(f"  Also failed from alternative: {e2}")

    # Process PSM CSVs to numpy
    train_csv = os.path.join(ds_dir, "train.csv")
    test_csv = os.path.join(ds_dir, "test.csv")
    label_csv = os.path.join(ds_dir, "test_label.csv")

    if os.path.exists(train_csv):
        import pandas as pd
        train_df = pd.read_csv(train_csv)
        # Drop timestamp column if present
        if 'timestamp_(min)' in train_df.columns:
            train_df = train_df.drop(columns=['timestamp_(min)'])
        elif train_df.columns[0].lower().startswith('time'):
            train_df = train_df.iloc[:, 1:]

        test_df = pd.read_csv(test_csv)
        if 'timestamp_(min)' in test_df.columns:
            test_df = test_df.drop(columns=['timestamp_(min)'])
        elif test_df.columns[0].lower().startswith('time'):
            test_df = test_df.iloc[:, 1:]

        # Fill NaN with 0 (standard preprocessing for PSM)
        train_data = train_df.values.astype(np.float32)
        test_data = test_df.values.astype(np.float32)
        train_data = np.nan_to_num(train_data, nan=0.0)
        test_data = np.nan_to_num(test_data, nan=0.0)

        label_df = pd.read_csv(label_csv)
        if label_df.shape[1] > 1:
            test_labels = label_df.iloc[:, -1].values.astype(np.int32)
        else:
            test_labels = label_df.values.flatten().astype(np.int32)

        np.save(os.path.join(ds_dir, "train.npy"), train_data)
        np.save(os.path.join(ds_dir, "test.npy"), test_data)
        np.save(os.path.join(ds_dir, "test_labels.npy"), test_labels)

        print(f"  PSM: train {train_data.shape}, test {test_data.shape}, "
              f"anomaly rate {test_labels.mean():.4f}")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("Downloading SMAP and MSL datasets...")
    print("=" * 60)
    download_smap_msl()

    print("\n" + "=" * 60)
    print("Downloading PSM dataset...")
    print("=" * 60)
    download_psm()

    print("\n" + "=" * 60)
    print("Dataset download complete.")
    print("SWaT requires manual registration — skipped.")
    print("=" * 60)
