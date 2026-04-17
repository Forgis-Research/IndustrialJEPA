#!/usr/bin/env python3
"""
Fix v2_train config on HuggingFace:
1. Re-normalize: compute norm on valid_length portion only, then zero-pad
2. Assign source-disjoint train/val/test splits per V2_TRAINING_SPEC
3. Add sample_weight field for source rebalancing
4. Upload as corrected v2_train with proper HF splits

Streams data to avoid filling disk. Processes in chunks and uploads per-split.
"""

import os
import sys
import json
import tempfile
import shutil

# Fix libstdc++ issue: preload the conda version
os.environ['LD_PRELOAD'] = '/opt/conda/lib/libstdc++.so.6'

import numpy as np
from collections import Counter, defaultdict
from datasets import load_dataset, Dataset, DatasetDict

DATASET_ID = "Forgis/Mechanical-Components"
WINDOW_LEN = 16384

# Source-disjoint split assignments per V2_TRAINING_SPEC.md
# SCA is special: split between train and test
SPLIT_MAP = {
    'cwru': 'train',
    'mfpt': 'train',
    'femto': 'train',
    'xjtu_sy': 'train',
    'ims': 'train',
    'oedi': 'train',
    'phm2009': 'train',
    'mcc5_thu': 'train',
    'seu': 'train',
    'mafaulda': 'train',
    'vbl_va001': 'train',
    # sca_pulpmill: split by episode (see below)
    'paderborn': 'validation',
    'ottawa_bearing': 'validation',
    'mendeley_bearing': 'test',
    # sca_pulpmill test portion handled below
}


def assign_split(source_id, episode_id=None, sca_test_episodes=None):
    """Assign split based on source_id. SCA is split by episode."""
    if source_id == 'sca_pulpmill':
        if sca_test_episodes and episode_id in sca_test_episodes:
            return 'test'
        return 'train'
    return SPLIT_MAP.get(source_id, 'train')


def fix_normalization(signal_arr, valid_length):
    """
    Re-normalize: compute stats on valid portion only, normalize, then zero-pad.
    Returns (fixed_signal, raw_stats_dict).
    """
    sig = np.array(signal_arr, dtype=np.float32)
    vl = min(valid_length, len(sig))
    valid = sig[:vl]

    # Compute raw stats on valid portion
    raw_mean = float(np.mean(valid))
    raw_std = float(np.std(valid))
    raw_max_abs = float(np.max(np.abs(valid)))

    # Instance normalize valid portion
    if raw_std > 1e-10:
        valid_normed = (valid - raw_mean) / raw_std
    else:
        valid_normed = valid - raw_mean

    # Build output: normalized valid + zero padding
    output = np.zeros(WINDOW_LEN, dtype=np.float32)
    output[:vl] = valid_normed

    raw_stats = {
        'mean': raw_mean,
        'std': raw_std,
        'max_abs': raw_max_abs,
    }
    return output.tolist(), raw_stats


def main():
    print("=" * 70)
    print("FIX v2_train: normalization + splits + sample_weight")
    print("=" * 70)

    # Step 1: First pass — collect SCA episode IDs to decide train/test split
    print("\nPass 1: Collecting SCA episode IDs...")
    sca_episodes = set()
    ds_stream = load_dataset(DATASET_ID, "v2_train", split="train", streaming=True)
    for row in ds_stream:
        if row.get('source_id') == 'sca_pulpmill':
            ep = row.get('episode_id')
            if ep:
                sca_episodes.add(ep)

    sca_episodes = sorted(sca_episodes)
    print(f"  SCA episodes: {sca_episodes}")

    # Reserve ~30% of SCA episodes for test
    n_test = max(1, len(sca_episodes) * 3 // 10)
    sca_test_episodes = set(sca_episodes[-n_test:])  # Last N episodes for test
    print(f"  SCA train episodes: {sorted(set(sca_episodes) - sca_test_episodes)}")
    print(f"  SCA test episodes: {sorted(sca_test_episodes)}")

    # Step 2: Count samples per source (for sample_weight)
    print("\nPass 2: Counting source distribution...")
    source_counts = Counter()
    ds_stream = load_dataset(DATASET_ID, "v2_train", split="train", streaming=True)
    for row in ds_stream:
        source_counts[row.get('source_id', 'unknown')] += 1

    total = sum(source_counts.values())
    n_sources = len(source_counts)
    # Weight = (total / n_sources) / source_count — makes each source contribute equally
    source_weights = {
        src: (total / n_sources) / cnt
        for src, cnt in source_counts.items()
    }
    print(f"  Total: {total}, Sources: {n_sources}")
    for src, cnt in source_counts.most_common():
        print(f"    {src:20s}: {cnt:>6} (weight={source_weights[src]:.4f})")

    # Step 3: Stream, fix, and collect into split buckets
    print("\nPass 3: Streaming, fixing normalization, assigning splits...")
    split_data = defaultdict(list)
    ds_stream = load_dataset(DATASET_ID, "v2_train", split="train", streaming=True)

    count = 0
    norm_fixes = 0
    for row in ds_stream:
        count += 1
        if count % 5000 == 0:
            print(f"  ...{count}/{total}")

        source_id = row.get('source_id', 'unknown')
        valid_length = row.get('valid_length', WINDOW_LEN)
        episode_id = row.get('episode_id')

        # Fix normalization
        fixed_signal, raw_stats = fix_normalization(row['signal'], valid_length)

        # Check if this was actually a fix
        old_sig = np.array(row['signal'], dtype=np.float32)
        old_valid_std = float(np.std(old_sig[:valid_length]))
        if abs(old_valid_std - 1.0) > 0.05:
            norm_fixes += 1

        # Assign split
        split = assign_split(source_id, episode_id, sca_test_episodes)

        # Build fixed row
        fixed_row = {
            'source_id': source_id,
            'signal': fixed_signal,
            'valid_length': valid_length,
            'health_state': row.get('health_state', 'unknown'),
            'fault_type': row.get('fault_type', 'unknown'),
            'fault_severity': row.get('fault_severity'),
            'rpm': row.get('rpm'),
            'episode_id': row.get('episode_id'),
            'episode_position': row.get('episode_position'),
            'rul_percent': row.get('rul_percent'),
            'is_transition': row.get('is_transition', False),
            'split': split,
            'raw_stats': json.dumps(raw_stats),
            'sample_weight': source_weights[source_id],
        }

        split_data[split].append(fixed_row)

    print(f"\n  Processed: {count}")
    print(f"  Normalization fixes: {norm_fixes}")
    for sp in ['train', 'validation', 'test']:
        print(f"  {sp}: {len(split_data[sp])}")

    # Step 4: Create HF DatasetDict and upload
    print("\nCreating HF DatasetDict...")
    ds_dict = {}
    for sp in ['train', 'validation', 'test']:
        if split_data[sp]:
            # Convert list of dicts to dict of lists
            columns = defaultdict(list)
            for row in split_data[sp]:
                for k, v in row.items():
                    columns[k].append(v)
            ds_dict[sp] = Dataset.from_dict(dict(columns))
            print(f"  {sp}: {len(ds_dict[sp])} samples")

    dataset_dict = DatasetDict(ds_dict)

    print("\nUploading to HuggingFace...")
    dataset_dict.push_to_hub(
        DATASET_ID,
        config_name="v2_train",
        commit_message="Fix v2_train: normalization on valid portion, source-disjoint splits, sample_weight",
    )
    print("DONE!")

    # Print verification stats
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    for sp in ['train', 'validation', 'test']:
        if sp in ds_dict:
            ds = ds_dict[sp]
            # Sample a few and check normalization
            for i in range(min(3, len(ds))):
                sig = np.array(ds[i]['signal'], dtype=np.float32)
                vl = ds[i]['valid_length']
                valid = sig[:vl]
                print(f"  {sp}[{i}] source={ds[i]['source_id']}: "
                      f"mean={np.mean(valid):.6f}, std={np.std(valid):.4f}, "
                      f"weight={ds[i]['sample_weight']:.4f}")


if __name__ == '__main__':
    main()
