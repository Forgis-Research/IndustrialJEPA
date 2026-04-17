"""
Download proprioception-rich subsets from Open X-Embodiment.

Targets ONLY datasets with meaningful proprioceptive state data,
discards images, and saves as numpy arrays ready for time series
pretraining (Mechanical-JEPA).

Requirements:
    pip install tensorflow tensorflow_datasets numpy

Usage:
    # Download 10 episodes per dataset (quick test)
    python datasets/downloaders/download_oxe_proprio.py --sample

    # Download 100 episodes per dataset
    python datasets/downloaders/download_oxe_proprio.py --n-episodes 100

    # Download a specific dataset only
    python datasets/downloaders/download_oxe_proprio.py --dataset toto --n-episodes 500

    # Download all proprio-rich datasets, 1000 episodes each
    python datasets/downloaders/download_oxe_proprio.py --n-episodes 1000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ============================================================
# Dataset registry: proprioception-rich OXE subsets
# ============================================================
# Verified via tfds.builder() introspection + GCS downloads (March 2026)

PROPRIO_DATASETS = {
    "toto": {
        "tfds_name": "toto",
        "robot": "Franka Emika Panda",
        "dof": 7,
        "state_key": "state",
        "state_dim": 7,
        "state_desc": "7x absolute joint angles (rad)",
        "action_type": "dict",
        "action_keys": ["world_vector", "rotation_delta", "open_gripper", "terminate_episode"],
        "total_episodes_approx": 2898,
        "control_freq_hz": 10,
        "extra_proprio_keys": [],
    },
    "stanford_kuka": {
        "tfds_name": "stanford_kuka_multimodal_dataset_converted_externally_to_rlds",
        "robot": "KUKA iiwa 7-DOF",
        "dof": 7,
        "state_key": "joint_pos",
        "state_dim": 7,
        "state_desc": "7x joint positions (rad)",
        "action_type": "tensor",
        "action_dim": 4,
        "total_episodes_approx": 3000,
        "control_freq_hz": 20,
        "extra_proprio_keys": [
            "joint_vel",          # (7,) joint velocities
            "ee_position",        # (3,) end-effector XYZ
            "ee_orientation",     # (4,) quaternion
            "ee_vel",             # (3,) EE velocity
            "ee_orientation_vel", # (3,) orientation velocity
            "ee_forces_continuous",  # (50, 6) force/torque window
            "contact",            # (50,) contact signal
            "state",              # (8,) = 7 joint + gripper
        ],
    },
    "berkeley_ur5": {
        "tfds_name": "berkeley_autolab_ur5",
        "robot": "Universal Robots UR5 (6-DOF)",
        "dof": 6,
        "state_key": "robot_state",
        "state_dim": 15,
        "state_desc": "15-dim: likely 6x joint pos + 6x joint vel + 3x EE",
        "action_type": "dict",
        "action_keys": ["world_vector", "rotation_delta", "gripper_closedness_action", "terminate_episode"],
        "total_episodes_approx": 1000,
        "control_freq_hz": 10,
        "extra_proprio_keys": [],
    },
    "berkeley_fanuc": {
        "tfds_name": "berkeley_fanuc_manipulation",
        "robot": "FANUC Mate 200iD (6-DOF)",
        "dof": 6,
        "state_key": "state",
        "state_dim": 13,
        "state_desc": "13-dim: 6x joint angles + 1x gripper + 6x joint velocities",
        "action_type": "tensor",
        "action_dim": 6,
        "total_episodes_approx": 415,
        "control_freq_hz": 10,
        "extra_proprio_keys": ["end_effector_state"],  # (7,)
    },
    "jaco_play": {
        "tfds_name": "jaco_play",
        "robot": "Kinova JACO (6-DOF + 2 fingers)",
        "dof": 6,
        "state_key": "joint_pos",
        "state_dim": 8,
        "state_desc": "8-dim: 6x joint angles + 2x finger positions",
        "action_type": "dict",
        "action_keys": ["world_vector", "gripper_closedness_action", "terminate_episode"],
        "total_episodes_approx": 1000,
        "control_freq_hz": 10,
        "extra_proprio_keys": [
            "end_effector_cartesian_pos",      # (7,)
            "end_effector_cartesian_velocity",  # (6,)
        ],
    },
    "fractal": {
        "tfds_name": "fractal20220817_data",
        "robot": "Google Everyday Robots (mobile manipulator)",
        "dof": 7,
        "state_key": "base_pose_tool_reached",
        "state_dim": 7,
        "state_desc": "7-dim: EE position (3) + quaternion (4)",
        "action_type": "dict",
        "action_keys": ["world_vector", "rotation_delta", "base_displacement_vector",
                        "base_displacement_vertical_rotation", "gripper_closedness_action",
                        "terminate_episode"],
        "total_episodes_approx": 87212,
        "control_freq_hz": 3,
        "extra_proprio_keys": [
            "gripper_closed",               # (1,)
            "gripper_closedness_commanded",  # (1,)
            "height_to_bottom",             # (1,)
            "rotation_delta_to_go",         # (3,)
            "vector_to_go",                 # (3,)
        ],
    },
    "maniskill": {
        "tfds_name": "maniskill_dataset_converted_externally_to_rlds",
        "robot": "Panda (sim, ManiSkill2)",
        "dof": 7,
        "state_key": "state",
        "state_dim": 18,
        "state_desc": "18-dim: 7x joint angles + 2x gripper + 7x joint vel + 2x gripper vel",
        "action_type": "tensor",
        "action_dim": 7,
        "total_episodes_approx": 30213,
        "control_freq_hz": 20,
        "extra_proprio_keys": [
            "base_pose",   # (7,)
            "tcp_pose",    # (7,)
        ],
    },
}

# DROID is NOT in the tfds registry but is available via GCS
# as a separate RLDS dataset at gs://gresearch/robotics/droid_100
DROID_INFO = {
    "robot": "Franka Emika Panda",
    "dof": 7,
    "total_episodes_approx": 76000,
    "control_freq_hz": 15,
    "state_fields": {
        "joint_position": 7,
        "cartesian_position": 6,
        "gripper_position": 1,
    },
    "note": "DROID is not in standard tfds; load with data_dir='gs://gresearch/robotics'",
}


def check_dependencies() -> bool:
    """Check if TensorFlow and tensorflow_datasets are available."""
    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds
        return True
    except ImportError as e:
        print(f"[ERR] Missing dependency: {e}")
        print("      Install: pip install tensorflow tensorflow_datasets")
        return False


def extract_proprio_from_episode(episode, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Extract proprioceptive state and action time series from one RLDS episode.

    Returns:
        states: (T, state_dim) numpy array
        actions: (T, action_dim) numpy array
    """
    steps = list(episode['steps'])
    states = []
    actions = []

    for step in steps:
        obs = step['observation']

        # Primary state
        primary = obs[config['state_key']].numpy()

        # Extra proprio channels
        extras = []
        for key in config.get('extra_proprio_keys', []):
            if key in obs:
                val = obs[key].numpy()
                if val.ndim == 0:
                    extras.append(np.array([float(val)]))
                elif val.ndim == 1:
                    extras.append(val)
                elif val.ndim == 2:
                    # e.g., ee_forces_continuous (50, 6) -> mean over window
                    extras.append(val.mean(axis=0))

        if extras:
            full_state = np.concatenate([primary] + extras)
        else:
            full_state = primary

        states.append(full_state)

        # Action extraction
        act = step['action']
        if hasattr(act, 'numpy'):
            actions.append(act.numpy())
        elif isinstance(act, dict):
            act_parts = []
            for k in sorted(act.keys()):
                v = act[k]
                if hasattr(v, 'numpy'):
                    val = v.numpy()
                    if val.ndim == 0:
                        act_parts.append(np.array([float(val)]))
                    else:
                        act_parts.append(val)
            if act_parts:
                actions.append(np.concatenate(act_parts))

    states = np.array(states) if states else np.array([])
    actions = np.array(actions) if actions else np.array([])
    return states, actions


def download_dataset(dataset_key: str, config: dict, output_dir: Path,
                     n_episodes: int = 100, verbose: bool = True) -> dict:
    """Download and extract proprioceptive data from one OXE dataset.

    Returns metadata dict with statistics.
    """
    import tensorflow_datasets as tfds

    tfds_name = config['tfds_name']
    if verbose:
        print(f"\n{'='*60}")
        print(f"Downloading: {dataset_key} ({config['robot']})")
        print(f"  tfds name: {tfds_name}")
        print(f"  State: {config['state_desc']}")
        print(f"  Requesting {n_episodes} episodes...")
        print(f"{'='*60}")

    t0 = time.time()
    try:
        ds = tfds.load(
            tfds_name,
            data_dir='gs://gresearch/robotics',
            split=f'train[:{n_episodes}]',
        )
    except Exception as e:
        print(f"  [ERR] Failed to load: {e}")
        return {"status": "failed", "error": str(e)}

    all_states = []
    all_actions = []
    ep_lengths = []
    state_dims = set()

    for i, episode in enumerate(ds):
        states, actions = extract_proprio_from_episode(episode, config)

        if states.size == 0:
            continue

        all_states.append(states)
        if actions.size > 0:
            all_actions.append(actions)
        ep_lengths.append(len(states))
        state_dims.add(states.shape[1] if states.ndim == 2 else 0)

        if verbose and (i + 1) % max(1, n_episodes // 10) == 0:
            print(f"  [{i+1}/{n_episodes}] episodes loaded")

    elapsed = time.time() - t0

    if not all_states:
        return {"status": "empty", "error": "No episodes extracted"}

    # Save as numpy
    ds_dir = output_dir / dataset_key
    ds_dir.mkdir(parents=True, exist_ok=True)

    # Save individual episodes
    for j, (st, ac) in enumerate(zip(all_states, all_actions or [None]*len(all_states))):
        np.save(ds_dir / f"ep{j:05d}_state.npy", st)
        if ac is not None and ac.size > 0:
            np.save(ds_dir / f"ep{j:05d}_action.npy", ac)

    # Save concatenated (for quick loading)
    # Pad to max length for batch processing
    max_len = max(ep_lengths)
    state_dim = all_states[0].shape[1]
    padded_states = np.zeros((len(all_states), max_len, state_dim), dtype=np.float32)
    for j, st in enumerate(all_states):
        padded_states[j, :len(st)] = st
    np.save(ds_dir / "all_states_padded.npy", padded_states)
    np.save(ds_dir / "episode_lengths.npy", np.array(ep_lengths))

    # Compute statistics
    all_concat = np.concatenate(all_states, axis=0)
    nan_count = int(np.isnan(all_concat).sum())
    const_channels = int((all_concat.std(axis=0) < 1e-8).sum())

    metadata = {
        "status": "success",
        "dataset_key": dataset_key,
        "robot": config['robot'],
        "dof": config['dof'],
        "n_episodes": len(all_states),
        "state_dim": state_dim,
        "state_desc": config['state_desc'],
        "ep_length_min": int(min(ep_lengths)),
        "ep_length_max": int(max(ep_lengths)),
        "ep_length_mean": float(np.mean(ep_lengths)),
        "total_timesteps": int(sum(ep_lengths)),
        "nan_count": nan_count,
        "constant_channels": const_channels,
        "state_mean": all_concat.mean(axis=0).tolist(),
        "state_std": all_concat.std(axis=0).tolist(),
        "state_min": all_concat.min(axis=0).tolist(),
        "state_max": all_concat.max(axis=0).tolist(),
        "download_time_sec": round(elapsed, 1),
        "control_freq_hz": config['control_freq_hz'],
    }

    # Save metadata
    with open(ds_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"\n  Results for {dataset_key}:")
        print(f"    Episodes:  {metadata['n_episodes']}")
        print(f"    State dim: {metadata['state_dim']}")
        print(f"    Ep length: {metadata['ep_length_min']}-{metadata['ep_length_max']} "
              f"(mean={metadata['ep_length_mean']:.0f})")
        print(f"    Timesteps: {metadata['total_timesteps']:,}")
        print(f"    NaN count: {metadata['nan_count']}")
        print(f"    Const channels: {metadata['constant_channels']}")
        print(f"    Time: {metadata['download_time_sec']:.1f}s")
        print(f"    Saved to: {ds_dir}")

    return metadata


def print_registry():
    """Print the dataset registry with key info."""
    print("\nProprioception-Rich OXE Datasets")
    print("=" * 90)
    header = f"{'Key':<18} {'Robot':<28} {'DOF':>3} {'State':>5} {'Episodes':>8} {'Hz':>4}"
    print(header)
    print("-" * 90)
    total_eps = 0
    for key, cfg in PROPRIO_DATASETS.items():
        n_extra = sum(1 for k in cfg.get('extra_proprio_keys', []))
        print(f"{key:<18} {cfg['robot']:<28} {cfg['dof']:>3} {cfg['state_dim']:>5} "
              f"{cfg['total_episodes_approx']:>8,} {cfg['control_freq_hz']:>4}")
        total_eps += cfg['total_episodes_approx']
    print("-" * 90)
    print(f"{'TOTAL':<18} {'':28} {'':>3} {'':>5} {total_eps:>8,}")
    print(f"\n  + DROID (not in tfds): ~{DROID_INFO['total_episodes_approx']:,} episodes, "
          f"{DROID_INFO['dof']}-DOF Franka, {DROID_INFO['control_freq_hz']} Hz")
    print(f"  = Grand total: ~{total_eps + DROID_INFO['total_episodes_approx']:,} episodes")


def main():
    parser = argparse.ArgumentParser(
        description="Download proprioception-rich OXE subsets for Mechanical-JEPA")
    parser.add_argument("--sample", action="store_true",
                        help="Download 10 episodes per dataset (quick test)")
    parser.add_argument("--n-episodes", type=int, default=100,
                        help="Number of episodes per dataset (default: 100)")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=list(PROPRIO_DATASETS.keys()),
                        help="Download a specific dataset only")
    parser.add_argument("--list", action="store_true",
                        help="List available datasets and exit")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent.parent / "data" / "oxe_proprio"),
                        help="Output directory")
    args = parser.parse_args()

    if args.list:
        print_registry()
        return

    if not check_dependencies():
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_episodes = 10 if args.sample else args.n_episodes
    print_registry()

    # Select datasets
    if args.dataset:
        datasets = {args.dataset: PROPRIO_DATASETS[args.dataset]}
    else:
        datasets = PROPRIO_DATASETS

    # Download
    results = {}
    for key, config in datasets.items():
        meta = download_dataset(key, config, output_dir, n_episodes=n_episodes)
        results[key] = meta

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    total_ts = 0
    total_ep = 0
    for key, meta in results.items():
        status = meta.get('status', 'unknown')
        if status == 'success':
            print(f"  {key:<20} {meta['n_episodes']:>5} eps, "
                  f"{meta['state_dim']:>2}D state, "
                  f"{meta['total_timesteps']:>8,} timesteps")
            total_ts += meta['total_timesteps']
            total_ep += meta['n_episodes']
        else:
            print(f"  {key:<20} FAILED: {meta.get('error', 'unknown')[:50]}")

    print(f"\n  Total: {total_ep} episodes, {total_ts:,} timesteps")
    print(f"  Output: {output_dir}")

    # Save combined metadata
    with open(output_dir / "download_summary.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
