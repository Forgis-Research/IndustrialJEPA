"""
Lean OXE downloader — streams one episode at a time, minimal memory.
Extracts proprio + actions only, discards images immediately.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

# Suppress TF warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import tensorflow_datasets as tfds

OUT = Path("/home/sagemaker-user/IndustrialJEPA/datasets/data/oxe_proprio")

DATASETS = {
    "toto": {
        "tfds": "toto",
        "state_key": "state",
        "extra_keys": [],
        "max_eps": 902,
    },
    "stanford_kuka": {
        "tfds": "stanford_kuka_multimodal_dataset_converted_externally_to_rlds",
        "state_key": "joint_pos",
        "extra_keys": ["joint_vel", "ee_position", "ee_vel", "ee_orientation", "ee_orientation_vel"],
        "max_eps": 3000,
    },
    "berkeley_ur5": {
        "tfds": "berkeley_autolab_ur5",
        "state_key": "robot_state",
        "extra_keys": [],
        "max_eps": 896,
    },
    "jaco_play": {
        "tfds": "jaco_play",
        "state_key": "joint_pos",
        "extra_keys": ["end_effector_cartesian_pos", "end_effector_cartesian_velocity"],
        "max_eps": 976,
    },
    "berkeley_fanuc": {
        "tfds": "berkeley_fanuc_manipulation",
        "state_key": "state",
        "extra_keys": ["end_effector_state"],
        "max_eps": 415,
    },
}


def extract_episode(ep, state_key, extra_keys):
    """Extract state + action from one episode. Minimal memory."""
    states, actions = [], []
    for step in ep['steps']:
        obs = step['observation']
        # Primary state
        st = obs[state_key].numpy().astype(np.float32)
        # Extra proprio
        for k in extra_keys:
            if k in obs:
                v = obs[k].numpy().astype(np.float32)
                if v.ndim == 0:
                    st = np.concatenate([st, [float(v)]])
                elif v.ndim == 1:
                    st = np.concatenate([st, v])
                elif v.ndim == 2:
                    st = np.concatenate([st, v.mean(axis=0)])
        states.append(st)
        # Action
        act = step['action']
        if hasattr(act, 'numpy'):
            actions.append(act.numpy().astype(np.float32))
        elif isinstance(act, dict):
            parts = []
            for k2 in sorted(act.keys()):
                v = act[k2]
                if hasattr(v, 'numpy'):
                    val = v.numpy()
                    if val.ndim == 0:
                        parts.append(np.array([float(val)], dtype=np.float32))
                    else:
                        parts.append(val.astype(np.float32))
            if parts:
                actions.append(np.concatenate(parts))
    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32) if actions else np.array([])


def download_one(name, cfg, n_eps=None):
    """Download one dataset."""
    max_eps = n_eps or cfg["max_eps"]
    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing_meta = out_dir / "metadata.json"
    if existing_meta.exists():
        with open(existing_meta) as f:
            m = json.load(f)
        if m.get("n_episodes", 0) >= max_eps * 0.9:
            print(f"  {name}: already have {m['n_episodes']} eps, skipping")
            return m

    print(f"\n{'='*50}")
    print(f"Downloading: {name} (max {max_eps} eps)")
    print(f"{'='*50}")

    t0 = time.time()
    ds = tfds.load(cfg["tfds"], data_dir='gs://gresearch/robotics', split='train')

    ep_lengths = []
    state_dim = None
    action_dim = None
    nan_total = 0

    for i, ep in enumerate(ds):
        if i >= max_eps:
            break

        states, actions = extract_episode(ep, cfg["state_key"], cfg["extra_keys"])

        if states.size == 0:
            continue

        np.save(out_dir / f"ep{i:05d}_state.npy", states)
        if actions.size > 0:
            np.save(out_dir / f"ep{i:05d}_action.npy", actions)

        ep_lengths.append(len(states))
        nan_total += int(np.isnan(states).sum())
        if state_dim is None:
            state_dim = states.shape[1]
            action_dim = actions.shape[1] if actions.ndim == 2 else 0

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eps_per_sec = (i + 1) / elapsed
            eta = (max_eps - i - 1) / eps_per_sec if eps_per_sec > 0 else 0
            print(f"  {i+1}/{max_eps} eps ({elapsed:.0f}s, {eps_per_sec:.1f} eps/s, ETA {eta:.0f}s)")

        # Aggressive cleanup every 100 episodes
        if (i + 1) % 100 == 0:
            gc.collect()

    elapsed = time.time() - t0
    n = len(ep_lengths)

    if n == 0:
        print(f"  FAILED: no episodes extracted")
        return None

    np.save(out_dir / "episode_lengths.npy", np.array(ep_lengths))

    meta = {
        "n_episodes": n,
        "state_dim": int(state_dim),
        "action_dim": int(action_dim),
        "ep_length_min": int(min(ep_lengths)),
        "ep_length_max": int(max(ep_lengths)),
        "ep_length_mean": float(np.mean(ep_lengths)),
        "total_timesteps": int(sum(ep_lengths)),
        "nan_count": nan_total,
        "download_time_sec": round(elapsed, 1),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  DONE: {n} eps, {state_dim}D state, {action_dim}D action, "
          f"{meta['total_timesteps']:,} ts, {elapsed:.0f}s")
    return meta


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    n_eps = int(sys.argv[2]) if len(sys.argv) > 2 else None

    if target == "all":
        for name, cfg in DATASETS.items():
            download_one(name, cfg, n_eps)
            gc.collect()
    elif target in DATASETS:
        download_one(target, DATASETS[target], n_eps)
    else:
        print(f"Unknown dataset: {target}. Options: {list(DATASETS.keys())} or 'all'")
        sys.exit(1)

    print("\n=== COMPLETE ===")
