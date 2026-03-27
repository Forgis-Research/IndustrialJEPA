"""
Fast OXE downloader via LeRobot HuggingFace mirrors.
Downloads proprio + actions only (no images), saves as numpy per episode.
"""
import os
os.environ['DATASETS_NO_TORCH'] = '1'

import json
import time
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset

OUT = Path("/home/sagemaker-user/IndustrialJEPA/datasets/data/oxe_proprio")

LEROBOT_DATASETS = {
    "toto": {
        "hf_name": "lerobot/toto",
        "robot": "Franka Panda 7-DOF",
        "role": "pretrain",
    },
    "stanford_kuka": {
        "hf_name": "lerobot/stanford_kuka_multimodal_dataset",
        "robot": "KUKA iiwa 7-DOF",
        "role": "transfer_target",
    },
    "berkeley_ur5": {
        "hf_name": "lerobot/berkeley_autolab_ur5",
        "robot": "UR5 6-DOF",
        "role": "transfer_target",
    },
    "jaco_play": {
        "hf_name": "lerobot/jaco_play",
        "robot": "Kinova JACO 6-DOF",
        "role": "transfer_target",
    },
    "berkeley_fanuc": {
        "hf_name": "lerobot/berkeley_fanuc_manipulation",
        "robot": "FANUC Mate 200iD 6-DOF",
        "role": "transfer_target",
    },
    "droid": {
        "hf_name": "lerobot/droid_100",
        "robot": "Franka Panda 7-DOF",
        "role": "pretrain",
    },
}


def download_one(name: str, cfg: dict):
    """Download one dataset from LeRobot HF mirror."""
    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    meta_file = out_dir / "metadata.json"
    if meta_file.exists():
        with open(meta_file) as f:
            m = json.load(f)
        if m.get("source") == "lerobot_hf" and m.get("n_episodes", 0) > 0:
            print(f"  {name}: already downloaded ({m['n_episodes']} eps), skipping")
            return m

    print(f"\n{'='*60}")
    print(f"Downloading: {name} ({cfg['hf_name']})")
    print(f"  Robot: {cfg['robot']}, Role: {cfg['role']}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        ds = load_dataset(cfg["hf_name"], split="train")
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return None

    total_rows = len(ds)
    print(f"  Loaded {total_rows:,} rows in {time.time()-t0:.1f}s")
    print(f"  Columns: {ds.column_names}")

    # Get dimensions from first row
    row0 = ds[0]
    state_dim = len(row0['observation.state'])
    action_dim = len(row0['action'])
    print(f"  State dim: {state_dim}, Action dim: {action_dim}")

    # Convert to numpy arrays grouped by episode
    print("  Converting to per-episode numpy...")
    t1 = time.time()

    # Extract all data at once (fast with Arrow backend)
    states_all = np.array(ds['observation.state'], dtype=np.float32)
    actions_all = np.array(ds['action'], dtype=np.float32)
    ep_indices = np.array(ds['episode_index'])
    rewards = np.array(ds['next.reward'], dtype=np.float32)
    dones = np.array(ds['next.done'])

    unique_eps = np.unique(ep_indices)
    n_episodes = len(unique_eps)
    print(f"  {n_episodes} episodes, converting... ({time.time()-t1:.1f}s)")

    ep_lengths = []
    ep_rewards = []
    nan_total = 0

    for i, ep_idx in enumerate(unique_eps):
        mask = ep_indices == ep_idx
        ep_states = states_all[mask]
        ep_actions = actions_all[mask]
        ep_reward = rewards[mask]

        np.save(out_dir / f"ep{i:05d}_state.npy", ep_states)
        np.save(out_dir / f"ep{i:05d}_action.npy", ep_actions)

        ep_lengths.append(len(ep_states))
        ep_rewards.append(float(ep_reward.sum()))
        nan_total += int(np.isnan(ep_states).sum())

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{n_episodes} episodes saved")

    # Save episode metadata
    np.save(out_dir / "episode_lengths.npy", np.array(ep_lengths))
    np.save(out_dir / "episode_rewards.npy", np.array(ep_rewards))

    # Compute stats
    const_channels = int((states_all.std(axis=0) < 1e-8).sum())
    state_mean = states_all.mean(axis=0).tolist()
    state_std = states_all.std(axis=0).tolist()
    state_min = states_all.min(axis=0).tolist()
    state_max = states_all.max(axis=0).tolist()
    action_mean = actions_all.mean(axis=0).tolist()
    action_std = actions_all.std(axis=0).tolist()

    elapsed = time.time() - t0
    meta = {
        "source": "lerobot_hf",
        "hf_name": cfg["hf_name"],
        "robot": cfg["robot"],
        "role": cfg["role"],
        "n_episodes": n_episodes,
        "total_rows": total_rows,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "ep_length_min": int(min(ep_lengths)),
        "ep_length_max": int(max(ep_lengths)),
        "ep_length_mean": float(np.mean(ep_lengths)),
        "total_timesteps": int(sum(ep_lengths)),
        "nan_count": nan_total,
        "constant_channels": const_channels,
        "state_mean": state_mean,
        "state_std": state_std,
        "state_min": state_min,
        "state_max": state_max,
        "action_mean": action_mean,
        "action_std": action_std,
        "success_rate": float(np.mean([r > 0 for r in ep_rewards])),
        "download_time_sec": round(elapsed, 1),
    }

    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    size_mb = sum(f.stat().st_size for f in out_dir.glob("*.npy")) / 1e6
    print(f"\n  DONE: {n_episodes} episodes, {state_dim}D state, {action_dim}D action")
    print(f"  Timesteps: {meta['total_timesteps']:,}, NaN: {nan_total}, Const: {const_channels}")
    print(f"  Success rate: {meta['success_rate']:.1%}")
    print(f"  Size on disk: {size_mb:.1f} MB, Time: {elapsed:.1f}s")
    return meta


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target == "all":
        results = {}
        for name, cfg in LEROBOT_DATASETS.items():
            meta = download_one(name, cfg)
            results[name] = meta
            if meta:
                print(f"  ✓ {name}: {meta['n_episodes']} eps")
            else:
                print(f"  ✗ {name}: FAILED")

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        total_eps = sum(m['n_episodes'] for m in results.values() if m)
        total_ts = sum(m['total_timesteps'] for m in results.values() if m)
        for name, m in results.items():
            if m:
                print(f"  {name:<20} {m['n_episodes']:>6} eps  {m['state_dim']:>2}D state  {m['action_dim']:>2}D action  {m['total_timesteps']:>8,} ts")
            else:
                print(f"  {name:<20} FAILED")
        print(f"  {'TOTAL':<20} {total_eps:>6} eps  {'':>2}  {'':>2}  {total_ts:>8,} ts")

        with open(OUT / "download_summary.json", "w") as f:
            json.dump({k: v for k, v in results.items() if v}, f, indent=2)

    elif target in LEROBOT_DATASETS:
        download_one(target, LEROBOT_DATASETS[target])
    else:
        print(f"Unknown: {target}. Options: {list(LEROBOT_DATASETS.keys())} or 'all'")
        sys.exit(1)
