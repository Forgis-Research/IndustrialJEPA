"""
Download Open X-Embodiment Dataset (Bridge V2 subset).

Full OXE is ~1.5 TB — impractical. This script downloads Bridge V2,
one of the most proprioceptive-rich sub-datasets (~40 GB for full,
~2 GB for 5% sample).

Requirements:
    pip install tensorflow tensorflow_datasets

Usage:
    python datasets/downloaders/download_open_x.py
    python datasets/downloaders/download_open_x.py --sample  # 100 trajectories only
    python datasets/downloaders/download_open_x.py --dataset fractal  # Google's Fractal dataset
"""

import argparse
import json
import sys
from pathlib import Path

AVAILABLE_SUBSETS = {
    "bridge_v2": {
        "tfds_name": "bridge_v2",
        "size_gb_approx": 40,
        "channels_proprioceptive": 12,  # joint positions + orientations
        "description": "WidowX robot arm, pick-and-place tasks, ~60k episodes",
    },
    "fractal": {
        "tfds_name": "fractal20220817_data",
        "size_gb_approx": 330,
        "channels_proprioceptive": 7,  # end-effector pose
        "description": "Google RT-1 training data, ~130k episodes",
    },
    "berkeley_autolab_ur5": {
        "tfds_name": "berkeley_autolab_ur5",
        "size_gb_approx": 85,
        "channels_proprioceptive": 20,  # joint pos/vel + force/torque
        "description": "UR5 robot, manipulation tasks, ~3k episodes",
    },
}


def check_dependencies() -> bool:
    """Check if tensorflow and tensorflow_datasets are available."""
    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds
        print(f"  [OK] TensorFlow {tf.__version__} available")
        print(f"  [OK] tensorflow_datasets available")
        return True
    except ImportError as e:
        print(f"  [ERR] Missing dependency: {e}")
        print(f"        Install: pip install tensorflow tensorflow_datasets")
        return False


def inspect_episode_structure(episode, max_steps: int = 5) -> dict:
    """Extract the structure of one episode for analysis."""
    steps = list(episode["steps"].take(max_steps))
    if not steps:
        return {"status": "empty"}

    step = steps[0]
    obs = step.get("observation", {})
    action = step.get("action", {})

    structure = {
        "n_steps_sampled": len(steps),
        "observation_keys": list(obs.keys()) if hasattr(obs, "keys") else str(type(obs)),
        "action_shape": str(action.shape) if hasattr(action, "shape") else str(type(action)),
        "has_image": "image" in obs if hasattr(obs, "keys") else False,
    }

    # Try to get proprioceptive shapes
    for key in ["state", "proprio", "joint_positions", "robot_state"]:
        if hasattr(obs, "keys") and key in obs:
            structure[f"proprio_{key}_shape"] = str(obs[key].shape)

    return structure


def download_sample(dataset_name: str, output_dir: Path,
                    n_episodes: int = 100) -> bool:
    """Download N episodes from a dataset as numpy arrays."""
    if not check_dependencies():
        return False

    try:
        import tensorflow_datasets as tfds
        import numpy as np

        info = AVAILABLE_SUBSETS.get(dataset_name, {})
        tfds_name = info.get("tfds_name", dataset_name)

        print(f"\n  [LOAD] Loading {n_episodes} episodes from {tfds_name}...")
        print(f"         (First call may trigger small metadata download)")

        # Load with data_dir to control where tfds stores data
        data_dir = output_dir / "tfds_cache"
        ds = tfds.load(
            tfds_name,
            split=f"train[:{n_episodes}]",
            data_dir=str(data_dir),
        )

        episodes_data = []
        structure_logged = False

        for i, episode in enumerate(ds):
            if not structure_logged:
                struct = inspect_episode_structure(episode)
                print(f"\n  [STRUCT] Episode structure:")
                for k, v in struct.items():
                    print(f"           {k}: {v}")
                structure_logged = True

            steps = list(episode["steps"])
            episode_data = {
                "n_steps": len(steps),
                "actions": [],
                "observations": [],
            }
            for step in steps:
                action = step.get("action", {})
                if hasattr(action, "numpy"):
                    episode_data["actions"].append(action.numpy())
                obs = step.get("observation", {})
                if hasattr(obs, "keys"):
                    obs_np = {
                        k: v.numpy() for k, v in obs.items()
                        if hasattr(v, "numpy") and v.dtype.name != "string"
                        and "image" not in k  # Skip images
                    }
                    episode_data["observations"].append(obs_np)

            episodes_data.append(episode_data)
            if (i + 1) % 10 == 0:
                print(f"  [PROG] {i + 1}/{n_episodes} episodes loaded")

        # Save as numpy format
        save_path = output_dir / f"{dataset_name}_sample_{n_episodes}ep.npz"
        save_dict = {}
        for i, ep in enumerate(episodes_data):
            if ep["actions"]:
                save_dict[f"ep{i:04d}_actions"] = np.array(ep["actions"])
            if ep["observations"] and ep["observations"][0]:
                for key in ep["observations"][0]:
                    try:
                        arr = np.array([obs[key] for obs in ep["observations"]
                                        if key in obs])
                        save_dict[f"ep{i:04d}_obs_{key}"] = arr
                    except Exception:
                        pass

        np.savez_compressed(save_path, **save_dict)
        size_mb = save_path.stat().st_size / 1024 / 1024
        print(f"\n  [SAVE] {n_episodes} episodes saved to {save_path} ({size_mb:.1f} MB)")

        # Print statistics
        n_steps_list = [ep["n_steps"] for ep in episodes_data]
        print(f"\n  Statistics:")
        print(f"    Episodes: {len(episodes_data)}")
        print(f"    Steps/episode: min={min(n_steps_list)}, "
              f"max={max(n_steps_list)}, "
              f"mean={sum(n_steps_list)/len(n_steps_list):.1f}")
        if episodes_data and episodes_data[0]["actions"]:
            action_shape = episodes_data[0]["actions"][0].shape
            print(f"    Action dim: {action_shape}")

        return True

    except Exception as e:
        print(f"  [ERR] Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_viability_analysis():
    """Print IndustrialJEPA viability analysis for OXE."""
    print("\n" + "=" * 60)
    print("VIABILITY ANALYSIS: Open X-Embodiment for IndustrialJEPA")
    print("=" * 60)
    print("""
Goal: Find Brain-JEPA analog (32k patients × 160 timesteps × 450 ROIs)

OXE comparison:
  Dimension          Brain-JEPA      OXE (Bridge V2)  Assessment
  ─────────────────  ──────────────  ───────────────  ──────────
  "Subjects"         32,000 pts      ~60,000 eps      OXE larger
  Timesteps/subject  160             50–200           Comparable
  "Channels"         450 ROIs        12–20            CRITICAL GAP
  Total tokens       ~2.3B           ~100M (no img)   Comparable

VERDICT: NOT VIABLE as Brain-JEPA analog.

Critical limitations:
  1. Channel count: 12–20 channels vs 450 ROIs (20–38x deficit)
     - Cannot demonstrate "many-channel" physics attention at scale
  2. Camera-centric: Most OXE use cases rely on RGB images, not proprioception
  3. Heterogeneous: Different robots have incompatible sensor spaces
  4. Format overhead: TFRecord/RLDS requires TensorFlow, not PyTorch-native

Better alternatives for Brain-JEPA scale:
  1. SWaT: 51 channels × 946k timesteps (continuous, 1 Hz)
  2. WADI: 127 channels × 1.2M timesteps (continuous, 1 Hz)  [best!]
  3. Multi-dataset concatenation: CWRU+Paderborn+Hydraulic = ~60 channels
  4. Electricity (UCI): 321 channels × 26k timesteps (if channel=sensor)
  5. Traffic: 862 channels × 17k timesteps
""")


def main():
    parser = argparse.ArgumentParser(
        description="Download Open X-Embodiment (Bridge V2 subset)")
    parser.add_argument("--sample", action="store_true",
                        help="Download 100 episodes only (recommended)")
    parser.add_argument("--n-episodes", type=int, default=100,
                        help="Number of episodes to download in sample mode")
    parser.add_argument("--dataset", choices=list(AVAILABLE_SUBSETS.keys()),
                        default="bridge_v2",
                        help="Which OXE sub-dataset to download")
    parser.add_argument("--analysis-only", action="store_true",
                        help="Print viability analysis without downloading")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent.parent / "data" / "open_x"),
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOpen X-Embodiment Downloader")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {output_dir}")
    print("-" * 60)

    if args.analysis_only:
        print_viability_analysis()
        return

    info = AVAILABLE_SUBSETS.get(args.dataset, {})
    print(f"Description: {info.get('description', 'N/A')}")
    print(f"Full size: ~{info.get('size_gb_approx', '?')} GB")
    print(f"Proprioceptive channels: {info.get('channels_proprioceptive', '?')}")

    n_episodes = args.n_episodes if args.sample else 1000
    if not args.sample:
        print(f"\n[WARN] Full download is very large. Downloading 1000 episodes.")
        print(f"       Use --sample for 100 episodes instead.")

    success = download_sample(args.dataset, output_dir, n_episodes=n_episodes)

    print_viability_analysis()

    if success:
        print(f"\nData saved to: {output_dir}")
    else:
        print(f"\n[FAIL] Download was unsuccessful.")
        print(f"       If TensorFlow is unavailable, use HuggingFace:")
        print(f'       from datasets import load_dataset')
        print(f'       ds = load_dataset("jxu124/OpenX-Embodiment", "bridge_v2")')


if __name__ == "__main__":
    main()
