"""
Deep OXE Data Analysis for Mechanical-JEPA.

Downloads samples from all relevant OXE subsets, verifies schema/dimensions,
analyzes actions thoroughly, checks label availability, and produces
comprehensive analysis figures.

Usage:
    python datasets/analysis/oxe_deep_analysis.py
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path
from collections import defaultdict

import numpy as np

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_datasets as tfds

# ============================================================
# Configuration
# ============================================================
OUTPUT_DIR = Path("/home/sagemaker-user/IndustrialJEPA/datasets/data/oxe_analysis")
FIGURE_DIR = Path("/home/sagemaker-user/IndustrialJEPA/datasets/analysis/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

N_EPISODES = 20  # per dataset for detailed analysis

DATASETS = {
    "toto": "toto",
    "stanford_kuka": "stanford_kuka_multimodal_dataset_converted_externally_to_rlds",
    "berkeley_ur5": "berkeley_autolab_ur5",
    "berkeley_fanuc": "berkeley_fanuc_manipulation",
    "jaco_play": "jaco_play",
    "maniskill": "maniskill_dataset_converted_externally_to_rlds",
    "fractal": "fractal20220817_data",
}

DROID_NAME = "droid_100"


def inspect_schema(tfds_name):
    """Get the full feature schema from tfds builder without downloading data."""
    try:
        builder = tfds.builder(tfds_name, data_dir='gs://gresearch/robotics')
        info = builder.info
        return {
            "description": info.description[:300] if info.description else "",
            "features": str(info.features),
            "splits": {k: v.num_examples for k, v in info.splits.items()},
            "size": str(info.dataset_size),
        }
    except Exception as e:
        return {"error": str(e)}


def deep_inspect_episode(episode, ds_key):
    """Deeply inspect a single episode, printing all observation and action fields."""
    result = {
        "observation_keys": {},
        "action_info": {},
        "other_step_keys": [],
        "n_steps": 0,
        "has_language": False,
        "has_reward": False,
        "has_is_terminal": False,
        "has_is_first": False,
        "has_is_last": False,
        "language_instruction": None,
    }

    steps = list(episode['steps'])
    result['n_steps'] = len(steps)

    if not steps:
        return result

    step0 = steps[0]

    # All keys in a step
    step_keys = list(step0.keys()) if hasattr(step0, 'keys') else []
    result['other_step_keys'] = step_keys

    # Check for reward, terminal, language
    if 'reward' in step0:
        result['has_reward'] = True
        try:
            rewards = [float(s['reward'].numpy()) for s in steps]
            result['reward_stats'] = {
                'min': float(np.min(rewards)),
                'max': float(np.max(rewards)),
                'mean': float(np.mean(rewards)),
                'nonzero_count': int(np.count_nonzero(rewards)),
            }
        except:
            pass

    for flag in ['is_terminal', 'is_first', 'is_last']:
        if flag in step0:
            result[f'has_{flag}'] = True

    # Language instruction
    if 'language_instruction' in step0:
        result['has_language'] = True
        try:
            lang = step0['language_instruction'].numpy()
            if isinstance(lang, bytes):
                lang = lang.decode('utf-8')
            result['language_instruction'] = str(lang)[:200]
        except:
            pass
    elif 'observation' in step0:
        obs = step0['observation']
        if hasattr(obs, 'keys'):
            for lk in ['natural_language_instruction', 'language_instruction', 'instruction']:
                if lk in obs:
                    result['has_language'] = True
                    try:
                        lang = obs[lk].numpy()
                        if isinstance(lang, bytes):
                            lang = lang.decode('utf-8')
                        result['language_instruction'] = str(lang)[:200]
                    except:
                        pass
                    break

    # Observation keys and shapes
    if 'observation' in step0:
        obs = step0['observation']
        if hasattr(obs, 'keys'):
            for key in sorted(obs.keys()):
                try:
                    val = obs[key]
                    if hasattr(val, 'numpy'):
                        v = val.numpy()
                        if isinstance(v, bytes):
                            result['observation_keys'][key] = f"string: '{v.decode('utf-8')[:80]}'"
                        elif v.ndim == 0:
                            result['observation_keys'][key] = f"scalar, dtype={v.dtype}, value={float(v):.6f}"
                        elif v.ndim >= 3 and v.shape[-1] in [1, 3, 4] and v.shape[0] > 32:
                            # Likely an image
                            result['observation_keys'][key] = f"IMAGE shape={v.shape}, dtype={v.dtype}"
                        else:
                            result['observation_keys'][key] = (
                                f"shape={list(v.shape)}, dtype={v.dtype}, "
                                f"range=[{float(v.min()):.6f}, {float(v.max()):.6f}], "
                                f"mean={float(v.mean()):.6f}"
                            )
                    elif hasattr(val, 'shape'):
                        result['observation_keys'][key] = f"shape={val.shape}, dtype={val.dtype}"
                    else:
                        result['observation_keys'][key] = str(type(val))
                except Exception as e:
                    result['observation_keys'][key] = f"ERROR: {e}"

    # Action info
    if 'action' in step0:
        act = step0['action']
        if hasattr(act, 'keys'):
            result['action_info']['_type'] = 'dict'
            for key in sorted(act.keys()):
                try:
                    v = act[key].numpy()
                    if v.ndim == 0:
                        result['action_info'][key] = f"scalar, dtype={v.dtype}, value={float(v):.6f}"
                    else:
                        result['action_info'][key] = (
                            f"shape={list(v.shape)}, dtype={v.dtype}, "
                            f"range=[{float(v.min()):.6f}, {float(v.max()):.6f}]"
                        )
                except Exception as e:
                    result['action_info'][key] = f"ERROR: {e}"
        elif hasattr(act, 'numpy'):
            v = act.numpy()
            result['action_info']['_type'] = 'tensor'
            result['action_info']['shape'] = list(v.shape)
            result['action_info']['dtype'] = str(v.dtype)
            result['action_info']['range'] = [float(v.min()), float(v.max())]
            result['action_info']['mean'] = float(v.mean())
            result['action_info']['values_sample'] = v.flatten()[:20].tolist()

    return result


def extract_full_episode_data(episode, ds_key):
    """Extract ALL numeric fields from an episode for analysis."""
    steps = list(episode['steps'])

    obs_arrays = defaultdict(list)
    action_data = []
    action_dict_data = defaultdict(list)  # Keep dict actions separate too
    rewards = []

    for step in steps:
        # Observations
        if 'observation' in step:
            obs = step['observation']
            if hasattr(obs, 'keys'):
                for key in obs.keys():
                    try:
                        v = obs[key].numpy()
                        if isinstance(v, bytes):
                            continue
                        if v.ndim == 0:
                            obs_arrays[key].append(np.array([float(v)]))
                        elif v.ndim == 1:
                            obs_arrays[key].append(v.astype(np.float64))
                        elif v.ndim == 2 and v.shape[0] * v.shape[1] <= 512:
                            obs_arrays[key].append(v.mean(axis=0).astype(np.float64))
                        # Skip images and large arrays
                    except:
                        pass

        # Actions
        if 'action' in step:
            act = step['action']
            if hasattr(act, 'keys'):
                act_parts = []
                for k in sorted(act.keys()):
                    try:
                        v = act[k].numpy()
                        if v.ndim == 0:
                            val = np.array([float(v)])
                            act_parts.append(val)
                            action_dict_data[k].append(val)
                        elif v.ndim == 1:
                            val = v.astype(np.float64)
                            act_parts.append(val)
                            action_dict_data[k].append(val)
                    except:
                        pass
                if act_parts:
                    action_data.append(np.concatenate(act_parts))
            elif hasattr(act, 'numpy'):
                v = act.numpy().astype(np.float64)
                if v.ndim == 1:
                    action_data.append(v)

        # Rewards
        if 'reward' in step:
            try:
                rewards.append(float(step['reward'].numpy()))
            except:
                pass

    # Convert to arrays
    obs_result = {}
    for key, vals in obs_arrays.items():
        try:
            arr = np.array(vals)
            if arr.ndim == 2:
                obs_result[key] = arr
        except:
            pass

    action_array = None
    if action_data:
        try:
            action_array = np.array(action_data)
            if action_array.ndim != 2:
                action_array = None
        except:
            pass

    # Also store dict actions
    action_dict_result = {}
    for k, vals in action_dict_data.items():
        try:
            arr = np.array(vals)
            if arr.ndim == 2:
                action_dict_result[k] = arr
        except:
            pass

    reward_array = np.array(rewards) if rewards else None

    return obs_result, action_array, reward_array, action_dict_result


def analyze_actions(action_arrays, state_arrays, ds_key):
    """Detailed action analysis."""
    actions = [a for a in action_arrays if a is not None and a.ndim == 2]
    if not actions:
        return {"status": "no_valid_actions"}

    all_actions = np.concatenate(actions, axis=0)
    action_dim = all_actions.shape[1]

    result = {
        "action_dim": action_dim,
        "n_timesteps": len(all_actions),
        "per_dim": {},
        "global_stats": {
            "mean": all_actions.mean(axis=0).tolist(),
            "std": all_actions.std(axis=0).tolist(),
            "min": all_actions.min(axis=0).tolist(),
            "max": all_actions.max(axis=0).tolist(),
        },
    }

    for d in range(action_dim):
        col = all_actions[:, d]
        result["per_dim"][f"dim_{d}"] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "median": float(np.median(col)),
            "pct_zero": float(np.mean(np.abs(col) < 1e-8) * 100),
            "pct_constant": float(np.mean(np.abs(col - col[0]) < 1e-8) * 100),
        }

    # Autocorrelation at lag 1 (smoothness)
    smoothness = []
    for ep_act in actions:
        if len(ep_act) > 5:
            for d in range(min(action_dim, 7)):
                if np.std(ep_act[:, d]) > 1e-8:
                    corr = np.corrcoef(ep_act[:-1, d], ep_act[1:, d])[0, 1]
                    if not np.isnan(corr):
                        smoothness.append(corr)
    result["mean_autocorrelation_lag1"] = float(np.mean(smoothness)) if smoothness else None

    # Action-state delta correlation
    correlations = []
    states = [s for s in state_arrays if s is not None and s.ndim == 2]
    for ep_act, ep_state in zip(actions, states):
        min_len = min(len(ep_act), len(ep_state)) - 1
        if min_len < 5:
            continue
        delta_state = ep_state[1:min_len+1] - ep_state[:min_len]
        act = ep_act[:min_len]
        min_dim = min(act.shape[1], delta_state.shape[1], 7)
        for d in range(min_dim):
            if np.std(act[:, d]) > 1e-8 and np.std(delta_state[:, d]) > 1e-8:
                try:
                    corr = np.corrcoef(act[:, d], delta_state[:, d])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
                except:
                    pass

    result["mean_action_state_delta_correlation"] = float(np.mean(np.abs(correlations))) if correlations else None
    result["action_state_corr_per_dim"] = [float(c) for c in correlations[:20]] if correlations else []

    return result


def analyze_dataset(ds_key, tfds_name, n_episodes=N_EPISODES):
    """Full analysis of one dataset."""
    print(f"\n{'='*70}")
    print(f"ANALYZING: {ds_key} ({tfds_name})")
    print(f"{'='*70}")

    result = {
        "ds_key": ds_key,
        "tfds_name": tfds_name,
        "schema": None,
        "episodes": [],
        "action_analysis": None,
        "label_info": {},
    }

    # 1. Schema inspection
    print("  [1/4] Inspecting schema...")
    schema = inspect_schema(tfds_name)
    result["schema"] = schema
    print(f"    Splits: {schema.get('splits', 'N/A')}")
    print(f"    Size: {schema.get('size', 'N/A')}")
    if 'features' in schema:
        # Print features truncated
        feat_str = schema['features']
        if len(feat_str) > 500:
            print(f"    Features (truncated): {feat_str[:500]}...")
        else:
            print(f"    Features: {feat_str}")

    # 2. Load episodes
    print(f"  [2/4] Loading {n_episodes} episodes...")
    t0 = time.time()
    try:
        ds = tfds.load(
            tfds_name,
            data_dir='gs://gresearch/robotics',
            split=f'train[:{n_episodes}]',
        )
    except Exception as e:
        print(f"    FAILED to load: {e}")
        result["error"] = str(e)
        return result

    # 3. Process all episodes
    all_obs = defaultdict(list)
    all_actions = []
    all_action_dicts = defaultdict(list)
    all_states = []
    all_rewards = []
    ep_lengths = []
    ep_inspections = []
    language_instructions = set()

    for i, episode in enumerate(ds):
        # Deep inspect first 3 episodes
        if i < 3:
            inspection = deep_inspect_episode(episode, ds_key)
            ep_inspections.append(inspection)
            if inspection.get('language_instruction'):
                language_instructions.add(inspection['language_instruction'])

        # Extract full data
        obs_data, act_data, rew_data, act_dict_data = extract_full_episode_data(episode, ds_key)

        for key, arr in obs_data.items():
            all_obs[key].append(arr)

        all_actions.append(act_data)
        all_rewards.append(rew_data)

        for k, v in act_dict_data.items():
            all_action_dicts[k].append(v)

        # Get primary state
        primary_state = None
        for sk in ['state', 'joint_pos', 'robot_state', 'base_pose_tool_reached',
                    'joint_position', 'proprio']:
            if sk in obs_data and obs_data[sk].ndim == 2:
                primary_state = obs_data[sk]
                break
        if primary_state is None:
            # Fall back to any non-image obs
            for k, v in obs_data.items():
                if v.ndim == 2 and v.shape[1] <= 50 and 'image' not in k.lower():
                    primary_state = v
                    break
        all_states.append(primary_state)

        if act_data is not None and act_data.ndim == 2:
            ep_lengths.append(len(act_data))
        elif primary_state is not None and primary_state.ndim == 2:
            ep_lengths.append(len(primary_state))

        if (i + 1) % 5 == 0:
            print(f"    [{i+1}/{n_episodes}] episodes processed")

    elapsed = time.time() - t0
    print(f"    Downloaded {len(ep_lengths)} episodes in {elapsed:.1f}s")

    # 4. Print detailed schema from first episode
    if ep_inspections:
        insp = ep_inspections[0]
        print(f"\n  === RAW SCHEMA for {ds_key} (from episode 0) ===")
        print(f"  Step keys: {insp['other_step_keys']}")
        print(f"\n  OBSERVATION KEYS:")
        for k, v in sorted(insp['observation_keys'].items()):
            print(f"    {k}: {v}")
        print(f"\n  ACTION INFO:")
        for k, v in sorted(insp['action_info'].items()):
            print(f"    {k}: {v}")

        # Print action dict keys statistics if dict-type
        if all_action_dicts:
            print(f"\n  ACTION DICT BREAKDOWN:")
            for k, ep_list in sorted(all_action_dicts.items()):
                valid = [a for a in ep_list if a is not None and a.ndim == 2]
                if valid:
                    concat = np.concatenate(valid, axis=0)
                    print(f"    {k}: dim={concat.shape[1]}, "
                          f"mean={concat.mean():.4f}, std={concat.std():.4f}, "
                          f"range=[{concat.min():.4f}, {concat.max():.4f}]")

        print(f"\n  METADATA:")
        print(f"    Has language: {insp['has_language']}")
        if insp.get('language_instruction'):
            print(f"    Language example: '{insp['language_instruction']}'")
        print(f"    Has reward: {insp['has_reward']}")
        if insp.get('reward_stats'):
            print(f"    Reward stats: {insp['reward_stats']}")
        print(f"    Has is_terminal: {insp['has_is_terminal']}")
        print(f"    Has is_first: {insp['has_is_first']}")
        print(f"    Has is_last: {insp['has_is_last']}")
        print(f"    Episode length: {insp['n_steps']} steps")

    result["inspections"] = ep_inspections
    result["language_instructions"] = list(language_instructions)

    # Episode length stats
    if ep_lengths:
        result["ep_length_stats"] = {
            "min": int(min(ep_lengths)),
            "max": int(max(ep_lengths)),
            "mean": float(np.mean(ep_lengths)),
            "std": float(np.std(ep_lengths)),
            "n_episodes": len(ep_lengths),
        }
        print(f"\n  EPISODE LENGTHS: min={min(ep_lengths)}, max={max(ep_lengths)}, "
              f"mean={np.mean(ep_lengths):.1f}, std={np.std(ep_lengths):.1f}")

    # Observation field statistics
    print(f"\n  === OBSERVATION FIELD STATISTICS ===")
    obs_stats = {}
    for key, ep_list in sorted(all_obs.items()):
        valid = [a for a in ep_list if a is not None and a.ndim == 2]
        if not valid:
            continue
        concat = np.concatenate(valid, axis=0)
        dim = concat.shape[1]
        if dim > 50 or 'image' in key.lower():
            continue
        stats = {
            "dim": dim,
            "n_timesteps": len(concat),
            "mean": concat.mean(axis=0).tolist(),
            "std": concat.std(axis=0).tolist(),
            "min": concat.min(axis=0).tolist(),
            "max": concat.max(axis=0).tolist(),
            "nan_count": int(np.isnan(concat).sum()),
            "const_channels": int((concat.std(axis=0) < 1e-8).sum()),
        }
        obs_stats[key] = stats
        print(f"    {key}: dim={dim}, timesteps={len(concat)}, "
              f"range=[{concat.min():.4f}, {concat.max():.4f}], "
              f"NaN={stats['nan_count']}, const_ch={stats['const_channels']}")
        # Print per-dimension stats for small dims
        if dim <= 10:
            for d in range(dim):
                print(f"      dim[{d}]: mean={concat[:, d].mean():.4f}, "
                      f"std={concat[:, d].std():.4f}, "
                      f"range=[{concat[:, d].min():.4f}, {concat[:, d].max():.4f}]")
    result["obs_stats"] = obs_stats

    # Action analysis
    print(f"\n  === ACTION ANALYSIS ===")
    act_analysis = analyze_actions(all_actions, all_states, ds_key)
    result["action_analysis"] = act_analysis
    if act_analysis.get("action_dim"):
        print(f"    Total action dim: {act_analysis['action_dim']}")
        print(f"    Total action timesteps: {act_analysis['n_timesteps']}")
        print(f"    Smoothness (autocorr lag-1): {act_analysis.get('mean_autocorrelation_lag1', 'N/A')}")
        print(f"    Action-state-delta |corr|: {act_analysis.get('mean_action_state_delta_correlation', 'N/A')}")
        print(f"    Per-dimension stats:")
        for d_key, d_stats in sorted(act_analysis.get("per_dim", {}).items()):
            print(f"      {d_key}: mean={d_stats['mean']:.4f}, std={d_stats['std']:.4f}, "
                  f"range=[{d_stats['min']:.4f}, {d_stats['max']:.4f}], "
                  f"%zero={d_stats['pct_zero']:.1f}%")
    else:
        print(f"    Status: {act_analysis.get('status', 'unknown')}")

    # Label availability
    result["label_info"] = {
        "has_language": any(i.get('has_language') for i in ep_inspections),
        "has_reward": any(i.get('has_reward') for i in ep_inspections),
        "has_is_terminal": any(i.get('has_is_terminal') for i in ep_inspections),
        "has_is_first": any(i.get('has_is_first') for i in ep_inspections),
        "has_is_last": any(i.get('has_is_last') for i in ep_inspections),
        "unique_language_instructions": len(language_instructions),
        "example_instructions": list(language_instructions)[:5],
        "reward_available": any(i.get('reward_stats', {}).get('nonzero_count', 0) > 0 for i in ep_inspections),
    }

    # Save raw data for plotting
    ds_out = OUTPUT_DIR / ds_key
    ds_out.mkdir(parents=True, exist_ok=True)

    for i in range(min(10, len(all_states))):
        if all_states[i] is not None and all_states[i].ndim == 2:
            np.save(ds_out / f"state_ep{i:03d}.npy", all_states[i].astype(np.float32))
        if all_actions[i] is not None and all_actions[i].ndim == 2:
            np.save(ds_out / f"action_ep{i:03d}.npy", all_actions[i].astype(np.float32))

    for key, ep_list in all_obs.items():
        valid = [a for a in ep_list if a is not None and a.ndim == 2]
        if valid and 'image' not in key.lower():
            concat = np.concatenate(valid[:5], axis=0)
            if concat.shape[1] <= 50:
                np.save(ds_out / f"obs_{key}.npy", concat.astype(np.float32))

    # Save metadata
    meta_path = ds_out / "analysis.json"
    try:
        serializable = json.loads(json.dumps(result, default=str))
        with open(meta_path, "w") as f:
            json.dump(serializable, f, indent=2)
    except Exception as e:
        print(f"    Warning: could not save analysis.json: {e}")

    return result


def try_droid():
    """Try loading DROID."""
    print(f"\n{'='*70}")
    print("ATTEMPTING DROID (droid_100)")
    print(f"{'='*70}")
    try:
        ds = tfds.load(
            'droid_100',
            data_dir='gs://gresearch/robotics',
            split='train[:5]',
        )
        print("  DROID loaded successfully!")
        for i, episode in enumerate(ds):
            if i == 0:
                inspection = deep_inspect_episode(episode, 'droid')
                print(f"\n  === RAW SCHEMA for DROID ===")
                print(f"  Step keys: {inspection['other_step_keys']}")
                print(f"\n  OBSERVATION KEYS:")
                for k, v in sorted(inspection['observation_keys'].items()):
                    if 'image' not in k.lower() or 'IMAGE' not in str(v):
                        print(f"    {k}: {v}")
                    else:
                        print(f"    {k}: [IMAGE]")
                print(f"\n  ACTION INFO:")
                for k, v in sorted(inspection['action_info'].items()):
                    print(f"    {k}: {v}")
                print(f"\n  Has language: {inspection['has_language']}")
                if inspection.get('language_instruction'):
                    print(f"  Language: '{inspection['language_instruction']}'")
                print(f"  Has reward: {inspection['has_reward']}")
                print(f"  Episode length: {inspection['n_steps']}")
                return inspection
    except Exception as e:
        print(f"  DROID failed: {e}")
        traceback.print_exc()
    return None


def estimate_download_sizes(all_results):
    """Estimate full download sizes for proprio-only extraction."""
    print(f"\n{'='*70}")
    print("DOWNLOAD SIZE ESTIMATES (PROPRIO-ONLY)")
    print(f"{'='*70}")

    estimates = {}
    for ds_key, res in all_results.items():
        if 'error' in res and ds_key != 'droid_100':
            continue
        schema = res.get('schema', {})
        splits = schema.get('splits', {})
        total_eps = splits.get('train', 0)

        ep_stats = res.get('ep_length_stats', {})
        mean_len = ep_stats.get('mean', 100)

        obs_stats = res.get('obs_stats', {})
        total_dim = sum(s['dim'] for k, s in obs_stats.items()
                       if 'image' not in k.lower() and s.get('dim', 0) <= 50)

        act_dim = res.get('action_analysis', {}).get('action_dim', 0)

        # float32: 4 bytes per value
        bytes_per_step = (total_dim + act_dim) * 4
        total_bytes = total_eps * mean_len * bytes_per_step
        total_mb = total_bytes / 1e6

        full_size_str = schema.get('size', 'unknown')

        est = {
            "total_episodes": total_eps,
            "mean_ep_length": round(mean_len),
            "total_proprio_dim": total_dim,
            "action_dim": act_dim,
            "proprio_only_mb": round(total_mb, 1),
            "full_rlds_size": full_size_str,
            "download_time_estimate": f"~{max(1, int(total_eps / 50))} min (at ~50 eps/min via GCS)",
        }
        estimates[ds_key] = est

        print(f"\n  {ds_key}:")
        print(f"    Total episodes: {total_eps:,}")
        print(f"    Mean ep length: {round(mean_len)}")
        print(f"    Proprio dims: {total_dim} (state) + {act_dim} (action) = {total_dim + act_dim}")
        print(f"    Proprio-only size: ~{total_mb:.1f} MB ({total_mb/1000:.2f} GB)")
        print(f"    Full RLDS size: {full_size_str}")
        print(f"    Download time: {est['download_time_estimate']}")

    total_mb = sum(e['proprio_only_mb'] for e in estimates.values())
    print(f"\n  {'='*50}")
    print(f"  TOTAL proprio-only: ~{total_mb:.0f} MB ({total_mb/1000:.1f} GB)")
    print(f"  Available disk: ~46 GB")
    print(f"  Fits on machine: {'YES easily' if total_mb < 10000 else 'YES' if total_mb < 40000 else 'TIGHT'}")
    print(f"  Note: Download from GCS requires streaming through TF, which downloads")
    print(f"  full RLDS records (including images). We only SAVE proprio fields.")

    return estimates


def create_figures(all_results):
    """Create comprehensive analysis figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    colors = {
        'toto': '#e74c3c',
        'stanford_kuka': '#3498db',
        'berkeley_ur5': '#2ecc71',
        'berkeley_fanuc': '#f39c12',
        'jaco_play': '#9b59b6',
        'maniskill': '#1abc9c',
        'fractal': '#95a5a6',
    }

    fig = plt.figure(figsize=(24, 22))
    gs = GridSpec(5, 4, figure=fig, hspace=0.4, wspace=0.3,
                  top=0.95, bottom=0.03, left=0.05, right=0.97)
    fig.suptitle('Open X-Embodiment: Deep Proprioceptive Analysis for Mechanical-JEPA',
                 fontsize=16, fontweight='bold', y=0.98)

    success = {k: v for k, v in all_results.items()
               if 'error' not in v and v.get('ep_length_stats')}

    # ============================================================
    # ROW 1: Data Landscape
    # ============================================================

    # 1a. Dataset sizes bubble chart
    ax1 = fig.add_subplot(gs[0, 0])
    for ds_key, res in success.items():
        ep_stats = res.get('ep_length_stats', {})
        mean_len = ep_stats.get('mean', 0)
        splits = res.get('schema', {}).get('splits', {})
        total_available = splits.get('train', ep_stats.get('n_episodes', 0))
        ax1.scatter(total_available, mean_len,
                   s=max(80, min(500, total_available / 50)),
                   c=colors.get(ds_key, 'gray'), alpha=0.7,
                   edgecolors='black', linewidth=0.5, zorder=5)
        ax1.annotate(ds_key, (total_available, mean_len),
                    fontsize=7, ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points')
    ax1.set_xlabel('Total Episodes (train split)', fontsize=8)
    ax1.set_ylabel('Mean Episode Length (steps)', fontsize=8)
    ax1.set_title('Dataset Scale', fontsize=10, fontweight='bold')
    ax1.set_xscale('log')
    ax1.tick_params(labelsize=7)
    ax1.grid(True, alpha=0.3)

    # 1b. State + action dimensionality
    ax2 = fig.add_subplot(gs[0, 1])
    ds_names_sorted = sorted(success.keys())
    state_dims = []
    action_dims = []
    for ds_key in ds_names_sorted:
        res = success[ds_key]
        obs_stats = res.get('obs_stats', {})
        total_proprio = sum(s['dim'] for k, s in obs_stats.items()
                          if 'image' not in k.lower() and s.get('dim', 0) <= 50)
        state_dims.append(total_proprio)
        action_dims.append(res.get('action_analysis', {}).get('action_dim', 0))

    x = np.arange(len(ds_names_sorted))
    width = 0.35
    ax2.bar(x - width/2, state_dims, width, label='Proprio Dim',
            color=[colors.get(n, 'gray') for n in ds_names_sorted], alpha=0.8,
            edgecolor='black', linewidth=0.5)
    ax2.bar(x + width/2, action_dims, width, label='Action Dim',
            color=[colors.get(n, 'gray') for n in ds_names_sorted], alpha=0.4,
            edgecolor='black', linewidth=0.5, hatch='//')
    ax2.set_xticks(x)
    ax2.set_xticklabels(ds_names_sorted, rotation=45, ha='right', fontsize=7)
    ax2.set_ylabel('Dimensions', fontsize=8)
    ax2.set_title('State & Action Dimensionality', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=7)
    ax2.tick_params(labelsize=7)
    ax2.grid(True, alpha=0.3, axis='y')

    # 1c. Episode length range
    ax3 = fig.add_subplot(gs[0, 2])
    ep_data = []
    ep_labels = []
    ep_colors_list = []
    for ds_key in ds_names_sorted:
        res = success[ds_key]
        ep_stats = res.get('ep_length_stats', {})
        if ep_stats:
            ep_data.append([ep_stats['min'], ep_stats['mean'], ep_stats['max']])
            ep_labels.append(ds_key)
            ep_colors_list.append(colors.get(ds_key, 'gray'))

    if ep_data:
        ep_arr = np.array(ep_data)
        y = np.arange(len(ep_labels))
        ax3.barh(y, ep_arr[:, 2] - ep_arr[:, 0], left=ep_arr[:, 0], height=0.6,
                color=ep_colors_list, alpha=0.3, edgecolor='black', linewidth=0.5)
        ax3.scatter(ep_arr[:, 1], y, c=ep_colors_list, zorder=5, s=60,
                   edgecolors='black', linewidth=0.5, marker='D')
        ax3.set_yticks(y)
        ax3.set_yticklabels(ep_labels, fontsize=7)
        ax3.set_xlabel('Steps per Episode', fontsize=8)
        ax3.set_title('Episode Length Range (diamond=mean)', fontsize=10, fontweight='bold')
        ax3.tick_params(labelsize=7)
        ax3.grid(True, alpha=0.3, axis='x')

    # 1d. Action space summary text
    ax4 = fig.add_subplot(gs[0, 3])
    lines = []
    for ds_key in ds_names_sorted:
        res = success[ds_key]
        insp_list = res.get('inspections', [])
        if not insp_list:
            continue
        insp = insp_list[0]
        act_info = insp.get('action_info', {})
        act_type = act_info.get('_type', 'unknown')
        act_dim = res.get('action_analysis', {}).get('action_dim', '?')
        autocorr = res.get('action_analysis', {}).get('mean_autocorrelation_lag1')
        act_corr = res.get('action_analysis', {}).get('mean_action_state_delta_correlation')

        line = f"{ds_key}:\n  type={act_type}, dim={act_dim}"
        if autocorr is not None:
            line += f"\n  smooth={autocorr:.3f}"
        if act_corr is not None:
            line += f", act-st={act_corr:.3f}"
        lines.append(line)

    ax4.text(0.05, 0.95, '\n'.join(lines), transform=ax4.transAxes,
             fontsize=6.5, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax4.set_title('Action Space Summary', fontsize=10, fontweight='bold')
    ax4.axis('off')

    # ============================================================
    # ROW 2: State trajectories (one subplot per dataset)
    # ============================================================
    ds_list = list(success.keys())[:4]
    for idx, ds_key in enumerate(ds_list):
        ax = fig.add_subplot(gs[1, idx])
        ds_dir = OUTPUT_DIR / ds_key
        state_files = sorted(ds_dir.glob("state_ep*.npy"))[:3]
        if state_files:
            for j, sf in enumerate(state_files):
                state = np.load(sf)
                n_dims = min(4, state.shape[1])
                for d in range(n_dims):
                    ls = ['-', '--', ':'][j % 3]
                    ax.plot(state[:, d], alpha=0.6, linewidth=0.8, linestyle=ls,
                           label=f'd{d}' if j == 0 else None)
        ax.set_title(f'{ds_key}: State', fontsize=9, fontweight='bold',
                    color=colors.get(ds_key, 'black'))
        ax.set_xlabel('Step', fontsize=7)
        ax.set_ylabel('Value (rad)', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=5, ncol=2, loc='best')

    # ============================================================
    # ROW 3: Action trajectories
    # ============================================================
    for idx, ds_key in enumerate(ds_list):
        ax = fig.add_subplot(gs[2, idx])
        ds_dir = OUTPUT_DIR / ds_key
        action_files = sorted(ds_dir.glob("action_ep*.npy"))[:3]
        if action_files:
            for j, af in enumerate(action_files):
                act = np.load(af)
                n_dims = min(4, act.shape[1])
                for d in range(n_dims):
                    ls = ['-', '--', ':'][j % 3]
                    ax.plot(act[:, d], alpha=0.6, linewidth=0.8, linestyle=ls,
                           label=f'd{d}' if j == 0 else None)
        else:
            ax.text(0.5, 0.5, 'No action data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10, color='gray')
        ax.set_title(f'{ds_key}: Actions', fontsize=9, fontweight='bold',
                    color=colors.get(ds_key, 'black'))
        ax.set_xlabel('Step', fontsize=7)
        ax.set_ylabel('Action Value', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
        if idx == 0 and action_files:
            ax.legend(fontsize=5, ncol=2, loc='best')

    # ============================================================
    # ROW 4: Cross-embodiment + correlation + labels
    # ============================================================

    # 4a-b. Normalized joint trajectories overlaid
    ax_cross = fig.add_subplot(gs[3, 0:2])
    for ds_key in ds_names_sorted:
        ds_dir = OUTPUT_DIR / ds_key
        state_files = sorted(ds_dir.glob("state_ep*.npy"))[:1]
        if state_files:
            state = np.load(state_files[0])
            s_min = state.min(axis=0, keepdims=True)
            s_max = state.max(axis=0, keepdims=True)
            s_range = s_max - s_min
            s_range[s_range < 1e-8] = 1
            norm = (state - s_min) / s_range
            t = np.linspace(0, 1, len(norm))
            ax_cross.plot(t, norm[:, 0], label=ds_key,
                         color=colors.get(ds_key, 'gray'),
                         alpha=0.8, linewidth=1.5)
    ax_cross.set_xlabel('Normalized Time [0,1]', fontsize=8)
    ax_cross.set_ylabel('Normalized Joint 0 [0,1]', fontsize=8)
    ax_cross.set_title('Cross-Embodiment: Joint 0 Trajectories (Normalized)',
                       fontsize=10, fontweight='bold')
    ax_cross.legend(fontsize=7, ncol=2)
    ax_cross.tick_params(labelsize=7)
    ax_cross.grid(True, alpha=0.3)

    # 4c. Action quality metrics
    ax_corr = fig.add_subplot(gs[3, 2])
    corr_ds = []
    corr_act_state = []
    corr_auto = []
    for ds_key in ds_names_sorted:
        res = success[ds_key]
        act_a = res.get('action_analysis', {})
        c = act_a.get('mean_action_state_delta_correlation')
        a = act_a.get('mean_autocorrelation_lag1')
        if c is not None or a is not None:
            corr_ds.append(ds_key)
            corr_act_state.append(c if c is not None else 0)
            corr_auto.append(a if a is not None else 0)

    if corr_ds:
        y = np.arange(len(corr_ds))
        ax_corr.barh(y - 0.15, corr_act_state, 0.3, label='|Act-dState| corr',
                     color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax_corr.barh(y + 0.15, corr_auto, 0.3, label='Act autocorr (lag-1)',
                     color='coral', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax_corr.set_yticks(y)
        ax_corr.set_yticklabels(corr_ds, fontsize=7)
        ax_corr.set_xlabel('Correlation', fontsize=8)
        ax_corr.set_title('Action Quality Metrics', fontsize=10, fontweight='bold')
        ax_corr.legend(fontsize=7)
        ax_corr.tick_params(labelsize=7)
        ax_corr.grid(True, alpha=0.3, axis='x')

    # 4d. Label availability matrix
    ax_labels = fig.add_subplot(gs[3, 3])
    label_cols = ['language', 'reward', 'terminal', 'first', 'last']
    label_matrix = []
    label_ds = []
    for ds_key in ds_names_sorted:
        res = success[ds_key]
        li = res.get('label_info', {})
        insps = res.get('inspections', [{}])
        insp0 = insps[0] if insps else {}
        row = [
            1 if li.get('has_language') else 0,
            1 if li.get('has_reward') else 0,
            1 if li.get('has_is_terminal') or insp0.get('has_is_terminal') else 0,
            1 if li.get('has_is_first') or insp0.get('has_is_first') else 0,
            1 if li.get('has_is_last') or insp0.get('has_is_last') else 0,
        ]
        label_matrix.append(row)
        label_ds.append(ds_key)

    if label_matrix:
        arr = np.array(label_matrix)
        ax_labels.imshow(arr, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax_labels.set_xticks(np.arange(len(label_cols)))
        ax_labels.set_xticklabels(label_cols, rotation=45, ha='right', fontsize=7)
        ax_labels.set_yticks(np.arange(len(label_ds)))
        ax_labels.set_yticklabels(label_ds, fontsize=7)
        ax_labels.set_title('Label Availability', fontsize=10, fontweight='bold')
        for i in range(len(label_ds)):
            for j in range(len(label_cols)):
                sym = 'Y' if arr[i, j] else 'N'
                color = 'white' if arr[i, j] else 'black'
                ax_labels.text(j, i, sym, ha='center', va='center',
                             fontsize=8, fontweight='bold', color=color)

    # ============================================================
    # ROW 5: Evaluation design + scale comparison
    # ============================================================

    # 5a-b. Evaluation taxonomy
    ax_eval = fig.add_subplot(gs[4, 0:2])
    eval_text = (
        "PROPOSED EVALUATION TAXONOMY FOR MECHANICAL-JEPA\n"
        "\n"
        "SANITY CHECKS (Easy):\n"
        "  1. Embodiment Classification: classify robot from latent z\n"
        "     Labels: dataset name | Metric: accuracy | Baseline: linear-on-raw\n"
        "  2. Gripper State Prediction: open/closed from latent\n"
        "     Labels: gripper channel threshold | Metric: binary acc\n"
        "  3. Motion Phase Detection: approach/grasp/lift/place\n"
        "     Labels: derive from gripper+velocity | Metric: F1\n"
        "\n"
        "MEDIUM (Transfer):\n"
        "  4. Cross-Embodiment Forecasting: pretrain Franka -> eval KUKA/UR5\n"
        "     Metric: MSE next-state | No extra labels needed\n"
        "  5. Few-Shot Task: 10 demos+pretrained vs 100 from scratch\n"
        "     Labels: task language instructions | Metric: trajectory MSE\n"
        "\n"
        "HIGH-IMPACT (Novel):\n"
        "  6. Action-Conditioned Forecasting: (s_t, a_t) -> s_{t+k}\n"
        "     Metric: multi-step rollout MSE | Core JEPA capability\n"
        "  7. Universal Dynamics: single encoder for all robots\n"
        "     Metric: cross-robot prediction quality\n"
        "  8. Contact/Force Prediction: predict KUKA forces from joint state\n"
        "     Labels: force channels (KUKA only) | Metric: MSE/correlation"
    )
    ax_eval.text(0.02, 0.98, eval_text, transform=ax_eval.transAxes,
                fontsize=6.5, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax_eval.set_title('Evaluation Design', fontsize=10, fontweight='bold')
    ax_eval.axis('off')

    # 5c-d. Scale comparison
    ax_scale = fig.add_subplot(gs[4, 2:4])
    categories = ['Subjects/\nTrajectories', 'Timesteps/\nInstance', 'Channels\n(proprio)', 'Total\nTokens (M)', 'Cross-Domain\nVariety']
    brain_vals = [32000, 160, 450, 2300, 1]
    oxe_vals = [200000, 120, 18, 31, 7]

    x = np.arange(len(categories))
    width = 0.35
    brain_log = [np.log10(max(v, 1)) for v in brain_vals]
    oxe_log = [np.log10(max(v, 1)) for v in oxe_vals]

    bars1 = ax_scale.bar(x - width/2, brain_log, width, label='Brain-JEPA',
                         color='mediumpurple', alpha=0.7, edgecolor='black', linewidth=0.5)
    bars2 = ax_scale.bar(x + width/2, oxe_log, width, label='OXE Mechanical-JEPA',
                         color='coral', alpha=0.7, edgecolor='black', linewidth=0.5)

    for bar, val in zip(bars1, brain_vals):
        ax_scale.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                     f'{val:,}', ha='center', va='bottom', fontsize=6)
    for bar, val in zip(bars2, oxe_vals):
        ax_scale.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                     f'{val:,}', ha='center', va='bottom', fontsize=6)

    ax_scale.set_xticks(x)
    ax_scale.set_xticklabels(categories, fontsize=7)
    ax_scale.set_ylabel('log10(value)', fontsize=8)
    ax_scale.set_title('Scale: Brain-JEPA vs OXE Mechanical-JEPA', fontsize=10, fontweight='bold')
    ax_scale.legend(fontsize=8)
    ax_scale.tick_params(labelsize=7)
    ax_scale.grid(True, alpha=0.3, axis='y')

    plt.savefig(FIGURE_DIR / 'oxe_deep_analysis.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\nFigure saved to {FIGURE_DIR / 'oxe_deep_analysis.png'}")
    plt.close()


def main():
    print("=" * 70)
    print("OXE DEEP PROPRIOCEPTIVE ANALYSIS FOR MECHANICAL-JEPA")
    print("=" * 70)

    all_results = {}

    # Analyze each dataset
    for ds_key, tfds_name in DATASETS.items():
        try:
            result = analyze_dataset(ds_key, tfds_name, n_episodes=N_EPISODES)
            all_results[ds_key] = result
        except Exception as e:
            print(f"\n  FATAL ERROR analyzing {ds_key}: {e}")
            traceback.print_exc()
            all_results[ds_key] = {"error": str(e)}

    # Try DROID
    droid_result = try_droid()
    if droid_result:
        all_results["droid_100"] = {
            "inspections": [droid_result],
            "schema": {"note": "loaded successfully"},
        }

    # Download size estimates
    estimates = estimate_download_sizes(all_results)

    # Create figures
    print(f"\n{'='*70}")
    print("CREATING COMPREHENSIVE ANALYSIS FIGURE")
    print(f"{'='*70}")
    try:
        create_figures(all_results)
    except Exception as e:
        print(f"Figure creation failed: {e}")
        traceback.print_exc()

    # Save comprehensive results
    summary_path = OUTPUT_DIR / "full_analysis_summary.json"
    try:
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nFull results saved to {summary_path}")
    except:
        pass

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    for ds_key in list(DATASETS.keys()) + (['droid_100'] if droid_result else []):
        res = all_results.get(ds_key, {})
        if 'error' in res:
            print(f"\n  {ds_key}: FAILED - {res['error'][:80]}")
            continue

        ep_stats = res.get('ep_length_stats', {})
        act = res.get('action_analysis', {})
        labels = res.get('label_info', {})

        print(f"\n  {ds_key}:")
        if ep_stats:
            print(f"    Episodes: {ep_stats.get('n_episodes', '?')}, "
                  f"Length: {ep_stats.get('min', '?')}-{ep_stats.get('max', '?')} "
                  f"(mean={ep_stats.get('mean', 0):.0f})")
        if act and act.get('action_dim'):
            print(f"    Action dim: {act['action_dim']}")
            if act.get('mean_autocorrelation_lag1') is not None:
                print(f"    Action smoothness (autocorr): {act['mean_autocorrelation_lag1']:.3f}")
            if act.get('mean_action_state_delta_correlation') is not None:
                print(f"    Action-state correlation: {act['mean_action_state_delta_correlation']:.3f}")
        if labels:
            print(f"    Labels: language={labels.get('has_language')}, "
                  f"reward={labels.get('has_reward')}, "
                  f"n_instructions={labels.get('unique_language_instructions', 0)}")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
