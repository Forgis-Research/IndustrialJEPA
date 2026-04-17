"""
Create comprehensive OXE deep analysis figure for Mechanical-JEPA.
Produces datasets/analysis/figures/oxe_deep_analysis.png
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# Load analysis results
ANALYSIS_PATH = Path("/home/sagemaker-user/IndustrialJEPA/datasets/data/oxe_analysis/deep_analysis.json")
FIGURE_PATH = Path("/home/sagemaker-user/IndustrialJEPA/datasets/analysis/figures/oxe_deep_analysis.png")

with open(ANALYSIS_PATH) as f:
    results = json.load(f)

# ============================================================
# Dataset metadata (manually verified from analysis output)
# ============================================================
DATASETS = {
    "toto":          {"robot": "Franka Panda",    "color": "#2196F3", "marker": "o"},
    "stanford_kuka": {"robot": "KUKA iiwa",       "color": "#F44336", "marker": "s"},
    "berkeley_ur5":  {"robot": "UR5",             "color": "#4CAF50", "marker": "^"},
    "berkeley_fanuc":{"robot": "FANUC",           "color": "#FF9800", "marker": "D"},
    "jaco_play":     {"robot": "JACO",            "color": "#9C27B0", "marker": "v"},
    "maniskill":     {"robot": "Panda (sim)",     "color": "#00BCD4", "marker": "P"},
    "fractal":       {"robot": "Everyday Robot",  "color": "#795548", "marker": "X"},
    "droid":         {"robot": "Franka (DROID)",  "color": "#3F51B5", "marker": "*"},
}

# Full-scale episode counts (from registry/schema, not just our sample)
FULL_EPISODES = {
    "toto": 902+101,
    "stanford_kuka": 3000,
    "berkeley_ur5": 1000,
    "berkeley_fanuc": 415,
    "jaco_play": 1000,
    "maniskill": 30213,
    "fractal": 87212,
    "droid": 76000,
}

# Action space descriptions (from schema inspection)
ACTION_SPACES = {
    "toto": "EE delta: world_vector(3) + rotation_delta(3) + open_gripper(1)",
    "stanford_kuka": "EE delta: position(3) + gripper(1) = 4D tensor",
    "berkeley_ur5": "EE delta: world_vector(3) + rotation_delta(3) + gripper(1) + terminate(1)",
    "berkeley_fanuc": "EE delta: dx,dy,dz + droll,dpitch,dyaw = 6D tensor",
    "jaco_play": "EE delta: world_vector(3) + gripper(1) + terminate(3)",
    "maniskill": "EE delta: position(3) + orientation_aa(3) + gripper(1) = 7D tensor",
    "fractal": "EE + base: world(3) + rot(3) + grip(1) + base_disp(2) + base_rot(1) + term(3)",
    "droid": "Joint vel: 6x joint velocities + 1x gripper = 7D tensor",
}

# State keys that matter for JEPA
PRIMARY_STATE = {
    "toto": ("state", 7),
    "stanford_kuka": ("joint_pos", 7),
    "berkeley_ur5": ("robot_state", 15),
    "berkeley_fanuc": ("state", 13),
    "jaco_play": ("joint_pos", 8),
    "maniskill": ("state", 18),
    "fractal": ("base_pose_tool_reached", 7),
    "droid": ("joint_position", 7),
}


def create_figure():
    fig = plt.figure(figsize=(24, 30))
    fig.suptitle("OXE Deep Analysis for Mechanical-JEPA", fontsize=20, fontweight='bold', y=0.98)

    # 6 rows of subplots
    gs = gridspec.GridSpec(6, 4, figure=fig, hspace=0.35, wspace=0.3,
                           top=0.96, bottom=0.03, left=0.06, right=0.97)

    # ================================================================
    # ROW 1: Data Landscape
    # ================================================================

    # 1a: Dataset sizes (bubble chart: episodes x mean_steps)
    ax1a = fig.add_subplot(gs[0, 0:2])
    for key in DATASETS:
        if key not in results or results[key].get('status') != 'success':
            continue
        r = results[key]
        ep_count = FULL_EPISODES.get(key, len(r['episode_lengths']['all']))
        mean_steps = r['episode_lengths']['mean']
        total_tokens = ep_count * mean_steps
        size = max(50, min(2000, total_tokens / 5000))
        ax1a.scatter(ep_count, mean_steps, s=size, c=DATASETS[key]['color'],
                    marker=DATASETS[key]['marker'], alpha=0.8, edgecolors='black', linewidths=0.5,
                    zorder=3, label=f"{DATASETS[key]['robot']} ({ep_count:,} eps)")
        ax1a.annotate(DATASETS[key]['robot'], (ep_count, mean_steps),
                     textcoords="offset points", xytext=(8, 5), fontsize=7)
    ax1a.set_xscale('log')
    ax1a.set_xlabel("Episodes (full dataset)", fontsize=9)
    ax1a.set_ylabel("Mean Steps/Episode", fontsize=9)
    ax1a.set_title("Dataset Scale (bubble = total tokens)", fontsize=11, fontweight='bold')
    ax1a.grid(True, alpha=0.3)

    # 1b: State dimensionality comparison
    ax1b = fig.add_subplot(gs[0, 2])
    ds_names = []
    state_dims = []
    colors = []
    for key in DATASETS:
        if key not in results or results[key].get('status') != 'success':
            continue
        ds_names.append(DATASETS[key]['robot'])
        state_dims.append(PRIMARY_STATE[key][1])
        colors.append(DATASETS[key]['color'])

    bars = ax1b.barh(range(len(ds_names)), state_dims, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1b.set_yticks(range(len(ds_names)))
    ax1b.set_yticklabels(ds_names, fontsize=8)
    ax1b.set_xlabel("Primary State Dim", fontsize=9)
    ax1b.set_title("State Dimensionality", fontsize=11, fontweight='bold')
    for i, (bar, dim) in enumerate(zip(bars, state_dims)):
        ax1b.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{dim}D", va='center', fontsize=8, fontweight='bold')
    ax1b.grid(True, alpha=0.3, axis='x')

    # 1c: Action space comparison
    ax1c = fig.add_subplot(gs[0, 3])
    act_dims = []
    act_types = []
    for key in DATASETS:
        if key not in results or results[key].get('status') != 'success':
            continue
        act_dim = results[key].get('total_action_dim', 0)
        act_dims.append(act_dim)
        # Classify action type
        if key == "droid":
            act_types.append("Joint Vel")
        elif key in ("stanford_kuka", "berkeley_fanuc", "maniskill"):
            act_types.append("EE Delta (tensor)")
        else:
            act_types.append("EE Delta (dict)")

    bars = ax1c.barh(range(len(ds_names)), act_dims, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1c.set_yticks(range(len(ds_names)))
    ax1c.set_yticklabels(ds_names, fontsize=8)
    ax1c.set_xlabel("Total Action Dim", fontsize=9)
    ax1c.set_title("Action Dimensionality", fontsize=11, fontweight='bold')
    for i, (bar, dim, atype) in enumerate(zip(bars, act_dims, act_types)):
        ax1c.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                 f"{dim}D ({atype})", va='center', fontsize=7)
    ax1c.grid(True, alpha=0.3, axis='x')

    # ================================================================
    # ROW 2: State Analysis
    # ================================================================

    # 2a: State value distributions (box plot per dataset)
    ax2a = fig.add_subplot(gs[1, 0:2])
    box_data = []
    box_labels = []
    box_colors = []
    for key in DATASETS:
        if key not in results or results[key].get('status') != 'success':
            continue
        obs = results[key].get('observations', {})
        state_key = PRIMARY_STATE[key][0]
        if state_key in obs and 'mean' in obs[state_key]:
            means = np.array(obs[state_key]['mean'])
            stds = np.array(obs[state_key]['std'])
            mins = np.array(obs[state_key]['min'])
            maxs = np.array(obs[state_key]['max'])
            # Create synthetic distribution from stats
            for d in range(len(means)):
                box_data.append([mins[d], means[d] - stds[d], means[d], means[d] + stds[d], maxs[d]])
            box_labels.append(DATASETS[key]['robot'])
            box_colors.append(DATASETS[key]['color'])

    # Simplified: plot mean +/- std per dataset
    positions = []
    pos = 0
    for key in DATASETS:
        if key not in results or results[key].get('status') != 'success':
            continue
        obs = results[key].get('observations', {})
        state_key = PRIMARY_STATE[key][0]
        if state_key in obs and 'mean' in obs[state_key]:
            means = np.array(obs[state_key]['mean'])
            stds = np.array(obs[state_key]['std'])
            dims = range(len(means))
            for d in dims:
                ax2a.errorbar(pos, means[d], yerr=stds[d], fmt=DATASETS[key]['marker'],
                            color=DATASETS[key]['color'], markersize=4, capsize=2, alpha=0.7)
                pos += 1
            pos += 2  # gap between datasets

    ax2a.set_xlabel("State Dimension Index (grouped by dataset)", fontsize=9)
    ax2a.set_ylabel("Value (rad or normalized)", fontsize=9)
    ax2a.set_title("State Channel Distributions (mean +/- std)", fontsize=11, fontweight='bold')
    # Add legend
    legend_handles = [mpatches.Patch(color=DATASETS[k]['color'], label=DATASETS[k]['robot'])
                     for k in DATASETS if k in results and results[k].get('status') == 'success']
    ax2a.legend(handles=legend_handles, fontsize=7, ncol=4, loc='upper right')
    ax2a.grid(True, alpha=0.3)

    # 2b: Episode length distribution
    ax2b = fig.add_subplot(gs[1, 2])
    ep_len_data = []
    ep_len_labels = []
    ep_len_colors = []
    for key in DATASETS:
        if key not in results or results[key].get('status') != 'success':
            continue
        ep_len_data.append(results[key]['episode_lengths']['all'])
        ep_len_labels.append(DATASETS[key]['robot'])
        ep_len_colors.append(DATASETS[key]['color'])

    bp = ax2b.boxplot(ep_len_data, vert=True, patch_artist=True, labels=ep_len_labels)
    for patch, color in zip(bp['boxes'], ep_len_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2b.set_ylabel("Steps/Episode", fontsize=9)
    ax2b.set_title("Episode Length Distribution", fontsize=11, fontweight='bold')
    ax2b.tick_params(axis='x', rotation=45, labelsize=7)
    ax2b.grid(True, alpha=0.3, axis='y')

    # 2c: Data quality summary
    ax2c = fig.add_subplot(gs[1, 3])
    quality_data = []
    for key in DATASETS:
        if key not in results or results[key].get('status') != 'success':
            continue
        obs = results[key].get('observations', {})
        state_key = PRIMARY_STATE[key][0]
        nan_count = obs.get(state_key, {}).get('nan_count', 0)
        const_ch = obs.get(state_key, {}).get('const_channels', 0)
        quality_data.append({
            'robot': DATASETS[key]['robot'],
            'nan': nan_count,
            'const': const_ch,
            'color': DATASETS[key]['color'],
        })

    y_pos = range(len(quality_data))
    for i, qd in enumerate(quality_data):
        ax2c.barh(i, 0 if qd['nan'] == 0 else qd['nan'], color='red', alpha=0.5, height=0.35, label='NaN' if i==0 else '')
        ax2c.barh(i + 0.35, qd['const'], color='orange', alpha=0.5, height=0.35, label='Const Ch' if i==0 else '')
        ax2c.text(0.5, i + 0.15, f"NaN={qd['nan']}, Const={qd['const']}", fontsize=7, va='center')

    ax2c.set_yticks([i + 0.15 for i in range(len(quality_data))])
    ax2c.set_yticklabels([qd['robot'] for qd in quality_data], fontsize=8)
    ax2c.set_xlabel("Count", fontsize=9)
    ax2c.set_title("Data Quality (Primary State)", fontsize=11, fontweight='bold')
    ax2c.legend(fontsize=7)
    ax2c.grid(True, alpha=0.3, axis='x')

    # ================================================================
    # ROW 3: Action Analysis
    # ================================================================

    # 3a: Action statistics per dataset
    ax3a = fig.add_subplot(gs[2, 0:2])
    act_info = []
    for key in DATASETS:
        if key not in results or results[key].get('status') != 'success':
            continue
        r = results[key]
        acts = r.get('actions', {})
        # Find the primary action (largest dim, skip terminate)
        best_key = None
        best_dim = 0
        for ak, av in acts.items():
            if 'terminate' in ak:
                continue
            if isinstance(av, dict) and 'dim' in av:
                if av['dim'] > best_dim:
                    best_key = ak
                    best_dim = av['dim']
        if best_key:
            av = acts[best_key]
            act_info.append({
                'robot': DATASETS[key]['robot'],
                'key': key,
                'act_key': best_key,
                'dim': av['dim'],
                'mean': np.array(av['mean']),
                'std': np.array(av['std']),
                'min': np.array(av['min']),
                'max': np.array(av['max']),
                'color': DATASETS[key]['color'],
                'autocorr': av.get('autocorrelation_lag1', None),
            })

    pos = 0
    for ai in act_info:
        dims = range(ai['dim'])
        for d in dims:
            ax3a.bar(pos, ai['std'][d], color=ai['color'], alpha=0.7, edgecolor='black', linewidth=0.3)
            ax3a.errorbar(pos, ai['mean'][d], yerr=[[ai['mean'][d] - ai['min'][d]], [ai['max'][d] - ai['mean'][d]]],
                        fmt='none', color='black', capsize=1, linewidth=0.5)
            pos += 1
        pos += 1  # gap

    ax3a.set_xlabel("Action Dimension (grouped by dataset)", fontsize=9)
    ax3a.set_ylabel("Std Dev / Range", fontsize=9)
    ax3a.set_title("Action Statistics (bars=std, whiskers=min/max)", fontsize=11, fontweight='bold')
    ax3a.legend(handles=legend_handles, fontsize=7, ncol=4, loc='upper right')
    ax3a.grid(True, alpha=0.3)

    # 3b: Action smoothness (autocorrelation)
    ax3b = fig.add_subplot(gs[2, 2])
    auto_labels = []
    auto_vals = []
    auto_colors = []
    for ai in act_info:
        if ai['autocorr']:
            valid_ac = [float(a) for a in ai['autocorr'] if not (isinstance(a, str) and 'nan' in a)]
            if valid_ac:
                auto_labels.append(ai['robot'])
                auto_vals.append(np.mean(valid_ac))
                auto_colors.append(ai['color'])

    if auto_vals:
        bars = ax3b.barh(range(len(auto_labels)), auto_vals, color=auto_colors, alpha=0.8,
                        edgecolor='black', linewidth=0.5)
        ax3b.set_yticks(range(len(auto_labels)))
        ax3b.set_yticklabels(auto_labels, fontsize=8)
        ax3b.set_xlabel("Mean Lag-1 Autocorrelation", fontsize=9)
        ax3b.set_title("Action Smoothness", fontsize=11, fontweight='bold')
        ax3b.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='threshold')
        for bar, val in zip(bars, auto_vals):
            ax3b.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                     f"{val:.3f}", va='center', fontsize=8)
        ax3b.set_xlim(0, 1.1)
        ax3b.grid(True, alpha=0.3, axis='x')

    # 3c: Action-state correlation
    ax3c = fig.add_subplot(gs[2, 3])
    corr_labels = []
    corr_vals = []
    corr_colors = []
    for key in DATASETS:
        if key not in results or results[key].get('status') != 'success':
            continue
        asc = results[key].get('action_state_correlation', {})
        if asc:
            for pair, info in asc.items():
                corr_labels.append(DATASETS[key]['robot'])
                corr_vals.append(info['mean_abs_corr'])
                corr_colors.append(DATASETS[key]['color'])

    if corr_vals:
        bars = ax3c.barh(range(len(corr_labels)), corr_vals, color=corr_colors, alpha=0.8,
                        edgecolor='black', linewidth=0.5)
        ax3c.set_yticks(range(len(corr_labels)))
        ax3c.set_yticklabels(corr_labels, fontsize=8)
        ax3c.set_xlabel("Mean |Corr(action_d, delta_state_d)|", fontsize=9)
        ax3c.set_title("Action -> Delta-State Correlation", fontsize=11, fontweight='bold')
        for bar, val in zip(bars, corr_vals):
            ax3c.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                     f"{val:.3f}", va='center', fontsize=8)
        ax3c.set_xlim(0, 0.7)
        ax3c.grid(True, alpha=0.3, axis='x')

    # ================================================================
    # ROW 4: Cross-Embodiment Comparison
    # ================================================================

    # 4a: Action type taxonomy
    ax4a = fig.add_subplot(gs[3, 0:2])
    ax4a.axis('off')

    # Create a table
    table_data = []
    for key in DATASETS:
        if key not in results or results[key].get('status') != 'success':
            continue
        r = results[key]
        state_key, state_dim = PRIMARY_STATE[key]
        act_dim = r.get('total_action_dim', '?')
        ep_min = r['episode_lengths']['min']
        ep_max = r['episode_lengths']['max']
        ep_mean = r['episode_lengths']['mean']
        full_eps = FULL_EPISODES.get(key, '?')

        # Action type
        aspace = ACTION_SPACES.get(key, "unknown")

        table_data.append([
            DATASETS[key]['robot'],
            f"{state_key} ({state_dim}D)",
            f"{act_dim}D",
            aspace.split(":")[0] if ":" in aspace else aspace[:35],
            f"{ep_min}-{ep_max} (avg {ep_mean:.0f})",
            f"{full_eps:,}",
        ])

    table = ax4a.table(cellText=table_data,
                       colLabels=["Robot", "Primary State", "Act Dim", "Action Type", "Steps/Ep", "Total Eps"],
                       loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.4)
    # Color header
    for j in range(6):
        table[0, j].set_facecolor('#E0E0E0')
        table[0, j].set_fontsize(8)
    # Color robot names
    for i, key in enumerate(k for k in DATASETS if k in results and results[k].get('status') == 'success'):
        table[i+1, 0].set_facecolor(DATASETS[key]['color'] + '33')  # Won't work with hex, use alpha
    ax4a.set_title("Complete Dataset Summary", fontsize=11, fontweight='bold', pad=20)

    # 4b: Cross-embodiment state range overlay
    ax4b = fig.add_subplot(gs[3, 2:4])
    for key in ["toto", "droid", "stanford_kuka", "jaco_play", "berkeley_fanuc"]:
        if key not in results or results[key].get('status') != 'success':
            continue
        obs = results[key].get('observations', {})
        state_key = PRIMARY_STATE[key][0]
        if state_key in obs and 'mean' in obs[state_key]:
            means = np.array(obs[state_key]['mean'])
            stds = np.array(obs[state_key]['std'])
            dims = min(7, len(means))  # Compare first 7 dims
            x = np.arange(dims)
            ax4b.fill_between(x, means[:dims] - stds[:dims], means[:dims] + stds[:dims],
                            alpha=0.2, color=DATASETS[key]['color'])
            ax4b.plot(x, means[:dims], color=DATASETS[key]['color'], linewidth=2,
                     marker=DATASETS[key]['marker'], markersize=5,
                     label=f"{DATASETS[key]['robot']} (joint 1-{dims})")

    ax4b.set_xlabel("Joint Index (first 7)", fontsize=9)
    ax4b.set_ylabel("Value (rad)", fontsize=9)
    ax4b.set_title("Cross-Embodiment Joint Range Comparison", fontsize=11, fontweight='bold')
    ax4b.legend(fontsize=7)
    ax4b.grid(True, alpha=0.3)

    # ================================================================
    # ROW 5: Evaluation Design
    # ================================================================
    ax5 = fig.add_subplot(gs[4, :])
    ax5.axis('off')

    eval_text = """
    PROPOSED EVALUATION TAXONOMY FOR MECHANICAL-JEPA
    ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

    SANITY CHECKS (Easy — proves representations capture basic structure)
    ├── Robot Embodiment Classification: Classify which robot from pretrained embeddings.
    │     Labels: trivially available (dataset source). Metric: Accuracy. Baseline: chance=12.5%
    ├── Gripper State Prediction: Open/closed from proprioceptive embedding.
    │     Labels: gripper channel or threshold on state. Metric: F1. Baseline: majority class
    └── Motion Phase Detection: Approach/grasp/lift/place from velocity+gripper patterns.
          Labels: derive from velocity magnitude + gripper transitions. Metric: Accuracy. Baseline: HMM

    MEDIUM TASKS (Require some signal extraction)
    ├── Task Type Classification: Pick vs place vs pour vs push.
    │     Labels: from language_instruction field (available in toto, ur5, jaco, fractal, droid). Metric: Accuracy
    ├── Next-Step Prediction: Given (state_t, action_t), predict state_{t+1}. Core JEPA task.
    │     Labels: self-supervised. Metric: MSE. Baseline: persistence (state_{t+1}=state_t)
    └── Contact Detection (KUKA only): Detect contact from force/torque signals.
          Labels: contact field available in KUKA dataset. Metric: AUC-ROC. Baseline: threshold on force norm

    HIGH-IMPACT (Transfer tasks — the paper's main contribution)
    ├── Cross-Embodiment Transfer: Pretrain on Franka (DROID+TOTO=79k eps) → fine-tune on UR5/KUKA/FANUC/JACO.
    │     Compare: pretrained+fine-tuned vs train-from-scratch. Metric: MSE improvement on held-out prediction
    ├── Few-Shot Task Learning: 100 demos from scratch vs 10 demos with pretrained encoder.
    │     Metric: task success rate or prediction MSE. Shows data efficiency gain.
    └── Multi-Step Rollout: Predict full trajectory from initial state + action sequence (k=5,10,20 steps).
          Metric: MSE at horizon k. Baseline: single-step autoregressive, physics simulator

    AVAILABLE LABELS SUMMARY:
    │  Language instructions:  toto(1), ur5(4+), jaco(16+), fractal(10+), droid(many)
    │  Reward signals:         all datasets (mostly sparse: 0.3-2% nonzero)
    │  Success/terminal:       all datasets (is_terminal at last step)
    │  Contact/force:          stanford_kuka only (contact + ee_forces_continuous)
    │  Robot type:             trivially available from dataset source
    │  Gripper state:          all datasets (from state or dedicated channel)
    │  Task phase:             derivable from velocity + gripper transitions (no images needed)
    """
    ax5.text(0.02, 0.95, eval_text, transform=ax5.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # ================================================================
    # ROW 6: Scale Comparison & Download Assessment
    # ================================================================
    ax6a = fig.add_subplot(gs[5, 0:2])
    ax6a.axis('off')

    scale_text = """
    DOWNLOAD SIZE & SCALE ASSESSMENT
    ──────────────────────────────────────────────────

    Dataset         Full Size    Proprio-Only Est.   Episodes
    ─────────────   ──────────   ─────────────────   ────────
    toto            127.7 GB     ~50 MB (state only)    1,003
    stanford_kuka   ~30 GB       ~20 MB                 3,000
    berkeley_ur5    ~50 GB       ~15 MB                 1,000
    berkeley_fanuc  ~15 GB       ~5 MB                    415
    jaco_play       ~20 GB       ~5 MB                  1,000
    maniskill       ~200 GB      ~200 MB               30,213
    fractal         ~500 GB      ~100 MB               87,212
    droid           1.7 TB       ~1 GB                 76,000

    TOTAL PROPRIO-ONLY: ~1.4 GB (fits easily on this machine)
    Total with images:  ~2.6 TB (NOT feasible locally)

    Strategy: Use tfds streaming (no local copy needed for
    proprio extraction). Extract state+action arrays only.
    Estimated extraction time: ~2-4 hours for all datasets.
    """
    ax6a.text(0.02, 0.95, scale_text, transform=ax6a.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    # 6b: Scale comparison with Brain-JEPA
    ax6b = fig.add_subplot(gs[5, 2:4])
    categories = ['Instances\n(subjects/traj)', 'Steps\nper instance', 'Channels\nper step', 'Total\ntokens']
    brain_vals = [32000, 160, 450, 2.3e9]
    oxe_vals = [200000, 120, 12, 2.9e8]  # average across datasets

    x = np.arange(len(categories))
    width = 0.35
    bars1 = ax6b.bar(x - width/2, [np.log10(v) for v in brain_vals], width,
                    label='Brain-JEPA', color='#E91E63', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax6b.bar(x + width/2, [np.log10(v) for v in oxe_vals], width,
                    label='OXE (Mechanical-JEPA)', color='#2196F3', alpha=0.8, edgecolor='black', linewidth=0.5)

    ax6b.set_ylabel('log10(count)', fontsize=9)
    ax6b.set_title('Scale: Brain-JEPA vs OXE Proprio', fontsize=11, fontweight='bold')
    ax6b.set_xticks(x)
    ax6b.set_xticklabels(categories, fontsize=8)
    ax6b.legend(fontsize=9)
    ax6b.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars1, brain_vals):
        ax6b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{val:,.0f}', ha='center', va='bottom', fontsize=7)
    for bar, val in zip(bars2, oxe_vals):
        ax6b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{val:,.0f}', ha='center', va='bottom', fontsize=7)

    # Save
    fig.savefig(str(FIGURE_PATH), dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to {FIGURE_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    create_figure()
    print("Done!")
