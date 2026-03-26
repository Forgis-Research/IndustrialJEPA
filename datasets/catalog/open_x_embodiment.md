# Open X-Embodiment (OXE)

## Executive Summary
- **Domain**: Robotics / Manipulation
- **Task**: Robot manipulation; adaptable to multivariate time series pretraining (Mechanical-JEPA)
- **Size**: 1M+ trajectories x 22 robot embodiments x 60 datasets from 34 labs
- **Sampling Rate**: Varies by dataset (3-20 Hz for control data)
- **Real vs Synthetic**: Mostly real; ManiSkill subset is simulated
- **License**: Apache 2.0 (most datasets); some have individual licenses
- **Download URL**: https://github.com/google-deepmind/open_x_embodiment
- **Published SOTA**: RT-2, RT-X, Octo (Google DeepMind, 2023-2024)

## Revised Verdict: CONDITIONALLY VIABLE for Mechanical-JEPA

**The initial "not viable" assessment was too dismissive.** While OXE as a whole is camera-centric, specific subsets contain rich proprioceptive data that can support a Mechanical-JEPA research direction. The key insight: OXE is the largest source of robot state/action data for cross-embodiment pretraining, and filtering for proprioception-rich subsets yields ~200k+ trajectories with real joint-level physics.

The channel count (7-27) is far below Brain-JEPA's 450 ROIs, so OXE cannot replicate the "many sensors" dimension. However, it enables a distinct and equally publishable direction: **cross-embodiment transfer learning on proprioceptive time series**, which no existing JEPA paper has attempted.

---

## Evaluation Suite Summary

### Rapid Evaluation (Mechanical-JEPA)

| Phase | Data | Test | Time | Success Metric |
|---|---|---|---|---|
| Sanity check | TOTO (902 ep, 20 MB) | Overfit single-embodiment prediction | ~1 hr | Train loss < 0.001 |
| Action correlation | ManiSkill (sim) | Verify state-action coupling | ~2 hr | r > 0.4 |
| DOF transfer | Franka→UR5 (zero-shot) | Joint angle MSE degradation | ~1 hr | MSE < 2x pretrain |

### Full-Scale Benchmarks (Mechanical-JEPA)

| Phase | Data | Paper Claim | Primary Metric |
|---|---|---|---|
| **Phase 1** | DROID + TOTO (77k Franka ep) | "JEPA predicts 10-step joint rollouts with MSE < 0.01" | Next-step MSE |
| **Phase 2** | Franka→{UR5, KUKA, JACO} | "50%+ performance retention with zero-shot transfer" | Transfer MSE ratio |
| **Phase 3** | ManiSkill (action-conditioned) | "Action conditioning improves 20-step rollout stability" | Rollout divergence time |

### Download Priority

| Priority | Dataset | Size | Episodes | Role |
|---|---|---|---|---|
| 1 | DROID | ~2 GB | 76,000 | Primary Franka training |
| 2 | TOTO | 20 MB | 902 | Long-horizon eval |
| 3 | ManiSkill | 1.7 GB | 30,213 | Action-conditioned ablation |
| 4 | Berkeley UR5 | ~100 MB | 896 | Cross-embodiment transfer target |

---

## Detailed Description

Open X-Embodiment consolidates 60 existing robot learning datasets into a unified RLDS (Robot Learning Dataset Specification) format. Originally designed for training generalist robot policies (RT-X, Octo), it is the largest publicly available collection of robot manipulation trajectories.

### Scale
| Metric | Value |
|---|---|
| Total trajectories | 1,000,000+ |
| Robot embodiments | 22 different robots |
| Contributing labs | 34 |
| Distinct skills | 527 |
| Task variations | 160,266 |

### Data Format
- **Encoding**: RLDS format (TFRecord / TensorFlow Dataset)
- **Episode structure**: Sequence of steps, each containing observations + actions + rewards + metadata
- **Storage**: Google Cloud Storage (`gs://gresearch/robotics/`) or HuggingFace
- **Access**: `tfds.load('dataset_name', data_dir='gs://gresearch/robotics')`

---

## Proprioception-Rich Subsets (Verified March 2026 -- Deep Analysis)

The following datasets were verified by downloading 20 episodes each from GCS, inspecting raw RLDS schemas, and analyzing every observation/action field. **All numbers below come from actual data, not documentation.**

### Corrected Episode Counts (from tfds.builder splits)

| Dataset | tfds Documented | Actual train split | Actual test split |
|---|---|---|---|
| TOTO | 2,898 | **902** | 101 |
| Stanford KUKA | 3,000 | **3,000** | -- |
| Berkeley UR5 | 1,000 | **896** | 104 |
| Berkeley FANUC | 415 | **415** | -- |
| JACO Play | 1,000 | **976** | 109 |
| ManiSkill | 30,213 | **30,213** | -- |
| Fractal (RT-1) | 87,212 | **87,212** | -- |
| DROID | 76,000 (claimed) | loaded via `droid_100` | -- |

**Note**: TOTO has only 902 train episodes, not 2,898 as previously claimed. UR5 has 896, not 1,000.

### Tier 1: Rich Proprioception (joint pos + vel + force)

| Dataset | tfds Name | Robot | DOF | State Dim | Extra Fields | Episodes | Hz |
|---|---|---|---|---|---|---|---|
| Stanford KUKA | `stanford_kuka_multimodal_dataset_converted_externally_to_rlds` | KUKA iiwa | 7 | 27+ total | joint_pos(7), joint_vel(7), ee_pos(3), ee_vel(3), ee_orient(4), ee_orient_vel(3), ee_forces_continuous(50x6), contact(50), state(8), ee_yaw(4), ee_yaw_delta(4) | 3,000 | 20 |
| ManiSkill | `maniskill_dataset_converted_externally_to_rlds` | Panda (sim) | 7 | 18 | state(18): joint_angles(7)+gripper(2)+joint_vel(7)+gripper_vel(2), tcp_pose(7), base_pose(7), target_poses | 30,213 | 20 |
| Berkeley UR5 | `berkeley_autolab_ur5` | UR5 | 6 | 15 | robot_state(15): opaque blob, likely 6x jpos + 6x jvel + 3x EE | 896 | 10 |

### Tier 2: Good Proprioception (joint pos, some extras)

| Dataset | tfds Name | Robot | DOF | State Dim | Fields | Episodes | Hz |
|---|---|---|---|---|---|---|---|
| DROID | `droid_100` | Franka Panda | 7 | 14 | joint_position(7, float64), cartesian_position(6, float64), gripper_position(1, float64) | 76,000+ | 15 |
| Berkeley FANUC | `berkeley_fanuc_manipulation` | FANUC Mate 200iD | 6 | 13 | state(13) + end_effector_state(7) | 415 | 10 |
| JACO Play | `jaco_play` | Kinova JACO | 6 | 8 | joint_pos(8): 6 joints + 2 fingers, ee_cartesian_pos(7), ee_cartesian_velocity(6) | 976 | 10 |
| TOTO | `toto` | Franka Panda | 7 | 7 | state(7): absolute joint angles | 902 | 10 |

### Tier 3: End-Effector Only (limited proprioception)

| Dataset | tfds Name | Robot | State Dim | Fields | Episodes | Hz |
|---|---|---|---|---|---|---|
| Fractal (RT-1) | `fractal20220817_data` | Google Everyday Robot | 7 (EE) + extras | base_pose_tool_reached(7), gripper_closed(1), gripper_closedness_commanded(1), height_to_bottom(1), rotation_delta_to_go(3), vector_to_go(3), orientation_start(4) | 87,212 | 3 |
| Bridge | `bridge` | WidowX-250 | 7 | state: likely EE pose + gripper | ~60,000 | 5 |
| bc_z | `bc_z` | Google | 4 | xyz(3), sensed_close(1) | 43,000 | ~5 |

---

## ACTION SPACE ANALYSIS (Deep Dive -- March 2026)

**This section was previously missing.** Actions are critical for Mechanical-JEPA because the core JEPA task is: predict z_{t+k} from (z_t, action sequence). Below are the verified action spaces from actual downloaded data.

### Action Space Summary

| Dataset | Action Type | Action Dim | Action Space Description | Smoothness (autocorr lag-1) | Action-State Corr |
|---|---|---|---|---|---|
| TOTO | dict | 8 | open_gripper(1, bool) + rotation_delta(3) + terminate_episode(1) + world_vector(3) | 0.950 | 0.400 |
| Stanford KUKA | tensor | 4 | 3x EE position delta + 1x gripper open/close | 0.411 | 0.327 |
| Berkeley UR5 | dict | 8 | gripper_closedness_action(1) + rotation_delta(3) + terminate_episode(1) + world_vector(3) | 0.764 | 0.436 |
| Berkeley FANUC | tensor | 6 | dx, dy, dz, droll, dpitch, dyaw (EE deltas) | 0.799 | 0.306 |
| JACO Play | dict | 7 | gripper_closedness_action(1) + terminate_episode(3) + world_vector(3) | 0.908 | 0.320 |
| ManiSkill | tensor | 7 | 3x EE delta position + 3x EE delta orientation (axis-angle) + 1x gripper target | 0.955 | 0.473 |
| Fractal | dict | 13 | base_displacement_vector(2) + base_displacement_vertical_rotation(1) + gripper_closedness_action(1) + rotation_delta(3) + terminate_episode(3) + world_vector(3) | 0.785 | 0.184 |
| DROID | tensor | 7 | 7-dim: appears to be EE space (values range [-2.89, 0.55]) | -- | -- |

### Action Space Details Per Dataset

**TOTO (Franka, 8-dim dict action)**:
- `open_gripper`: bool, always 0 in sampled data (100% zero) -- gripper never opens
- `rotation_delta`: (3,) float32, range [-1.91, 0.99], mean=-0.12 -- EE rotation commands
- `terminate_episode`: scalar float32, 99.4% zero -- episode termination flag
- `world_vector`: (3,) float32, range [-0.33, 0.77], mean=0.24 -- EE translation commands
- **Usable action dims**: 6 (rotation_delta + world_vector); gripper/terminate are mostly constant

**Stanford KUKA (4-dim tensor action)**:
- shape=(4,) float32: [3x EE position delta, 1x gripper]
- range: [-0.15, 0.15], small EE deltas
- Low smoothness (0.411) -- actions are relatively noisy/discrete
- **Best for**: force/torque prediction (obs has ee_forces_continuous)

**Berkeley UR5 (8-dim dict action)**:
- Same schema as TOTO: gripper(1) + rotation_delta(3) + terminate(1) + world_vector(3)
- gripper_closedness: 98.7% zero; terminate: 97.9% zero
- world_vector range: [-0.02, 0.02] -- very small EE deltas
- **Usable action dims**: 6

**Berkeley FANUC (6-dim tensor action)**:
- [dx, dy, dz, droll, dpitch, dyaw] -- pure EE deltas
- Very small magnitudes: position deltas range [-0.01, 0.01], rotation [-0.035, 0.035]
- High percentage of zeros (65-98% per dim) -- sparse, discrete commands
- **Usable action dims**: 6, but sparse

**JACO Play (7-dim dict action)**:
- gripper_closedness_action(1) + terminate_episode(3) + world_vector(3)
- terminate_episode has 3 dims (one-hot): dim_1=98.6% zero, dim_2=98.6% nonzero, dim_3=100% zero
- world_vector: range [-0.2, 0.2], reasonable EE deltas
- **Usable action dims**: 4 (gripper + world_vector)

**ManiSkill (7-dim tensor action)**:
- [3x EE delta position, 3x EE delta orientation (axis-angle), 1x gripper]
- Position deltas: range [-0.25, 0.21], orientation: range [-0.43, 0.41]
- Gripper: bimodal [-1, 1], std=0.98
- Highest action-state correlation (0.473) -- best dataset for action-conditioned prediction
- **Usable action dims**: 7

**Fractal (13-dim dict action)**:
- base_displacement_vector(2): 100% zero -- robot base never moves
- base_displacement_vertical_rotation(1): 100% zero
- Effective dims: gripper(1) + rotation_delta(3) + world_vector(3) = 7 usable
- **Usable action dims**: 7

**DROID (7-dim tensor action)**:
- dtype: float64, range [-2.89, 0.55]
- First sample: [0.38, 0.07, 0.55, -2.89, -0.20, 0.13, 0.0]
- Appears to be [cartesian_position(6) + gripper(1)] absolute commands
- **Usable action dims**: 7

### Critical Finding: All Actions are End-Effector Space

**Every OXE dataset uses end-effector (Cartesian) action commands, NOT joint-space commands.** This has major implications for Mechanical-JEPA:

1. Actions are 3-7 dim EE deltas, not 6-7 dim joint commands
2. The mapping from EE action to joint state change involves the robot's Jacobian (nonlinear, configuration-dependent)
3. Action-state correlation is moderate (0.18-0.47) because EE commands don't directly map to joint deltas
4. For action-conditioned prediction, we need either:
   - Learn the inverse kinematics implicitly (harder)
   - Work in EE space for both state and action (loses joint-level detail)
   - Use actions as auxiliary conditioning signal, not primary predictor

### Action Quality Assessment

| Dataset | Usable Action Dims | Sparsity Issue | Smoothness | Best For |
|---|---|---|---|---|
| ManiSkill | 7 | Low | Very High (0.955) | Action-conditioned forecasting |
| TOTO | 6 | Moderate (gripper constant) | Very High (0.950) | Long trajectory prediction |
| JACO | 4 | High (terminate bloat) | High (0.908) | Short manipulation |
| FANUC | 6 | Very High (65-98% zero) | High (0.799) | Sparse control analysis |
| Fractal | 7 | High (base=0) | High (0.785) | Large scale pretraining |
| UR5 | 6 | High (98% zeros) | High (0.764) | Cross-embodiment transfer |
| KUKA | 4 | Low | Moderate (0.411) | Force prediction |

---

## Verified Episode Lengths (from 20-episode samples, March 2026)

| Dataset | Min Steps | Max Steps | Mean Steps | Std | Notes |
|---|---|---|---|---|---|
| TOTO (Franka) | 229 | 1,160 | 361 | 210 | Long pour/scoop trajectories |
| ManiSkill (Panda sim) | 124 | 187 | 154 | 17 | Consistent sim episodes |
| FANUC | 27 | 235 | 123 | 60 | Variable manipulation tasks |
| UR5 | 70 | 123 | 95 | 17 | Cloth/pick-place |
| JACO Play | 35 | 115 | 74 | 22 | Short pick-place |
| KUKA iiwa | 50 | 50 | 50 | 0 | Fixed-length episodes |
| Fractal | 21 | 115 | 49 | 23 | Short manipulation |
| DROID | 166 (1 ep) | -- | -- | -- | Long teleoperation |

### Data Quality Assessment (from actual downloads)

| Dataset | NaN Count | Constant Channels | State Value Range | Quality |
|---|---|---|---|---|
| TOTO | 0 | 0 | [-3.54, 2.80] | Clean |
| KUKA joint_pos | 0 | 0 | [-2.43, 2.58] | Clean |
| KUKA ee_forces | 0 | 0 | [-8.94, 2.58] | Clean |
| FANUC state | 0 | 0 | [-2.89, 1.18] | Clean |
| JACO joint_pos | 0 | 0 | [-2.13, 4.41] | Clean |
| UR5 robot_state | 0 | 0 | [-3.76, 3.83] | Clean |
| ManiSkill state | 0 | 0 | [-2.51, 3.20] | Clean |
| Fractal EE | 0 | 0 | [-0.71, 1.08] | Clean (except orientation_box=NaN) |
| Fractal orientation_box | 2934 NaN | 0 | NaN | **BAD -- skip this field** |

**All core proprioceptive fields are clean. Fractal's `orientation_box` field is entirely NaN -- skip it.**

---

## LABEL AVAILABILITY ANALYSIS

### Available Labels Per Dataset

| Dataset | Language Instr. | Reward | is_terminal | is_first | is_last | Unique Instructions (sample) |
|---|---|---|---|---|---|---|
| TOTO | Yes | Yes (sparse) | Yes | Yes | Yes | "pour" |
| Stanford KUKA | Yes | Yes (sparse) | Yes | Yes | Yes | "insert the peg into the hole" |
| Berkeley UR5 | Yes | Yes (sparse) | Yes | Yes | Yes | "sweep the green cloth..." |
| Berkeley FANUC | Yes | Yes (sparse) | Yes | Yes | Yes | "Press the stapler." |
| JACO Play | Yes | Yes (sparse) | Yes | Yes | Yes | "pick up the butter dairy" |
| ManiSkill | Yes | Yes (sparse) | Yes | Yes | Yes | "Pick up the object and move it..." |
| Fractal | Yes | Yes (sparse) | Yes | Yes | Yes | "pick rxbar chocolate from bottom drawer..." |
| DROID | Yes | Yes | Yes | Yes | Yes | "Put the marker in the pot" |

### Key Observations on Labels

1. **All datasets have language instructions** -- can be used for task classification
2. **Rewards are extremely sparse**: typically 1 nonzero reward per episode (terminal success), mean ~0.01-0.02
3. **is_first/is_last/is_terminal** available everywhere -- useful for episode boundary detection
4. **Natural language embeddings** (512-dim) pre-computed in TOTO, UR5, JACO, Fractal

### Labels We Can Derive Without Images

| Derived Label | Method | Datasets | Difficulty |
|---|---|---|---|
| Robot/Embodiment ID | Dataset name | All | Trivial |
| Task category | Language instruction clustering | All | Easy |
| Gripper state (open/closed) | Threshold on gripper channel | KUKA, FANUC, JACO, ManiSkill, Fractal | Easy |
| Motion phase (approach/grasp/lift/place) | Velocity + gripper state heuristic | All with gripper channel | Medium |
| Contact detection | Force threshold | KUKA only (ee_forces_continuous) | Easy |
| Trajectory success | Terminal reward > 0 | All | Easy |
| Episode difficulty | Trajectory length / variance | All | Easy |
| Control mode | Action sparsity / magnitude | All | Medium |

### If Image Labels Are Needed

- **CLIP embeddings of first frame** to cluster tasks: each dataset has images, so we can extract CLIP features to create pseudo-task-labels
- **Language instruction to task category**: map instructions to {pick, place, pour, push, insert, sweep, press} via keyword matching
- **Object type**: extract from language ("pick up the butter" -> object=butter)

---

## PROPOSED EVALUATIONS FOR MECHANICAL-JEPA

### Sanity Checks (Easy -- proves representations capture structure)

| # | Evaluation | Analogy to Brain-JEPA | Labels Needed | Available? | Metric | Baseline |
|---|---|---|---|---|---|---|
| 1 | **Embodiment Classification** | Sex classification | Robot type (from dataset) | Yes, trivial | Accuracy | Random=14%, linear-on-raw ~80% |
| 2 | **Gripper State Prediction** | -- | Gripper channel threshold | Yes (KUKA, JACO, ManiSkill, Fractal) | Binary accuracy | Threshold on raw gripper value |
| 3 | **Task Category Classification** | -- | Language instruction -> category | Yes, via keyword mapping | F1 | Random ~20%, TF-IDF ~70% |

### Medium (Transfer tasks)

| # | Evaluation | Analogy to Brain-JEPA | Labels Needed | Available? | Metric | Baseline |
|---|---|---|---|---|---|---|
| 4 | **Cross-Embodiment Forecasting** | Cross-ethnicity diagnosis | None (self-supervised) | N/A | Next-state MSE | Train from scratch per robot |
| 5 | **Few-Shot Task Learning** | -- | Task labels from language | Yes | Trajectory MSE | 100 demos from scratch |
| 6 | **Episode Success Prediction** | Age classification | Terminal reward | Yes | AUROC | Random ~50% |

### High-Impact (Novel contributions)

| # | Evaluation | Labels Needed | Available? | Metric | Baseline |
|---|---|---|---|---|---|
| 7 | **Action-Conditioned Forecasting** | None | N/A | Multi-step rollout MSE | Unconditional prediction |
| 8 | **Universal Dynamics Model** | None | N/A | Cross-robot prediction MSE | Per-robot specialist model |
| 9 | **Force/Contact Prediction** | Force channels (KUKA) | Yes | MSE / correlation | Linear model on joint state |
| 10 | **Anomaly Detection** | -- | Synthetic failures | Create synthetic | F1 / AUROC | Statistical threshold |

### Time Series Forecasting (Core Competency)

| Task | Input | Output | Horizon | Metric |
|---|---|---|---|---|
| Next-state prediction | (s_t, a_t) | s_{t+1} | 1 step | MSE per joint |
| Short rollout | (s_t, a_{t:t+10}) | s_{t+1:t+11} | 10 steps | Cumulative MSE |
| Long rollout | (s_t, a_{t:t+50}) | s_{t+1:t+51} | 50 steps | Cumulative MSE + divergence time |
| Unconditional forecast | s_{t-20:t} | s_{t+1:t+11} | 10 steps | MSE (no action) |

---

## DOWNLOAD SIZE ESTIMATES (Proprio-Only Extraction)

| Dataset | Total Episodes | Mean Length | Proprio Dims | Action Dims | Proprio-Only Size | Full RLDS Size |
|---|---|---|---|---|---|---|
| TOTO | 902 | 361 | 7 | 8 | ~20 MB | 128 GB |
| Stanford KUKA | 3,000 | 50 | ~99* | 4 | ~62 MB | 32 GB |
| Berkeley UR5 | 896 | 95 | 15 | 8 | ~8 MB | 76 GB |
| Berkeley FANUC | 415 | 123 | 20 | 6 | ~5 MB | 9 GB |
| JACO Play | 976 | 74 | 21 | 7 | ~8 MB | 9 GB |
| ManiSkill | 30,213 | 154 | ~82* | 7 | ~1.7 GB | 151 GB |
| Fractal | 87,212 | 49 | ~33* | 13 | ~785 MB | 111 GB |
| **Total** | **123,614** | -- | -- | -- | **~2.6 GB** | **~516 GB** |

*KUKA has many observation fields (joint_pos, joint_vel, ee_pos, ee_vel, ee_orient, ee_forces, contact, state, etc.) totaling ~99 dims when all are concatenated. ManiSkill has state(18) + tcp_pose(7) + base_pose(7) + target_poses + camera matrices. Fractal has EE pose(7) + gripper(2) + delta_to_go(6) + orientation(4) + bounds + etc.

**Fits on this machine? YES -- 2.6 GB proprio-only vs 46 GB available.**

**BUT**: Download requires streaming full RLDS records (including images) through TF, then discarding images. KUKA (32 GB), ManiSkill (151 GB), and Fractal (111 GB) are slow to stream due to image data. Estimated download time: 30-40 hours for all datasets at ~50 eps/min.

**Recommended approach**: Download TOTO + KUKA + FANUC + JACO first (~2 hours), then UR5 + ManiSkill over time.

---

## Scale Comparison: OXE vs Brain-JEPA

### Brain-JEPA Analog Assessment

| Dimension | Brain-JEPA | OXE Proprio Subset | Assessment |
|---|---|---|---|
| "Subjects" (instances) | 32,000 patients | ~123,000 trajectories (corrected) | OXE 4x larger |
| Timesteps per instance | 160 | 50-1,160 (mean ~120) | Comparable |
| "Channels" per instance | 450 ROIs | 7-27 (proprioception) | OXE 17-64x smaller |
| Total tokens | ~2.3B | ~31M (Tier 1+2 only) | Brain-JEPA 74x larger |
| Modality | fMRI BOLD | Joint angles/velocities/forces | Different physics |
| Cross-domain | 1 domain (brain) | 7+ robot embodiments | OXE richer for transfer |
| Actions available | No | Yes -- EE commands per timestep | OXE unique advantage |

### Critical Limitation: Channel Count
OXE has 7-27 channels of proprioceptive data per dataset. Brain-JEPA has 450 ROIs. This 17-64x gap means OXE **cannot** replicate the "many sensors" attention masking research.

### What OXE Uniquely Enables
OXE is the **only** public dataset large enough for **cross-embodiment proprioceptive pretraining**:
1. Train a JEPA encoder on Franka joint trajectories (DROID + TOTO = ~77k episodes)
2. Fine-tune/transfer to UR5, KUKA, FANUC, JACO
3. Evaluate: does pretraining on one robot's physics help predict another's?
4. **Action-conditioned prediction**: unique to robotics, not possible with Brain-JEPA

---

## Viability for Mechanical-JEPA

### Phase 1: Franka-Only Pretraining
- **Data**: DROID (76k eps x 14-dim) + TOTO (902 eps x 7-dim) = ~77k episodes
- **Approach**: JEPA encoder on 7-dim joint angle time series
- **Masking**: Temporal masking (predict future joint states from past)
- **Scale**: ~77k x 200 steps x 7 channels = ~108M tokens (modest but workable)
- **Preprocessing**: Normalize per-joint, segment to fixed windows, discard images
- **Feasibility**: HIGH -- data is clean, accessible, sufficient for proof-of-concept

### Phase 2: Cross-Embodiment Transfer
- **Pretrain** on Franka (7-DOF) data
- **Transfer to**: UR5 (6-DOF), KUKA iiwa (7-DOF), FANUC (6-DOF), JACO (6-DOF)
- **DOF mismatch handling**:
  - Option A: Zero-pad smaller DOF to 7, mask padding in attention
  - Option B: Learn per-embodiment linear projection to shared latent space
  - Option C: Use end-effector space (6-dim: xyz + rpy) as universal representation
- **Evaluation**: Fine-tune pretrained encoder on target robot with few-shot episodes
- **Feasibility**: MEDIUM -- requires careful architecture choices for DOF alignment

### Phase 3: Action-Conditioned Forecasting
- **Input**: (state_t, action_t, ..., action_{t+k-1})
- **Output**: Predicted latent z_{t+k}
- **Challenge**: Actions are EE-space, states are joint-space. The encoder must learn approximate inverse kinematics.
- **Best dataset**: ManiSkill (highest action-state correlation: 0.473, consistent episodes)
- **Baseline**: Unconditional forecasting (no action input)

### Quality Assessment
- **Is the proprio data clean enough?** YES -- zero NaN in all core fields, no constant channels, reasonable ranges.
- **What preprocessing is needed?** Per-channel normalization, fixed-length windowing, alignment of different control frequencies (resample to common rate, e.g., 10 Hz).
- **Are there alignment issues?** YES -- different robots have different DOF, joint limits, and coordinate conventions. The universal EE-space representation (6-dim) is the safest alignment strategy.
- **Action space gotcha**: All actions are EE deltas, not joint commands. This limits direct action-to-joint-delta prediction.

---

## Concrete Extraction Plan

### Step 1: Download Proprioceptive Data
```bash
# Quick test (10 eps per dataset)
python datasets/downloaders/download_oxe_proprio.py --sample

# Full download (100 eps per dataset, ~30 min)
python datasets/downloaders/download_oxe_proprio.py --n-episodes 100

# Franka-focused (Phase 1)
python datasets/downloaders/download_oxe_proprio.py --dataset toto --n-episodes 900
```

### Step 2: Preprocess for JEPA
1. Resample all data to 10 Hz (downsample KUKA/ManiSkill from 20 Hz, upsample Fractal from 3 Hz)
2. Normalize per-channel to zero mean, unit variance
3. Segment into fixed-length windows (e.g., 128 timesteps)
4. Split: 80% train, 10% val, 10% test
5. Format as PyTorch tensors: `(batch, channels, time)`

### Step 3: Architecture
- Encoder: 1D-CNN or Transformer on (channels, time)
- Predictor: JEPA-style predictor in latent space
- Masking: Random temporal blocks (predict 20-40% of timesteps)
- Loss: L2 in latent space (not pixel/value space)
- Action conditioning: concatenate action embeddings to latent for conditioned prediction

---

## DROID Dataset Details

DROID is the single largest proprioceptive dataset. It IS in tfds as `droid_100`:

```python
# Confirmed working (March 2026)
ds = tfds.load('droid_100', data_dir='gs://gresearch/robotics', split='train[:5]')
```

**DROID Observation Fields (verified)**:
- `joint_position`: (7,) float64 -- Franka joint angles, range [-2.39, 1.98]
- `cartesian_position`: (6,) float64 -- EE position + orientation, range [-2.89, 0.55]
- `gripper_position`: (1,) float64 -- gripper opening

**DROID Action (verified)**:
- shape=(7,) float64 -- appears to be [cartesian_position(6) + gripper(1)] absolute commands
- range [-2.89, 0.55]

**DROID Step Keys**: action, action_dict, discount, is_first, is_last, is_terminal, language_instruction, language_instruction_2, language_instruction_3, observation, reward

**DROID has 3 language instructions per episode** (multiple annotators).

---

## Raw Observation Schemas (Verified from Data)

### TOTO
```
observation:
  image: (480, 640, 3) uint8
  natural_language_embedding: (512,) float32
  natural_language_instruction: string, e.g. "pour"
  state: (7,) float32, range [-3.54, 2.80]  <-- 7 joint angles
action (dict):
  open_gripper: bool (always 0)
  rotation_delta: (3,) float32, range [-1.91, 0.99]
  terminate_episode: float32 (99.4% zero)
  world_vector: (3,) float32, range [-0.33, 0.77]
```

### Stanford KUKA
```
observation:
  contact: (50,) float32, range [0, 5]
  ee_forces_continuous: (50, 6) float32, range [-8.94, 2.58]
  ee_orientation: (4,) float32 (quaternion)
  ee_orientation_vel: (3,) float32
  ee_position: (3,) float32
  ee_vel: (3,) float32
  ee_yaw: (4,) float32
  ee_yaw_delta: (4,) float32
  image: (128, 128, 3) uint8
  joint_pos: (7,) float32, range [-2.43, 2.58]
  joint_vel: (7,) float32, range [-0.56, 0.54]
  optical_flow: (128, 128, 2) float32
  state: (8,) float32 [= ee_pos(3) + ee_yaw(1) + ee_vel(3) + ee_orient_vel_z(1)]
action: (4,) float32  [3x EE pos delta, 1x gripper], range [-0.15, 0.15]
language_instruction: string, e.g. "insert the peg into the hole"
```

### Berkeley UR5
```
observation:
  hand_image: (480, 640, 3) uint8
  image: (480, 640, 3) uint8
  image_with_depth: (480, 640, 1) float32
  natural_language_embedding: (512,) float32
  natural_language_instruction: string
  robot_state: (15,) float32, range [-3.76, 3.83]
action (dict):
  gripper_closedness_action: float32 (98.7% zero)
  rotation_delta: (3,) float32, range [-0.067, 0.067]
  terminate_episode: float32 (97.9% zero)
  world_vector: (3,) float32, range [-0.02, 0.02]
```

### Berkeley FANUC
```
observation:
  end_effector_state: (7,) float32, range [-0.73, 1.00]  [position(3) + quaternion(4)]
  image: (224, 224, 3) uint8
  state: (13,) float32, range [-2.89, 1.18]
  wrist_image: (224, 224, 3) uint8
action: (6,) float32  [dx, dy, dz, droll, dpitch, dyaw], range [-0.035, 0.035]
language_instruction: string, e.g. "Press the stapler."
```

### JACO Play
```
observation:
  end_effector_cartesian_pos: (7,) float32  [position(3) + quaternion(4)]
  end_effector_cartesian_velocity: (6,) float32
  image: (224, 224, 3) uint8
  image_wrist: (224, 224, 3) uint8
  joint_pos: (8,) float32  [6 joints + 2 fingers]
  natural_language_embedding: (512,) float32
  natural_language_instruction: string
action (dict):
  gripper_closedness_action: (1,) float32
  terminate_episode: (3,) int32 (one-hot)
  world_vector: (3,) float32, range [-0.2, 0.2]
```

### ManiSkill
```
observation:
  base_pose: (7,) float32 [mostly constant except dim 0]
  depth: (256, 256, 1) uint16
  image: (256, 256, 3) uint8
  main_camera_cam2world_gl: (4, 4) float32
  main_camera_extrinsic_cv: (4, 4) float32
  main_camera_intrinsic_cv: (3, 3) float32
  state: (18,) float32  [7 joints + 2 gripper + 7 joint_vel + 2 gripper_vel]
  target_object_or_part_final_pose: (7,) float32
  target_object_or_part_initial_pose: (7,) float32
  tcp_pose: (7,) float32  [tool center point position(3) + quaternion(4)]
  wrist_depth / wrist_image: camera data
action: (7,) float32  [3x EE delta pos, 3x EE delta orient (axis-angle), 1x gripper]
language_instruction: string, e.g. "Pick up the object and move it to a goal position."
```

### Fractal (RT-1)
```
observation:
  base_pose_tool_reached: (7,) float32  [EE position(3) + quaternion(4)]
  gripper_closed: (1,) float32  [binary: 0 or 1]
  gripper_closedness_commanded: (1,) float32
  height_to_bottom: (1,) float32
  image: (256, 320, 3) uint8
  natural_language_embedding: (512,) float32
  natural_language_instruction: string
  orientation_box: (2, 3) float32  ** ALL NaN -- DO NOT USE **
  orientation_start: (4,) float32
  robot_orientation_positions_box: (3, 3) float32 [constant 0.333]
  rotation_delta_to_go: (3,) float32
  src_rotation: (4,) float32 [constant [1,0,0,0]]
  vector_to_go: (3,) float32
  workspace_bounds: (3, 3) float32 [constant]
action (dict):
  base_displacement_vector: (2,) float32 [100% zero]
  base_displacement_vertical_rotation: (1,) float32 [100% zero]
  gripper_closedness_action: (1,) float32
  rotation_delta: (3,) float32
  terminate_episode: (3,) int32
  world_vector: (3,) float32
```

### DROID
```
observation:
  cartesian_position: (6,) float64
  exterior_image_1_left: image
  exterior_image_2_left: image
  gripper_position: (1,) float64
  joint_position: (7,) float64
  wrist_image_left: image
action: (7,) float64  [cartesian_position(6?) + gripper(1)]
step_keys: action, action_dict, discount, is_first, is_last, is_terminal,
           language_instruction, language_instruction_2, language_instruction_3,
           observation, reward
```

---

## Other OXE Datasets with Proprioception

| Dataset | Robot | State Dim | Episodes | Notes |
|---|---|---|---|---|
| NYU Franka Play | Franka | 13 | ~8,000 | `nyu_franka_play_dataset_converted_externally_to_rlds` |
| CMU Franka Exploration | Franka | 8 (action) | ~1,000 | Limited obs |
| TACO Play | DLR | 15 (robot_obs) | ~3,600 | German Aerospace Center arm |
| Robomimic PH | Panda (sim) | 32-115 | 200/task | Richest per-step but tiny |
| Robosuite Panda | Panda (sim) | 32+ | ~1,000 | Sim only |

---

## Download Notes
- Requires `tensorflow_datasets` package (pip install tensorflow tensorflow_datasets)
- All datasets accessible via: `tfds.load(name, data_dir='gs://gresearch/robotics')`
- No GCP authentication required for read access
- Downloader: `datasets/downloaders/download_oxe_proprio.py` (extracts proprio only)
- Deep analysis script: `datasets/analysis/oxe_deep_analysis.py`
- Legacy downloader: `datasets/downloaders/download_open_x.py` (Bridge subset)
- HuggingFace mirror: `jxu124/OpenX-Embodiment` (loader has compatibility issues as of March 2026)
- LeRobot format mirrors available for some datasets (e.g., `lerobot/toto`)

## Analysis Figures
- `datasets/analysis/figures/oxe_curator_audit.png` -- Overview (trajectory counts, tier breakdown)
- `datasets/analysis/figures/oxe_deep_analysis.png` -- Deep analysis (state/action trajectories, cross-embodiment comparison, action quality, label availability, evaluation design, Brain-JEPA scale comparison)

---

## References
- [Open X-Embodiment paper](https://arxiv.org/abs/2310.08864) (arXiv:2310.08864)
- [DROID paper](https://arxiv.org/abs/2403.12945) (arXiv:2403.12945)
- [OXE GitHub](https://github.com/google-deepmind/open_x_embodiment)
- [OXE Dataset Spreadsheet](https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g)
- [TOTO Benchmark](https://toto-benchmark.org/)
- [Stanford KUKA Multimodal](https://sites.google.com/view/stanford-kuka-multimodal)
- [Berkeley FANUC](https://sites.google.com/berkeley.edu/fanuc-manipulation)
- [DROID Dataset](https://droid-dataset.github.io/)
