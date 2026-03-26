# Open X-Embodiment (OXE)

## Executive Summary
- **Domain**: Robotics / Manipulation
- **Task**: Robot manipulation (originally), adaptable to multivariate time series pretraining
- **Size**: 1M+ trajectories × 22 robot embodiments × 60 datasets from 34 labs
- **Sampling Rate**: Varies by dataset (typically 5–30 Hz for control data)
- **Real vs Synthetic**: Real — collected from real physical robots
- **License**: Apache 2.0 (most datasets); some have individual dataset licenses
- **Download URL**: https://github.com/google-deepmind/open_x_embodiment
- **Published SOTA**: RT-2, RT-X (Google DeepMind, 2023)

## Detailed Description

Open X-Embodiment is a massive robotics dataset consolidating 60 existing robot learning datasets into a unified format using the RLDS (Robot Learning Dataset Specification) standard. Originally designed for training generalist robot policies (RT-X), it is the largest publicly available collection of robot manipulation trajectories.

### Scale
| Metric | Value |
|---|---|
| Total trajectories | 1,000,000+ |
| Robot embodiments | 22 different robots |
| Contributing labs | 34 |
| Distinct skills | 527 |
| Task variations | 160,266 |

### Data Format
- **Encoding**: RLDS format (TFRecord/TensorFlow Dataset)
- **Episode structure**: Sequence of steps, each containing observations + actions + rewards + metadata
- **Storage**: Google Cloud Storage (gs://gresearch/robotics/) or HuggingFace datasets

### Action Space (Standard)
- 7-dimensional: (x, y, z, roll, pitch, yaw, gripper_opening)
- Represents delta end-effector pose + gripper state

### Observation Space (Varies by Dataset)
| Modality | Availability |
|---|---|
| RGB images (workspace camera) | Nearly all datasets |
| Wrist camera RGB | Many datasets |
| Joint positions | Some datasets |
| Joint velocities | Some datasets |
| Joint torques | Few datasets |
| End-effector pose | Most datasets |

### Proprioceptive-Rich Sub-Datasets
The following OXE datasets have significant proprioceptive data:
| Dataset Name | Channels | Notes |
|---|---|---|
| Berkeley AutoLab UR5 | ~20 | Joint pos/vel + force/torque |
| JACO Play | ~14 | 6-DOF joint data |
| Bridge Dataset v2 | ~12 | WidowX joint angles |
| Fractal (Google) | 7 | Action-only proprioception |
| KUKA IIWA | ~14 | Joint pos + gripper |

## Relevance to IndustrialJEPA

### Brain-JEPA Scale Comparison
Brain-JEPA uses: **32,000 patients × 160 timesteps × 450 ROIs**

OXE comparison:
| Dimension | Brain-JEPA | OXE | Assessment |
|---|---|---|---|
| "Subjects" (instances) | 32,000 patients | 1M+ trajectories | OXE much larger |
| Timesteps per instance | 160 | 50–500 per trajectory | Comparable |
| "Channels" per instance | 450 ROIs | 7–20 (proprioception) | OXE much smaller |
| Total tokens | ~2.3B | ~10–50B | OXE larger |

### Critical Limitation: Channel Count
OXE has **7–20 channels** of proprioceptive data per dataset. Brain-JEPA has 450 ROIs. This 20–60x difference means OXE cannot replicate the "many sensors" dimension of Brain-JEPA. For physics-informed channel grouping, 7–20 channels is also insufficient compared to industrial sensor arrays.

### Data Format Barrier
- RLDS/TFRecord format requires TensorFlow to load
- Not natively compatible with PyTorch time series pipelines
- Extraction requires custom ETL: TFRecord → numpy/pandas
- Each dataset within OXE has different channel definitions — no unified "21 sensors" like C-MAPSS

### Practical Viability as Pretraining Corpus
**Low** for IndustrialJEPA goals:
1. Images are the dominant modality — most papers use RGB frames, not joint data
2. Action dimension (7) is far too few channels for attention masking research
3. Format conversion overhead is substantial
4. Trajectories are heterogeneous — different robots have incompatible sensor spaces

### What OXE IS good for
- Pretraining a **robot policy** (not a forecasting model)
- Fine-tuning on specific manipulation tasks
- Cross-embodiment transfer learning for robot control

### Verdict
**Not viable** as the Brain-JEPA analog for IndustrialJEPA. The channel count is too low (7–20 vs. 450), format is camera-centric, and the physics structure (joint mechanics) differs from industrial sensor arrays. See FEMTO/N-CMAPSS/MIMIC alternatives for large-scale pretraining.

## Better Alternatives for Large-Scale Pretraining
1. **N-CMAPSS**: 8 sub-datasets, 18 sensors, realistic flight cycles (~100k+ cycles total)
2. **MIMII/DCASE industrial sound**: 41,000+ clips, multiple machines
3. **SWaT (Secure Water Treatment)**: 51 sensors, 11 days continuous at 1s resolution = ~950k timesteps
4. **Fleet-level bearing data**: Multiple CWRU + Paderborn + XJTU-SY concatenated (~20 sensors across datasets)

## Download Notes
- Requires `tensorflow_datasets` package
- Sample download: `tfds.load('fractal20220817_data', split='train[:10%]')`
- Full OXE: ~1.5 TB total storage; impractical for this machine
- Google Cloud: `gsutil -m cp -r gs://gresearch/robotics/bridge_v2 .`
- HuggingFace: `load_dataset("jxu124/OpenX-Embodiment", "bridge_v2")`
- Downloader: `datasets/downloaders/download_open_x.py` (downloads Bridge subset only)
