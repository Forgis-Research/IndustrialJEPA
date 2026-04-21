# Bearing Fault Datasets

Aggregated bearing fault data from 4 public datasets for fault detection experiments.

## Overview

| Dataset | Source | Channels | Fault Types | Sampling | Size |
|---------|--------|----------|-------------|----------|------|
| **Paderborn** | Paderborn Univ. | 8 (multimodal) | Healthy, OR, IR, Combined | 64 kHz | 33 bearings |
| **CWRU** | Case Western | 2-3 (vibration) | Healthy, OR, IR, Ball | 12 kHz | 40 files¹ |
| **IMS** | NASA/Cincinnati | 4-8 (vibration) | Run-to-failure | 20 kHz | 3 test runs |
| **XJTU-SY** | Xi'an Jiaotong | 2 (vibration) | Progressive degradation | 25.6 kHz | 15 bearings | ⚠️ Unavailable |

## Dataset Details

### 1. Paderborn (Recommended for Physics Masking)

**Best for**: Multi-modal learning, physics-aware masking

| Property | Value |
|----------|-------|
| Bearings | 33 (6 healthy, 27 faulty) |
| Channels | 8 in **4 physics groups** |
| Measurements | 20 files × 4 sec per bearing |
| Download | https://groups.uni-paderborn.de/kat/BearingDataCenter/ |

**Physics Groups**:
| Group | Channels | Modality |
|-------|----------|----------|
| Vibration (radial) | a1, a2 | Accelerometers |
| Vibration (axial) | a3, v1 | Accelerometer + velocity |
| Thermal/mechanical | temp1, torque | Temperature + torque |
| Electrical | ia, ib | Motor current phases |

### 2. CWRU (Most Cited Benchmark)

**Best for**: Comparison with prior work

| Property | Value |
|----------|-------|
| Files | 40 (representative subset¹) |
| Total samples | ~6M (8.5 minutes at 12 kHz) |
| Channels | 2-3 (Drive End, Fan End, Base) |
| Fault sizes | 0.007", 0.014", 0.021" diameter |
| Load conditions | 0, 1, 2, 3 HP |
| Download | Auto (direct HTTP) |

¹ *The full CWRU dataset has hundreds of files. We download a representative 40-file subset covering all fault types × sizes × loads.*

### 3. IMS (Run-to-Failure)

**Best for**: Prognostics, RUL prediction

| Property | Value |
|----------|-------|
| Test runs | 3 |
| Duration | Weeks of continuous operation |
| Failure modes | Inner race, outer race, roller |
| Download | Manual (Kaggle, requires account) |

### 4. XJTU-SY (Progressive Degradation) — ⚠️ UNAVAILABLE

**Status**: IEEE DataPort URL returns 404 (as of March 2026)

| Property | Value |
|----------|-------|
| Bearings | 15 (5 per operating condition) |
| Conditions | 3 speed/load combinations |
| Channels | 2 (horizontal + vertical acceleration) |
| Download | ❌ IEEE DataPort page is dead |

**Alternative**: Use IMS dataset for run-to-failure experiments instead.

## Replication Process

### Step 1: Install Dependencies

```bash
pip install numpy scipy pandas matplotlib
# For RAR extraction (Paderborn): install unrar, 7z, or bsdtar
```

### Step 2: Download Datasets

```bash
cd mechanical-jepa/data/bearings

# Paderborn (sample = 3 bearings)
python prepare_bearing_dataset.py --download --sample --dataset paderborn

# CWRU (sample = 4 files)
python prepare_bearing_dataset.py --download --sample --dataset cwru

# All datasets, full download
python prepare_bearing_dataset.py --download --dataset all
```

### Step 3: Process into Unified Format

```bash
python prepare_bearing_dataset.py --process
```

Creates:
- `bearing_episodes.parquet` - All measurements with labels
- `statistics.json` - Dataset statistics

## Unified Schema

All datasets are processed into a common format:

```python
# Episode format
{
    'dataset': str,        # 'paderborn', 'cwru', 'ims', 'xjtu'
    'bearing_id': str,     # Unique identifier
    'fault_type': str,     # 'healthy', 'outer_race', 'inner_race', 'ball', 'combined'
    'fault_label': int,    # 0=healthy, 1=outer, 2=inner, 3=ball, 4=combined
    'channels': List[str], # Available channel names
    'n_samples': int,      # Number of timesteps
    'sampling_rate': int,  # Hz
}

# Window format (for training)
{
    'dataset': str,
    'bearing_id': str,
    'fault_label': int,
    'window_data': np.ndarray,  # (window_size, n_channels)
    'channel_stats': dict,      # Per-channel RMS, std, etc.
}
```

## Classification Tasks

| Task | Classes | Datasets |
|------|---------|----------|
| Binary fault detection | Healthy vs Faulty | All |
| 4-class fault type | Healthy/OR/IR/Ball | CWRU, Paderborn |
| Severity estimation | Damage size | CWRU (3 sizes) |
| RUL prediction | Continuous | IMS, XJTU-SY |

## Current Status (Verified)

| Dataset | Status | Episodes | Samples | Duration |
|---------|--------|----------|---------|----------|
| **CWRU** | Downloaded | 40 | 6.1M | 8.5 min |
| **IMS** | Downloaded | 9,464 | 193.8M | 2.7 hours |
| Paderborn | Needs RAR tool | - | - | - |
| XJTU-SY | ❌ URL dead | - | - | - |

**Total**: 9,504 episodes, ~200M samples, ~2.8 hours

Run `python prepare_bearing_dataset.py --verify` to check current status.

## Citation

```bibtex
@inproceedings{lessmeier2016paderborn,
  title={Condition monitoring of bearing damage},
  author={Lessmeier, Christian and others},
  booktitle={PHM Europe},
  year={2016}
}

@article{smith2015cwru,
  title={Rolling element bearing diagnostics using the CWRU bearing data},
  author={Smith, Wade A and Randall, Robert B},
  journal={Mechanical Systems and Signal Processing},
  year={2015}
}

@misc{lee2007ims,
  title={IMS bearing data},
  author={Lee, Jay and others},
  howpublished={NASA Prognostics Data Repository},
  year={2007}
}

@article{wang2018xjtu,
  title={A hybrid prognostics approach},
  author={Wang, Biao and others},
  journal={Mechanical Systems and Signal Processing},
  year={2018}
}
```
