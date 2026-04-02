# Overnight Task: Create v2 Training-Ready Config + Enrich Remaining Sources

**Date:** 2026-04-02
**HuggingFace Dataset:** https://huggingface.co/datasets/Forgis/Mechanical-Components

## Current State

- **RAW dataset**: ~12K samples, 9.5 GB, 16 sources across bearings + gearboxes configs
- **34 parquet files** in bearings/, 5 in gearboxes/, 1 in source_metadata/
- **No v2 training-ready config yet** — this is the primary deliverable

## Environment

```bash
# .env location
/home/sagemaker-user/IndustrialJEPA/.env
# Contains: HF_TOKEN, KAGGLE_USERNAME, KAGGLE_API_TOKEN

# Load in Python:
from dotenv import load_dotenv
load_dotenv("/home/sagemaker-user/IndustrialJEPA/.env")
```

## TASK 1: Create v2_train Config (PRIMARY — do this first)

Create a new HuggingFace config `v2_train` that is ready for direct model training.

### Spec (from V2_TRAINING_SPEC.md)

| Parameter | Value |
|-----------|-------|
| Sampling rate | 12,800 Hz |
| Window length | 16,384 samples (1.28 seconds) |
| Channels | 1 (single vibration channel per sample) |
| Normalization | Per-sample instance norm (zero-mean, unit-variance) |
| Data type | float32 |

### How to build it

1. Download each parquet shard from HF (use `hf_hub_download`)
2. For each sample:
   a. Select primary vibration channel (first vibration channel in `channel_modalities`)
   b. Resample to 12,800 Hz using `scipy.signal.resample_poly` (see resampling table below)
   c. Window into 16,384-sample segments (non-overlapping from middle of signal)
   d. Normalize: subtract mean, divide by std
   e. Store raw stats (mean, std) in `raw_stats` field
3. Assign splits (source-disjoint):
   - **Train**: cwru, mfpt, femto, xjtu_sy, ims, oedi, phm2009, mcc5_thu, seu, mafaulda, vbl_va001, sca_pulpmill (train)
   - **Val**: paderborn, ottawa_bearing
   - **Test**: mendeley_bearing, sca_pulpmill (test)
4. Upload as `v2_train` config with train/val/test splits

### Resampling Table

| Source | Native Hz | Target Hz | Method |
|--------|----------|-----------|--------|
| cwru | 12,000 | 12,800 | resample_poly(16, 15) |
| mcc5_thu | 12,800 | 12,800 | None |
| mfpt | 48,828 | 12,800 | resample then truncate |
| femto | 25,600 | 12,800 | decimate(2) |
| xjtu_sy | 25,600 | 12,800 | decimate(2) |
| ims | 20,480 | 12,800 | resample_poly(8, 12.8→simplify) |
| paderborn | 64,000 | 12,800 | decimate(5) |
| mendeley | 25,600 | 12,800 | decimate(2) |
| oedi | 40,000 | 12,800 | resample_poly(16, 50) |
| phm2009 | 66,667 | 12,800 | resample |
| seu | 5,120 | 12,800 | resample_poly(5, 2) |
| mafaulda | 50,000 | 12,800 | resample_poly(16, 62.5→round) |
| ottawa | 42,000 | 12,800 | resample |
| sca | 8,192 | 12,800 | resample_poly(25, 16) |
| vbl | 20,000 | 12,800 | resample_poly(16, 25) |

**Note**: For non-integer ratios, use `scipy.signal.resample` (FFT-based) instead of `resample_poly`.

### v2 Per-Sample Schema

```python
{
    "source_id": str,
    "sample_id": str,
    "signal": list[float],       # 1D, 16384 floats, normalized
    "valid_length": int,          # Actual samples before zero-padding (16384 if no padding)
    "health_state": str,          # healthy | faulty | degrading | unknown
    "fault_type": str,
    "fault_severity": str|None,
    "rpm": int|None,
    "episode_id": str|None,
    "episode_position": float|None,
    "rul_percent": float|None,
    "is_transition": bool,
    "split": str,                 # train | val | test
    "raw_stats": str,             # JSON: {"mean": float, "std": float}
}
```

### Memory Management

Process one source at a time:
1. Download source parquets from HF
2. Process all samples from that source → write v2 parquet shard
3. Delete downloaded parquets
4. Move to next source

Upload all v2 shards at the end (or incrementally if disk gets tight).

## TASK 2: Enrich MCC5-THU (if time permits)

The current MCC5-THU gearbox data only has 3 gearbox vibration channels. The raw files have 8 columns:
- `speed` (key phase)
- `torque` (multiply by 6 for actual Nm)
- `motor_vibration_x/y/z` (3 motor channels — **not currently included**)
- `gearbox_vibration_x/y/z` (what we have)

Re-download from Mendeley (DOI: 10.17632/p92gj2732w.2), re-process with all 6 vibration channels + speed/torque in slow_signals, and replace the existing gearboxes data.

**Download URL**: `https://data.mendeley.com/public-files/datasets/p92gj2732w/files/8c075bf0-d8f5-42f0-a8dc-365f5be0b909/file_downloaded`
**Size**: 6.4 GB zip → 16 GB extracted (480 CSV files, 60s each at 12.8 kHz)

## TASK 3: Add Tsinghua Motor Dataset (if time permits)

Motor-specific faults: rotor unbalance, stator winding shorts, compound faults.
- Mendeley: https://data.mendeley.com/datasets/6s3dggj9mw/1
- Would create new `motors` config
- Contains vibration + current + torque + key-phase

## Disk Constraints

- 34 GB free
- v2 processing needs: download shards (~9.5GB) + write v2 shards (~2-3GB) ≈ 13GB
- MCC5-THU re-download: 6.4GB zip + processing ≈ 10GB
- Sequential: process v2 first, clean up, then do MCC5-THU

## Success Criteria

By morning:
- [ ] `v2_train` config exists on HuggingFace with train/val/test splits
- [ ] All 16 sources represented in v2 at 12.8 kHz / 16384 samples
- [ ] Source-disjoint splits correctly assigned
- [ ] Loadable: `load_dataset("Forgis/Mechanical-Components", "v2_train", split="train")`
- [ ] Documented in README

## If Something Fails

- If a source can't be resampled cleanly, skip it and note in progress.log
- If disk runs out, upload v2 shards incrementally and delete as you go
- If HF upload fails, retry with exponential backoff
- The v2 config is the #1 priority — everything else is bonus
