# v2 Training-Ready Config Specification

## Design Decisions (from TRAINING_READY_RESEARCH.md)

Based on analysis of MOMENT, Chronos, MOIRAI, BearLLM/MBHM, and HEAR benchmarks.

## Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Config name** | `v2_train` | Separate from raw configs |
| **Sampling rate** | 12,800 Hz | Clean decimation from most sources. Captures up to 6.4 kHz. |
| **Window length** | 16,384 samples | Power of 2 (2^14). 1.28 seconds at 12.8 kHz. |
| **Channels** | Vibration only | Drop current, acoustic, tachometer, temperature |
| **Channel count** | 1 (single-channel) | Average/select primary vibration channel |
| **Normalization** | Per-sample instance norm | Zero-mean, unit-variance per sample. Store raw stats for reversal. |
| **Data type** | float32 | Standard |

## Resampling Plan

| Source | Native Rate | Decimation | Method |
|--------|------------|------------|--------|
| CWRU | 12,000 Hz | Resample 12000→12800 (×16/15) | scipy.signal.resample_poly |
| MCC5-THU | 12,800 Hz | None | Direct use |
| MFPT | 48,828 Hz | ÷3.81 → 12,800 | scipy.signal.resample |
| FEMTO | 25,600 Hz | ÷2 → 12,800 | scipy.signal.decimate(2) |
| XJTU-SY | 25,600 Hz | ÷2 → 12,800 | scipy.signal.decimate(2) |
| IMS | 20,480 Hz | Resample → 12,800 (×5/8) | scipy.signal.resample_poly |
| Paderborn | 64,000 Hz | ÷5 → 12,800 | scipy.signal.decimate(5) |
| Mendeley | 25,600 Hz | ÷2 → 12,800 | scipy.signal.decimate(2) |
| OEDI | 40,000 Hz | Resample → 12,800 (×16/50) | scipy.signal.resample_poly |
| PHM2009 | 66,667 Hz | Resample → 12,800 | scipy.signal.resample |
| SEU | 5,120 Hz | Upsample ×2.5 → 12,800 | scipy.signal.resample_poly |
| MAFAULDA | 50,000 Hz | Resample → 12,800 | scipy.signal.resample |
| Ottawa | 42,000 Hz | Resample → 12,800 | scipy.signal.resample |
| SCA | 8,192 Hz | Upsample → 12,800 (×25/16) | scipy.signal.resample_poly |
| VBL | 20,000 Hz | Resample → 12,800 (×16/25) | scipy.signal.resample_poly |

## Windowing

- Fixed 16,384 samples per window
- For signals longer than 16,384: extract non-overlapping windows from middle
- For signals shorter than 16,384: zero-pad right, add `valid_length` field
- For run-to-failure episodes: slide window with 50% overlap to capture degradation trajectory

## Channel Selection

For multi-channel sources, select the primary vibration channel:
- CWRU: DE_accel (drive end)
- Paderborn: vibration_1
- SEU: planetary_y (most sensitive direction per readme)
- MAFAULDA: underhang_rad (radial = most common fault direction)
- Ottawa: accelerometer (col 0)
- VBL: accel_x (first axis)
- Others: first vibration channel

## Split Strategy

**Primary: Source-disjoint (cross-domain evaluation)**
- Train: CWRU, MFPT, FEMTO, XJTU-SY, IMS, OEDI, PHM2009, MCC5-THU, SEU, MAFAULDA, VBL, SCA-train
- Val: Paderborn, Ottawa
- Test: Mendeley, SCA-test

**Secondary: Episode-disjoint within-source**
- For run-to-failure: entire trajectories in same split
- For classification: random 70/15/15 but never within same recording

## v2 Per-Sample Schema

```python
{
    "source_id": str,
    "sample_id": str,
    "signal": list[float],       # 1D array, 16384 samples at 12.8 kHz
    "valid_length": int,          # Actual signal length before padding
    "health_state": str,
    "fault_type": str,
    "fault_severity": float|None,
    "rpm": int|None,
    "episode_id": str|None,
    "episode_position": float|None,
    "rul_percent": float|None,
    "is_transition": bool,
    "split": str,                 # train | val | test
    "raw_stats": str,             # JSON: {"mean": float, "std": float, "max_abs": float}
}
```

## Implementation Notes

- Use `scipy.signal.resample_poly` for integer-ratio resampling (anti-aliasing built in)
- Apply low-pass filter before decimation to prevent aliasing
- Verify Nyquist: no bearing fault characteristic frequency should exceed 6.4 kHz
- For JEPA pretraining: mask 30% of 256-sample patches (64 patches per window)
