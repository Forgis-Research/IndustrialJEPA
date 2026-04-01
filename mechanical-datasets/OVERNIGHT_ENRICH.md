# Overnight Task: Enrich Mechanical-Components Dataset

**Date:** 2026-04-01
**HuggingFace Dataset:** https://huggingface.co/datasets/Forgis/Mechanical-Components

## Current State

| Config | Samples | Sources | Issue |
|--------|---------|---------|-------|
| bearings | 6,919 | 7 | Dominated by FEMTO (3,569). Only 384 Paderborn, 40 CWRU |
| gearboxes | 1,085 | 3 | **Way too small** vs bearings. Need 5x more |
| motors | 0 | 0 | **Empty config. Must create** |

## Goals (Priority Order)

### 1. FIX: Add missing modalities to existing datasets

We left sensor data on the table during the first pass. Go back and add:

**MCC5-THU (gearboxes):** Currently only 3 gearbox vibration channels. The files have 8 columns:
- `speed` - key phase signal (use for RPM estimation)
- `torque` - torque on input shaft (multiply by 6 for actual Nm)
- `motor_vibration_x/y/z` - 3 motor drive-end channels (**not included!**)
- `gearbox_vibration_x/y/z` - what we have

**Action:** Re-process MCC5-THU to include all 6 vibration channels (motor + gearbox) plus speed and torque as slow_signals. This enriches the existing 956 samples without downloading anything new.

**Mendeley (bearings):** Currently only vibration (4 channels). The source has:
- `current_R/S/T` - 3-phase motor current at 100 kHz (**not included!**)
- `rpm` files with time-varying speed profiles (**partially used** - avg RPM only)

**Action:** Re-process Mendeley to add current channels. Since current is at 100kHz vs vibration at 25.6kHz, store current in separate samples or downsample to match. Also store the full RPM profile in slow_signals.

### 2. NEW: Gearbox datasets (close the gap)

**SEU Gearbox+Bearing Dataset (Southeast University)** - HIGHEST PRIORITY
- Source: https://github.com/cathysiyu/Mechanical-datasets
- Size: ~200 MB (quick!)
- 8 channels: motor vib, planetary gearbox XYZ, motor torque, parallel gearbox XYZ
- Fault types: normal, missing tooth, root fault, surface fault, chipped tooth (gear); inner/outer/ball/combined (bearing)
- 12 kHz sampling, 2 operating conditions
- **Multi-component drivetrain** - exactly what we need for cross-component learning

### 3. NEW: Motor datasets (create the motors config)

**MAFAULDA (Machinery Fault Database)** - UFRJ Brazil
- Kaggle: https://www.kaggle.com/datasets/uysalserkan/fault-induction-motor-dataset (~2.5GB)
- Full: https://www.kaggle.com/datasets/vuxuancu/mafaulda-full (~13GB)
- 8 channels: tachometer, 3-axis underhang accel, 3-axis overhang accel, microphone
- 50 kHz, 5-second recordings
- Faults: imbalance (333), horizontal misalignment (197), vertical misalignment (301), bearing faults (1,071), normal (49)
- **1,951 samples** - new fault categories (imbalance, misalignment)!
- Start with Kaggle subset (2.5GB), use full if disk allows

**Electric Motor Vibrations (Kaggle)** - tiny, quick win
- Source: https://www.kaggle.com/datasets/amirberenji/electric-motor-vibrations-dataset
- Size: 40 MB
- Triaxial vibration, 17+ scenarios
- Faults: mechanical (misalignment/imbalance), electrical (resistance faults), combined
- **Electrical faults** - completely new fault category

### 4. NEW: Unique bearing datasets

**Ottawa Bearing Dataset** - acoustic + vibration
- Source: https://data.mendeley.com/datasets/y2px5tg92h/1
- Size: ~2 GB
- Sensors: accelerometer + **microphone** (acoustic) + load cell + hall effect
- 42 kHz, 10-second recordings
- 20 bearings, **cage faults** (rare!), three health stages per bearing
- **Acoustic data** - first non-vibration-non-current modality

**SCA Pulp Mill Industrial Dataset** - real-world data
- Source: https://data.mendeley.com/datasets/tdn96mkkpt/2
- Size: ~500 MB
- **Real industrial data** from a pulp mill (not lab)
- Naturally occurring faults over months of operation
- 11 cases with 4 months normal + 4 months to failure

**DIRG/PoliTo Aerospace Bearings**
- Source: https://zenodo.org/records/3559553
- Size: 3.2 GB
- **Up to 30,000 RPM** (10x faster than anything we have)
- Quantified spall sizes (enables fault size estimation)
- Variable speed AND load

### 5. NEW: More large bearing datasets

**VBL-VA001 Lab-Scale Dataset**
- Source: https://zenodo.org/records/7006575
- Size: 3.8 GB
- **4,000 samples** (triaxial, 20 kHz)
- Normal, bearing fault, misalignment, unbalance

**KAIST Run-to-Failure**
- Source: https://data.mendeley.com/datasets/5hcdd3tdvb/6
- Size: ~2 GB
- Vibration + temperature, 25.6 kHz
- 128-hour complete degradation trajectory

---

## Processing Order (by priority and size)

| # | Task | Size | Config | New Samples | Why |
|---|------|------|--------|-------------|-----|
| 1 | Re-process MCC5-THU (add motor vib + speed/torque) | 0 (from zip) | gearboxes | 956 (enriched) | Missing modalities |
| 2 | SEU gearbox+bearing | ~200MB | gearboxes+bearings | ~5,000+ | Balance gearbox config |
| 3 | Electric Motor Vibrations (Kaggle) | 40MB | motors | ~20 | Quick win, create motors config |
| 4 | MAFAULDA (Kaggle subset) | 2.5GB | motors | ~1,951 | Major motor dataset |
| 5 | Re-process Mendeley (add current) | 0 (re-download ~20GB) | bearings | 280 (enriched) | Missing modalities |
| 6 | Ottawa bearing (acoustic) | ~2GB | bearings | ~60 | Acoustic modality |
| 7 | SCA Pulp Mill | ~500MB | bearings | ~22 | Real industrial data |
| 8 | VBL-VA001 | 3.8GB | bearings | ~4,000 | Sample count |
| 9 | DIRG aerospace | 3.2GB | bearings | ~100s | High-speed regime |
| 10 | KAIST run-to-failure | ~2GB | bearings | ~129 | Temperature + vibration |

---

## Environment

```bash
# .env file location
~/dev/IndustrialJEPA/.env
# OR
~/IndustrialJEPA/.env

# Contains: HF_TOKEN, KAGGLE_USERNAME, KAGGLE_API_TOKEN
```

## Schema Reminder

Two-level schema. For each new source:
1. Add entry to `source_metadata` config
2. Add samples to appropriate component config (bearings/gearboxes/motors)
3. Use `source_id` as foreign key

Per-sample fields:
```python
{
    "source_id": str,           # FK to source_metadata
    "sample_id": str,
    "original_file": str,
    "signal": list[list[float]],  # (n_channels, signal_length)
    "n_channels": int,
    "channel_names": list[str],
    "channel_modalities": list[str],  # "vibration", "current", "acoustic", "tachometer"
    "slow_signals": str|None,   # JSON: {"temperature_c": 45.2, "rpm_profile": [...]}
    "health_state": str,        # healthy | faulty | degrading | unknown
    "fault_type": str,
    "fault_severity": float|None,
    "rpm": int|None,
    "load": float|None,
    "load_unit": str|None,
    "episode_id": str|None,
    "episode_position": float|None,
    "cumulative_runtime_hours": float|None,
    "rul_percent": float|None,
    "is_transition": bool,
    "transition_type": str|None,
    "split": str,
    "extra_metadata": str,      # JSON
}
```

## Disk Constraint

- 37 GB free. Process sequentially: download, process, upload, delete.
- Mendeley re-download is 20GB - do this last or skip if disk is tight.
- MCC5-THU zip is already deleted; re-download needed (~6.4GB zip).

## Success Criteria

By morning:
- [ ] MCC5-THU enriched with all 6 vibration + speed/torque channels
- [ ] SEU gearbox data added (gearboxes config >5,000 samples)
- [ ] `motors` config created with MAFAULDA + Electric Motor data
- [ ] At least 2 new bearing datasets added (Ottawa acoustic, VBL-VA001)
- [ ] source_metadata updated with all new sources
- [ ] Progress logged to progress.log

## If Something Fails

- Log error to progress.log, skip to next dataset
- Kaggle mirrors often work when primary sources don't
- Don't get stuck on Mendeley re-download (20GB) - skip if disk tight
- Keep making progress through the list
