# Mechanical Component Vibration Dataset Collection

Comprehensive multi-source mechanical vibration dataset for training Mechanical-JEPA.

**HuggingFace Dataset**: https://huggingface.co/datasets/Forgis/Mechanical-Components

## Current State: ~12,000 samples | 9.5 GB | 16 sources | 5 component types

### Bearings (~10,000 samples, 10 sources)

| Source | Samples | Sensors | Key Feature |
|--------|---------|---------|-------------|
| CWRU | 40 | vibration | Standard benchmark |
| MFPT | 20 | vibration | Variable load |
| FEMTO | 3,569 | vibration + temperature | Run-to-failure (RUL) |
| Mendeley | 280 | vibration | Speed transitions (action-conditioning) |
| XJTU-SY | 1,370 | vibration | Run-to-failure (RUL) |
| IMS | 1,256 | vibration | Run-to-failure (RUL) |
| Paderborn | 384 | vibration + current | Real + artificial faults |
| MAFAULDA | 800 | vibration + acoustic + tach | **Shaft faults** (imbalance, misalignment) |
| Ottawa | 180 | vibration + acoustic | Cage faults, 3 health stages |
| SCA Pulp Mill | 2,663 | vibration | **Real industrial** data |
| VBL-VA001 | 800 | vibration (triaxial) | Misalignment, unbalance |
| SEU | 140 | 8-ch drivetrain | Cross-component rig |

### Gearboxes (~1,225 samples, 4 sources)

| Source | Samples | Sensors | Key Feature |
|--------|---------|---------|-------------|
| OEDI | 20 | vibration (4-ch) | Healthy vs gear crack |
| PHM 2009 | 109 | vibration + tachometer | Challenge data |
| MCC5-THU | 956 | vibration | Speed/load transitions |
| SEU | 140 | 8-ch drivetrain | Cross-component rig |

### Component Types

- **Bearings**: inner_race, outer_race, ball, cage, compound, degrading
- **Gears**: gear_crack, gear_wear, missing_tooth, tooth_break
- **Shafts**: imbalance, misalignment (horizontal/vertical)
- **Drivetrains**: Combined motor+gearbox+bearing (SEU)
- **Industrial**: Real naturally-occurring faults (SCA)

### Sensor Modalities

- Vibration (accelerometer) — all 16 sources
- Motor current — Paderborn
- Acoustic (microphone) — MAFAULDA, Ottawa
- Tachometer — MAFAULDA, PHM2009, OEDI
- Temperature — FEMTO
- Torque — SEU, MCC5-THU

## Schema

Two-level: `source_metadata` (one row per source) + `bearings`/`gearboxes` (per-sample, linked by `source_id`).

```python
from datasets import load_dataset
bearings = load_dataset("Forgis/Mechanical-Components", "bearings", split="train")
sources = load_dataset("Forgis/Mechanical-Components", "source_metadata", split="train")
```

## v2 Training-Ready Config (Planned)

Standardized for direct model training: 12.8 kHz, 16384 samples (1.28s), vibration-only, instance norm, source-disjoint splits. See `TRAINING_READY_RESEARCH.md` for design rationale.

## Key Files

- `datasets_inventory.md` — Original 10 datasets with download links
- `NEW_DATASETS_INVENTORY.md` — 34 additional candidate datasets
- `TRAINING_READY_RESEARCH.md` — v2 config design research
- `LITERATURE_REVIEW.md` — SOTA methods and justification for Mechanical-JEPA
- `OVERNIGHT_TASK.md` — Full schema specification
- `OVERNIGHT_ENRICH.md` — Enrichment task plan
