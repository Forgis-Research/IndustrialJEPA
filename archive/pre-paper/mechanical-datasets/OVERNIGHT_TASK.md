# Overnight Data Curation Task: Universal Mechanical Component Dataset

**Goal**: Build a comprehensive, well-structured mechanical vibration dataset from open sources for training Mechanical-JEPA.

**HuggingFace Dataset**: https://huggingface.co/datasets/Forgis/Mechanical-Components

---

## CRITICAL: Resource Constraints

### Disk Space Management
- **Total available**: 50GB
- **Max for this task**: 20GB (GPU training uses rest)
- **Target working set**: ≤10GB at any time

### Strategy: Sequential Upload & Delete
```
For each dataset:
1. Download to raw/ (watch size!)
2. Process to unified format
3. Validate & sanity check
4. Upload to HuggingFace
5. DELETE local files
6. Move to next dataset
```

**NEVER accumulate >10GB locally. Upload and delete before downloading next large dataset.**

### Disk Monitoring
```bash
# Check disk usage frequently
df -h /
du -sh raw/ processed/

# If approaching 10GB, STOP and upload first
```

---

## Phase 1: Research & Inventory (Deep Dive)

### Priority 1: Bearing Datasets → `bearings` config
| Dataset | Size | Channels | Faults | Download |
|---------|------|----------|--------|----------|
| CWRU | ~500MB | 1-2 (DE, FE) | Ball, Inner, Outer Race | https://engineering.case.edu/bearingdatacenter |
| MFPT | ~100MB | 1 | Artificial + Natural | https://www.mfpt.org/fault-data-sets/ |
| IMS/NASA | ~6GB | 2 per bearing (X,Y) | Natural degradation | https://data.nasa.gov/dataset/ims-bearings |
| XJTU-SY | ~5GB | 2 (H, V) | Run-to-failure | https://github.com/hustcxl/PHM_datasets |
| FEMTO/PRONOSTIA | ~2GB | 2 (H, V) | Accelerated degradation | PHM 2012 Challenge |
| Paderborn (PU) | ~20GB | 1 | Artificial + Natural | https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter/ |

**Note**: For Paderborn (20GB), process in batches - download subset, process, upload, delete, repeat.

### Priority 2: Gearbox Datasets → `gearboxes` config
| Dataset | Size | Channels | Faults | Download |
|---------|------|----------|--------|----------|
| OEDI Gearbox | ~500MB | 4 | Gear crack | https://data.openei.org/submissions/623 |
| PHM 2009 Gearbox | ~1GB | 2 | Mixed gear+bearing | https://phmsociety.org/public-data-sets/ |
| MCC5-THU | ~2GB | Multiple | Variable conditions | https://github.com/liuzy0708/MCC5-THU-Gearbox-Benchmark-Datasets |

### Priority 3: Motor/Pump Datasets → `motors` config
| Dataset | Size | Channels | Notes | Download |
|---------|------|----------|-------|----------|
| NLN-EMP | ~3GB | Multiple | Vibration + current + temperature | https://data.4tu.nl/datasets/2b61183e-c14f-4131-829b-cc4822c369d0 |

**Important**: Include ALL sensor modalities (vibration, current, temperature). Tag each channel with its modality.

### Meta-Resources
- [awesome-bearing-dataset](https://github.com/VictorBauler/awesome-bearing-dataset)
- [PHM_datasets](https://github.com/hustcxl/PHM_datasets)
- [PHM Society](https://phmsociety.org/public-data-sets/)

---

## Phase 2: Two-Level Schema (Efficient, No Duplicates)

### Data Type: Multi-Modal Sensors
Include ALL available sensor types - they provide complementary views of component health:

| Sensor Type | Typical Rate | What It Captures |
|-------------|--------------|------------------|
| Accelerometer | 10-50 kHz | Vibration, fault frequencies |
| Current sensor | 10-100 kHz | Motor load, electrical faults |
| Tachometer | Varies | RPM, shaft position |
| Temperature | 1 Hz or slower | Thermal degradation, friction |
| Acoustic emission | 100+ kHz | Early-stage cracks |

**Handling different time scales:**
- Store each modality with its own `sampling_rate_hz`
- Fast signals (vibration, current): high-frequency arrays
- Slow signals (temperature): can be scalar metadata OR low-freq array
- Channel names describe modality: `["accel_x", "accel_y", "current_A", "temp_bearing"]`

### HuggingFace Structure
```
Forgis/Mechanical-Components/
├── source_metadata/     # One row per source dataset (~10 rows)
│   └── data.parquet
├── bearings/            # Samples, link via source_id
│   └── train.parquet
├── gearboxes/           # Samples, link via source_id
│   └── train.parquet
└── motors/              # Samples, link via source_id
    └── train.parquet
```

Load with:
```python
from datasets import load_dataset

# Load samples
samples = load_dataset("Forgis/Mechanical-Components", "bearings", split="train")

# Load source metadata (no duplicates!)
sources = load_dataset("Forgis/Mechanical-Components", "source_metadata", split="train")
sources_dict = {s["source_id"]: s for s in sources}

# Get full info
sample = samples[0]
source = sources_dict[sample["source_id"]]
print(f"Sampling rate: {source['sampling_rate_hz']} Hz")
```

---

### Level 1: Source Metadata (one row per source dataset)

Config: `source_metadata`

```python
{
    # === IDENTIFICATION ===
    "source_id": "cwru",                    # Primary key
    "full_name": "Case Western Reserve University Bearing Dataset",
    "url": "https://engineering.case.edu/bearingdatacenter",
    "license": "public_domain",
    "citation": "CWRU Bearing Data Center",

    # === SIGNAL PROPERTIES (constant for this source) ===
    "sampling_rate_hz": 12000,              # Primary sampling rate
    "signal_duration_sec": 10.0,

    # === SENSOR MODALITIES AVAILABLE ===
    "available_modalities": ["vibration", "current"],  # What sensor types are in this source
    "modality_details": {
        "vibration": {"channels": ["accel_x", "accel_y"], "rate_hz": 12000},
        "current": {"channels": ["current_A", "current_B"], "rate_hz": 12000},
        "temperature": {"channels": ["temp_bearing"], "rate_hz": 1},  # Slow!
    },

    # === COMPONENT INFO ===
    "component_type": "bearing",            # bearing | gear | gearbox
    "component_subtype": "ball_bearing",    # ball_bearing | roller_bearing | spur_gear | helical_gear
    "component_context": "test_rig",        # test_rig | electric_motor | gearbox_assembly
    "manufacturer": "SKF",
    "model": "6205-2RS",

    # === SCALE FACTORS (for normalization) ===
    "vibration_baseline_g": 0.1,
    "vibration_max_g": 10.0,

    # === DATA AVAILABILITY FLAGS ===
    "has_episodes": false,                  # true for run-to-failure
    "has_transitions": false,               # true for MCC5-THU, Mendeley
    "has_continuous_severity": false,       # true for prognostics
    "has_timestamps": false,

    # === AVAILABLE LABELS ===
    "available_fault_types": ["healthy", "inner_race", "outer_race", "ball"],
}
```

---

### Level 2: Samples (only per-sample varying data)

Config: `bearings`, `gearboxes`, or `motors`

```python
{
    # === LINK TO SOURCE ===
    "source_id": "cwru",                    # Foreign key to source_metadata
    "sample_id": "cwru_105_0",
    "original_file": "105.mat",

    # === SIGNAL (multi-modal) ===
    "signal": [[0.1, 0.2, ...], [...]],     # Shape: (n_channels, signal_length)
    "n_channels": 2,
    "channel_names": ["DE_accel", "FE_accel"],  # Descriptive names
    "channel_modalities": ["vibration", "vibration"],  # What type each channel is

    # === SLOW SIGNALS (optional - for temperature, etc.) ===
    "slow_signals": {                       # null if not available
        "temperature_c": 45.2,              # Scalar if single value
        "temperature_array": [44.1, 44.5, 45.2],  # Array if time series
        "temperature_rate_hz": 0.1,
    },

    # === LABELS ===
    "health_state": "faulty",               # healthy | faulty | degrading | unknown
    "fault_type": "inner_race",             # Standardized taxonomy
    "fault_severity": null,                 # 0-1 continuous, null if binary

    # === OPERATING CONDITIONS (measured) ===
    "rpm": 1750,
    "load": 2.0,
    "load_unit": "hp",

    # === EPISODE / TEMPORAL (for prognostics) ===
    "episode_id": null,                     # e.g., "ims_test1_bearing3"
    "episode_position": null,               # 0-1 normalized position
    "cumulative_runtime_hours": null,
    "rul_percent": null,                    # Remaining useful life (0-1)

    # === TRANSITIONS (MCC5-THU, Mendeley only) ===
    "is_transition": false,
    "transition_type": null,                # ramp_up | ramp_down | step_change

    # === SPLIT ===
    "split": "train",

    # === EXTRAS ===
    "extra_metadata": {
        "fault_diameter_inches": 0.021,
        "sensor_positions": ["drive_end", "fan_end"],
    }
}
```

---

### Component Taxonomy

**Primary Components:**
```
bearing
├── ball_bearing
└── roller_bearing

gear
├── spur_gear
├── helical_gear
└── bevel_gear

gearbox (assembly)
```

**Component Context:**
- `test_rig` - Standalone lab setup
- `electric_motor` - Bearing inside motor
- `gearbox_assembly` - Component inside gearbox

**Fault Types:**

| Component | Fault Types |
|-----------|-------------|
| Bearings | `healthy`, `inner_race`, `outer_race`, `ball`, `cage`, `compound`, `degrading`, `unknown` |
| Gears | `healthy`, `gear_crack`, `gear_wear`, `tooth_break`, `pitting`, `compound`, `unknown` |
| General | `imbalance`, `misalignment`, `looseness`, `bent_shaft`, `coupling_wear` |

---

### What's Available Per Dataset

| Dataset | Episodes | Transitions | Severity | Timestamps | Best For |
|---------|----------|-------------|----------|------------|----------|
| CWRU | No | No | Binary | No | Classification |
| IMS | Yes | No | Implicit | Yes | RUL prediction |
| XJTU-SY | Yes | No | Implicit | Yes | RUL prediction |
| FEMTO | Yes | No | Implicit | Yes | RUL prediction |
| Paderborn | No | No | Discrete | No | Multi-condition |
| MCC5-THU | Yes | **Yes** | Discrete | No | Action-conditioning |
| Mendeley | Yes | **Yes** | Binary | ~ | Varying speed |
| PHM2009 | No | No | Continuous | No | Multi-condition |

**Key**: Only MCC5-THU and Mendeley have speed transitions for action-conditioning.

---

### Signal Storage Notes

- **Two-level schema**: Source metadata separate from samples (no duplicates)
- **Variable channels**: 2D array `(n_channels, length)` - Parquet handles efficiently
- **Nulls are fine**: Parquet stores nulls efficiently
- **Episode fields**: Populated for prognostics datasets, null for snapshots

---

## Phase 3: Process Each Dataset (Sequential!)

### Workflow for Each Dataset

```python
# 1. DOWNLOAD (monitor size!)
download_dataset(name, raw_dir)
print(f"Downloaded: {get_dir_size(raw_dir)}GB")

# 2. PROCESS to unified format
process_to_unified(raw_dir, processed_dir, metadata_schema)

# 3. VALIDATE
run_sanity_checks(processed_dir)
visualize_samples(processed_dir)

# 4. UPLOAD to HuggingFace
upload_to_hf(
    processed_dir,
    repo_id="Forgis/Mechanical-Components",
    subset_name=name  # e.g., "cwru", "ims"
)

# 5. DELETE local files
shutil.rmtree(raw_dir / name)
shutil.rmtree(processed_dir / name)
print(f"Cleaned up. Current disk: {get_disk_usage()}GB")

# 6. NEXT dataset
```

### Order of Processing (by size, smallest first)

| # | Dataset | Size | Config | Expected Metadata |
|---|---------|------|--------|-------------------|
| 1 | MFPT | ~100MB | bearings | fault_type, health_state |
| 2 | CWRU | ~500MB | bearings | fault_type, severity, rpm, load, manufacturer, model |
| 3 | OEDI | ~500MB | gearboxes | fault_type, load |
| 4 | PHM 2009 | ~1GB | gearboxes | fault_type, rpm, load |
| 5 | MCC5-THU | ~2GB | gearboxes | fault_type, severity, rpm, load |
| 6 | FEMTO | ~2GB | bearings | health_state (degrading→failure), rpm |
| 7 | NLN-EMP | ~3GB | motors | fault_type, rpm (vibration channels only!) |
| 8 | XJTU-SY | ~5GB | bearings | health_state (run-to-failure), rpm, load |
| 9 | IMS | ~6GB | bearings | health_state (run-to-failure) |
| 10 | Paderborn | ~20GB | bearings | fault_type, rpm, load, manufacturer (process in batches!) |

**Note**: "Expected Metadata" = what we expect to extract. Many fields will still be null - that's fine.

---

## Phase 4: Quality Checks (Per Dataset)

Before uploading, verify:
- [ ] Signal statistics (mean ≈ 0, reasonable std)
- [ ] Sampling rate matches documentation
- [ ] Label distribution logged
- [ ] No NaN or infinite values
- [ ] Representative samples visualized
- [ ] Metadata complete for all samples

---

## Phase 5: HuggingFace Upload

### Upload by Component Config
```python
from datasets import Dataset, concatenate_datasets
import os

HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "Forgis/Mechanical-Components"

def upload_to_config(dataset: Dataset, config_name: str, append: bool = True):
    """
    Upload dataset to a specific config (bearings, gearboxes, motors).
    If append=True, loads existing data and concatenates.
    """
    if append:
        try:
            # Load existing data for this config
            from datasets import load_dataset
            existing = load_dataset(REPO_ID, config_name, split="train", token=HF_TOKEN)
            dataset = concatenate_datasets([existing, dataset])
            print(f"Appending to existing {config_name} config ({len(existing)} + {len(dataset)} samples)")
        except Exception as e:
            print(f"No existing {config_name} config, creating new: {e}")

    # Push to hub
    dataset.push_to_hub(
        REPO_ID,
        config_name=config_name,  # "bearings", "gearboxes", or "motors"
        split="train",
        token=HF_TOKEN,
    )
    print(f"Uploaded {len(dataset)} samples to {REPO_ID}/{config_name}")

# Example usage:
# upload_to_config(cwru_dataset, "bearings", append=True)
# upload_to_config(phm2009_dataset, "gearboxes", append=True)
```

### Workflow: Process → Upload → Delete → Next
```python
# For each source dataset:
raw_data = download_dataset("CWRU")          # Download
processed = process_to_schema(raw_data)       # Convert to unified schema
validate_dataset(processed)                   # Sanity checks
upload_to_config(processed, "bearings")       # Upload to HF (appends)
cleanup_local(["raw/cwru", "processed/cwru"]) # Delete local files
# Repeat for next dataset...
```

### Dataset Card Updates
After each upload, update the HF README with:
- Source datasets included in each config
- Total samples per config
- Statistics (channels, sampling rates, fault distribution)
- Citations for all source datasets

---

## Success Criteria

1. **Coverage**: At least 5 source datasets uploaded
2. **Quality**: All samples validated, no corruption
3. **Metadata**: Complete for every sample
4. **Disk**: Never exceeded 10GB local
5. **Documentation**: Full provenance on HF

---

## Logging & Progress

Create `progress.log` with:
```
[TIMESTAMP] Starting MFPT download...
[TIMESTAMP] MFPT downloaded: 98MB
[TIMESTAMP] MFPT processed: 1,234 samples
[TIMESTAMP] MFPT validated: OK
[TIMESTAMP] MFPT uploaded to HF
[TIMESTAMP] MFPT cleaned up. Disk: 2.1GB
[TIMESTAMP] Starting CWRU download...
...
```

---

## Troubleshooting

### Download fails
- Try alternative mirrors (Kaggle often has copies)
- Check if registration required
- Document and skip, move to next

### Disk approaching limit
- STOP immediately
- Upload whatever is processed
- Delete raw/ and processed/
- Continue

### Upload fails
- Check HF_TOKEN is set
- Check internet connection
- Retry with exponential backoff

---

## Environment Setup

```bash
# Ensure HF token is set
export HF_TOKEN=hf_xxxxx

# Verify
python -c "from huggingface_hub import HfApi; print(HfApi().whoami())"

# Install dependencies
pip install huggingface_hub datasets scipy pandas numpy tqdm requests
```
