# V19 Dataset Readiness Audit

(Compiled 2026-04-21 via Explore subagent + manual checks.)

## Priority order for V19 execution

1. **PSM** — immediate (NPY files, normalized, binary labels).
2. **MBA** — medical ECG, xlsx→numpy conversion needed (small effort).
3. **SMD** — 28 machines, TSV format, per-machine train/test/labels.
4. **Paderborn** — 80 .mat files per bearing condition (K001/KA01/KI01); 16008 × 7 ch.
5. **NAB** — 7 CSVs, labels in JSON timestamp windows (needs alignment).
6. **Hydraulic** — multi-rate sensors (1/10/100 Hz), 2205 cycles × 43680 attrs.
7. **CWRU** — minimal, deprioritised (Paderborn covers mechanical better).

## Per-dataset

### PSM (server metrics, MTS-JEPA's 4th benchmark)
 - Path: `paper-replications/mts-jepa/data/PSM/`
 - `train.npy` (132481, 25), `test.npy` (87841, 25), `test_labels.npy` (87841,) binary.
 - Already normalised. Drop-in replacement for smap_msl.py loader.

### MBA (MIT-BIH Arrhythmia ECG - medical)
 - Path: `paper-replications/mts-jepa/data/tranad_repo/data/MBA/`
 - `train.xlsx`, `test.xlsx`, `labels.xlsx` (159K / 159K / 12K).
 - PhysioNet source. 360 Hz ECG. Needs xlsx→numpy conversion.
 - Univariate ECG signal + arrhythmia labels.

### SMD (Server Machine Dataset)
 - Path: `paper-replications/mts-jepa/data/tranad_repo/data/SMD/`
 - 28 train files (7.8-9.4M each), 28 test files, 28 label files.
 - 38 normalised features, TSV format. Per-machine alignment.

### NAB (Numenta Anomaly Benchmark)
 - Path: `paper-replications/mts-jepa/data/tranad_repo/data/NAB/`
 - 7 CSVs (53K-716K), univariate (timestamp, value).
 - Labels in `labels.json` as timestamp ranges - needs alignment to row indices.

### Paderborn bearing (real mechanical vibration)
 - Path: `datasets/data/paderborn/{K001,KA01,KI01}/`
 - 80 .mat files per condition (~8.4M each), 667M per bearing total.
 - Each .mat: `Y` shape (16008, 7) — 16008 samples × 7 accelerometer channels.
 - Labels implicit (K001=healthy, KA01=outer-race, KI01=inner-race).

### Hydraulic UCI
 - Path: `datasets/data/hydraulic/`
 - 14 TXT files. 2205 cycles × multi-rate sensors (PS1-6 @100Hz, EPS1, FS1-2 @10Hz, TS1-4/CE/CP/SE/VS1 @1Hz).
 - `profile.txt`: 5 targets (cooler%, valve%, pump leakage, accumulator bar, stable).

### CWRU
 - Path: `datasets/data/cwru/` — 4 .mat files (2.8-3.8M).
 - Single-column vibration, ~121K-243K samples each.
 - 4 classes: normal, IR007, B007, OR007. Single load condition only.

## Pipeline template

All datasets reduce to:
```python
def load_<dataset>() -> dict:
    return {
        'train': np.ndarray (T, C),
        'test': np.ndarray (T, C),
        'labels': np.ndarray (T,) binary or multi-class,
        'n_channels': int,
        'name': str,
        'anomaly_rate': float or None,
        'sampling_hz': float or None,
    }
```

This matches `data/smap_msl.py::load_smap`. New loaders go in `data/` alongside it.
