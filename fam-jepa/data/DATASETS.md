# FAM Dataset Audit

Compiled 2026-04-21 from source-code inspection of all five dataset loaders.
This document is authoritative for the paper pipeline.

---

## 1. Summary Table

| Dataset | Task | Train shape | Test shape | Channels | Sampling rate | Event type | Anomaly/event rate | Loader |
|---------|------|-------------|------------|----------|---------------|------------|--------------------|--------|
| C-MAPSS FD001 | RUL | 20631 rows, 26 cols raw → 100 engines × var-len, 14 ch | 100 engines × var-len, 14 ch | 14 (from 21) | 1 cycle (~flight-cycle) | Engine failure (RUL=0) | 100% engines eventually fail | `v11/data_utils.py` |
| C-MAPSS FD002 | RUL | 26 cols raw → 260 engines, 14 ch | 259 engines, 14 ch | 14 | 1 cycle | Engine failure | 100% | same |
| C-MAPSS FD003 | RUL | 26 cols raw → 100 engines, 14 ch | 100 engines, 14 ch | 14 | 1 cycle | Engine failure | 100% | same |
| C-MAPSS FD004 | RUL | 26 cols raw → 249 engines, 14 ch | 248 engines, 14 ch | 14 | 1 cycle | Engine failure | 100% | same |
| SMAP | Anomaly | (135183, 25) | (427617, 25) | 25 | ~1 Hz telemetry | Spacecraft anomaly | 12.8% (per docstring) | `data/smap_msl.py` |
| MSL | Anomaly | (58317, 55) | (73729, 55) | 55 | ~1 Hz telemetry | Spacecraft anomaly | 10.5% (per docstring) | `data/smap_msl.py` |
| PSM | Anomaly | (132481, 25) | (87841, 25) | 25 (after drop) | 1 Hz server metrics | Server metric anomaly | ~27% (per docstring) | `data/psm.py` |
| MBA | Anomaly | (7680, 2) | (7680, 2) | 2 | 360 Hz ECG | Arrhythmia | TBD (±20-sample windows) | `v19/phase2_mba.py::load_mba` |

Notes on shapes marked "per docstring": shapes are asserted in code comments, not runtime-verified assertions. Exact counts should be confirmed by running the loaders against actual data files.

---

## 2. Dataset Details

### 2.1 C-MAPSS FD001–FD004

**Loader:** `fam-jepa/experiments/v11/data_utils.py`

#### A. Data shape

Raw file layout: space-separated, no header, 26 columns (`engine_id`, `cycle`, `op1–3`, `s1–21`).
Two trailing columns in the raw files are NaN and are dropped by `dropna(axis=1, how='all')`.
The final raw shape is therefore 26 columns (confirmed by `sanity_check_fd001` assertion: `train_df.shape == (20631, 26)`).

After sensor selection, each engine sequence becomes `(T_engine, 14)`.

FD001 verified shapes (from `sanity_check_fd001`):
- Train: 20631 total rows, 100 engines
- Test: 13096 total rows, 100 engines
- RUL array: (100,) — one scalar per test engine (RUL at last observed cycle)

FD002/FD003/FD004 shapes are not asserted in code; from the NASA dataset documentation:
- FD002: 260 train engines, 259 test engines
- FD003: 100 train engines, 100 test engines
- FD004: 249 train engines, 248 test engines

#### B. Preprocessing

1. **Sensor selection:** 7 near-constant sensors (`s1, s5, s6, s10, s16, s18, s19`) are hardcoded as dropped. Constant list is not data-driven — it is based on domain knowledge from prior literature. `SELECTED_SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]`, giving `N_SENSORS = 14`.

2. **Normalization strategy:**
   - FD001, FD003 (single operating condition): global min-max per sensor, computed on the training split only (`fit_normalizer(per_condition=False)`).
   - FD002, FD004 (six operating conditions): per-condition min-max using KMeans(n_clusters=6, seed=42) on the three op-setting columns. The KMeans object is stored in `stats['_kmeans']` for use at test time.
   - Scale: [0, 1] (min-max). Zero-variance sensors are mapped to 0.0.

3. **No resampling.** Original cycle resolution retained.

4. **No NaN handling needed** — CMAPSS data has no missing values; the `dropna(axis=1)` only removes the structural trailing empty columns.

5. **Patch size:** Not defined in this loader. `patch_length=1` is used when `TrajectoryJEPA` is instantiated in `phase2_mba.py` (and presumably in all CMAPSS experiment scripts). This means each input timestep is treated as one token with no temporal patch merging.

6. **RUL labels:** Piecewise-linear capped at `RUL_CAP = 125`. For a sequence of length T, `rul[t] = min(T - t, 125)`. Labels count down from min(T, 125) to 1.

#### C. Splits

The loader uses a **per-engine random split** of the labeled training data:
- 85% of engine IDs → `train_engines`
- 15% of engine IDs → `val_engines`
- All test-file engines → `test_engines`

Split is seeded (`rng = np.random.default_rng(42)`). Minimum 1 validation engine.

For **pretraining** (unsupervised), `CMAPSSPretrainDataset` uses the `train_engines` dict and samples random `(t, k)` pairs — no RUL labels consumed.

For **finetuning**, `CMAPSSFinetuneDataset` uses `train_engines` with multiple cut-points per engine (`n_cuts_per_engine=5`), attaching normalized RUL as a regression target.

For **testing**, `CMAPSSTestDataset` uses each test engine's full sequence and the ground-truth RUL array from `RUL_FD00X.txt`.

Standard literature split: the CMAPSS files already define a canonical train/test partition. The val split (15% of training engines) is this project's addition and is consistent with the literature convention of not using test labels for threshold tuning.

#### D. Event labels for FAM/BCE loss

The loader currently outputs **RUL regression targets** (scalar, normalized to [0,1] by dividing by `RUL_CAP=125`). To convert to per-window binary labels `y(t, Δt)` for BCE:

```
y(engine_e, timestep_t, horizon_k) = 1  if  RUL(t) <= k  else  0
```

This conversion is implemented in `grey_swan_metrics.py::f1_at_horizon`:
```python
y_true_bin = (true_rul <= k).astype(int)
y_pred_bin = (pred_rul <= k).astype(int)
```

The FAM pipeline should call `evaluate_rul_run(pred, target, horizons=(10, 20, 30, 50))` which returns both RMSE (legacy) and F1@k for all four horizons.

#### E. Code quality

- `sanity_check_fd001()` exists and covers FD001 shapes + engine counts. No equivalent for FD002/003/004.
- No `if __name__ == '__main__'` block. The file is a library module only.
- **HARDCODED PATH:** `CMAPSS_DATA_DIR = "/home/sagemaker-user/IndustrialJEPA/datasets/data/cmapss/6. Turbofan Engine Degradation Simulation Data Set"` — will break on any machine that is not the SageMaker instance. Must be replaced with an environment-variable or relative-path pattern before paper experiments.
- No TODO or placeholder code.
- Collate functions (`collate_pretrain`, `collate_finetune`, `collate_test`) handle variable-length sequences via zero-padding. Mask convention: `True = padding`.

---

### 2.2 SMAP

**Loader:** `fam-jepa/data/smap_msl.py::load_smap()`

#### A. Data shape

From code comments (docstring, not runtime-asserted):
- Train: `(135183, 25)`
- Test: `(427617, 25)`
- Labels: `(427617,)` binary int32
- Anomaly rate: ~12.8%
- Channels: 25 (no channel dropping performed)

#### B. Preprocessing

1. **Normalization:** Z-score per channel, computed on train only.
   ```python
   mu = train.mean(axis=0, keepdims=True)   # shape (1, 25)
   std = train.std(axis=0, keepdims=True) + 1e-6
   train = (train - mu) / std
   test  = (test  - mu) / std
   ```
   The epsilon `1e-6` prevents division by zero but does not drop constant channels — a constant channel will be normalized to 0 everywhere, not removed.

2. **No channel dropping.** All 25 channels are retained regardless of variance.

3. **No resampling.** Data is loaded as-is from `.npy` files (already at the original telemetry rate from the OmniAnomaly/MTS-JEPA preprocessing pipeline).

4. **No NaN handling.** The `.npy` files from the OmniAnomaly source are assumed clean. No `nan_to_num` call is made.

5. **Window / patch:** `WINDOW_SIZE = 100` timesteps. `patch_length=1` in model instantiation. `TRAIN_STRIDE = 10` (sparse sampling for efficiency), `STRIDE = 1` (full overlap for test scoring).

#### C. Splits

SMAP has a natural **temporal train/test split** — the `.npy` files already separate train and test. There is no val split defined in this loader.

For **pretraining**, `AnomalyPretrainDataset` samples random `(ctx_len, horizon)` windows from `data['train']`. N_samples defaults to 50,000.

For **finetuning / evaluation**, the full `data['test']` array is scored against `data['labels']`. There is no separate labeled finetuning split — the anomaly detection approach is unsupervised (threshold is set at the 95th percentile of scores on the first 10% of test data, treated as "approximately normal").

Standard split: matches OmniAnomaly / MTS-JEPA / Anomaly Transformer convention (the `.npy` files are their preprocessed output).

#### D. Event labels for FAM/BCE

SMAP labels are already per-timestep binary: `labels[t] = 1` if timestep t is anomalous.

For BCE window labels, a window ending at time t is positive if any timestep in that window is labeled anomalous:
```python
window_label = int(labels[t - W : t].any())
```

For the unified TTE framework, `grey_swan_metrics.labels_to_first_event_tte(labels)` converts the binary array to time-to-first-event. Since SMAP has multiple anomaly segments, "first event" refers strictly to the first anomaly onset — subsequent anomaly segments are ignored by this function.

The `evaluate_anomaly_detection()` function in `smap_msl.py` computes per-timestep scores using L1 prediction error in embedding space, then thresholds at the 95th percentile of normal-period scores. It calls `anomaly_metrics()` which returns both non-PA F1 (primary) and PA F1 (for literature comparison).

#### E. Code quality

- `if __name__ == '__main__'` block present. Prints shapes and anomaly rate. Tests DataLoader batch shape. PASSES for infrastructure verification.
- **HARDCODED PATH:** `SMAP_DATA_DIR = Path('/home/sagemaker-user/IndustrialJEPA/paper-replications/mts-jepa/data/SMAP')`. Same SageMaker path issue as C-MAPSS.
- No TODO or placeholder code.
- `evaluate_anomaly_detection()` has a **hardcoded sys.path insert** (`sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/fam-jepa')`) — this is an absolute SageMaker path and will break on other machines. The import of `anomaly_metrics` should use a proper relative import or package install.
- Mask convention in `collate_anomaly_pretrain`: `True = valid` (opposite of C-MAPSS convention where `True = padding`). This inconsistency must be handled carefully when mixing datasets in a unified pipeline.

---

### 2.3 MSL

**Loader:** `fam-jepa/data/smap_msl.py::load_msl()`

#### A. Data shape

From code comments:
- Train: `(58317, 55)`
- Test: `(73729, 55)`
- Labels: `(73729,)` binary int32
- Anomaly rate: ~10.5%
- Channels: 55

#### B. Preprocessing

Identical to SMAP: Z-score per channel on train stats, no channel dropping, no resampling, no NaN handling. `WINDOW_SIZE = 100`.

#### C. Splits

Same structure as SMAP — natural temporal split from `.npy` files. No explicit val split. Pretraining on `data['train']`, evaluation on `data['test']` with auto-threshold on first 10% of test scores.

#### D. Event labels

Identical approach to SMAP. Per-timestep binary labels from `test_labels.npy`.

#### E. Code quality

Same file as SMAP, same issues. Both HARDCODED PATH (`MSL_DATA_DIR`) and `sys.path.insert` in `evaluate_anomaly_detection()`. The `__main__` block covers both SMAP and MSL.

---

### 2.4 PSM

**Loader:** `fam-jepa/data/psm.py::load_psm()`

#### A. Data shape

From `v19/datasets_audit.md` (code comments match):
- Train: `(132481, 25)` — but channel count is after constant-channel dropping
- Test: `(87841, 25)`
- Labels: `(87841,)` binary int32
- Anomaly rate: ~27%
- Channels: 25 stated in docstring; actual count may change if `drop_constant` removes any

Note: The docstring says 87K/87K but train is actually 132K. The discrepancy is in the docstring ("87K train / 87K test") — the `datasets_audit.md` entry gives the correct shape as `(132481, 25)` for train.

#### B. Preprocessing

1. **NaN handling:** `np.nan_to_num(df.values, nan=0.0)` — NaNs in CSV are replaced with 0. This is a silent imputation that affects normalization statistics. NaN distribution not documented.

2. **Constant channel dropping:** `var = train.var(axis=0); keep = var > 1e-8`. Applied before normalization. Number of dropped channels is logged to stdout but not persisted. The output channel count depends on the data and may differ from 25.

3. **Normalization:** Z-score per channel on train stats (identical to SMAP/MSL). Applied after channel dropping.

4. **Timestamp column:** Dropped from CSV by scanning for `'time'` in column name (case-insensitive). Not needed for `.npy` path.

5. **Length alignment:** `min_len = min(len(test), len(labels))` handles the known off-by-one between test CSV and label CSV.

6. **CSV-to-NPY caching:** If `.npy` files don't exist, converts from CSV and saves. Subsequent runs skip CSV parsing.

7. **Patch size:** Same as SMAP — `WINDOW_SIZE = 100`, `patch_length=1`.

#### C. Splits

Same as SMAP/MSL: natural train/test split from files. No explicit val split. Pretraining on `data['train']`, evaluation on `data['test']`.

`get_psm_dataloader()` imports `AnomalyPretrainDataset` and `collate_anomaly_pretrain` from `smap_msl.py` via relative import (`from .smap_msl import ...`). This requires the `data/` directory to be a proper Python package with an `__init__.py`. The current `__init__.py` is effectively empty (1 line), which is sufficient for the relative import to work.

#### D. Event labels

Same as SMAP/MSL: per-timestep binary from `test_labels.npy`. Same BCE window label construction applies.

#### E. Code quality

- `if __name__ == '__main__'` present. Checks availability with `check_psm_available()` before loading.
- **HARDCODED PATH:** `PSM_DATA_DIR = Path('/home/sagemaker-user/IndustrialJEPA/paper-replications/mts-jepa/data/PSM')`.
- `check_psm_available()` is a useful guard but only checks for file presence — does not validate shapes or label alignment.
- No `sys.path.insert` issues (no evaluation code in this file).
- The `nan_to_num(nan=0.0)` zero-imputation is undocumented. For PSM specifically, NaN values in server metrics typically represent sensor dropouts — zero is likely a poor substitute. This should be investigated.
- No TODO or placeholder code.

---

### 2.5 MBA (MIT-BIH Arrhythmia ECG)

**Loader:** `fam-jepa/experiments/v19/phase2_mba.py::load_mba()`

**There is no standalone loader in `fam-jepa/data/`.** The loading code is embedded in the experiment script.

#### A. Data shape

From `load_mba()` and `v19/datasets_audit.md`:
- Source files: `train.xlsx`, `test.xlsx`, `labels.xlsx`
- After `tr.values[1:, 1:]`: drops the first row (header artifact) and first column (sample index)
- Train: `(7680, 2)` — 7680 timesteps, 2 ECG channels
- Test: `(7680, 2)`
- Labels: `(7680,)` binary int32, expanded ±20 samples around arrhythmia annotation indices
- Channels: 2
- Sampling rate: 360 Hz (MIT-BIH standard)
- Anomaly rate: not printed in code; depends on number of annotation events and their density after ±20 expansion

The `v19/datasets_audit.md` notes "159K / 159K / 12K" rows in the raw Excel files, suggesting the 7680-row shape comes from subsampling or a specific subset — this discrepancy is unexplained in the code and should be investigated.

#### B. Preprocessing

1. **Normalization:** TranAD-style **min-max per channel**, computed on train only.
   ```python
   def normalize3(train, mn=None, mx=None):
       if mn is None: mn = train.min(axis=0)   # per-channel min
       if mx is None: mx = train.max(axis=0)   # per-channel max
       return (train - mn) / (mx - mn + 1e-8), mn, mx
   ```
   Scale: [0, 1]. The epsilon `1e-8` prevents zero-division for constant channels.
   This differs from SMAP/MSL/PSM which use Z-score — an inconsistency in normalization strategy across the paper's datasets.

2. **No channel dropping.** Both channels retained.

3. **No resampling.** 360 Hz is kept as-is. This means 1 timestep = 1/360 s, compared to SMAP/MSL which operate at ~1 Hz. The model sees very different temporal scales across datasets — relevant for pretraining curriculum.

4. **Label expansion:** Annotation indices from `labels.xlsx` are expanded to `±20` samples:
   ```python
   for i in range(-20, 21):
       idx = ls_idx + i
       idx = idx[(idx >= 0) & (idx < te.shape[0])]
       labels[idx] = 1
   ```
   This follows TranAD's evaluation protocol. A single annotation event produces a 41-sample anomaly window.

5. **Patch size:** `patch_length=1` (same as all other datasets).

#### C. Splits

Same structure as SMAP/MSL — `load_mba()` returns `train` and `test` arrays; there is no separate val split. Pretraining on `data['train']`, evaluation on `data['test']` with the same 95th-percentile threshold strategy.

The `SEEDS = [42, 123, 456]` and 3-run averaging in `main()` serve as the variance estimate in lieu of a separate val split.

#### D. Event labels

Per-timestep binary labels after ±20 expansion, identical format to SMAP/MSL/PSM. Same BCE window-label construction applies.

#### E. Code quality

- **No standalone loader file in `data/`.** The `load_mba()` function lives inside the experiment script and cannot be imported by other experiments without importing the entire `phase2_mba.py` (which has heavyweight imports and side effects).
- **HARDCODED PATH:** `MBA_PATH = Path('/home/sagemaker-user/IndustrialJEPA/paper-replications/mts-jepa/data/tranad_repo/data/MBA')`.
- `load_mba()` has no `if __name__ == '__main__'` guard; the main block is at the bottom of the experiment script.
- **Unexplained shape discrepancy:** `datasets_audit.md` says 159K rows in the raw Excel files; `load_mba()` produces 7680-row train/test. The `values[1:, 1:]` slicing only removes one row and one column — it cannot reduce 159K to 7680. This suggests the 7680-row figure refers to a pre-processed subset of the full MIT-BIH dataset, not the raw PhysioNet recording. The xlsx files may already be the subset. This must be verified before the paper.
- **Normalization inconsistency:** min-max (TranAD protocol) vs. Z-score (SMAP/MSL/PSM). The paper should either justify this difference or standardize all datasets to one normalization.

---

## 3. SMD (not in paper scope — audited for completeness)

**Loader:** `fam-jepa/data/smd.py::load_smd()`

SMD was in v19 experiments (`phase4_smd.py`) but is not in the five paper datasets. The loader is well-structured, supports both combined `.npy` and per-machine directory layouts, has `check_smd_available()`, and a `__main__` block. Same hardcoded SageMaker path issue. The loader could be promoted to a paper dataset if needed — no structural blockers.

---

## 4. Issues Summary

### Critical (will break on non-SageMaker machines)

| Issue | Files affected | Fix |
|-------|---------------|-----|
| Hardcoded `/home/sagemaker-user/...` data paths in every loader | `v11/data_utils.py`, `smap_msl.py`, `psm.py`, `smd.py`, `v19/phase2_mba.py` | Replace with `os.environ.get('INDUSTRIAL_JEPA_DATA', '...')` or a central `config.py` with a `DATASETS_ROOT` variable |
| `sys.path.insert` with absolute SageMaker path | `smap_msl.py::evaluate_anomaly_detection()` | Use package-relative import: `from mechanical_jepa.evaluation.grey_swan_metrics import anomaly_metrics` |

### Structural (affect paper reproducibility)

| Issue | Files affected | Fix |
|-------|---------------|-----|
| MBA has no standalone loader in `data/` | `v19/phase2_mba.py` | Extract `load_mba()` to `data/mba.py`, add path config, `__main__` test block |
| MBA 7680 vs. 159K row discrepancy | `v19/phase2_mba.py`, `datasets_audit.md` | Confirm the xlsx files are already a subset; document which subset and why |
| Normalization inconsistency: MBA uses min-max, others use Z-score | `v19/phase2_mba.py` vs. `smap_msl.py`, `psm.py` | Standardize to Z-score across all five datasets (or document the asymmetry as intentional TranAD protocol) |
| PSM NaN imputation with zero | `psm.py` | Investigate NaN distribution in PSM CSVs; use median or forward-fill if NaN = sensor dropout |
| Mask convention is inverted between C-MAPSS and anomaly loaders | `v11/data_utils.py` (True=padding) vs. `smap_msl.py` (True=valid) | Pick one convention; document it; add assertion in model forward pass |

### Minor (data quality)

| Issue | Files affected | Fix |
|-------|---------------|-----|
| PSM channel count after constant-drop is not persisted or asserted | `psm.py` | Log and assert final channel count |
| SMAP/MSL shape claims in docstring are not runtime-asserted | `smap_msl.py` | Add `assert train.shape == (135183, 25)` after load |
| No sanity checks for FD002/003/004 | `v11/data_utils.py` | Extend `sanity_check_fd001` pattern to all four subsets |
| `__init__.py` in `data/` is empty | `data/__init__.py` | Add explicit exports: `from .smap_msl import load_smap, load_msl` etc. |

---

## 5. Recommended Fixes (priority order)

### Fix 1: Central path configuration

Create `fam-jepa/data/config.py`:
```python
import os
from pathlib import Path

_ROOT = Path(os.environ.get('INDUSTRIAL_JEPA_DATA',
             '/home/sagemaker-user/IndustrialJEPA'))

CMAPSS_DIR = _ROOT / 'datasets/data/cmapss/6. Turbofan Engine Degradation Simulation Data Set'
SMAP_DIR   = _ROOT / 'paper-replications/mts-jepa/data/SMAP'
MSL_DIR    = _ROOT / 'paper-replications/mts-jepa/data/MSL'
PSM_DIR    = _ROOT / 'paper-replications/mts-jepa/data/PSM'
MBA_DIR    = _ROOT / 'paper-replications/mts-jepa/data/tranad_repo/data/MBA'
SMD_DIR    = _ROOT / 'paper-replications/mts-jepa/data/SMD'
```

Replace all hardcoded paths with imports from this file.

### Fix 2: Standalone MBA loader

Create `fam-jepa/data/mba.py` by extracting `load_mba()` from `phase2_mba.py`, adding Z-score normalization (to match SMAP/MSL/PSM), verifying the 7680-row shape, and adding a `__main__` test block.

### Fix 3: Standardize mask convention

In `collate_anomaly_pretrain` (smap_msl.py), change mask to `True = padding` to match C-MAPSS collators. This affects the model's attention mask interpretation — verify the model handles the convention correctly before changing.

### Fix 4: Fix sys.path in smap_msl.py

Replace:
```python
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/fam-jepa')
from evaluation.grey_swan_metrics import anomaly_metrics
```
with:
```python
from ..evaluation.grey_swan_metrics import anomaly_metrics
```
(assuming `fam-jepa` is installed as a package, or adjust import depth as needed).

### Fix 5: Populate data/__init__.py

```python
from .smap_msl import load_smap, load_msl, get_smap_dataloader, get_msl_dataloader
from .psm import load_psm, get_psm_dataloader
from .smd import load_smd, get_smd_dataloader
# from .mba import load_mba  # after Fix 2
```

---

## 6. Label Conversion Reference for BCE Loss

All five datasets ultimately need `y(t, Δt) ∈ {0, 1}` for the FAM BCE loss where y=1 means "an event will occur within Δt steps of timestep t".

| Dataset | Raw label format | Conversion to y(t, Δt) | Function |
|---------|-----------------|------------------------|----------|
| C-MAPSS | RUL scalar per engine-timestep (float, capped at 125) | `y = (RUL[t] <= Δt)` | `grey_swan_metrics.f1_at_horizon(pred, true_rul, k=Δt)` |
| SMAP | Per-timestep binary `(427617,)` | `y = labels[t:t+Δt].any()` — is any timestep in next Δt anomalous? | `grey_swan_metrics.labels_to_first_event_tte` + `event_detection(tte, pred_tte, horizon=Δt)` |
| MSL | Per-timestep binary `(73729,)` | Same as SMAP | Same |
| PSM | Per-timestep binary `(87841,)` | Same as SMAP | Same |
| MBA | Per-timestep binary `(7680,)` after ±20 expansion | Same as SMAP | Same |

The canonical unified call is `evaluate_event_prediction(time_to_event, pred_time_to_event, window_size, n_windows)` in `grey_swan_metrics.py`, which runs both detection and timing stages for all horizons.

For C-MAPSS: `time_to_event = rul_array` directly (already time-to-failure in cycles).
For anomaly datasets: `time_to_event = labels_to_first_event_tte(labels)` to convert binary labels to time-to-first-event.
