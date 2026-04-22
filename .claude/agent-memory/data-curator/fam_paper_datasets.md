---
name: FAM paper dataset loader audit
description: Key findings from April 2026 audit of all five FAM paper dataset loaders — shapes, issues, and the mask convention mismatch
type: project
---

Five datasets confirmed for the FAM paper: C-MAPSS FD001-004, SMAP, MSL, PSM, MBA.

**Why:** Paper needs reproducible, portable loaders before experiments can run.

**How to apply:** Use this as a checklist when writing or fixing loaders.

## Verified shapes (from code, not runtime)

- C-MAPSS FD001: 20631 rows × 26 cols raw → 100 engines × var-len × 14 sensors (sanity_check asserted in code)
- SMAP: train (135183, 25), test (427617, 25), labels (427617,) binary — from docstring, not asserted
- MSL: train (58317, 55), test (73729, 55), labels (73729,) binary — from docstring, not asserted
- PSM: train (132481, 25), test (87841, 25), labels (87841,) binary — from datasets_audit.md
- MBA: train (7680, 2), test (7680, 2), labels (7680,) binary — unexplained discrepancy with datasets_audit.md citing 159K raw Excel rows

## Critical blockers

1. ALL loaders have hardcoded `/home/sagemaker-user/...` paths. Fix: central `data/config.py` reading `INDUSTRIAL_JEPA_DATA` env var.
2. `smap_msl.py::evaluate_anomaly_detection()` has hardcoded `sys.path.insert` to SageMaker path.
3. MBA has no standalone loader in `data/` — `load_mba()` is buried in `v19/phase2_mba.py`.
4. MBA 7680 vs 159K row discrepancy is unexplained.

## Normalization inconsistency

MBA uses **min-max** (TranAD protocol). SMAP, MSL, PSM use **Z-score**. Paper should standardize or justify.

## Mask convention mismatch

C-MAPSS collators: `True = padding` (token is masked OUT).
SMAP/MSL/PSM `collate_anomaly_pretrain`: `True = valid` (token is kept).
This will silently invert attention masks if datasets are mixed. Must be resolved before any cross-dataset training.

## Label conversion for BCE

- C-MAPSS: `y(t, k) = (RUL[t] <= k)` via `grey_swan_metrics.f1_at_horizon`
- Anomaly datasets: `labels_to_first_event_tte(labels)` then `event_detection(tte, pred, horizon=k)`
- Unified call: `evaluate_event_prediction(tte, pred_tte, window_size, n_windows)`

## Audit document

Full audit at: `mechanical-jepa/data/DATASETS.md`
