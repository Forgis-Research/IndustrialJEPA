# FAM Dataset Characterization

Reference for all datasets used in the paper. Informs data loading,
evaluation protocol, and SOTA comparisons.

---

## Cross-Dataset Summary

| Dataset | Domain | Entities | Ch | Rate | Event type | Label source | Our AUPRC (v22) | SOTA metric | SOTA value | Citation currency |
|---------|--------|----------|----|------|-----------|-------------|-----------------|-------------|------------|-------------------|
| FD001 | Turbofan | 100/100 engines | 14 | 1/cycle | Degradation | Simulation | 0.945±0.016 | RMSE | 10.61 (STAR '24) | RMSE |
| FD002 | Turbofan | 260/259 engines | 14 | 1/cycle | Degradation (multi-cond) | Simulation | 0.955±0.009 | RMSE | 13.47 (STAR '24) | RMSE |
| FD003 | Turbofan | 100/100 engines | 14 | 1/cycle | Degradation (multi-fault) | Simulation | 0.932±0.010 | RMSE | 10.71 (STAR '24) | RMSE |
| SMAP | Spacecraft | 55 entities | 25 | ~1/min | Anomaly onset | Expert (NASA) | 0.290±0.042 | PA-F1 | 0.336 (MTS-JEPA '26) | PA-F1 |
| MSL | Spacecraft | 27 entities | 55 | ~1/min | Anomaly onset | Expert (NASA) | 0.237±0.077 | PA-F1 | 0.336 (MTS-JEPA '26) | PA-F1 |
| PSM | Server | 1 stream | 25 | 1/min | Server incident | Engineer (eBay) | 0.417±0.113 | PA-F1 | 0.616 (MTS-JEPA '26) | PA-F1 |
| SMD | Server | 28 machines | 38 | 1/min | Machine anomaly | Engineer (Tsinghua) | 0.196±0.025 | PA-F1 | 0.925 (AT '22, diff split) | PA-F1 |
| MBA | Cardiac | 48 recordings | 2 | 275 Hz | Arrhythmia | Cardiologist (BIH) | 0.784±0.024 | F1 (supervised) | ~0.95 (InceptionTime) | No AD benchmark |
| Sepsis | ICU | 40K patients | 34 | 1/hour | Sepsis onset | Criterion (PhysioNet) | 0.096±0.019 | AUROC | 0.85 (InceptionTime '21) | AUROC |

---

## Per-Dataset Details

### C-MAPSS FD001 / FD002 / FD003

- **Scale**: 100–260 train engines, 100–259 test. 128–525 cycles per engine (mean ~206). 14 sensors.
- **Event**: RUL (remaining useful life) — degradation to failure. Piece-wise linear labels capped at 125 cycles.
- **Labels**: Simulation ground truth from NASA. No annotation noise.
- **SOTA**: STAR (Sensors 2024) — sparse Transformer with adaptive reconfiguration, supervised on RUL regression.
- **RMSE**: STAR 10.61 (FD001), 13.47 (FD002), 10.71 (FD003). Our pred-FT: 17.1, 12.4, 16.2.
- **Problems**: (1) Synthetic data. (2) RMSE measures point prediction at end-of-sequence; our surface AUPRC measures ranking quality across all timesteps and horizons — different tasks. (3) FD002 multi-condition, FD003 multi-fault are harder.
- **Tokens at P=16**: 8–23 per engine. Shortest engines (128 cycles) are at minimum viable token count.

### SMAP

- **Scale**: 55 independent spacecraft telemetry entities. 500–15K timesteps each. 25 channels.
- **Event**: Anomalous spacecraft behavior — sensor glitches, command faults, subsystem failures.
- **Labels**: Expert annotation by NASA anomaly review team, post-hoc. Binary per-timestep. Anomaly rate ~12.8% in test.
- **SOTA**: MTS-JEPA PA-F1 = 0.336. Our PA-F1 = 0.792 (from surfaces). Our non-PA F1 = 0.440.
- **Problems**: (1) PA-F1 inflation: non-PA F1 = 0.184 vs PA-F1 = 0.792. (2) Intra-entity split: 2.1% anomaly in ft_train vs 28.2% in ft_test (13x shift). (3) AUROC near chance (0.43). (4) Published baselines concatenate entities globally — our intra-entity split is stricter.

### MSL

- **Scale**: 27 entities. 500–8K timesteps each. 55 channels. Anomaly rate ~10.5%.
- **Event/Labels**: Same as SMAP (NASA expert annotation).
- **SOTA**: MTS-JEPA PA-F1 = 0.336. Our PA-F1 = 0.516. Our non-PA F1 = 0.330.
- **Problems**: (1) 55 channels is high-dim. (2) Short entities. (3) High seed variance (std=0.077).

### PSM

- **Scale**: Single continuous server pool stream. 132K train, 87K test. 25 channels.
- **Event**: Server pool incidents (performance degradation, resource exhaustion).
- **Labels**: Retrospective annotation by eBay engineers. Anomaly rate ~27% in test.
- **SOTA**: MTS-JEPA PA-F1 = 0.616. RANSynCoders PA-F1 = 0.714. Our PA-F1 = 0.619.
- **Problems**: (1) Single entity — no entity generalization. (2) 17.7% → 39.9% anomaly rate shift train→test. (3) Channels have distinct semantics (CPU/memory/disk/network).

### SMD

- **Scale**: 28 server machines. ~25K timesteps each. 38 channels. Anomaly rate ~4.2%.
- **Event**: Server machine anomalies.
- **Labels**: Annotated by Tsinghua operations engineers.
- **SOTA**: Anomaly Transformer PA-F1 = 0.925 (different split protocol!). Our PA-F1 = 0.608. Our non-PA F1 = 0.262.
- **Problems**: (1) AT uses global split, we use intra-entity — not comparable. (2) PA inflation: non-PA 0.262 vs PA 0.608.

### MBA (MIT-BIH Arrhythmia)

- **Scale**: 48 ECG recordings, 30 min each at 275 Hz. 2 channels. ~650K samples per recording.
- **Event**: Cardiac arrhythmia — ventricular ectopic beats, AF episodes, bundle branch blocks.
- **Labels**: Beat-by-beat annotation by expert cardiologists (Moody & Mark 2001).
- **SOTA**: InceptionTime supervised per-beat F1 ≈ 0.95. No established anomaly-detection AUPRC benchmark.
- **Problems**: (1) PA-F1 = 0.999 (ceiling — long arrhythmia segments). (2) Only 2 channels. (3) No published AUPRC benchmark to compare against. (4) We are the first to frame this as anomaly prediction with AUPRC.

### PhysioNet 2019 Sepsis

- **Scale**: 40K ICU patients. 6–336 hours per stay (mean ~40h). 34 clinical variables (after dropping static/admin).
- **Event**: Sepsis onset (Sepsis-3 criteria). Label = 1 for [t_sepsis-6h, t_sepsis+3h].
- **Labels**: Criterion-based (SOFA score increase ≥ 2 + suspected infection). ~2.2% positive timesteps. ~8% of patients develop sepsis.
- **SOTA**: AUROC 0.85 (InceptionTime), 0.78 (MGP-AttTCN). Challenge utility ≈ 0.38 (top leaderboard).
- **AUPRC in literature**: 0.15–0.30 (due to 2% prevalence). Our: 0.096 ± 0.019.
- **Problems**: (1) >60% missingness in lab values. (2) Label noise ~15-20%. (3) SOFA features not accessible from raw channels. (4) Our AUROC 0.698 is 8-15 pp below SOTA — expected without clinical domain knowledge. (5) **P=1 required** (hourly data, stays too short for P=16).

---

## Protocol Comparability Warnings

1. **C-MAPSS**: AUPRC (ours) vs RMSE (SOTA) measure different things. Include legacy RMSE for comparability.
2. **SMAP/MSL/PSM/SMD**: Published SOTA uses PA-F1. Report our PA-F1 (from surfaces) alongside non-PA F1 and AUPRC. State PA inflation explicitly.
3. **SMD**: AT's 0.925 uses global split; our 0.608 uses intra-entity. Not directly comparable — footnote required.
4. **MBA**: No AUPRC benchmark exists. We are the first in this framing.
5. **Sepsis**: AUROC is the comparable metric. AUPRC too prevalence-dependent to compare across papers.

---

## Data Loader Notes for V24

All loaders must pass `normalize=False` — RevIN in the encoder handles normalization.

| Dataset | Loader | normalize=False needed? | Split type |
|---------|--------|------------------------|------------|
| C-MAPSS | `v11/data_utils.py:load_cmapss_subset()` | Yes (defaults to True) | Per-engine train/test |
| SMAP | `data/smap_msl.py:split_smap_entities()` | Yes | Intra-entity chrono |
| MSL | `data/smap_msl.py:split_msl_entities()` | Yes | Intra-entity chrono |
| PSM | `data/psm.py:load_psm()` | Yes | Chrono with gap |
| SMD | `data/smd.py:split_smd_entities()` | Yes | Intra-entity chrono |
| MBA | `data/mba.py:load_mba()` | Yes | Chrono with gap |
| Sepsis | `data/sepsis.py` | Check | Patient-level |
