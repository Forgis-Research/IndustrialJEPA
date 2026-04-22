# C-MAPSS Dataset Analysis Report

Generated: 2026-04-10 23:48

## Overview

C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) contains 4 subsets of
turbofan engine run-to-failure trajectories from NASA.

## Inventory

| Subset | Train Eng | Test Eng | Min Cycles | Max Cycles | Mean Cycles | Op Conds |
|:------:|:---------:|:--------:|:----------:|:----------:|:-----------:|:--------:|
| FD001 | 100 | 100 | 128 | 362 | 206.3 | 1 |
| FD002 | 260 | 259 | 128 | 378 | 206.8 | 6 |
| FD003 | 100 | 100 | 145 | 525 | 247.2 | 1 |
| FD004 | 249 | 248 | 128 | 543 | 246.0 | 6 |

## Sensor Selection

Following STAR (Fan et al. 2024) convention, 7 near-constant sensors are dropped.
On FD001 (single operating condition), these sensors show std~0 or only 2 unique values:
- Dropped: s1, s5, s6, s10, s16, s18, s19 (7 sensors)
- Selected: s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21 (14 sensors)

Validation: selected sensors have mean |Spearman rho| with cycle position significantly
higher than dropped sensors. See sensor_informativeness_fd001.png.

## Normalization Strategy

- FD001/FD003 (single condition): global min-max per sensor on training data only
- FD002/FD004 (6 conditions): per-operating-condition min-max (KMeans k=6)
  This is critical for multi-condition subsets where raw sensor values cluster by condition.

## Key Observations

1. Engine lengths vary substantially within each subset (FD001: 128-362 cycles). Variable-length
   handling (padding + mask) is essential.
2. FD002/FD004 have 6 distinct operating conditions visible in op-setting scatter plots.
   Per-condition normalization is mandatory for these subsets.
3. A large fraction of early cycles have RUL > 125 and map to the constant plateau region
   under the piecewise-linear cap. This is the "healthy" phase.
4. Degradation trends in s2, s9, s14 are clearly visible in trajectory plots - these sensors
   show consistent monotonic trends across engines within each subset.
5. FD001 and FD003 share the same operational regime; FD002 and FD004 are multi-condition.

## Pitfalls

- Do NOT normalize test data using test statistics (use training stats only)
- Do NOT split train/test by cycles - split by engine_id
- FD002/FD004: apply per-condition normalization or sensor ranges will span multiple modes
- Last-window-per-engine evaluation is canonical for C-MAPSS (not sliding window)

## Figures

- episode_length_distributions.png - histograms of engine lengths
- sensor_informativeness_fd001.png - Spearman rho ranking
- operating_conditions_fd002.png - op condition clustering
- degradation_trajectories.png - sample degradation trajectories
- rul_distribution.png - RUL label histograms
