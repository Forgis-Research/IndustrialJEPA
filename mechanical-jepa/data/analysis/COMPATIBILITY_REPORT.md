# Dataset Compatibility Report

Generated: 2026-04-09

Analysis: 8 bearing sources, 12800Hz target SR, 1024-sample windows


## Summary Table

| Source | Centroid (Hz) | Kurtosis | RMS | Crest Factor | KL vs FEMTO | Compatible? |
|--------|--------------|---------|-----|-------------|-------------|-------------|
| cwru            |     2699 |    4.57 |  1.000 |       4.80 |       1.47 | YES          |
| mfpt            |     2753 |   12.39 |  1.000 |       6.49 |       0.54 | MARGINAL     |
| ims             |     2827 |    0.60 |  1.000 |       3.98 |       0.73 | YES          |
| mafaulda        |      173 |    2.91 |  1.000 |       2.83 |       3.04 | MARGINAL     |
| ottawa          |     1074 |    3.30 |  1.000 |       4.56 |       0.99 | YES          |
| femto           |     2453 |    0.99 |  1.000 |       4.22 |       0.28 | YES          |
| xjtu_sy         |     1987 |    0.16 |  1.000 |       3.49 |       0.28 | YES          |
| paderborn       |     3323 |    2.40 |  1.000 |       4.72 |       0.67 | YES          |

## Per-Source Statistics

### CWRU
- **Verdict**: COMPATIBLE
- Windows analyzed: 300
- Spectral centroid: 2699 ± 695 Hz
- Kurtosis: 4.57 ± 6.18
- RMS: 1.000 ± 0.000
- Crest factor: 4.80 ± 1.65
- Band energies: 0-500Hz: 6.3%, 500-2kHz: 14.4%, 2-5kHz: 78.7%, 5kHz+: 0.6%

### MFPT
- **Verdict**: MARGINAL
- **Reason**: high kurtosis diff (11.8)
- Windows analyzed: 300
- Spectral centroid: 2753 ± 440 Hz
- Kurtosis: 12.39 ± 16.99
- RMS: 1.000 ± 0.000
- Crest factor: 6.49 ± 3.13
- Band energies: 0-500Hz: 3.6%, 500-2kHz: 37.6%, 2-5kHz: 48.1%, 5kHz+: 10.7%

### IMS
- **Verdict**: COMPATIBLE
- Windows analyzed: 300
- Spectral centroid: 2827 ± 426 Hz
- Kurtosis: 0.60 ± 2.18
- RMS: 1.000 ± 0.000
- Crest factor: 3.98 ± 1.03
- Band energies: 0-500Hz: 9.2%, 500-2kHz: 31.3%, 2-5kHz: 46.4%, 5kHz+: 13.0%

### MAFAULDA
- **Verdict**: MARGINAL
- **Reason**: moderate PSD divergence (KL=3.0); large spectral centroid diff (2047Hz>1500Hz)
- Windows analyzed: 300
- Spectral centroid: 173 ± 50 Hz
- Kurtosis: 2.91 ± 1.72
- RMS: 1.000 ± 0.000
- Crest factor: 2.83 ± 0.34
- Band energies: 0-500Hz: 93.9%, 500-2kHz: 5.0%, 2-5kHz: 1.0%, 5kHz+: 0.1%

### OTTAWA
- **Verdict**: COMPATIBLE
- Windows analyzed: 300
- Spectral centroid: 1074 ± 649 Hz
- Kurtosis: 3.30 ± 6.67
- RMS: 1.000 ± 0.000
- Crest factor: 4.56 ± 2.03
- Band energies: 0-500Hz: 31.1%, 500-2kHz: 56.0%, 2-5kHz: 10.8%, 5kHz+: 2.2%

### FEMTO
- **Verdict**: COMPATIBLE
- **Reason**: reference source (RUL target)
- Windows analyzed: 300
- Spectral centroid: 2453 ± 564 Hz
- Kurtosis: 0.99 ± 2.02
- RMS: 1.000 ± 0.000
- Crest factor: 4.22 ± 1.04
- Band energies: 0-500Hz: 10.5%, 500-2kHz: 40.8%, 2-5kHz: 38.2%, 5kHz+: 10.5%

### XJTU_SY
- **Verdict**: COMPATIBLE
- **Reason**: reference source (RUL target)
- Windows analyzed: 300
- Spectral centroid: 1987 ± 785 Hz
- Kurtosis: 0.16 ± 0.46
- RMS: 1.000 ± 0.000
- Crest factor: 3.49 ± 0.40
- Band energies: 0-500Hz: 10.4%, 500-2kHz: 56.1%, 2-5kHz: 23.8%, 5kHz+: 9.7%

### PADERBORN
- **Verdict**: COMPATIBLE
- Windows analyzed: 300
- Spectral centroid: 3323 ± 642 Hz
- Kurtosis: 2.40 ± 3.42
- RMS: 1.000 ± 0.000
- Crest factor: 4.72 ± 1.29
- Band energies: 0-500Hz: 4.8%, 500-2kHz: 23.8%, 2-5kHz: 46.1%, 5kHz+: 25.3%

## Recommended Source Groups

- **Group A (bearing RUL — primary targets)**: femto, xjtu_sy, ims
- **Group B (compatible bearing faults)**: cwru, ottawa, paderborn
- **Group C (marginal — structural differences)**: mfpt, mafaulda
- **Group D (incompatible — exclude from joint pretraining)**: 

## Pretraining Recommendation

Based on the analysis:

**Recommended pretraining group**: cwru, ims, ottawa, femto, xjtu_sy, paderborn

**Exclude from pretraining**:
- MAFAULDA: spectral centroid 173Hz (vs 2453Hz for FEMTO). 93.9% of energy in 0-500Hz band vs 10.5% for FEMTO. The encoder would need to simultaneously represent 173Hz centrifugal pump signals and 2453Hz bearing vibrations — these are incompatible objectives.
- MFPT: extreme kurtosis spread (12.4 ± 17.0 vs 1.0 ± 2.0 for FEMTO). The high variance kurtosis distribution suggests impulse-dominated signals that are structurally different from run-to-failure data.

**Key findings**:
1. Instance normalization makes RMS comparable across sources, but does NOT equalize spectral shapes.
2. Sources with very different spectral centroids pull the JEPA encoder in conflicting directions.
3. Kurtosis differences indicate different signal statistics that instance norm does not address.
4. MAFAULDA (centrifugal pump) has the most different signal characteristics from bearing sources.

## Pairwise Compatibility Matrices

See plots in `data/analysis/plots/`:

- `psd_comparison.png` — Average PSD per source
- `amplitude_distributions.png` — Amplitude histograms
- `compatibility_kl.png` — PSD KL divergence matrix
- `compatibility_wasserstein.png` — Wasserstein distance matrix
- `compatibility_centroid.png` — Spectral centroid difference matrix
- `signal_stats.png` — Per-source kurtosis, centroid, RMS, crest factor
- `band_energies.png` — Band energy distribution per source