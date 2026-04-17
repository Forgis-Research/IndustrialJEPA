# Research: Best Practices for Training-Ready Multi-Source Vibration Datasets

**Date:** 2026-03-31
**Purpose:** Inform the design of a "v2 training-ready" config for Forgis/Mechanical-Components

---

## 1. How Existing Benchmarks Handle Multi-Source Data

### 1.1 HEAR 2021 (Holistic Evaluation of Audio Representations)

The HEAR benchmark (NeurIPS 2021) evaluated audio representations across 19 diverse downstream tasks from 16 datasets spanning speech, environmental sound, and music.

**Key design decisions:**
- **Single canonical sampling rate:** All audio resampled to **48 kHz** regardless of source
- **Standardized format:** All datasets normalized to "a common human-readable format" via `hearpreprocess`
- **Common embedding API:** Models must accept raw audio at 48 kHz and produce embeddings
- **Splits preserved:** Original train/test splits from each source dataset are respected

**Lesson for us:** HEAR chose the *highest* common rate (48 kHz) to avoid information loss. This is the audio community's standard approach -- resample up, not down.

Source: https://hearbenchmark.com/ and https://arxiv.org/abs/2203.03022

### 1.2 Google Speech Commands (HuggingFace)

- **Fixed sampling rate:** 16 kHz (all recordings)
- **Fixed duration:** 1 second per utterance (16,000 samples)
- **Padding:** Shorter clips padded, silence sliced to 1-second windows
- **Schema:** `{audio: {array, sampling_rate}, label, speaker_id, utterance_id}`
- **Splits:** Pre-defined train/validation/test by speaker (no speaker leakage)

**Lesson for us:** Fixed length + fixed rate makes batching trivial. Speaker-disjoint splits prevent leakage -- analogous to our need for source-disjoint or episode-disjoint splits.

Source: https://huggingface.co/datasets/google/speech_commands

### 1.3 BearLLM / MBHM Dataset (The Closest Precedent)

The **MBHM (Multimodal Bearing Health Management)** dataset on HuggingFace is the most directly relevant precedent. Published at AAAI 2025, it combines **9 bearing datasets** (CWRU, DIRG, HIT, IMS, JNU, JUST, MFPT, PU, XJTU) with **262 working conditions** and sampling rates from 12-100 kHz.

**How they solved the multi-rate problem -- Discrete Cosine Normalization (DCN):**
1. Apply Discrete Cosine Transform (DCT) to convert time-domain signal to frequency domain
2. Pad (with zeros) or truncate to exactly **24,000 frequency components**
3. Normalize amplitude to [-1, 1] range with scale factor beta = 0.01
4. Store a fault-free reference signal and compute residual (Fres = Fquery - Freference)

**Key properties:**
- Fixed output length: 24,000 (frequency domain, not time domain)
- Frequency-domain representation preserves fault signatures regardless of original sampling rate
- Amplitude normalization to [-1, 1]
- 7:2:1 train/val/test split ratio
- Reference signal operations restricted to training data (prevents leakage)

**Limitations:**
- Frequency-domain representation loses temporal structure (phase, transient timing)
- Not suitable for prediction tasks (JEPA needs time-domain for next-state prediction)
- The 24,000 length is arbitrary and manually chosen

Source: https://huggingface.co/datasets/SIA-IDE/MBHM and https://arxiv.org/abs/2408.11281

---

## 2. How Time Series Foundation Models Handle Multi-Source Data

### 2.1 MOMENT (CMU, ICML 2024) -- The Time Series Pile

**Dataset:** 13 domains, 13M time series, 1.23 billion timestamps from 4 major repositories (Informer, Monash, UCR/UEA, TSB-UAD).

**Handling different sampling rates:**
- **They don't explicitly model temporal resolution.** Quote: "We did not explicitly model the temporal resolution of time series, since this information is often unavailable outside of time series forecasting datasets."
- Instead, they rely on **reversible instance normalization** to handle diverse temporal distributions

**Fixed input length:**
- **T = 512 time steps** (fixed)
- Longer series: sub-sampled to 512
- Shorter series: zero-padded on the left

**Patching:**
- 64 patches of 8 time steps each (disjoint, no stride)
- 30% of patches masked randomly during pretraining (replaced with learnable [MASK] token)

**Normalization:**
- **Reversible instance normalization** (per-sample zero-mean, unit-variance, reversed at output)
- This is critical: it means the model never sees absolute amplitudes, only relative patterns

**Splits:**
- Use predefined splits when available
- Otherwise: 60% train, 10% val, 30% test
- Only training splits used for pretraining

Source: https://arxiv.org/abs/2402.03885

### 2.2 Chronos (Amazon, 2024) -- Tokenization Approach

**Dataset:** Large collection of public time series + synthetic data from Gaussian processes (KernelSynth + TSMix augmentation).

**How they handle variable domains:**
- **Tokenize by scaling + quantization:** Scale each series by its absolute mean, then quantize into 4,096 uniformly spaced bins
- This converts any time series (regardless of domain, rate, or amplitude) into a sequence of discrete tokens
- Special tokens: PAD (padding/missing values), EOS (end of sequence)

**Key insight:** By scaling by absolute mean *per series*, Chronos achieves implicit normalization. The model learns patterns, not absolute values.

**Variable length:** Handled natively by transformer architecture with PAD tokens.

**Chronos-2 (Oct 2025):** Extended to multivariate and covariate-informed forecasting.

Source: https://arxiv.org/abs/2403.07815

### 2.3 MOIRAI (Salesforce, 2024) -- LOTSA Dataset

**Dataset:** LOTSA (Large-scale Open Time Series Archive) -- **27.6 billion observations** across 105 datasets in 9 domains.

**How they handle multiple frequencies:**
- **Multiple patch sizes:** 5 different input/output projection layers for patch sizes {8, 16, 32, 64, 128}
- High-frequency data gets larger patches (compresses more time steps)
- Low-frequency data gets smaller patches (preserves granularity)
- Patch size selected based on data frequency

**Variable length:**
- Sequence length uniformly sampled during training (min 2 timesteps, max 512 tokens)
- **Sequence packing** to minimize padding waste (reduced padding from 61% to 0.38%)

**Normalization:** Instance normalization + RMSNorm internally.

**Multivariate handling:** "Any-variate attention" with Rotary Position Embeddings for temporal position + learnable binary attention biases for variate identity. Flattens all variates into a single sequence.

**Splits:** Held out entire datasets for OOD evaluation (not random within-dataset splits).

Source: https://arxiv.org/abs/2402.02592

### 2.4 TimesFM (Google, 2024)

- **100B+ real-world time points** (Google Trends, Wikipedia pageviews)
- **Patch-based:** Groups of contiguous time points treated as tokens
- **Decoder-only transformer:** Predicts (i+1)-th patch given i-th output
- **MSE loss** (not cross-entropy like Chronos)

Source: https://github.com/google-research/timesfm

---

## 3. Synthesis: What the Literature Tells Us

### 3.1 Sampling Rate Strategy

| Approach | Used By | Pros | Cons |
|----------|---------|------|------|
| Resample everything to one rate | HEAR (48kHz), Speech Commands (16kHz) | Simple batching, consistent | Loses info if downsampling, wastes space if upsampling |
| Frequency-domain normalization | BearLLM/MBHM (DCN to 24K) | Rate-agnostic | Loses temporal structure, bad for prediction |
| Don't model rate explicitly | MOMENT | Simple | Ignores physical meaning of temporal resolution |
| Multiple patch sizes per rate | MOIRAI (5 patch sizes) | Adapts to data | Complex architecture |
| Per-sample scaling/tokenization | Chronos | Domain-agnostic | Quantization loss |

**For vibration/JEPA specifically:**
- We need **time-domain** data (frequency-domain like BearLLM won't work for next-state prediction)
- Bearing fault characteristic frequencies range from ~50 Hz (cage faults at low speed) to ~5 kHz (resonance bands)
- By Nyquist, 12 kHz captures up to 6 kHz -- sufficient for characteristic frequencies and first resonance bands
- 25.6 kHz captures up to 12.8 kHz -- better for high-frequency resonance bands and early fault detection
- Higher rates (48 kHz, 100 kHz) are overkill for most bearing faults

### 3.2 Window Length Strategy

| Dataset/Paper | Window Length | Samples | Duration at Native Rate |
|---------------|-------------|---------|------------------------|
| CWRU standard practice | 1024 samples | 1024 | ~85 ms at 12 kHz |
| CWRU with overlap | 1024 / stride 256 | 1024 | ~85 ms at 12 kHz |
| XJTU-SY native | 32,768 samples | 32,768 | 1.28 sec at 25.6 kHz |
| BearLLM/MBHM | 24,000 (freq domain) | 24,000 | N/A (frequency domain) |
| MOMENT | 512 time steps | 512 | Varies |
| MOIRAI | Up to 512 tokens | Varies | Varies |
| General vibration lit | 1024-10,000 | Varies | ~0.1-1.0 sec |

**For bearing fault diagnosis:**
- One shaft revolution at 1800 RPM = 33 ms. At 12 kHz = 400 samples.
- Need at least 3-5 revolutions to see periodic fault patterns = ~100-170 ms = 1,200-2,000 samples at 12 kHz
- Standard CWRU practice: 1,024 samples (~85 ms) is actually *too short* for reliable fault frequency extraction at low speeds
- 1 second is a sweet spot: captures ~30 revolutions at 1800 RPM, plenty for fault pattern visibility
- At 12 kHz: 1 second = 12,000 samples. At 25.6 kHz: 1 second = 25,600 samples.

### 3.3 Normalization Strategy

| Approach | Used By | Description |
|----------|---------|-------------|
| Instance normalization | MOMENT, MOIRAI | Per-sample zero-mean, unit-variance. Reversed at output. |
| Absolute mean scaling | Chronos | Divide by absolute mean of the series |
| Amplitude normalization to [-1, 1] | BearLLM | Global min-max per sample |
| Z-score (global) | Many ML papers | Mean/std from training set |

**Consensus:** Per-sample (instance) normalization is the dominant approach in foundation models. It makes the model amplitude-invariant, which is essential when combining sources with different sensor gains, mounting conditions, and units.

### 3.4 Split Strategy

| Approach | Used By | When |
|----------|---------|------|
| By source dataset (OOD) | MOIRAI | Entire datasets held out for zero-shot eval |
| By speaker/subject | Speech Commands | Prevents identity leakage |
| Predefined splits | MOMENT | Respects original dataset creators |
| Temporal (horizontal) | MOMENT, MOIRAI | Long series split in time |
| By episode | Best practice for RUL | Never mix samples from same run-to-failure episode |

**For our dataset, splits should be:**
- **Never random within a recording** (this is the CWRU leakage problem documented by Braga et al.)
- **By source for cross-domain evaluation** (train on 5 sources, test on 2 held-out sources)
- **By episode for RUL data** (entire degradation trajectories in same split)
- **Multiple split schemes** as separate configs (within-source splits + cross-source splits)

---

## 4. Recommended "v2 Training-Ready" Config

Based on all the above research, here is the recommended design:

### 4.1 Fixed Sampling Rate: **12,800 Hz (12.8 kHz)**

**Rationale:**
- 12.8 kHz is a power-of-two-friendly rate (2^7 * 100) that works well with FFT and patching
- Captures frequencies up to 6.4 kHz (Nyquist) -- sufficient for all bearing/gear characteristic frequencies and primary resonance bands
- CWRU is natively 12 kHz (minimal resampling)
- MCC5-THU is natively 12.8 kHz (no resampling needed)
- XJTU-SY at 25.6 kHz: downsample by exactly 2x (clean decimation)
- IMS at 20 kHz: resample by 16/25 ratio
- Paderborn at 64 kHz: downsample by exactly 5x
- FEMTO at 25.6 kHz: downsample by exactly 2x
- MFPT varies (48.8 kHz for faults, 97.7 kHz for baselines): resample down
- Mendeley at 25.6 kHz: downsample by exactly 2x
- Close to the 12 kHz CWRU standard that dominates the literature
- 25.6 kHz would be better for high-frequency resonance but doubles storage and most sources need upsampling

**Alternative considered:** 25,600 Hz (25.6 kHz)
- Pro: Native for XJTU-SY, FEMTO, Mendeley (3 major sources)
- Pro: Better high-frequency coverage for early fault detection
- Con: CWRU (12 kHz) and MCC5-THU (12.8 kHz) need upsampling (adds interpolated data, not real information)
- Con: Doubles sample count per window vs 12.8 kHz
- Con: MOMENT showed that not modeling temporal resolution explicitly still works

**Recommendation:** Provide both. A `v2_12k` and `v2_25k` config, with 12.8 kHz as the default for training and 25.6 kHz as an option for sources that natively support it.

### 4.2 Fixed Window Length: **16,384 samples (1.28 seconds at 12.8 kHz)**

**Rationale:**
- Power of 2 (2^14) -- optimal for FFT, clean patching for transformers
- 1.28 seconds captures ~38 shaft revolutions at 1800 RPM -- plenty for fault pattern visibility
- Matches XJTU-SY native snapshot length (1.28 sec at 25.6 kHz = 32,768 samples, which downsampled 2x = 16,384)
- At 12.8 kHz: 16,384 samples = 1.28 seconds
- Cleanly divisible into patches: 64 patches of 256, 128 patches of 128, 256 patches of 64, etc.
- Long enough for RUL degradation signature visibility
- Short enough for efficient batching (16,384 float32 = 64 KB per channel)

**Handling sources with less data:**
- CWRU recordings are ~10 seconds at 12 kHz. Resample to 12.8 kHz, then slice into non-overlapping 16,384-sample windows. Yields ~7-8 windows per recording.
- Short recordings: zero-pad on the right with a `valid_length` field indicating actual signal length

**Handling sources with more data:**
- IMS, XJTU-SY snapshots are longer: slice into multiple non-overlapping windows per snapshot
- FEMTO 0.1-second snippets (2,560 samples at 25.6 kHz = 1,280 at 12.8 kHz): too short for 16,384. Options:
  - (a) Concatenate consecutive snippets within same episode to fill 16,384 (with gap markers)
  - (b) Zero-pad to 16,384 with `valid_length=1280`
  - (c) Exclude from training-ready config (keep in raw config only)
  - **Recommend (b)** with valid_length field, so masking can ignore padding

### 4.3 Vibration-Only (Single Modality)

**Drop:** current, temperature, tachometer signals from training-ready config.

**Rationale:**
- JEPA pretraining should learn from the most universally available modality
- Only 2 of 10 sources have current (Paderborn, Mendeley); only 1 has tachometer (PHM 2009)
- Including rare modalities creates severe class imbalance in channel space
- Multi-modal data is preserved in the raw v1 config for future multi-modal experiments
- Audio foundation models (HEAR, speech_commands) similarly focus on single-modality (audio waveform only)

**Exception:** Keep multi-axis vibration. If a source has X, Y, Z accelerometer channels, keep all of them. Channel count varies (1-8), which is fine -- MOIRAI's "any-variate attention" shows this is solvable.

### 4.4 Normalization Strategy: **Per-Sample Instance Normalization**

Following the consensus from MOMENT, MOIRAI, and Chronos:

```python
# Per-sample, per-channel normalization
for each channel c in sample:
    mean_c = mean(signal[c])
    std_c = std(signal[c])
    signal_normalized[c] = (signal[c] - mean_c) / max(std_c, 1e-8)
    # Store mean_c and std_c in metadata for reversibility
```

**Store both raw and normalized:**
- `signal_raw`: Original amplitude (for physics-based analysis, envelope extraction)
- `signal_normalized`: Zero-mean, unit-variance per channel (for model input)
- `signal_mean`: Per-channel mean (for reversal)
- `signal_std`: Per-channel std (for reversal)

**Why not global normalization?**
- Different sources have wildly different amplitude scales (g vs m/s^2 vs raw ADC counts)
- Different sensor sensitivities and mounting conditions
- Per-sample normalization is standard in all recent foundation models
- Makes the model learn *patterns* not *absolute amplitudes*

### 4.5 Train/Val/Test Split Strategy

**Primary split: By source (cross-domain evaluation)**

```
Train sources:  CWRU, MFPT, FEMTO, XJTU-SY, IMS, MCC5-THU, OEDI
Val sources:    Paderborn (different lab, different fault creation method)
Test sources:   Mendeley (variable speed -- hardest generalization)
```

**Secondary split: Within-source (standard evaluation)**
- For each source, split by recording/episode (never random within recording)
- 70% train, 15% val, 15% test
- Episode-disjoint for run-to-failure data (FEMTO, XJTU-SY, IMS)
- Condition-disjoint where possible (train on some RPM/load, test on others)

**Implement as separate configs:**
- `v2_cross_domain` -- split by source (for transfer learning evaluation)
- `v2_standard` -- within-source splits (for standard benchmarking)
- Both use the same preprocessed signals, different split assignments

### 4.6 Proposed Schema for v2 Training-Ready Config

```python
{
    # === IDENTIFICATION ===
    "sample_id": "cwru_105_w0",
    "source_id": "cwru",

    # === SIGNAL (fixed size, normalized) ===
    "signal": [[...], [...]],           # (n_channels, 16384) float32, normalized
    "signal_raw": [[...], [...]],       # (n_channels, 16384) float32, original amplitude
    "n_channels": 2,
    "channel_names": ["DE_accel", "FE_accel"],
    "valid_length": 16384,              # Actual signal length before padding (for masking)

    # === NORMALIZATION METADATA (for reversibility) ===
    "channel_means": [0.012, -0.003],   # Per-channel means subtracted
    "channel_stds": [0.45, 0.31],       # Per-channel stds divided

    # === LABELS ===
    "health_state": "faulty",           # healthy | faulty | degrading
    "fault_type": "inner_race",
    "fault_severity": null,             # 0-1 continuous

    # === OPERATING CONDITIONS ===
    "rpm": 1750,
    "load": 2.0,
    "load_unit": "hp",

    # === TEMPORAL ===
    "episode_id": null,
    "episode_position": null,           # 0.0-1.0
    "rul_percent": null,

    # === TRANSITIONS ===
    "is_transition": false,
    "transition_type": null,

    # === SPLIT ASSIGNMENTS ===
    "split_cross_domain": "train",      # train|val|test by source
    "split_standard": "train",          # train|val|test within source

    # === SIGNAL PROPERTIES (constant per source, denormalized here for convenience) ===
    "sampling_rate_hz": 12800,          # Always 12800 in v2
    "window_duration_sec": 1.28,        # Always 1.28 in v2
    "original_sampling_rate_hz": 12000, # Before resampling
}
```

---

## 5. Summary of Key Decisions

| Decision | Choice | Justification |
|----------|--------|---------------|
| Sampling rate | 12,800 Hz | Clean decimation from most sources, sufficient for fault frequencies, FFT-friendly |
| Window length | 16,384 samples (1.28 sec) | Power-of-2, matches XJTU-SY native, good fault visibility |
| Modality | Vibration only | Universal availability, consistent with foundation model practice |
| Normalization | Per-sample instance norm | Consensus from MOMENT/MOIRAI/Chronos, handles amplitude diversity |
| Raw preservation | Store both raw + normalized | Enables physics analysis and reversibility |
| Primary split | By source | Cross-domain generalization is the real test (Braga et al., 2022) |
| Secondary split | Within-source, episode-disjoint | Prevents recording leakage, standard evaluation |
| Short signal handling | Zero-pad + valid_length field | Simple, masking-compatible |
| Multi-channel | Keep all vibration axes | MOIRAI any-variate attention proves variable channels work |

---

## 6. Key References

### HuggingFace Dataset Precedents
- [MBHM (BearLLM)](https://huggingface.co/datasets/SIA-IDE/MBHM) -- 9 bearing datasets unified via DCN, AAAI 2025
- [Google Speech Commands](https://huggingface.co/datasets/google/speech_commands) -- Fixed 16kHz, 1-second windows, speaker-disjoint splits
- [HEAR 2021 Benchmark](https://hearbenchmark.com/) -- All audio resampled to 48 kHz, common API

### Time Series Foundation Models
- [MOMENT](https://arxiv.org/abs/2402.03885) -- Time Series Pile, T=512, instance norm, 64 patches of 8, ICML 2024
- [Chronos](https://arxiv.org/abs/2403.07815) -- Tokenization via scaling+quantization into 4096 bins, Amazon 2024
- [MOIRAI/LOTSA](https://arxiv.org/abs/2402.02592) -- 27.6B observations, 5 patch sizes for multi-frequency, any-variate attention, Salesforce 2024
- [TimesFM](https://github.com/google-research/timesfm) -- 100B+ time points, patch-as-token, decoder-only, Google 2024
- [Chronos-2](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting) -- Univariate to universal forecasting, Oct 2025

### Bearing Dataset Methodology
- [BearLLM paper](https://arxiv.org/abs/2408.11281) -- DCN normalization, 9-dataset unification, AAAI 2025
- [Braga et al., 2022](https://doi.org/10.1016/j.ymssp.2022.109095) -- CWRU data leakage analysis, accuracy drops from 99% to 36% under controlled domain shift
- [awesome-bearing-dataset](https://github.com/VictorBauler/awesome-bearing-dataset) -- Comprehensive collection of bearing fault datasets
- [HUST bearing dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC10327369/) -- 99 raw vibration signals, 6 defect types, 5 bearing types

### Vibration Signal Processing
- Standard CWRU segmentation: 1024 samples with stride 256 at 12 kHz
- XJTU-SY native: 32,768 samples at 25.6 kHz (1.28 seconds)
- Bearing resonance bands typically 1-5 kHz; 12.8 kHz sampling captures up to 6.4 kHz (Nyquist)
