# Overnight Autoresearch Prompt: Mechanical-JEPA v2

## Fix Predictor Collapse & Achieve Real Cross-Dataset Transfer

Use this prompt with the ml-researcher agent for overnight autonomous research.

---

## Prompt

```
Run autoresearch overnight on the Mechanical-JEPA bearing fault detection project.

## Context

You are continuing research on JEPA (Joint Embedding Predictive Architecture) for
industrial bearing fault detection. This is analogous to Brain-JEPA (NeurIPS 2024
Spotlight) but applied to vibration signals instead of fMRI.

**Working directory:** `/home/sagemaker-user/IndustrialJEPA/mechanical-jepa`

### Where We Stand

**The good:**
- CWRU linear probe: 80.4% +/- 2.6% (3 seeds), MLP probe: 96.1%
- Cross-dataset CWRU->IMS transfer is statistically significant (p=0.00003)
- Best config: embed_dim=512, 100 epochs, mask_ratio=0.5, mean-pool

**The problem:**
- The PREDICTOR HAS COLLAPSED. Diagnostic scripts (`quick_diagnose.py`,
  `diagnose_jepa.py`) show the predictor outputs nearly identical embeddings
  regardless of which position it's asked to predict. It ignores positional
  information entirely.
- This means the pretraining objective is partially broken: the model minimizes
  loss by predicting an average patch embedding, not position-specific content.
- The encoder still learns useful features (because it must provide good context),
  but we're leaving massive performance on the table.
- Cross-dataset transfer to IMS is only +2-4%, and FFT baseline hits 100% on
  binary IMS tasks.

**Root cause hypothesis:**
The predictor (2-layer transformer, predictor_dim=embed_dim//2) with learnable
positional embeddings is too weak or too homogeneous to differentiate positions.
The mask tokens all start identical, get similar positional offsets (small learned
pos embeddings), pass through only 2 transformer layers, and collapse to similar
outputs. The loss still decreases because predicting the mean is a valid (but lazy)
minimum.

### Architecture Reference

The predictor in `src/models/jepa.py` (JEPAPredictor, line ~208):
- input_proj: Linear(embed_dim -> predictor_dim)  [predictor_dim = embed_dim//2]
- mask_token: single learnable token, expanded for all mask positions
- pos_embed: learnable (1, n_patches, predictor_dim), trunc_normal_(std=0.02)
- 2 transformer blocks (predictor_dim, 4 heads, mlp_ratio=4, dropout=0.1)
- output_proj: Linear(predictor_dim -> embed_dim)

The encoder (JEPAEncoder, line ~110):
- PatchEmbed1D: Linear(n_channels * patch_size -> embed_dim)
- Learnable pos_embed + CLS token
- 4 transformer blocks

Loss: MSE on L2-normalized predictions vs targets (cosine-normalized MSE).

## FILES TO READ FIRST

Read ALL of these before running any experiment:

1. `jepa-lit-review/jepa_sota_review.md` -- **CRITICAL**: Recent JEPA SOTA review
   covering C-JEPA (Feb 2026) and ThinkJEPA (Mar 2026). Key insights:
   - C-JEPA: object-level masking as latent interventions — maps directly to
     component-level masking for machinery. "Structured masking > random masking."
   - ThinkJEPA: dual-temporal pathways (fast for vibration, slow for trends).
   - Both: frozen encoders + learned predictors works; explicit factorization
     of information sources beats flat architectures.
   - Specific Mechanical-JEPA recommendations on lines 219-237.
2. `autoresearch/LITERATURE_REVIEW.md` -- Broader JEPA SOTA review with table of
   ALL JEPA variants (I-JEPA, V-JEPA, TS-JEPA, MTS-JEPA, Brain-JEPA, C-JEPA)
   and their collapse prevention mechanisms. Section "Direction 3" has root cause
   analysis of what our implementation gets wrong + specific fix recommendations
   (codebook bottleneck, multi-resolution, high masking, L1 loss, VICReg).
   Between this file and #1, the literature review is already halfway done.
3. `autoresearch/READING_LIST.md` -- Curated paper list with links
4. `autoresearch/mechanical_jepa/EXPERIMENT_LOG.md` -- all prior results (15+ experiments)
5. `autoresearch/mechanical_jepa/LESSONS_LEARNED.md` -- critical insights
6. `autoresearch/mechanical_jepa/program.md` -- original research plan
7. `mechanical-jepa/src/models/jepa.py` -- FULL model code, understand every line
8. `mechanical-jepa/src/models/jepa_enhanced.py` -- enhanced masking strategies
9. `mechanical-jepa/train.py` -- training loop, loss, optimizer, LR schedule
10. `mechanical-jepa/src/data/bearing_dataset.py` -- data pipeline
11. `mechanical-jepa/ims_transfer.py` -- cross-dataset transfer evaluation
12. `mechanical-jepa/quick_diagnose.py` -- the diagnostic that found the collapse
13. `mechanical-jepa/diagnose_jepa.py` -- full diagnostic with real data

### DATASET SCALE AWARENESS

When choosing architectures and hyperparameters from the literature, ALWAYS consider
that our datasets are TINY compared to what these papers use:

| Dataset | Samples | Windows | Bearings |
|---------|---------|---------|----------|
| CWRU | ~500K samples | ~2,400 windows | 40 bearings |
| IMS | ~60M samples | ~3,000 windows | 4 bearings |
| Paderborn | ~15M samples | ~600 windows (3 bearings) | 3-33 bearings |

Compare this to papers:
- I-JEPA: ImageNet (1.3M images)
- V-JEPA: VideoMix2M (2M videos)
- Brain-JEPA: 10K+ fMRI subjects
- MTS-JEPA: Large multivariate time series datasets
- wav2vec 2.0: 960h of speech (~350M samples)

**Implications for architecture choices:**
- We CANNOT use massive models. 4-layer encoder, 2-4 layer predictor is right-sized.
- Regularization matters more (dropout, weight decay, early stopping)
- High mask ratios (70-90%) that work for V-JEPA may overtax a small dataset —
  the model doesn't see enough data per epoch. Our 50% may be closer to optimal.
- Codebook bottleneck (MTS-JEPA) is interesting but the codebook needs enough
  data to learn meaningful discrete states. Start small (e.g., 64 codes).
- Pretrained encoders (Round 6) must be SMALL models — a 300M parameter wav2vec
  would overfit instantly on 2,400 windows. Look for base/tiny variants.
- L1 loss (from TS-JEPA) is a cheap experiment worth trying early.

## YOUR MISSION

Fix the predictor collapse, prove it's fixed with diagnostics, and demonstrate
materially improved cross-dataset transfer. Follow the cycle:

    Literature Review -> Sharp Experiments -> Validate & Analyze -> Repeat

### =======================================================================
### ROUND 1: UNDERSTAND THE PROBLEM (Literature + Diagnostics)
### =======================================================================

#### 1A. Deep Literature Review

START by reading `jepa-lit-review/jepa_sota_review.md` AND
`autoresearch/LITERATURE_REVIEW.md` — together they cover C-JEPA, ThinkJEPA,
I-JEPA, V-JEPA, TS-JEPA, MTS-JEPA, Brain-JEPA with collapse prevention
mechanisms and specific fix recommendations. Use them as your foundation and
UPDATE with new findings. Then search the web for:

1. **I-JEPA (Assran et al., CVPR 2023)** -- the original image JEPA
   - How does their predictor work? What depth, what positional encoding?
   - Do they report predictor collapse? What prevents it?
   - Key: they use a NARROW predictor (small dim) but SUFFICIENT depth

2. **Brain-JEPA (NeurIPS 2024 Spotlight)**
   - Their predictor architecture for time series
   - Spatiotemporal masking strategies (Cross-ROI, Cross-Time, Double-Cross)
   - How they handle positional encoding (Brain Gradient + sinusoidal)

3. **V-JEPA (Bardes et al., 2024)** -- video JEPA
   - Predictor design for spatiotemporal data
   - Any collapse prevention mechanisms

4. **"Predictor collapse" in JEPA literature**
   - Search for: "JEPA predictor collapse", "mask prediction collapse",
     "representation collapse self-supervised"
   - VICReg, Barlow Twins, DINO -- how do they prevent collapse?
   - What specific mechanisms (variance regularization, contrastive loss,
     asymmetric architectures) are known to work?

5. **JEPA for time series / vibration / industrial**
   - Any existing work applying JEPA to 1D time series?
   - TS2Vec, TNC, TS-TCC -- how do other time series SSL methods work?
   - What do they do differently from masked prediction?

6. **Sinusoidal vs learnable positional encodings**
   - When does each work better?
   - Frequency-based positional encoding for time series
   - Rotary positional embeddings (RoPE) -- applicable here?

Document key findings. Extract specific architectural choices and hyperparameters
from papers. Note what I-JEPA's predictor depth/dim/pos-encoding actually is.

#### 1B. Diagnostic Baseline

Before changing anything, establish the collapse baseline:

```bash
cd /home/sagemaker-user/IndustrialJEPA/mechanical-jepa

# Find best existing checkpoint
ls -la checkpoints/

# Run quick diagnostic on best checkpoint
python quick_diagnose.py --checkpoint checkpoints/<best_checkpoint>.pt

# Run full diagnostic if data is available
python diagnose_jepa.py --checkpoint checkpoints/<best_checkpoint>.pt
```

Record these numbers precisely. They are the "before" measurements:
- Predictor position variance (currently near 0)
- Prediction spread ratio (pred_std / target_std)
- Per-position cosine similarity pattern
- Encoder diversity metrics (should be healthy)

### =======================================================================
### ROUND 2: FIX THE PREDICTOR (Sharp PoC Experiments)
### =======================================================================

The goal is SHORT, TARGETED experiments. Train for 30 epochs max initially.
Always use `--seed 42` for comparability. ALWAYS use wandb (do NOT pass --no-wandb).

#### 2A. Modify the Predictor Architecture

Edit `src/models/jepa.py` to support these variations. Add CLI flags to `train.py`
so experiments are easy to run without editing code each time.

**Experiment 2A-1: Sinusoidal positional encoding in predictor**
- Replace learnable pos_embed with fixed sinusoidal encoding
- Hypothesis: Learnable pos embeddings collapse to similar values during training.
  Sinusoidal encoding provides guaranteed position discrimination.
- Implementation: Standard sine/cosine positional encoding from "Attention is All
  You Need", scaled to predictor_dim.

**Experiment 2A-2: Increase predictor depth (2 -> 4 layers)**
- Hypothesis: 2 layers is insufficient for the predictor to learn position-dependent
  transformations. More depth gives it capacity to differentiate positions.
- Keep predictor_dim the same (embed_dim//2).

**Experiment 2A-3: Separate mask tokens per position**
- Instead of one shared mask_token expanded to all positions, use N separate
  learnable tokens (one per patch position).
- Hypothesis: Shared mask token + weak pos encoding = collapse. Distinct tokens
  per position guarantee initial diversity.

**Experiment 2A-4: Variance regularization on predictions**
- Add a variance term to the loss: penalize low variance across predicted positions.
- Implementation: `var_loss = max(0, threshold - predictions.var(dim=1).mean())`
- Total loss = MSE_loss + lambda * var_loss
- Hypothesis: Explicitly prevents the collapse shortcut.

**Experiment 2A-5: L1 loss instead of MSE (from TS-JEPA)**
- Replace F.mse_loss with F.l1_loss on normalized predictions vs targets
- Cheapest possible experiment — one line change
- Hypothesis: L1 is more robust to outliers and may reduce collapse tendency
  (MSE heavily penalizes large errors, incentivizing "safe" mean predictions)

**Experiment 2A-6: Codebook bottleneck (from MTS-JEPA)**
- Add a vector quantization layer between predictor output and loss
- Predictions get snapped to nearest codebook entry before loss computation
- Small codebook: 64-128 entries of dimension embed_dim
- Hypothesis: Forces discrete, diverse predictions — can't collapse to a
  single continuous mean. This is what MTS-JEPA uses for collapse prevention.
- NOTE: Keep codebook small given our dataset size.

**Experiment 2A-7: Channel-level masking (inspired by C-JEPA)**
- Instead of masking random patches across all channels, mask entire CHANNELS
  for certain time positions (or mask all time for one channel).
- From C-JEPA: "structured masking > random masking" and "object-level masking
  forces learning of cross-object dependencies."
- In our case: mask channel 0 (drive-end accel), predict from channels 1-2.
  Forces the model to learn cross-channel (cross-sensor) relationships.
- NOTE: Only works if n_channels > 1 in the patch embedding. May need to
  modify PatchEmbed1D to keep channels separate before masking.

**Experiment 2A-8: Combined fix (best individual fixes from 2A-1 through 2A-7)**
- Combine the most promising individual fixes.

For each experiment:
1. Train 30 epochs, seed 42, embed_dim=512, wandb enabled
2. Run quick_diagnose.py on the checkpoint IMMEDIATELY after training
3. Check: Did predictor position variance increase? Did spread ratio improve?
4. Record linear probe accuracy AND diagnostic metrics
5. Log to EXPERIMENT_LOG.md

#### 2B. Quick Decision Gate

After 2A experiments (should take ~1-2 hours total):
- Which fix(es) improved predictor position variance the most?
- Which fix(es) improved linear probe accuracy?
- Did any fix make things worse?
- Select the best 1-2 approaches for deeper investigation.

### =======================================================================
### ROUND 3: VALIDATE THE FIX (Depth + Multi-Seed)
### =======================================================================

#### 3A. Scale Up the Best Fix

Take the winning approach from Round 2 and:

1. **Train 100 epochs** (the known optimal duration for CWRU)
2. **3-seed validation** (seeds 42, 123, 456)
3. **Run full diagnostics** on each checkpoint
4. **Compare to old best** (80.4% +/- 2.6% linear, 96.1% MLP)

```bash
# Example (adjust flags based on Round 2 winners)
python train.py --epochs 100 --seed 42 --embed-dim 512 --predictor-pos sinusoidal
python train.py --epochs 100 --seed 123 --embed-dim 512 --predictor-pos sinusoidal
python train.py --epochs 100 --seed 456 --embed-dim 512 --predictor-pos sinusoidal
```

#### 3B. Critical Analysis

For each trained model, answer:
- Is the predictor collapse actually fixed? (Check diagnostics, not just accuracy)
- Does linear probe accuracy improve, or only MLP probe?
- If linear probe improved: the features are better organized (more linearly separable)
- If only MLP improved: features are richer but still tangled
- Does the gap between linear and MLP probe shrink? (It should, if collapse is fixed)
- What does the per-class breakdown look like? Does outer_race (the hardest class) improve?

#### 3C. Ablation: What Actually Mattered?

If the combined fix won, ablate:
- Remove sinusoidal pos encoding, keep rest -> how much does accuracy drop?
- Remove variance reg, keep rest -> how much drops?
- Remove depth increase, keep rest -> how much drops?

This tells us the MECHANISM, not just the result.

### =======================================================================
### ROUND 4: CROSS-DATASET TRANSFER (The Real Test)
### =======================================================================

This is where the rubber meets the road. Prior transfer was +2-4% on IMS.
With a fixed predictor, can we do materially better?

#### 4A. CWRU -> IMS Transfer

Using the best fixed-predictor checkpoint:

```bash
python ims_transfer.py --checkpoint checkpoints/<best_fixed>.pt --seeds 42,123,456
```

Test on:
- IMS binary (healthy vs failure) -- Test 1 and Test 2
- IMS 3-class (healthy / degrading / failure)
- Compare: old JEPA (collapsed predictor) vs new JEPA (fixed) vs random init
- Compare: JEPA vs FFT baseline (the FFT baseline gets 100% -- can we close the gap?)

#### 4B. IMS Self-Pretrain with Fixed Predictor

Also pretrain on IMS itself with the fixed predictor architecture:

```bash
python ims_pretrain.py --epochs 50 --seed 42 --predictor-pos sinusoidal  # adjust flags
```

This gives the new upper bound. Compare:
- Old upper bound (collapsed predictor, IMS self-pretrain): 73.2% +/- 1.1%
- New upper bound (fixed predictor, IMS self-pretrain): ???
- New cross-dataset (fixed predictor, CWRU pretrain): ???

#### 4C. Few-Shot Transfer

If transfer improves, test data efficiency:
- How many IMS labeled samples does JEPA need to match random init with full data?
- Test N = 20, 50, 100, 200 labeled samples
- This is the compelling story: "JEPA needs 50 labeled samples to match 200 from scratch"

### =======================================================================
### ROUND 5: SPECTRAL / FFT INPUT EXPERIMENTS
### =======================================================================

IMPORTANT FRAMING: The goal is a GENERAL-PURPOSE vibration/machinery encoder,
NOT a bearing-specific model. Everything here should transfer to gearboxes,
turbines, pumps, compressors — any rotating machinery. Do not overfit on
bearing-specific tricks. The FFT is interesting because vibration fault
signatures live in the frequency domain across ALL rotating machinery.

#### 5A. Literature Review: FFT/Spectral Inputs for Self-Supervised Learning

Deep web search for:

1. **JEPA / masked prediction on frequency-domain inputs**
   - Has anyone tried JEPA on FFT/STFT/spectrogram inputs instead of raw time series?
   - How does AudioMAE (Huang et al., 2022) handle spectrograms? (Masked autoencoder
     on mel spectrograms — directly relevant architecture)
   - How does SSAST (Gong et al., 2022) do self-supervised audio spectrogram
     transformers? What pretraining objective do they use?

2. **Frequency-domain representations for machinery diagnostics**
   - Search: "spectrogram CNN bearing fault", "STFT transformer vibration"
   - What spectral features matter? (Envelope spectrum, order tracking, BPFO/BPFI)
   - Are these features general across machinery types, or bearing-specific?

3. **Data leakage considerations with FFT in time series**
   - CRITICAL: If we're predicting future patches, we CANNOT use FFT computed
     over the future. The FFT must be computed ONLY on past/current data.
   - Search: "causal spectrogram", "online STFT", "sliding window FFT"
   - How do audio models handle this? (They typically process fixed-length clips,
     no causality issue — but we're doing masked prediction on a window)

4. **Dual-domain / hybrid representations**
   - Search: "time-frequency representation learning", "dual domain transformer"
   - Papers that combine raw waveform + spectral features
   - Does one domain complement the other?

5. **Pretrained audio/vibration models**
   - Are there pretrained models on industrial vibration data?
   - Can audio pretraining (AudioSet, speech) transfer to vibration?
   - Search: "pretrained vibration model", "audio foundation model transfer
     industrial", "wav2vec vibration"

KEY QUESTION TO ANSWER: Should the encoder see raw time series, FFT, STFT,
or some combination? What does the literature say about which representation
makes self-supervised pretraining learn the most transferable features?

#### 5B. Design Principles (NO Leakage)

The JEPA setup: we have a window of signal (e.g., 4096 samples). We split it
into patches (e.g., 16 patches of 256 samples). We mask some patches and predict
their embeddings from the visible ones.

**Where FFT can and cannot be used:**

SAFE (no leakage — FFT computed on the ENTIRE window, which is the model's input):
- The model sees the whole 4096-sample window at once. Computing FFT/STFT on this
  window is fine — it's the model's input, not future data. The JEPA objective
  predicts masked EMBEDDINGS, not raw future signal values.
- Think of it this way: the encoder transforms the input into patch embeddings.
  Whether the input is raw time series or FFT is just a preprocessing choice.
  The masking happens at the embedding level, not the input level.

SAFE approaches:
1. **FFT of each patch independently**: Each 256-sample patch -> 128-bin FFT.
   Feed magnitude spectrum as input instead of raw waveform. No leakage because
   each patch's FFT uses only that patch's data.
2. **STFT over the window**: Sliding FFT with hop size = patch size. Each "patch"
   is now a spectral frame. Same masking logic applies.
3. **Dual-stream**: One encoder branch for raw time series, one for spectral.
   Fuse or use one to predict the other.

UNSAFE (leakage):
- Using FFT of the FULL window to predict individual patch content, if the FFT
  implicitly contains information about masked patches. BUT: in JEPA, the target
  encoder sees all patches anyway (it's not autoregressive). The asymmetry is
  context encoder (sees visible) vs target encoder (sees all). So the concern is
  about the CONTEXT encoder's input, not the target.
- Actually, even for context: the context encoder only processes VISIBLE patches.
  If each visible patch independently has its own FFT, there's no leakage.

BOTTOM LINE: Per-patch FFT or per-patch STFT is clean. No leakage concerns.
The masked patches are masked at the embedding level — their spectral content
is never shown to the context encoder.

#### 5C. Spectral Input Experiments

**Experiment 5C-1: Per-patch FFT magnitude as input**
- Replace raw time-domain patches with FFT magnitude spectrum
- Each patch: 256 samples -> FFT -> 128 magnitude bins (discard phase or keep)
- Modify PatchEmbed1D to accept spectral input: Linear(n_channels * 128, embed_dim)
- Or: keep patch_size=256 but the "signal" is now the FFT magnitude
- Train 30 epochs, seed 42, compare to raw time-domain baseline
- Hypothesis: Spectral representation makes fault signatures more salient and
  easier for the transformer to learn. Faults show up as specific frequency peaks.

**Experiment 5C-2: STFT spectrogram patches**
- Compute STFT over the window (e.g., 256-point FFT, hop=256, giving 16 frames)
- Each patch is now a spectral frame: (n_channels, n_freq_bins)
- The transformer processes a sequence of spectral frames
- This is essentially what AudioMAE does on mel spectrograms
- Hypothesis: STFT preserves temporal structure within the spectral domain,
  giving the model both frequency and time information.

**Experiment 5C-3: Dual-domain input (raw + FFT concatenated)**
- Each patch: concatenate raw 256 samples + 128 FFT magnitude bins = 384 features
- Or: two-stream encoder with cross-attention
- Hypothesis: Time domain captures transients (impulses), frequency domain captures
  periodic patterns. Together they're more informative than either alone.

**Experiment 5C-4: Spectral input + best predictor fix from Round 2-3**
- Combine the winning predictor fix with the winning spectral representation
- This is the "full package" experiment
- 3-seed validation if promising

**Experiment 5C-5: FFT baseline sanity check**
- Simple FFT + linear classifier (no JEPA, no transformer)
- This is the floor. If JEPA with spectral input can't beat this, something's wrong.
- Also: FFT + random forest / SVM as comparison
- This tells us how much the REPRESENTATION LEARNING adds beyond raw spectral features

For each experiment: wandb logging, diagnostics, log to EXPERIMENT_LOG.md.

#### 5D. Critical Analysis of Spectral Experiments

After running 5C:
- Does spectral input improve WITHIN-dataset (CWRU) accuracy?
- Does spectral input improve CROSS-dataset transfer (CWRU->IMS)?
- Does the linear-vs-MLP probe gap shrink? (Spectral features might be more
  linearly separable)
- Is the improvement GENERAL (all fault types) or specific (e.g., only outer race)?
- Does the predictor collapse problem change with spectral input?
- MOST IMPORTANT: Does spectral JEPA beat spectral-only baseline (FFT + classifier)?
  If not, the self-supervised learning isn't adding value on top of spectral features.

### =======================================================================
### ROUND 6: PRETRAINED ENCODERS & TRANSFER FROM RELATED DOMAINS
### =======================================================================

IMPORTANT: We want to stand on the shoulders of giants. If there are well-cited,
open-source pretrained models that could serve as our encoder backbone, we should
try them. This is NOT about bearing-specific models — it's about general-purpose
time series or vibration encoders that have been validated in the literature.

#### 6A. Deep Literature Review: Pretrained Models for Vibration/Time Series

Search thoroughly for:

1. **Foundation models for time series (general)**
   - **TimesFM** (Google, 2024) -- pretrained on 100B time points
   - **Chronos** (Amazon, 2024) -- T5-based time series foundation model
   - **MOMENT** (CMU, 2024) -- "A Family of Open Time-Series Foundation Models"
   - **Timer** (2024) -- generative pretrained transformer for time series
   - Which of these have open weights on HuggingFace?
   - Which have been tested on industrial/vibration data?
   - What's their encoder architecture? Can we extract embeddings?

2. **Audio/vibration pretrained models**
   - **wav2vec 2.0** (Meta) -- self-supervised speech, but audio = vibration at
     different frequencies. Has anyone used it for machinery diagnostics?
   - **HuBERT** (Meta) -- similar to wav2vec, discrete token prediction
   - **AudioMAE** -- masked autoencoder for audio spectrograms
   - **BEATs** (Microsoft) -- audio pretraining with acoustic tokenizers
   - Search: "wav2vec vibration", "audio pretrained model industrial",
     "speech model transfer mechanical"

3. **Vibration-specific pretrained models**
   - Search: "pretrained model vibration analysis", "foundation model predictive
     maintenance", "self-supervised vibration feature extraction"
   - Check HuggingFace for any vibration/industrial models
   - Check GitHub for well-cited repos (>100 stars) with pretrained weights

4. **Transfer learning from audio to vibration**
   - Search: "audio to vibration transfer learning", "cross-domain audio industrial"
   - Key question: Audio models are pretrained at 16kHz speech / 44kHz music.
     Vibration data is 12-64kHz. Is the frequency range compatible?
   - Does resampling to match the pretrained model's expected rate help?

FOR EACH CANDIDATE, record:
- Paper citation count / venue (must be reputable: NeurIPS, ICML, ICLR, etc.)
- HuggingFace model ID (if available)
- Encoder architecture (transformer? CNN? hybrid?)
- Input format expected (raw waveform? spectrogram? patches?)
- Embedding dimensionality
- Whether fine-tuning or feature extraction is recommended
- Any reported results on industrial/vibration tasks

#### 6B. Selection Criteria

Only use pretrained models that meet ALL of:
1. Published at a top venue OR >200 citations
2. Open-source weights available (HuggingFace or GitHub)
3. Encoder produces fixed-size embeddings we can probe
4. Input format is compatible (1D waveform or spectrogram)
5. Not trained on data that overlaps with CWRU/IMS/Paderborn (no leakage)

#### 6C. Pretrained Encoder Experiments

**Experiment 6C-1: Use pretrained encoder as JEPA backbone**
- Replace our custom JEPAEncoder with a pretrained encoder (e.g., wav2vec 2.0 base)
- Keep the JEPA predictor and training objective
- Pretrain on CWRU with the JEPA objective (fine-tune encoder + train predictor)
- Hypothesis: A pretrained encoder starts with better representations, so JEPA
  pretraining converges faster and learns richer features.

**Experiment 6C-2: Feature extraction (no JEPA, just probe)**
- Use pretrained model as frozen feature extractor
- Extract embeddings on CWRU, train linear probe
- This is the "ceiling" for what the pretrained model already knows
- Compare: pretrained frozen vs our JEPA-trained encoder vs random init

**Experiment 6C-3: Pretrained encoder + JEPA fine-tuning + spectral input**
- If both spectral input (Round 5) and pretrained encoder help individually,
  combine them
- This is the "kitchen sink" experiment — only if individual components work

**Experiment 6C-4: Cross-dataset transfer with pretrained backbone**
- Take best pretrained setup, evaluate on IMS and Paderborn
- Compare to our custom JEPA encoder transfer
- Does the pretrained backbone help with domain gap?

#### 6D. Critical Analysis

- Is the pretrained encoder actually better, or just bigger? Control for model size.
- If pretrained encoder wins: is it because of pretraining or architecture?
  (Test: same architecture, random init, same JEPA training)
- If pretrained encoder loses: the domain gap (speech/audio vs vibration) may be
  too large. Document this — it's a useful negative result.
- How does compute cost compare? A huge pretrained model that's 5% better but
  10x slower is not practical for deployment.

### =======================================================================
### ROUND 7: SECOND LITERATURE PASS + REMAINING ADVANCED EXPERIMENTS
### =======================================================================

Based on Round 3-6 results, do a targeted literature review on what worked:

#### 7A. Conditional Experiments

**If spectral input helped (Round 5):**
- Try mel-spectrogram instead of raw FFT (compresses frequency axis, focuses on
  perceptually relevant bands — but "perceptually relevant" for machines, not humans)
- Try learnable filterbanks (SincNet-style) instead of fixed FFT
- Try different FFT sizes (128, 256, 512) and hop sizes

**If pretrained encoder helped (Round 6):**
- Try different pretrained models (wav2vec vs HuBERT vs MOMENT)
- Try different layers of the pretrained model (early vs late features)
- Try adapter-based fine-tuning (LoRA) instead of full fine-tuning

**If transfer is still modest:**
- Resampling: Normalize all datasets to same sampling rate before anything
- Contrastive + predictive hybrid loss (InfoNCE + MSE)
- Adversarial domain adaptation during fine-tuning

#### 7B. Unconditional Experiments (always try these)

- **Temporal block masking** (from jepa_enhanced.py) with best config
- **Cross-channel masking**: mask entire channels, predict from remaining
- Best config so far + 200 epochs (check if longer training helps with fixed predictor)

### =======================================================================
### ROUND 8: PADERBORN DATASET INTEGRATION & TRANSFER
### =======================================================================

Paderborn is the HARDEST transfer test: 64kHz sampling (vs 12kHz CWRU), 8 channels
(vibration + motor current + temperature + torque), completely different test rig.
Published cross-domain CWRU->Paderborn accuracy is 88-93% (supervised CNNs with
domain adaptation). If JEPA transfer works here, it's a real result.

#### 8A. Integrate Paderborn Data Loading

**Current state:**
- 3 sample bearings downloaded: K001 (healthy), KA01 (outer race), KI01 (inner race)
- Located at: `/home/sagemaker-user/IndustrialJEPA/datasets/data/paderborn/`
- Each has 20 MAT files (~4s recordings at 64kHz, 8 channels)
- `prepare_bearing_dataset.py` has `process_paderborn()` — but it looks for data
  in `mechanical-jepa/data/bearings/raw/paderborn/`, not `datasets/data/paderborn/`
- `bearing_dataset.py` has NO `load_paderborn_signal()` — returns None for 'paderborn'
- `bearing_episodes.parquet` has NO Paderborn entries

**You must implement these steps IN ORDER:**

**Step 1: Symlink or copy Paderborn data to expected location**
```bash
# Link the already-downloaded data to where prepare_bearing_dataset.py expects it
ln -sf /home/sagemaker-user/IndustrialJEPA/datasets/data/paderborn/* \
  /home/sagemaker-user/IndustrialJEPA/mechanical-jepa/data/bearings/raw/paderborn/
# Create the raw/paderborn dir first if needed
mkdir -p /home/sagemaker-user/IndustrialJEPA/mechanical-jepa/data/bearings/raw/paderborn
```

**Step 2: Run process step to add Paderborn to parquet**
```bash
cd /home/sagemaker-user/IndustrialJEPA/mechanical-jepa
python data/bearings/prepare_bearing_dataset.py --process
```
Verify: `bearing_episodes.parquet` should now contain paderborn entries.
If the process step doesn't pick up paderborn, debug why (check paths, check
PADERBORN_BEARINGS dict in prepare_bearing_dataset.py).

**Step 3: Implement `load_paderborn_signal()` in `bearing_dataset.py`**

Add this function following the pattern of `load_cwru_signal()` and `load_ims_signal()`:

```python
def load_paderborn_signal(data_dir: Path, bearing_id: str, measurement_id: str) -> Optional[np.ndarray]:
    """Load Paderborn bearing signal.

    Paderborn MAT files have a struct with .Y attribute containing sensor data.
    8 channels: a1, a2, a3 (vibration), v1 (velocity), temp1, torque, phase_a, phase_b
    Sampling rate: 64kHz (vibration) / 4kHz (motor current)

    Returns: (n_samples, n_channels) array or None
    """
    mat_file = data_dir / 'raw' / 'paderborn' / bearing_id / f'{measurement_id}.mat'
    if not mat_file.exists():
        return None
    try:
        data = loadmat(str(mat_file), squeeze_me=True, struct_as_record=False)
        for key in data.keys():
            if not key.startswith('_'):
                val = data[key]
                if hasattr(val, 'Y'):
                    sensor_data = val.Y
                    if sensor_data.ndim == 2:
                        return sensor_data.astype(np.float32)
        return None
    except Exception:
        return None
```

Then add the paderborn case in `_load_signal()`:
```python
elif dataset == 'paderborn':
    signal = load_paderborn_signal(self.data_dir, bearing_id, measurement_id)
```

**Step 4: Handle sampling rate mismatch**

CRITICAL: Paderborn is 64kHz, CWRU is 12kHz. With patch_size=256:
- CWRU: 256/12000 = 21.3ms per patch
- Paderborn: 256/64000 = 4.0ms per patch (5x shorter!)

Options (try in order of simplicity):
a) **Resample Paderborn to 12kHz** using scipy.signal.resample or decimate.
   Add this in `load_paderborn_signal()` or in `__getitem__()`.
   This is the cleanest approach — same temporal scale per patch.
b) **Use larger patch size for Paderborn** (256 * 64/12 ~ 1365 -> round to 1280).
   Messier, requires architecture changes.
c) **Just use it as-is** and see what happens. The model might learn anyway.

Try option (a) first. If resampling, also only use the 3 vibration channels
(a1, a2, a3) to match CWRU's 3-channel format. Or use all 8 and pad CWRU to 8.
Be pragmatic — start with 3 vibration channels resampled to 12kHz.

**Step 5: Verify with a smoke test**
```bash
python train.py --epochs 2 --no-wandb --dataset-filter paderborn
```
If --dataset-filter doesn't exist, add it or use the data pipeline directly.
The point is: can we load Paderborn data, create patches, and run a forward pass?

#### 8B. Download More Paderborn Bearings (if needed)

3 bearings (1 per class) is marginal for train/test splits. Consider downloading
more — the download script supports it:
```bash
python data/bearings/prepare_bearing_dataset.py --download --dataset paderborn
```
This downloads all 33 bearings (~5.4GB). Only do this if you have time and disk space.
With 3 bearings you can still do leave-one-bearing-out evaluation.

#### 8C. Paderborn Transfer Experiments

Once data loading works, run the transfer experiments:

**Experiment 8C-1: CWRU -> Paderborn transfer (3-class: healthy/OR/IR)**
- Load best CWRU-pretrained checkpoint (with fixed predictor from Round 3)
- Linear probe on Paderborn (train on 2 bearings, test on 1, rotate)
- Compare: JEPA-pretrained vs random init
- 3 seeds minimum

**Experiment 8C-2: Paderborn self-pretrain (upper bound)**
- Pretrain JEPA on Paderborn data (no labels)
- Linear probe on same Paderborn task
- This shows how much room there is

**Experiment 8C-3: CWRU+IMS -> Paderborn (multi-source transfer)**
- Pretrain on CWRU AND IMS combined
- Transfer to Paderborn
- Does more pretraining data help?

**Self-check for Paderborn experiments:**
- [ ] Did resampling work correctly? Verify signal shapes and frequencies.
- [ ] Is the train/test split by bearing (not by window)?
- [ ] Are we comparing JEPA vs random init fairly (same architecture, same eval)?
- [ ] With only 3 bearings and 20 files each, is the sample size adequate?
- [ ] Does the FFT baseline beat everything here too? (Check it.)
- [ ] Are results on wandb?

### =======================================================================
### ROUND 9: FINAL ANALYSIS & DOCUMENTATION
### =======================================================================

#### 9A. Update Jupyter Notebook

Update `notebooks/03_results_analysis.ipynb` with:

1. **New section: "Predictor Collapse & Fix"**
   - Before/after diagnostic visualizations
   - PCA of predictions vs targets (collapsed vs fixed)
   - Per-position cosine similarity (collapsed vs fixed)

2. **Updated results table**
   - All configurations: old JEPA, fixed JEPA, random init, FFT baseline
   - CWRU and IMS results side by side
   - 3-seed mean +/- std for all

3. **Spectral input results section**
   - Raw vs FFT vs STFT vs dual-domain comparison table
   - Spectral JEPA vs FFT-only baseline (the key question)
   - Visualization: what do spectral patch embeddings look like?

4. **Pretrained encoder results section**
   - Which pretrained models were tested, with citations
   - Comparison: custom JEPA vs pretrained backbone vs frozen features
   - Architecture diagrams if helpful

5. **t-SNE visualizations**
   - CWRU embeddings: old vs fixed JEPA vs spectral JEPA
   - IMS embeddings: CWRU-pretrained (fixed) vs random init
   - Paderborn embeddings (if transfer worked)
   - Color by fault type / degradation stage

6. **Confusion matrices**
   - Best model on CWRU (4-class)
   - Best model on IMS (binary + 3-class)
   - Paderborn results (if available)

7. **Cross-dataset transfer summary table**
   - All source->target combinations (CWRU->IMS, CWRU->Paderborn, multi-source)
   - All encoder variants (custom JEPA, spectral JEPA, pretrained backbone)
   - JEPA vs random init vs FFT baseline for each

8. **The story**: Clear narrative:
   problem (predictor collapse) -> fix -> spectral input -> pretrained encoders
   -> cross-dataset transfer -> what works, what doesn't, and why

#### 9B. Update Documentation

- EXPERIMENT_LOG.md: All new experiments with full details
- LESSONS_LEARNED.md: New insights about predictor collapse and fix
- README.md: Update with new best results if improved

#### 9C. Final Commit

Commit all changes with descriptive message summarizing findings.

### =======================================================================
### ROUND 10: HUGGINGFACE MECHANICAL-COMPONENTS DATASET (BONUS — AFTER 3+ HOURS)
### =======================================================================

**IMPORTANT CONTEXT:** A separate agent is building a new HuggingFace dataset
in parallel tonight: `Forgis/Mechanical-Components`. This dataset will contain
bearing data (potentially from multiple sources for cross-dataset transfer) AND
other mechanical components (gearboxes, pumps, etc.) — exactly what we need to
test general-purpose transferability.

**TREAT WITH EXTREME CAUTION.** This dataset is being constructed concurrently.
It may be incomplete, have formatting issues, or change during the night.

#### 10A. Check Dataset Status (only after Rounds 1-9 are done, or after 3+ hours)

```python
import os
os.environ['HF_TOKEN'] = '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'  # from .env

from huggingface_hub import HfApi
api = HfApi()

# Check if dataset exists and what's in it
try:
    info = api.dataset_info("Forgis/Mechanical-Components")
    print(f"Dataset exists: {info.id}")
    print(f"Last modified: {info.lastModified}")
    print(f"Files: {[f.rfilename for f in info.siblings]}")
except Exception as e:
    print(f"Dataset not ready yet: {e}")
```

Also try:
```python
from datasets import load_dataset
ds = load_dataset("Forgis/Mechanical-Components", token="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")
print(ds)
# Check: what columns? what splits? how many rows? what component types?
```

#### 10B. Sanity Check Before Using

If the dataset loads, perform thorough sanity checks:

1. **Schema check**: What columns exist? (signal data, labels, metadata?)
2. **Data quality**: Are signals the right shape? Any NaN/inf values?
3. **Label distribution**: How many samples per component type? Per fault type?
4. **Sampling rates**: What sampling rate(s)? Consistent across entries?
5. **Component types**: What mechanical components are included beyond bearings?
6. **Overlap check**: Is CWRU/IMS/Paderborn data included? If so, which bearings?
   DO NOT use overlapping data for transfer experiments — that's leakage.
7. **Size**: Is it big enough to be useful? (A dataset with 10 samples is not helpful)

If ANY of these checks fail or look suspicious, STOP and document what you found.
Do NOT run experiments on unreliable data.

#### 10C. Experiments (only if sanity checks pass)

**Experiment 10C-1: Cross-component transfer**
- Pretrain JEPA on bearing data from this dataset
- Transfer to non-bearing components (gearboxes, pumps, etc.)
- This is the ultimate generality test — does JEPA learn features that
  transfer across different mechanical component types?
- Compare: JEPA-pretrained vs random init on non-bearing tasks

**Experiment 10C-2: Expanded bearing pretraining**
- If the dataset has bearings from new sources (beyond CWRU/IMS/Paderborn):
  pretrain on ALL bearing data, test on held-out sources
- More pretraining data should improve representations

**Experiment 10C-3: Update the notebook**
- Add HF dataset results to the analysis notebook
- Include cross-component transfer results
- Update the summary tables

#### 10D. Self-Check for HF Dataset

- [ ] Is the dataset actually complete and stable? (Check last modified time)
- [ ] Are there enough samples per class for meaningful experiments?
- [ ] Have I checked for overlap with CWRU/IMS/Paderborn?
- [ ] Are the signal formats compatible with our data pipeline?
- [ ] Am I using the HF token from .env (not hardcoding it in committed code)?
- [ ] Results logged to wandb and EXPERIMENT_LOG.md?

## GLOBAL RULES

### Experiment Discipline
- **ALWAYS use wandb** (never pass --no-wandb). Project: 'mechanical-jepa'
- **Log EVERY experiment** to EXPERIMENT_LOG.md, even failures
- **3+ seeds** for any claim. 1 seed for initial exploration only.
- **Never tune on test set**
- **Run diagnostics after every training run** (quick_diagnose.py at minimum)
- **Short experiments first** (30 epochs), scale up only what works

### General-Purpose Design Philosophy
- The encoder must work for ALL rotating machinery, not just bearings
- Do NOT add bearing-specific features (e.g., BPFO/BPFI frequency formulas)
- Spectral features should be learned, not hand-engineered
- If something helps on CWRU but hurts on IMS/Paderborn, it's overfitting
- The test of generality: would this work on a gearbox? A pump? A turbine?

### Code Changes
- Add CLI flags for new features (don't hardcode experimental changes)
- Keep backward compatibility (old checkpoints should still load)
- Comment WHY, not WHAT

### Commit Protocol
- Commit after each round completes
- Commit message format: "Exp N-M: [brief description of finding]"
- Push after each round

### Self-Criticism Checklist (ask yourself after every experiment)
- [ ] Is this improvement real or could it be noise? (Check error bars)
- [ ] Am I comparing fairly? (Same seeds, same epochs, same eval protocol)
- [ ] Could a simpler baseline explain this? (FFT, random features, etc.)
- [ ] Does the diagnostic actually show the collapse is fixed?
- [ ] Am I fooling myself with the MLP probe? (It can fit anything)
- [ ] Would this result survive peer review?

### Stopping Conditions

Stop and write final summary when:
1. All 10 rounds complete (or time limit reached)
2. You achieve >85% CWRU linear probe (fixed predictor should unlock this)
3. You achieve >5% transfer gain on IMS (double the current +2-4%)
4. Spectral input experiments complete with clear conclusion
5. At least one pretrained encoder tested
6. Paderborn integration complete with at least one transfer experiment
7. HF dataset checked (if available) with at least one experiment
8. You've been running for 12+ hours
9. You hit an irrecoverable error

**Priority order if running out of time:**
Rounds 1-4 (predictor fix + IMS transfer) are MUST-DO.
Round 5 (spectral) is HIGH priority.
Round 6 (pretrained encoders) is MEDIUM priority.
Rounds 7-8 (advanced + Paderborn) are STRETCH.
Round 9 (documentation) is MUST-DO — always leave time for this.
Round 10 (HF dataset) is BONUS — only if time remains after everything else.

### Expected Timeline (rough)
- Round 1 (Literature + Diagnostics): 30-60 min
- Round 2 (PoC predictor fix): 60-90 min
- Round 3 (Validation): 60-90 min
- Round 4 (IMS Transfer): 60-90 min
- Round 5 (Spectral/FFT experiments): 90-120 min
- Round 6 (Pretrained encoders): 60-90 min
- Round 7 (Advanced experiments): 60-90 min
- Round 8 (Paderborn integration + experiments): 90-120 min
- Round 9 (Documentation + notebook): 60-90 min
- Round 10 (HF dataset — bonus): 60-90 min
- Total: 8-14 hours

Good luck. Be rigorous. Be honest. Fix this predictor.
```

---

## Pre-Flight Checklist

Before starting the overnight run, verify:

- [ ] Dataset downloaded: `ls mechanical-jepa/data/bearings/bearing_episodes.parquet`
- [ ] IMS data available: `ls mechanical-jepa/data/bearings/ims/`
- [ ] Paderborn data exists: `ls /home/sagemaker-user/IndustrialJEPA/datasets/data/paderborn/`
- [ ] Smoke test passes: `cd mechanical-jepa && python train.py --epochs 2 --no-wandb`
- [ ] Diagnostics work: `python quick_diagnose.py --checkpoint checkpoints/<latest>.pt`
- [ ] WandB authenticated: `python -c "import wandb; print(wandb.api.api_key[:8])"`
- [ ] Git is clean: `git status`
- [ ] GPU available: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] scipy available: `python -c "from scipy.io import loadmat; print('OK')"`
- [ ] HF token available: `grep HF_TOKEN /home/sagemaker-user/IndustrialJEPA/.env`
- [ ] HF datasets library: `python -c "from datasets import load_dataset; print('OK')"`

## How to Launch

```bash
# In Claude Code, run:
# Use the ml-researcher agent. Run autoresearch overnight on the Mechanical-JEPA
# project. Read autoresearch/mechanical_jepa/OVERNIGHT_PROMPT.md for full instructions.
```

## Expected Outcomes

After a successful overnight run:

1. **Predictor collapse diagnosed and fixed** with before/after diagnostics
2. **CWRU accuracy improved** beyond 80.4% linear probe baseline
3. **IMS transfer improved** beyond +2-4% (target: +5-10%)
4. **Spectral input tested** — FFT/STFT as encoder input, with clear conclusion
   on whether it helps JEPA learn more transferable features
5. **Pretrained encoder tested** — at least one well-cited model (wav2vec, MOMENT,
   etc.) compared against custom JEPA encoder, with dataset-scale-appropriate sizing
6. **Paderborn dataset integrated** — loading, resampling, parquet metadata working
7. **Paderborn transfer tested** — CWRU->Paderborn with best model variant
8. **HF Mechanical-Components dataset** — checked, sanity-validated, and if ready:
   cross-component transfer experiment (bearing -> gearbox/pump)
9. **Ablation results** showing which components matter (predictor fix, spectral
   input, pretrained backbone)
10. **Updated notebook** with comprehensive analysis across all experiments,
    all datasets, all encoder variants
11. **All experiments on wandb** for inspection
