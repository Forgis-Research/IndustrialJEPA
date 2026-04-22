# V18 Plan: E2E Fine-Tuning, Honest Probing, Accelerated Curriculum, MTS-JEPA Comparison

## Motivation

V17 showed that LogUniform k + fixed-window target + curriculum SIGReg achieves 15.38 frozen RMSE (from 17.81 in v2). But:
1. **No E2E fine-tuning was tested** — v11's E2E was 13.80 with the old architecture. V17's improvements should compound with E2E.
2. **The v2 baseline (17.81) used a broken probe protocol** — 15 val points, no weight decay. The improvement over v2 may be inflated or deflated; we need to re-probe v11 honestly.
3. **The curriculum schedule was conservative** — 100/150/200 with a hard EMA cutoff causing a 4x loss spike at ep151. Faster transition + gradual EMA fade should converge sooner.
4. **F1 is only reported for v17** — need F1/AUC-PR across all configurations for unified comparison.
5. **MTS-JEPA comparison is incomplete** — paper claims 62.5% SMAP PA-F1 (v15) but v17 got 0.219. Need rigorous head-to-head.

## Architecture (unchanged from V17)

- Encoder: causal transformer, 2L, d=256, 4 heads, attention pooling → h_past (256-d)
- Predictor: MLP, horizon-conditioned via sinusoidal PE, (h_past, k) → γ(k) (256-d)
- Target: ema_encoder(x_{t+k:t+k+w}), w=10 fixed, k ~ LogU[1, K_max]
- Collapse prevention: EMA 0.99 → curriculum SIGReg on predictor output

## Experiments (in execution order)

### Phase 0: Honest re-probe of v11/v2 checkpoints (~30 min)

**Goal**: determine whether v2's 17.81 frozen RMSE is real or sandbagged by the broken probe.

**Approach**:
1. Check if local checkpoints in `checkpoints/` are v2 pretrained weights (load, check architecture match)
2. If yes: re-probe with the v17 Phase 2 protocol (WD=1e-2, n_cuts_per_engine=10 on val, 3 seeds)
3. If no local v2 ckpts: pretrain v2 from scratch (k ~ U[5,30], 200 epochs, ~30 min) and probe honestly
4. Report both old-protocol and new-protocol RMSE side by side

**Output**: `experiments/v18/phase0_honest_reprobe.json`

**This is critical**: if v2 honest-probe is actually ~15.5, then v17's "improvement" is mostly probe fix, not architecture. If v2 honest-probe is still ~17-18, v17 genuinely helps.

### Phase 1: V17 pretrain + E2E fine-tuning (~2 hours)

**Goal**: combine v17 architectural improvements with E2E fine-tuning to beat v11's 13.80.

**Step 1 — Pretrain** (rerun v17 Phase 1):
- Same config: LogU k ∈ [1, 150], w=10, EMA 0.99, 200 epochs, cosine LR from 3e-4
- 3 seeds (42, 123, 456)
- Save checkpoints at ep50, ep100, ep150, ep200 (for curriculum and ablation)
- **Honest probe every 10 epochs** (WD=1e-2, val n_cuts=10)

**Step 2 — E2E fine-tuning** from best pretrained checkpoint:
- Unfreeze encoder + predictor, add linear head on h_past
- Same protocol as v11: MSE loss, LR=1e-4 (10x lower than pretrain), 50 epochs
- Label efficiency sweep: 100%, 20%, 10%, 5% (5 seeds each)
- Compare to v11 E2E at each budget

**Step 3 — F1 evaluation** for every configuration:
- Binary task: "fails within k cycles?" for k ∈ {10, 20, 30, 50}
- Report F1, precision, recall, AUC-PR at each k
- Also report RMSE for backward compatibility

**Output**: `experiments/v18/phase1_results.json`

**Success criteria**:
- Frozen (honest probe): ≤ 15.5 (confirming v17)
- E2E @ 100%: < 13.80 (beat v11)
- E2E @ 5%: < 21.53 (beat v2 frozen, showing SSL value)

### Phase 2: Accelerated curriculum SIGReg (~1.5 hours)

**Goal**: test whether the curriculum can be compressed and the EMA transition smoothed.

**Three schedules to compare** (all from Phase 1 checkpoints):

| Schedule | EMA phase | SIGReg ramp | EMA fade | SIGReg-only | Total |
|----------|-----------|-------------|----------|-------------|-------|
| A (v17 original) | 0-100 | 100-150 (λ 0→0.05) | hard cutoff at 150 | 150-200 | 200 ep |
| B (compressed) | 0-75 | 75-100 (λ 0→0.05) | hard cutoff at 100 | 100-150 | 150 ep |
| C (gradual fade) | 0-75 | 75-100 (λ 0→0.05) | momentum 0.99→1.0 over 100-120 | 120-150 | 150 ep |

Schedule C is the key novelty: instead of a binary EMA on/off switch, linearly ramp EMA momentum from 0.99 to 1.0 over 20 epochs. At momentum=1.0, the target network is frozen (= stop-grad on stale weights). Then switch to stop-grad on the live encoder.

```python
# Schedule C: gradual EMA fade
if epoch <= 75:
    use_ema = True; momentum = 0.99; lam_sig = 0
elif epoch <= 100:
    use_ema = True; momentum = 0.99
    lam_sig = LAMBDA_SIG_MAX * (epoch - 75) / 25
elif epoch <= 120:
    use_ema = True
    momentum = 0.99 + 0.01 * (epoch - 100) / 20  # 0.99 → 1.0
    lam_sig = LAMBDA_SIG_MAX
else:
    use_ema = False  # stop-grad on live encoder
    lam_sig = LAMBDA_SIG_MAX
```

**SIGReg placement**: predictor only (v17 showed encoder placement destroys RUL signal).

**Evaluation**: honest frozen probe + E2E fine-tune (100% labels only) for each schedule. Track loss curves to measure spike severity at transition.

**Output**: `experiments/v18/phase2_curriculum_results.json`

**Success**: Schedule B or C matches A quality in fewer epochs. C avoids the 4x loss spike.

### Phase 3: Reviewer-Writer loop (~1.5 hours)

**Goal**: simulate a NeurIPS review cycle to identify paper weaknesses before submission.

**Protocol**:
1. Launch `neurips-reviewer` agent (4 independent reviews of current `paper-neurips/paper.tex`)
2. Synthesize reviews: extract top-3 weaknesses across reviewers
3. Launch `paper-writer` agent to address the weaknesses (rewrite sections, add experiments)
4. If reviews identify missing experiments: design and run them (budget: 30 min compute each)
5. Re-review the revised paper (1 more round)

**Expected review concerns** (based on known weaknesses):
- SMAP results are bad — paper claims 62.5% PA-F1 but v17 got 21.9%. Which is in the paper?
- Only FD001 as primary benchmark — need FD003/FD004 numbers
- Comparison to MTS-JEPA is incomplete (different tasks, different metrics)
- Frozen probe vs E2E gap — is the architecture or the probe doing the work?
- CI/seed variance with only 3 seeds

**Output**: `experiments/v18/reviewer_synthesis.md`, updated `paper-neurips/paper.tex`

### Phase 4: MTS-JEPA head-to-head comparison table (~1 hour)

**Goal**: rigorous, fair comparison between our method and MTS-JEPA.

**Dimensions of comparison**:

| Dimension | Our method (FAM/Trajectory JEPA) | MTS-JEPA |
|-----------|----------------------------------|----------|
| Architecture | Causal transformer, 2L, 1.26M params | Channel-independent transformer, 6L, ~5-8M params |
| Prediction target | Single h_future (attention-pooled) | Dual-resolution codebook (fine + coarse) |
| Collapse prevention | EMA → SIGReg curriculum | Soft codebook + dual entropy + EMA |
| Masking | Causal temporal split (variable length) | Fixed context/target windows |
| Horizon | Stochastic k ~ LogU[1, 150] | Fixed: one window ahead |
| Loss | L1 (1 term) | 7+ weighted terms |
| Downstream eval | Linear probe on frozen encoder | MLP classifier on flattened codes |

**Experiments to run**:
1. **Same-protocol SMAP evaluation**: pretrain our method on SMAP with MTS-JEPA's window=100, stride=1 protocol. Use both PA-F1 and non-PA F1. Report honestly.
2. **Lead-time analysis**: what fraction of our "detections" are continuations vs genuine predictions? Apply the same lead-time decomposition we did for MTS-JEPA replication.
3. **C-MAPSS as differentiator**: MTS-JEPA has no C-MAPSS numbers. Report our C-MAPSS results alongside "N/A" for MTS-JEPA. This is our unique contribution.
4. **Parameter efficiency**: compare performance per parameter. We're 4-6x smaller.
5. **Training cost**: compare wall-clock time, GPU hours.

**Important**: if our SMAP numbers are bad (likely, given v17's 0.038 non-PA F1), be honest. Frame the contribution as: "JEPA for prognostics (RUL/TTE), not anomaly detection. MTS-JEPA wins on anomaly detection; we win on prognostics. Different tasks, complementary methods."

**Output**: `experiments/v18/mtsjepa_comparison.md`, LaTeX table for paper

### Phase 5: Cross-sensor attention with v17 improvements (~1.5 hours, stretch)

**Goal**: combine v14's iTransformer-style cross-sensor attention with v17's LogUniform k + fixed-window target.

**Why revisit**: v14 cross-sensor (14.98 frozen at 100%) was the best frozen result before v17. But it used v2's k~U[5,30]. With v17's LogU k and honest probing, it might improve further. The training instability (2/3 seeds diverge without sensor embeddings) might be addressable with curriculum SIGReg.

**Config**:
- Sensor-as-token: 14 tokens per cycle, learnable sensor-ID embeddings (required for stability)
- Alternating temporal causal + cross-sensor attention, 2 layer pairs, d=128, 4 heads
- LogU k ∈ [1, 150], w=10 (v17)
- Curriculum SIGReg on predictor (v17 Phase 3)
- 150 epochs (Schedule B or C from Phase 2)

**Evaluation**: frozen probe + E2E, FD001 only (100% and 20% labels).

**Output**: `experiments/v18/phase5_cross_sensor_results.json`

**Success**: frozen < 14.98 at 100%, std < 1.0 at 20% (stability improvement over v14).

### Phase 6: Multi-subset evaluation (FD001 + FD003 + FD004) (~1 hour, stretch)

**Goal**: show v17/v18 improvements generalize beyond FD001.

Using best pretrained checkpoint from Phase 1:
- In-domain pretrain + probe on FD003 (2 fault modes) and FD004 (6 conditions + 2 faults)
- Report frozen + E2E + F1 at k=30

**Reference numbers** (v14 full-sequence):

| Subset | V2 frozen | V14 frozen | V14 E2E |
|--------|-----------|------------|---------|
| FD001 | 17.81 | 15.70 | 14.32 |
| FD003 | 19.25 | 18.39 | 13.67 |
| FD004 | 29.35 | 28.08 | 25.27 |

**Output**: `experiments/v18/phase6_multisubset_results.json`

## Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| w (target window) | 10 | V17 |
| K_max (C-MAPSS) | 150 | V17 |
| k sampling | LogU[1, K_max] | V17 |
| EMA momentum | 0.99 | V2 |
| SIGReg λ | 0.05 | V15/V17 |
| SIGReg placement | predictor output | V17 Phase 3 |
| Probe WD | 1e-2 | V17 Phase 2 fix |
| Probe val n_cuts | 10 | V17 Phase 2 fix |
| E2E LR | 1e-4 | V11 |
| E2E epochs | 50 | V11 |
| Pretrain LR | 3e-4 | V2 |
| Pretrain epochs | 200 (or 150 for accelerated) | V17 / V18 |

## What we're NOT changing

- Encoder architecture (causal transformer, 2L, d=256, attention pooling)
- Predictor architecture (MLP, horizon-conditioned)
- L1 prediction loss
- Full-history context from cycle 1
- Dataset-specific tokenization (1 token per cycle for C-MAPSS)
- FD001 as primary benchmark (but adding FD003/FD004 evaluation)
