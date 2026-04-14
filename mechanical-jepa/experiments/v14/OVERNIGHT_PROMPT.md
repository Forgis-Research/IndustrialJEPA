# V14 Overnight Session

**Goal**: Paper-quality analysis, architectural experiments, honest reframing,
and theory. This session is about making the NeurIPS submission bulletproof,
not chasing RMSE.

**Commit cadence**: ~10 commits, roughly every hour. Push after every 1-2
commits. Never let >2 unpushed commits accumulate.

**W&B**: All training runs logged to `wandb.init(project="industrialjepa")`.
VRAM/RAM/disk every 60s via psutil. Tag runs `v14-{phase}-{name}`.

**Output**: All results under `mechanical-jepa/experiments/v14/`. Generate
a Quarto notebook `mechanical-jepa/notebooks/14_v14_analysis.qmd` by end
of session (use .qmd not .ipynb, code-fold: true, self-contained: true).

---

## What we know from v11-v13

- V2 (d=256, L=2, 1.26M params) is the primary encoder config
- E2E 14.23 ± 0.39 on FD001, frozen 17.81, STAR 12.19 (gap is architectural)
- From-scratch ablation: pretraining contributes +8.8 RMSE at 100%, +15.6 at 10%
- Length-vs-content: encoder reads sensor content, not just length
- STAR label sweep: JEPA frozen beats STAR at 5% labels (24.47 vs 24.55)
- H.I. R²=0.926 and frozen RMSE 17.81 are the SAME result on different eval
  protocols. H.I. is piecewise-linear rescaling of RUL — not a separate finding.
  The R² is inflated by the healthy plateau (~40% of cycles trivially correct)
  and by evaluating on in-distribution val engines vs harder canonical test set.
- Frozen encoder tracks better (rho=0.856) than E2E (0.804)
- Deeper (V4, L=4) improves frozen but hurts E2E — frozen-vs-E2E trade-off

---

## Phase 1 — Paper honesty fix: H.I. vs RUL reframing [~30 min]

**Problem**: The paper currently presents H.I. recovery (R²=0.926) as "the
headline result" and a separate contribution from the RMSE. It's not — H.I.
and RUL are deterministic transforms of each other:

```
H.I.(t) = 1 - RUL(t)/125    (when RUL < 125)
RUL(t) = 125 × (1 - H.I.(t))
```

The R²=0.926 looks better than RMSE 17.81 because: (a) healthy-plateau
cycles (~40%) are trivially correct and inflate R², (b) eval is on
in-distribution val engines vs harder canonical test set, (c) Ridge
regression on all cycles has more training signal than a probe on sampled
cut points.

**What to do**:
1. In `paper-neurips/paper.tex`, reframe Section 4.3 (H.I. Recovery):
   - Keep the result, but present it as a **representation diagnostic**,
     not a separate prediction method or a separate contribution
   - Add a sentence: "H.I. and RUL are deterministic transforms of each
     other; R²=0.926 and frozen RMSE 17.81 measure the same encoder
     capability under different evaluation protocols. The R² is reported
     as a representation quality diagnostic, not as an alternative to
     RMSE evaluation."
   - Move the headline from "H.I. recovery" to the from-scratch ablation
     (+8.8 RMSE, the genuinely separate finding) and the STAR crossover
     at 5% labels
2. Update the abstract accordingly
3. Update contribution bullet 1 to be honest about what H.I. recovery
   actually demonstrates vs what it doesn't

Do NOT make a big deal of this correction. Just fix it precisely and move
on. The paper is stronger for being honest.

---

## Phase 2 — Full-sequence prediction experiment [~3-4h]

**Hypothesis**: Currently the JEPA predicts future latent embeddings
$\hat{h}_{t+1:t+k}$ from past context $h_{1:t}$. But the target encoder
sees ONLY the future window $x_{t+1:t+k}$, not the full history. What if
both the context and target encoder see the WHOLE sequence $x_{1:t+k}$,
just with different roles?

**New objective**: Given $x_{1:t}$ (context), predict the EMA target
encoder's representation of $x_{1:t+k}$ (the WHOLE sequence up to t+k,
not just the future slice). This means:
- Context encoder: causal over $x_{1:t}$ → $h_\text{past}$ (unchanged)
- Target encoder: processes $x_{1:t+k}$ (full sequence including past)
  → $h_\text{full}$ (NEW — this sees the whole trajectory)
- Predictor: $h_\text{past}, k$ → $\hat{h}_\text{full}$
- Loss: $\|$predictor($h_\text{past}$, $k$) $- \text{sg}[\text{target}(x_{1:t+k})]$$\|_1$

**Rationale**: The current objective asks "given the past, predict what
the future looks like." The new objective asks "given the past, predict
what the whole-trajectory-including-future looks like." The target
representation now includes the past (which the predictor already has)
plus the future (which it must predict). This might produce more
informative gradients because the target is richer.

**Implementation**:
- Modify `TargetEncoder.forward` to accept the concatenated sequence
  $x_{1:t+k}$ instead of just $x_{t+1:t+k}$
- In `CMAPSSPretrainDataset`, return `(past, full_sequence, k, t)` instead
  of `(past, future, k, t)`
- Everything else stays the same (EMA, predictor, loss)

**Evaluation**: Pretrain with new objective (200 epochs), evaluate with
frozen probe + E2E at 100% labels, 5 seeds. Compare directly to V2
baseline (same architecture, same hyperparams, only the prediction
target changes).

Output: `experiments/v14/full_sequence_prediction.json`

**Kill criterion**: If frozen probe doesn't improve vs V2 frozen (17.81),
the objective change doesn't help. Revert.

---

## Phase 3 — Cross-sensor attention (iTransformer-style) [~4-5h]

**Motivation**: Currently all 14 sensors are concatenated into a single
vector per cycle, then projected to d_model. The attention operates
over cycles (temporal attention). STAR's key advantage is two-stage
attention (temporal THEN sensor-wise). The iTransformer paper inverts
the standard transformer: it treats each variate (sensor) as a token
and applies attention across variates, capturing cross-sensor
dependencies directly.

**Experiment**: Implement a cross-sensor attention variant:

### 3a. Sensor-as-token encoder

Instead of `x^{(t)} ∈ R^{14}` → project → single token per cycle:
- Each cycle produces 14 tokens (one per sensor), each projected to
  d_model via a per-sensor linear layer
- Temporal attention: within each sensor, attend across cycles
  (like current architecture but per-sensor)
- Cross-sensor attention: within each cycle, attend across 14 sensors
- Alternate temporal and cross-sensor layers (like STAR's two-stage)

This is a direct adaptation of iTransformer to our setting. The
architecture becomes: SensorProject (14 × Linear) → [TemporalAttn +
CrossSensorAttn] × L layers → pool → h_past.

### 3b. Attention map analysis

After training 3a, extract and visualize:
- **14×14 cross-sensor attention map**, averaged across:
  - All cycles of an average-length engine
  - The healthy phase (first 60% of cycles)
  - The degradation phase (last 40% of cycles)
- Compare healthy vs degradation attention patterns. Do sensors attend
  to different peers during degradation? Which sensor pairs have the
  strongest attention weights?
- Overlay with known C-MAPSS physics: s2 (total temp LPC), s3 (total
  temp HPC), s4 (total temp LPT), s7 (total press HPC), s11 (static
  press at HPC outlet), s12 (fuel flow), s15 (bypass ratio), s21
  (bleed enthalpy). Sensor pairs that are physically coupled (e.g.,
  HPC sensors s3/s7/s11) should attend to each other.

Output:
- `experiments/v14/cross_sensor_results.json` (RMSE frozen + E2E)
- `experiments/v14/cross_sensor_attention_maps.json` (attention weights)
- `analysis/plots/v14/cross_sensor_attention_healthy.png`
- `analysis/plots/v14/cross_sensor_attention_degradation.png`
- `analysis/plots/v14/cross_sensor_attention_diff.png`

**Evaluation**: Same as Phase 2 — pretrain 200 epochs, frozen + E2E at
100%, 5 seeds. Does cross-sensor attention close the STAR gap?

---

## Phase 4 — C-MAPSS data analysis (concise, publication-quality) [~1h]

Generate a small set of maximally informative plots. Aim for 4-6 plots
total that a reader needs to understand C-MAPSS and our method. Every
plot must have a clear point — no decorative figures.

### 4a. Dataset overview (1 figure, 2-3 panels)

- Panel A: 3-4 raw sensor trajectories (s2, s7, s12, s21) for ONE
  representative engine (pick one near median length ~200 cycles).
  Y-axis: normalized sensor value. X-axis: cycle. Show the degradation
  trend clearly.
- Panel B: Same sensors for ONE extreme engine (longest, ~360 cycles).
  Show that degradation onset is later but the pattern is similar.
- Panel C: RUL label distribution across all training engines. Show the
  piecewise-linear cap at 125. Annotate the healthy plateau fraction.

### 4b. Method illustration (1 figure, 2-3 panels)

- Panel A: Schematic of the trajectory prediction task. Show an engine
  trajectory split at cutoff t. Past (blue) → context encoder → h_past.
  Future (red) → target encoder → h_future. Predictor: h_past → ĥ_future.
  Keep it clean — this replaces paragraphs of text.
- Panel B: What the probe sees. Show h_past as a dot in latent space,
  colored by H.I./RUL. Multiple engines at different degradation stages.
  The linear probe finds the direction that separates healthy from
  degraded.
- Panel C: Prediction vs true RUL for one test engine across all cut
  points. Show the model tracking degradation within a single engine.

### 4c. Key result visualization (1 figure, 2 panels)

- Panel A: Label-efficiency curve. JEPA E2E, JEPA frozen, STAR, LSTM.
  X-axis: label budget (log scale). Y-axis: RMSE. Show the crossover
  where JEPA frozen beats STAR at 5%.
- Panel B: From-scratch ablation. Pretrained vs random init at each
  label budget. Show the growing delta as labels decrease.

All plots: publication quality, matplotlib with consistent style, font
size ≥ 10pt, no unnecessary gridlines, save as both PNG and PDF.

Output: `analysis/plots/v14/` + PDF copies in `notebooks/plots/`.

---

## Phase 5 — Paper review and update [~1h]

Read `paper-neurips/paper.tex` end-to-end. For each section, assess:
1. Is the claim supported by current evidence?
2. Is the evidence presented honestly?
3. What is missing for NeurIPS spotlight?

### Specific updates:

**Abstract**: Lead with the from-scratch ablation and label-efficiency
crossover, not H.I. recovery. H.I. is a diagnostic, not the headline.

**Contributions**: Reorder:
1. From-scratch ablation (+8.8 to +15.6 RMSE) — strongest quantitative SSL evidence
2. Label-efficiency crossover (beats STAR at 5%, matches at 20%) — grey-swan pitch
3. Tracking verification (5 diagnostics) — methodological contribution
4. H.I. recovery as representation diagnostic — supporting evidence, not headline

**Experiments**: Add v13 results (from-scratch, STAR sweep, length-vs-content).
These are currently only in experiments/v13/RESULTS.md, not in the paper.

**New figure**: The from-scratch ablation deserves its own figure (pretrained
vs random init across label budgets). This is the money plot.

**Blue \plannedc{} additions** for realistic NeurIPS scope:
- Phase 2 result (full-sequence prediction) if it improves
- Phase 3 result (cross-sensor attention) if it improves
- Cross-domain transfer (C-MAPSS → bearings or vice versa)
- Theoretical analysis from Phase 6

**Narrative for spotlight**: The paper should argue that self-supervised
trajectory prediction is a **sufficient condition** for learning
degradation-aware representations, with formal connections to slow
feature analysis and information-theoretic arguments (Phase 6). The
empirical evidence is strong enough; what's missing for spotlight is
theoretical grounding and a second domain.

---

## Phase 6 — Theory: why does trajectory prediction learn degradation? [~2h]

This is the section that could elevate the paper from accept to spotlight.
The empirical observation is clear (trajectory prediction → degradation
tracking). The question is: WHY?

### 6a. Slow Feature Analysis connection

**Argument**: The trajectory prediction objective forces the encoder to
represent features whose predicted future value differs from the present.
Features that change slowly (degradation) are harder to predict than
features that change fast (noise, operating-condition oscillations).
The optimal predictor therefore allocates representational capacity
to slow features — exactly the degradation dynamics.

This is a formal connection to Slow Feature Analysis (Wiskott & Sejnowski,
2002). SFA extracts the slowest-varying signals from a time series. Our
trajectory JEPA does something similar: by predicting future latent states,
it implicitly identifies the directions in representation space along
which the signal changes most predictably (= slowest, = degradation).

**Formalize**: Write out the objective, show that minimizing L1 prediction
error in latent space is equivalent to maximizing mutual information
between h_past and h_future, which is maximized by features that vary
slowly and predictably across the trajectory.

### 6b. Information-theoretic argument

**Argument**: The JEPA objective maximizes $I(h_\text{past}; h_\text{future})$
(mutual information between past and future representations). By the
Data Processing Inequality, $I(h_\text{past}; h_\text{future}) \leq
I(x_{1:t}; x_{t+1:t+k})$. The mutual information between past and
future sensor readings is dominated by the slow-varying component
(degradation state), not the fast-varying component (sensor noise,
operating-condition transients). Therefore the encoder is incentivized
to represent the slow component — which is the health index.

**Formalize**: State as a proposition with assumptions:
- Assumption 1: Sensor readings decompose as $x^{(t)} = f(\text{H.I.}(t),
  \epsilon_t)$ where H.I. varies slowly and $\epsilon$ varies fast.
- Assumption 2: $\epsilon$ is approximately i.i.d. across cycles.
- Proposition: The representation that maximizes next-state prediction
  accuracy concentrates on the H.I. component.

### 6c. Why frozen beats E2E on tracking

**Argument**: E2E fine-tuning with MSE loss on capped RUL introduces a
bias toward the label distribution. ~40% of training labels are at the
cap (125 cycles), creating a strong prior toward predicting "healthy."
The E2E encoder shifts its representation to accommodate this prior,
sacrificing tracking fidelity (rho drops from 0.856 to 0.804). The
frozen encoder is uncontaminated by this label bias.

**Formalize**: Show that the MSE gradient under the capped-RUL label
distribution has a larger component in the "predict healthy" direction
than in the "track degradation" direction, because the plateau dominates
the loss surface.

### Output

Write the theory as a self-contained section draft. Save as:
- `experiments/v14/theory_draft.md` (markdown)
- Include in the Quarto notebook as a theory section
- If formal enough, include in `paper-neurips/paper.tex` (new section
  between Method and Experiments, or as a subsection of Analysis)

---

## Phase 7 — Quarto notebook [~1h, end of session]

Compile all v14 findings into `mechanical-jepa/notebooks/14_v14_analysis.qmd`.

Structure:
1. TL;DR callout (like the A2P walkthrough)
2. C-MAPSS data analysis (Phase 4 plots with explanation)
3. Full-sequence prediction experiment (Phase 2 result)
4. Cross-sensor attention experiment (Phase 3 result + attention maps)
5. Theory: why trajectory prediction learns degradation (Phase 6)
6. Updated label-efficiency comparison (with v13 STAR + from-scratch)
7. Paper review notes (Phase 5 findings)

Use `.qmd` format, `code-fold: true`, `self-contained: true`, `theme: cosmo`.

---

## Execution timeline

```
T+0:00  Phase 1 (paper H.I. reframe, 30 min) — COMMIT + PUSH
T+0:30  Launch Phase 2 (full-sequence pretrain, background, ~3h)
T+0:30  Phase 4 (data analysis plots, 1h) — COMMIT + PUSH
T+1:30  Phase 6 (theory, 2h) — COMMIT + PUSH
T+3:00  Collect Phase 2 results — COMMIT + PUSH
T+3:00  Launch Phase 3a (cross-sensor pretrain, background, ~4h)
T+3:30  Phase 5 (paper review + update, 1h) — COMMIT + PUSH
T+4:30  Phase 6 continued if needed
T+5:00  Phase 3b (attention map analysis, after 3a finishes) — COMMIT + PUSH
T+7:00  Collect Phase 3 results — COMMIT + PUSH
T+7:00  Phase 7 (Quarto notebook, 1h) — COMMIT + PUSH
T+8:00  Session wrap-up, final RESULTS.md — COMMIT + PUSH
```

---

## Kill criteria

- Phase 2 (full-sequence): if frozen RMSE > 18.5 (worse than V2 by > 0.7), revert
- Phase 3 (cross-sensor): if frozen RMSE > 18.5, revert (but keep attention maps)
- Phase 6 (theory): if the SFA/MI argument doesn't formalize cleanly, keep it
  as an informal "intuition" section rather than forcing weak math

---

## Self-check reminders

From the ml-researcher Internal Consistency Audit:
- Reconcile every artifact from the same run (metric + plot must agree)
- Run the trivial feature-regressor baseline if any new architecture is tested
- If a new result beats V2, verify with the v12 tracking diagnostics (rho, pred_std, shuffle)
- Do NOT present inflated numbers (the H.I. R²=0.926 lesson)

---

## One-sentence success criterion

**By morning, the paper is honestly reframed, two architectural experiments
have run, the C-MAPSS dataset is illustrated with publication-quality plots,
a theoretical argument for WHY trajectory prediction learns degradation
exists in draft form, and everything is in a Quarto notebook — committed
and pushed.**
