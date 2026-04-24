# FAM Architecture Specification

Definitive reference for tokenization, pretraining, and finetuning.
All code and paper text must be consistent with this document.

---

## Notation

| Symbol | Meaning |
|--------|---------|
| $t$ | Current time (end of context) |
| $\Delta t$ | Horizon — length of the future interval |
| $C$ | Number of channels (sensors) |
| $P$ | Patch length (consecutive timesteps per token) |
| $d$ | Embedding dimension (256) |
| $\mathbf{x}_{\leq t}$ | Context: all observations up to time $t$ |
| $\mathbf{x}_{(t, t+\Delta t]}$ | Target: observations in the interval $(t, t{+}\Delta t]$ |
| $\mathbf{h}_t$ | Context embedding (output of encoder) |
| $\mathbf{h}^*_{(t, t+\Delta t]}$ | Target embedding (output of target encoder) |
| $p(t, \Delta t)$ | Predicted probability of event in $(t, t{+}\Delta t]$ |

---

## 1. Tokenizer

All channels, all datasets, same tokenizer. **P=16 is fixed globally** —
no dataset-specific tuning. This follows MOMENT (P=8 fixed) and TimesFM
(P=32 fixed) rather than PatchTST (P tuned per dataset). One tokenizer
strengthens the "one architecture" claim.

```
Raw input: (context_len, C)           variable length

Step 1 — RevIN (Reversible Instance Normalization):
  Per context, per channel: subtract mean, divide by std.
  Store (mean, std) for optional denormalization.
  This removes distribution shift without requiring training-set statistics.

Step 2 — Patch:
  Group P=16 consecutive timesteps.
  (context_len, C) → (ceil(context_len / 16), C × 16)
  Pad last patch with zeros if context_len not divisible by 16.

Step 3 — Project:
  Linear(C × 16, d)
  (N_tokens, C × 16) → (N_tokens, d)

Step 4 — Positional encoding:
  + sinusoidal PE(patch_index)

Result: N_tokens = ceil(context_len / 16) tokens of dimension d.
```

### What P=16 means per dataset

| Dataset       | Channels | Sample rate | Tokens for typical context     |
|---------------|----------|-------------|--------------------------------|
| C-MAPSS       | 14       | 1/cycle     | 192 cycles → 12 tokens         |
| SMAP / MSL    | 25 / 55  | ~1 Hz       | 512 steps → 32 tokens          |
| PSM           | 25       | 1/min       | 512 steps → 32 tokens          |
| SMD           | 38       | 1/min       | 512 steps → 32 tokens          |
| MBA (ECG)     | 2        | 275 Hz      | 512 steps → 32 tokens          |
| Sepsis        | 40       | 1/hour      | 48 hours → 3 tokens            |

P=16 is not a dataset-specific hyperparameter — there are **zero**
dataset-specific hyperparameters.

---

## 2. Pretraining (self-supervised)

Given a raw time series of length $T$:

```
Full series:  [═══════════════════════════════════] T steps

Sample cut point t and horizon Δt:

              [────── context ──────][── target ──]
              x[0 : t]               x(t : t+Δt]
              variable length         Δt steps
```

- **Context** = $\mathbf{x}_{\leq t}$: all observations up to time $t$.
  Variable length (full history for C-MAPSS, sliding window for streams).
- **Target** = $\mathbf{x}_{(t, t+\Delta t]}$: the **next $\Delta t$ steps**
  immediately after $t$. No gap. No skip.
- $\Delta t \sim \text{LogUniform}[1, \Delta t_\text{max}]$: sampled randomly.
  During pretraining, the target x(t : t+Δt] is tokenized with the same
  P=16 tokenizer (last patch zero-padded if Δt is not a multiple of P).
  This is fine — the target encoder handles variable-length input.

### Forward pass

```
context  x[0:t]       → tokenizer → causal transformer  → h_t           (d,)
target   x(t:t+Δt]    → tokenizer → bidir transformer   → h*_(t,t+Δt]  (d,)

predictor(h_t, Δt)    → ĥ_(t,t+Δt]   (d,)

Loss = ‖ĥ − sg(h*)‖₁  +  λ_var · var_reg
```

The predictor learns: "given the context representation and a horizon Δt,
predict the representation of the next Δt steps."

### Key: pretraining and finetuning use the same interval

The target represents "what happens in $(t, t+\Delta t]$."
The finetuning head classifies whether that interval contains an event.
Same interval, same semantics.

---

## 3. Finetuning (pred-FT)

Freeze encoder. Train predictor + event head on labeled data.
Output is parameterized as a **discrete hazard → CDF** to guarantee
monotonicity by construction (not post-hoc enforcement).

```
context  x[0:t]  → frozen encoder → h_t  (d,)

For each Δt_k ∈ H = {Δt_1 < ... < Δt_K}:
  predictor(h_t, Δt_k) → ĥ_k → event_head → hazard logit

λ_k = σ(hazard_logit_k)              conditional hazard ∈ (0,1)
S_k = ∏_{j≤k} (1 - λ_j)             survival function (non-increasing)
p(t, Δt_k) = 1 - S_k                 CDF (non-decreasing)
```

**Interpretation**: λ_k = P(event in (Δt_{k-1}, Δt_k] | no event before Δt_{k-1}).
Each λ_k is an arbitrary function of h_t — no distributional assumptions.

**Label**: $y(t, \Delta t) = \mathbb{1}[\text{event occurs in } (t, t{+}\Delta t]]$

**Loss**: $\sum_{k} w^+ \cdot \text{BCE}(p(t, \Delta t_k), y(t, \Delta t_k))$

where $w^+ = N_{neg} / N_{pos}$ handles class imbalance. Gradients flow
through `cumprod` — interval j gets gradient signal from all horizons k ≥ j.

### Horizons

Horizons are **independent of P**. The predictor takes Δt as a continuous
scalar — it learns a smooth mapping from Δt to representations during
pretraining, and can be evaluated at any Δt during finetuning. No
constraint that Δt be a multiple of P.

$\mathcal{H}$ is chosen per dataset to cover the relevant event timescale:

| Dataset       | Horizons $\mathcal{H}$                    | Units  |
|---------------|-------------------------------------------|--------|
| C-MAPSS       | {1, 5, 10, 20, 50, 100, 150}             | cycles |
| SMAP/MSL/PSM/SMD/MBA | {1, 5, 10, 20, 50, 100, 150, 200} | steps  |
| Sepsis        | {1, 2, 3, 6, 12, 24, 48}                 | hours  |

### Monotonicity (by construction)

$p(t, \Delta t)$ is a discrete CDF of time-to-event. Monotonicity
is guaranteed by the hazard parameterization:

$$\Delta t_2 > \Delta t_1 \implies p(t, \Delta t_2) \geq p(t, \Delta t_1)$$

Each $(1-\lambda_j) \in (0,1)$, so the survival product is non-increasing,
and $1 - \text{product}$ is non-decreasing. Zero violations by design.

This is standard discrete-time survival analysis (DeepHit, DRSA) — not
an assumption about the data, but a reparameterization that makes the
CDF constraint structural.

### Derived quantities

Probability of event in an arbitrary future interval:

$$P(\text{event in } (t{+}a,\; t{+}b]) = p(t, b) - p(t, a) \quad \text{for } b > a$$

This is a CDF difference. No additional model evaluation needed.

---

## 4. Inference

Sliding context with stride = 1 (every timestep gets a prediction):

```
For each t in test set:
  context = x[max(0, t-C_max) : t]   (variable length, up to C_max)
  → encoder → predictor at all Δt ∈ H → p(t, Δt)

Stack → p_surface  (N, K)     predicted probabilities
        y_surface  (N, K)     ground truth

AUPRC = average_precision(p_surface.flat, y_surface.flat)
```

---

## 5. Context length

| Dataset   | Context strategy | Rationale |
|-----------|-----------------|-----------|
| C-MAPSS   | Full history (up to ~362 cycles) | Degradation has a lifecycle start |
| SMAP/MSL  | Sliding, max 512 steps | Continuous monitoring, no lifecycle |
| PSM/SMD   | Sliding, max 512 steps | Continuous monitoring |
| MBA       | Sliding, max 512 steps | Continuous monitoring |
| Sepsis    | Full stay (up to ~336 hours) | ICU stay has a clear start |

Variable-length context is handled by:
- Causal transformer (no fixed-length constraint)
- Padding mask for batching
- Sinusoidal PE (generalizes to any length)

### Minimum and maximum context

**Minimum**: 128 timesteps (8 tokens at P=16). Below this the transformer
degenerates — self-attention with <8 tokens learns nothing an MLP couldn't.
Sequences shorter than 128 steps should be padded to 128 or excluded.

**Maximum sliding context**: 512 timesteps (32 tokens) for continuous
monitoring datasets. This is a compute/memory choice, not architectural.

**Operating bandwidth**: the architecture targets data sampled between
~1/minute and ~1 kHz. Datasets outside this range (e.g. bearings at
20+ kHz) should be downsampled before tokenization. Datasets at very
low rates (e.g. hourly) must have sequences of at least 128 steps to
produce enough tokens.

### Implication for Sepsis

Sepsis at 1/hour with P=16 requires stays ≥ 128 hours (~5.3 days).
Most ICU stays are shorter. Two options:
1. Use P=1 for Sepsis (honest exception to P=16 global rule)
2. Exclude stays < 128 hours (loses most of the dataset)

Decision: **use P=1 for Sepsis.** The paper states "P=16 for all datasets
with sufficient temporal resolution." Sepsis at 1/hour is below the
resolution floor where P=16 is meaningful.

---

## 6. Model dimensions

| Component | Architecture | Params |
|-----------|-------------|--------|
| Context encoder | Causal Transformer (d=256, L=2, H=4, d_ff=256) + RevIN + PatchEmbed | ~850K |
| Target encoder | Bidirectional Transformer (same dims) + attention pool | ~920K |
| Predictor | MLP: Linear(257→256) → GELU → Linear(256→256) → GELU → Linear(256→256) | ~198K |
| Event head | LayerNorm(256) + Linear(256→1) | ~513 |
| **Total** | | **~2.16M** |

Trainable during pred-FT: predictor + event head = ~198K params.

### Predictor design

The predictor is currently a 2-layer MLP that takes (h_t, Δt) and outputs
a predicted future representation. It learns a smooth mapping from horizon
to representation space during pretraining, then is repurposed during
finetuning to produce representations the event head can classify.

**Pretraining role**: learn temporal dynamics — how representations evolve.
Short Δt → small perturbation. Long Δt → larger displacement. The predictor
forces the encoder to produce representations that capture actual dynamics.

**Finetuning role**: make the target event visible. The pretrained predictor
already knows what "Δt=50 steps into the future" looks like in representation
space. Finetuning bends this toward event detection.

**Open question**: the predictor may benefit from a small transformer
architecture rather than MLP, especially for learning nonlinear regime
transitions (degradation onset, pre-failure acceleration). Worth ablating
in a future session.

---

## 7. Code

Canonical implementation lives in two files:
- `fam-jepa/model.py` — architecture only (FAM class and all components)
- `fam-jepa/train.py` — datasets, training loops, evaluation

Experiment scripts import from these. No model code in experiment dirs.
