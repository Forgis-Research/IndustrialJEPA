# V17 Plan: Generalized Horizon + Trajectory Probing + Curriculum SIGReg

## Motivation

V2 (v11) trains the predictor on k ~ U[5, 30], coupling offset and target length. This limits inference to a 30-step look-ahead and makes large-k targets diffuse. V17 fixes this and introduces two new ideas: (1) using the predictor at probe time, and (2) curriculum learning from EMA to SIGReg.

## Key design principles

1. **Full history from cycle 1.** The context encoder always sees x_{1:t}, not a sliding window. Cumulative degradation (total energy, wear) is only visible from the full history. The causal transformer + attention pooling handles variable length natively.

2. **The predictor is a latent dynamics model.** It maps (h_past, k) → γ(k), the predicted latent state k steps ahead. This asset should be used at inference, not discarded.

3. **All grey swan tasks reduce to:** "when does the predicted trajectory enter a critical region?" One encoder, one predictor, one linear probe per event type.

## Architecture Changes

### 1. Separate offset from target window

```
Old:  target = ema_encoder(x_{t+1 : t+k})      # length varies with k
New:  target = ema_encoder(x_{t+k : t+k+w})     # fixed length w, offset k
```

- **k** = temporal offset (how far ahead). Sampled from LogUniform[1, K_max].
- **w** = target window size (fixed, e.g. 10 cycles). Always the same.

Prediction task: "what will a w-step snapshot look like, k steps from now?"

The target quality is constant across all k (always a w-step summary). Only the prediction difficulty varies with k. This is how I-JEPA works spatially — fixed-size target patch at varying positions.

### 2. LogUniform horizon sampling

Replace `k ~ U[5, 30]` with `k ~ LogUniform[1, K_max]`.

```python
k = int(torch.exp(torch.empty(1).uniform_(0, math.log(K_max))).item())
k = max(1, min(k, available_future - w))
```

- Equal training weight per order of magnitude: k ∈ [1,3] same as [30,100]
- Short horizons: easy, strong gradients, ground the model
- Long horizons: hard, teach extrapolation
- K_max = min(150, available_future_length - w) for C-MAPSS
- Training cost unchanged (still one (t, k) pair per step)

This also serves as the multi-scale mechanism: small k captures spike-level dynamics, large k captures drift-level dynamics. No multi-resolution architecture needed.

### 3. Trajectory probing (use the predictor at inference)

Instead of probing h_past alone, probe the predicted latent trajectory:

```
h       = encoder(x_{1:t})              # frozen
γ(k)    = predictor(h, k)               # frozen, for any k

probe(γ(k)) → s(k)                      # "proximity to event at horizon k"

TTE     = min { k : s(k) > τ }          # sweep k at inference
```

One probe per event type. Probe is a linear layer. Trained on (γ(k), y_k) pairs.
The k-dependence comes from the predictor, not the probe.

Direct prediction (skip), not autoregressive rollout — no error accumulation, no domain mismatch between predictor output and encoder input space.

For RUL (regression), alternative: concatenate γ at a fixed k-set:
```
probe([h, γ(5), γ(10), γ(20), γ(50), γ(100)]) → RUL
```

### 4. Curriculum EMA → SIGReg

**Idea**: EMA bootstraps the encoder into a good basin (stable, proven). Then transition to SIGReg (no target network, simpler). V15 showed SIGReg fails from cold start but achieves isotropy — the issue was optimization, not the objective. A warm start from EMA should land in a stable SIGReg basin.

**Schedule** (graduated, not abrupt):
```
Epoch   0-100:  EMA (momentum 0.99) + λ_sig = 0         # standard V2 pretraining
Epoch 100-150:  EMA (momentum 0.99) + λ_sig ramps 0→0.05 # introduce SIGReg gently
Epoch 150-200:  No EMA (stop-grad only) + λ_sig = 0.05   # SIGReg maintains isotropy
```

The target in epoch 150+ becomes: h* = stop_grad(encoder(x_{t+k:t+k+w})) — same encoder, no separate target network. SIGReg prevents collapse.

**Two SIGReg placement variants** (test both):

**(a) SIGReg after encoder** — SIGReg(h_past = encoder(x_{1:t}))
- Ensures encoder output space is isotropic
- Standard placement (same as V15)
- Relevant if probing h_past directly

**(b) SIGReg after predictor** — SIGReg(γ(k) = predictor(h, k))
- Ensures predicted latent states are isotropic
- Novel: this is the space where trajectory probes operate in V17
- More directly relevant to downstream performance
- The predictor might collapse γ(k) even if h_past is diverse — this prevents it

Hypothesis: (b) is better for V17 because the probe reads γ(k), not h_past.

## Experiments (in execution order)

### Phase 1: V17 baseline — LogUniform k + fixed-window target
- Retrain V2 architecture with k ~ LogU[1, 150], w=10
- EMA only (no SIGReg yet), 200 epochs
- Frozen probe on h_past → compare to V2 baseline (17.81)
- 3 seeds, FD001
- **Success**: frozen probe ≤ 17.81

### Phase 2: Trajectory probing
- Freeze encoder + predictor from Phase 1
- Train linear probe on concatenated [h, γ(5), γ(10), γ(20), γ(50), γ(100)]
- Compare to Phase 1 frozen probe (h_past only, 256-dim) and V2 baseline
- 3 seeds × 3 probe seeds
- **Success**: trajectory probe < 15.0 (close the frozen-E2E gap of 14.23)

### Phase 3: Curriculum EMA → SIGReg
- From Phase 1 checkpoint at epoch 100, continue with graduated schedule
- Test both variants: (a) SIGReg after encoder, (b) SIGReg after predictor
- Evaluate frozen probe and trajectory probe at epoch 200
- Compare to Phase 1 (EMA-only) at epoch 200
- 3 seeds each variant
- **Success**: matches Phase 1 frozen quality without target network

### Phase 4: TTE via trajectory sweep
- Using best checkpoint from Phase 1-3
- Train event-boundary probe: probe(γ(k)) → p(sensor s14 exceeds 3σ by horizon k)
- Sweep k = 1..150 at inference, find crossing point
- Evaluate RMSE on C-MAPSS TTE ground truth
- **Success**: TTE numbers exist (paper section 5.4 stops being future work)

### Phase 5: SMAP anomaly detection
- Pretrain V17 on SMAP (k ~ LogU[1, 500], w=10)
- Anomaly score: prediction error at multiple k values
- Compare non-PA F1 to V15 baseline (6.9%)
- **Success**: non-PA F1 > 0.10

### Phase 6: Quarto notebook
- Concise educational walkthrough of V17 results
- Architecture diagram (pretrain → probe → infer)
- Key plots: LogUniform k distribution, trajectory probe vs h_past probe, curriculum SIGReg transition, TTE sweep visualization
- Save as notebooks/17_v17_analysis.qmd

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| w (target window) | 10 | ~3-5% of typical C-MAPSS engine life |
| K_max (C-MAPSS) | 150 | ~50% of median engine life |
| K_max (SMAP) | 500 | ~1% of test sequence length |
| LogU sampling | k = exp(U[0, ln(K_max)]) | Equal weight per order of magnitude |
| EMA momentum | 0.99 | V2 default, proven |
| SIGReg λ | 0.05 | V15 default |
| Curriculum switch | epoch 150 | After 75% of training |
| SIGReg ramp | epochs 100-150 | 50-epoch linear ramp |
| Probe k-set (RUL) | {5, 10, 20, 50, 100} | Covers short to long horizon |
| w ablation | {5, 10, 20} | If time permits after main phases |

## What we're NOT changing

- Encoder architecture (causal transformer, 2L, d=256, attention pooling)
- Predictor architecture (MLP, horizon-conditioned via sinusoidal PE)
- L1 loss
- Full-history context from cycle 1 (variable length, not windowed)
- Dataset-specific tokenization (1 token per cycle for C-MAPSS)
