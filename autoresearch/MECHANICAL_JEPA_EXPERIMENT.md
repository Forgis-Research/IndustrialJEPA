# Mechanical-JEPA: Full Experiment Roadmap

**Goal:** Determine if JEPA (Joint Embedding Predictive Architecture) can learn transferable physics representations from industrial sensor data.

---

## The Core Hypothesis

> "Predicting in latent space (JEPA) forces the model to learn the underlying physics/dynamics, not sensor noise. These physics representations should transfer across machines that share the same underlying physics."

---

## Why JEPA for Physical Systems?

### Standard Prediction
```
Sensors(t) ──► Model ──► Sensors(t+1)

Problem: Must model noise, sensor quirks, calibration differences
         These DON'T transfer.
```

### JEPA Prediction
```
Sensors(t) ──► Encoder ──► Latent(t)
                              │
                          Predictor
                              │
                              ▼
                         Latent(t+1) ◄── Target Encoder ◄── Sensors(t+1)

Benefit: Latent space can ignore noise, capture dynamics.
         Dynamics DO transfer.
```

---

## Proof of Concept (PoC) Gate

**Before investing in full implementation, validate the core assumption:**

### PoC Question
> "Does predicting in latent space actually learn better physics than predicting raw sensors?"

### PoC Experiment

```
Dataset: AURSAD (Robot 1 only, simpler)
Comparison:
  A) Raw prediction: Encoder → MLP → Sensors(t+1)
  B) Latent prediction: Encoder → Predictor → Latent(t+1), then Decoder → Sensors(t+1)
  C) JEPA: Encoder → Predictor → Latent(t+1), Target Encoder provides target

Metrics:
  - Forecasting MSE (all should be similar)
  - Latent space quality:
    - Linear probe on latent space (can you predict physics quantities?)
    - t-SNE visualization (does it cluster by physical state?)
  - Transfer (Robot 1 → Robot 2):
    - Which approach transfers better?
```

### PoC Success Criteria

| Metric | Pass | Fail |
|--------|------|------|
| JEPA forecasting | Within 10% of raw prediction | >20% worse |
| Linear probe accuracy | >70% | <50% |
| Transfer improvement | JEPA 20%+ better than raw | JEPA worse |

**If PoC fails → abandon this direction.**
**If PoC passes → proceed to full implementation.**

---

## Roadmap

### Stage 1: Literature & Theory (Day 1)

- [ ] Deep read of I-JEPA, V-JEPA papers
- [ ] Understand what makes JEPA different from contrastive
- [ ] Study World Models literature (Dreamer, etc.)
- [ ] Define: What is "physics" in latent space?
- [ ] Define: How do we measure if latent space captures physics?

**Deliverable:** `autoresearch/JEPA_LITERATURE_NOTES.md`

### Stage 2: PoC Implementation (Day 2-3)

- [ ] Implement simple JEPA for time series:
  ```python
  class SimpleJEPA(nn.Module):
      def __init__(self):
          self.encoder = TransformerEncoder(...)
          self.predictor = MLP(...)
          self.target_encoder = EMA(self.encoder)

      def forward(self, x_past, x_future):
          z_past = self.encoder(x_past)
          z_pred = self.predictor(z_past)
          z_target = self.target_encoder(x_future).detach()
          loss = F.mse_loss(z_pred, z_target)
          return loss
  ```

- [ ] Implement baselines (raw prediction, autoencoder)
- [ ] Run on AURSAD Robot 1

**Deliverable:** `autoresearch/experiments/jepa_poc.py`

### Stage 3: PoC Evaluation (Day 3-4)

- [ ] Compare forecasting accuracy
- [ ] Linear probe evaluation
- [ ] Visualization of latent space
- [ ] Transfer test: Robot 1 → Robot 2

**Deliverable:** PoC results in `autoresearch/EXPERIMENT_LOG.md`

### Stage 4: Decision Gate

```
IF PoC passes:
    → Proceed to Stage 5
ELSE:
    → Document why it failed
    → Either abandon or pivot
```

### Stage 5: Full Implementation (Day 5-7)

- [ ] Add sparse graph learning to JEPA
- [ ] Implement proper masking strategies
- [ ] Multi-scale prediction (multiple horizons)
- [ ] Architecture search (encoder depth, predictor type)

### Stage 6: Full Evaluation (Day 8-10)

- [ ] Run on all datasets (ETT, C-MAPSS, AURSAD→Voraus)
- [ ] Ablation studies
- [ ] Comparison against all baselines
- [ ] Interpretability analysis

### Stage 7: Paper Writing (Day 11-14)

- [ ] Main results
- [ ] Ablations
- [ ] Analysis / insights
- [ ] Related work positioning

---

## Architecture Details

### Version 1: Vanilla Mechanical-JEPA

```python
class MechanicalJEPA(nn.Module):
    """
    JEPA for industrial sensor time series.
    Predicts future latent states, not raw sensors.
    """

    def __init__(
        self,
        n_sensors: int,
        d_model: int = 64,
        n_layers: int = 3,
        patch_size: int = 16,
        mask_ratio: float = 0.5,
        ema_decay: float = 0.996,
    ):
        super().__init__()

        # Patch embedding (like ViT)
        self.patch_embed = nn.Linear(patch_size * n_sensors, d_model)

        # Context encoder (processes visible patches)
        self.context_encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=4,
            n_layers=n_layers,
        )

        # Predictor (predicts masked patch representations)
        self.predictor = TransformerEncoder(
            d_model=d_model,
            n_heads=4,
            n_layers=2,  # Smaller than encoder
        )

        # Target encoder (EMA of context encoder)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.ema_decay = ema_decay
        self.mask_ratio = mask_ratio

    def forward(self, x):
        """
        x: [B, T, N_sensors] - full time series

        1. Patchify
        2. Mask some patches
        3. Encode visible patches (context encoder)
        4. Predict masked patches (predictor)
        5. Get target from full sequence (target encoder)
        6. Loss = MSE(predicted, target) for masked patches
        """
        B, T, N = x.shape

        # Patchify: [B, T, N] -> [B, N_patches, D]
        patches = self.patchify(x)

        # Generate mask
        mask = self.generate_mask(patches)  # True = masked

        # Visible patches
        visible = patches[~mask].reshape(B, -1, self.d_model)

        # Encode visible
        z_visible = self.context_encoder(visible)

        # Predict masked
        # (Add mask tokens, then predict)
        z_pred = self.predictor(z_visible, mask)

        # Target: encode full sequence with EMA encoder
        with torch.no_grad():
            self._update_ema()
            z_target = self.target_encoder(patches)

        # Loss: MSE on masked positions only
        loss = F.mse_loss(
            z_pred[mask],
            z_target[mask]
        )

        return loss

    def _update_ema(self):
        for p_online, p_target in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            p_target.data = (
                self.ema_decay * p_target.data +
                (1 - self.ema_decay) * p_online.data
            )
```

### Version 2: With Graph Learning

```python
class GraphMechanicalJEPA(MechanicalJEPA):
    """
    Adds learned sparse graph to capture sensor interactions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Graph learner
        self.graph_learner = SparseGraphLearner(
            n_sensors=self.n_sensors,
            d_model=self.d_model,
            sparsity=0.1,  # 10% of edges
        )

        # GNN for message passing
        self.gnn = GraphAttentionNetwork(
            d_model=self.d_model,
            n_layers=2,
        )

    def forward(self, x):
        # ... same as before, but add:

        # Learn graph structure
        adj = self.graph_learner(z_visible)

        # Apply GNN before prediction
        z_graph = self.gnn(z_visible, adj)

        # Predict using graph-enhanced features
        z_pred = self.predictor(z_graph, mask)

        # ... rest same
```

---

## Masking Strategies

### Option A: Random Patch Masking (I-JEPA style)
```
[P1][P2][P3][P4][P5][P6][P7][P8]
[  ][P2][  ][P4][  ][P6][  ][P8]  ← Visible (50%)
[P1][  ][P3][  ][P5][  ][P7][  ]  ← Predict
```

### Option B: Future Prediction (Autoregressive)
```
[P1][P2][P3][P4][P5][P6][P7][P8]
[P1][P2][P3][P4][  ][  ][  ][  ]  ← Visible (past)
[  ][  ][  ][  ][P5][P6][P7][P8]  ← Predict (future)
```

### Option C: Sensor Masking (Cross-sensor)
```
Sensors: [S1, S2, S3, S4, S5, S6]
Visible: [S1, S2, __, S4, __, S6]
Predict: [__, __, S3, __, S5, __]
```

**Recommendation:** Start with Option B (future prediction) — most relevant for forecasting.

---

## Evaluation Protocol

### 1. Forecasting Quality
```
Train JEPA (self-supervised)
Add linear head for forecasting
Evaluate MSE on held-out data
```

### 2. Transfer Learning
```
Train on Source (e.g., AURSAD)
Freeze encoder
Evaluate on Target (e.g., Voraus):
  a) Zero-shot (just the frozen encoder + new head)
  b) Fine-tuned (update encoder with target data)
  c) Compare to training from scratch on target
```

### 3. Representation Quality
```
Linear Probe:
  - Freeze encoder
  - Train linear classifier on latent space
  - Predict: anomaly/normal, operating mode, etc.

Downstream Tasks:
  - Anomaly detection
  - RUL prediction
  - Fault classification
```

### 4. Interpretability
```
Visualizations:
  - t-SNE of latent space (colored by physical state)
  - Attention maps (what does encoder focus on?)
  - Learned graph (if using graph version)

Physics Alignment:
  - Do similar physical states cluster?
  - Does the graph match known kinematics?
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| JEPA doesn't help forecasting | Medium | High | PoC gate early |
| Latent space doesn't capture physics | Medium | High | Linear probe evaluation |
| Transfer doesn't improve | Medium | High | Compare multiple transfer scenarios |
| Training instability | Low | Medium | EMA, careful hyperparams |
| Too slow to train | Low | Low | Start small, scale up |

---

## Timeline

| Day | Focus | Deliverable |
|-----|-------|-------------|
| 1 | Literature review | JEPA_LITERATURE_NOTES.md |
| 2-3 | PoC implementation | jepa_poc.py |
| 3-4 | PoC evaluation | PoC results |
| 4 | **Decision gate** | Go/no-go |
| 5-7 | Full implementation | Full model |
| 8-10 | Evaluation | Results |
| 11-14 | Paper | Draft |

---

## Success Looks Like

**For PoC:**
- JEPA matches raw prediction on forecasting
- Linear probe shows latent space captures physical states
- Transfer to Robot 2 improves 20%+

**For Full Paper:**
- Beat iTransformer on transfer tasks
- Interpretable latent space (visualizations)
- Competitive single-dataset performance
- Works across domains (robots, turbines, etc.)

**For NeurIPS:**
- Novel application of JEPA to industrial time series
- Strong empirical results on transfer
- Insight into what latent space learns
- Practical value for industrial applications
