# V29 Session — 3 New Datasets + Transformer Predictor Ablation

**Usage**: Paste as opening prompt to a Claude Code session on the GPU VM.
**Working directory**: `IndustrialJEPA/fam-jepa/`
**Duration**: OVERNIGHT (10-12 hours on A10G). Use ALL available time.
**Prereqs**: Read CLAUDE.md, `fam-jepa/ARCHITECTURE.md`, `fam-jepa/model.py`,
`fam-jepa/train.py`, `experiments/RESULTS.md`, v27 + v28 SESSION_SUMMARY.md

---

## Architecture (clean, minimal arbitrary choices)

Use the v27 canonical architecture throughout:

| Component | Value | Justification |
|-----------|-------|---------------|
| Encoder | Causal transformer, d=256, L=2, H=4 | Unchanged since v17 |
| Target encoder | EMA (τ=0.99), bidirectional + attention pool | Default |
| Predictor | 2-layer MLP, hidden=256, 198K params | v24 ablation |
| Event head | LayerNorm + Linear(256→1), hazard CDF | v26, zero violations |
| Patch size | P=16 | PatchTST standard; adaptive tokenization is future work |
| Pretraining loss | L1 on L2-normalized reps + variance regularizer | Default |
| Finetuning horizons | Sparse {1,5,10,20,50,100,150} (streaming) or {1,5,10,20,50,100,150,200} | Proven sufficient |
| Output | h_t = last token of causal encoder | Standard for causal transformers |

**Two dataset-dependent choices (independent of each other):**

| Choice | Criterion | Options |
|--------|-----------|---------|
| **Context length** | Does the entity have a known start? | Full history (lifecycle) / Sliding 512 (streaming) |
| **Normalization** | Is the predictive signal in drift or shape? | `none` + global z-score (drift) / `revin` (shape) |

These happen to correlate in our datasets (lifecycle = drift, streaming = shape)
but are logically independent. A streaming sensor with slow drift (e.g., ETTm
seasonal trend) might need `none`; a lifecycle entity with abrupt shape-based
faults might need `revin`.

| Dataset | Context | norm_mode | Signal type |
|---------|---------|-----------|-------------|
| C-MAPSS | Full history | `none` | Slow sensor drift |
| SMAP/MSL/PSM/SMD | Sliding 512 | `revin` | Local anomaly patterns |
| MBA | Sliding 512 | `revin` | Waveform shape |
| GECCO/BATADAL | Sliding 512 | `revin` | Local anomaly patterns |
| CHB-MIT | Sliding 512 | `revin` | Preictal spectral changes (shape) |
| SKAB | Sliding 512 | `revin` | Pressure/vibration patterns |
| ETTm1 | Sliding 512 | `revin` | Thermal shape (try `none` too if drift suspected) |

**IMPORTANT**: `finetune_forward` returns CDF probabilities. Do NOT apply sigmoid.

---

## Evaluation framework (MANDATORY for every run)

Every experiment MUST report:

1. **Per-horizon AUROC + AUPRC** at every horizon
2. **Mean per-horizon AUROC** (honest primary metric)
3. **Pooled AUPRC with base-rate baseline** (for context, not as headline)
4. **Prediction gap** `p̄(y=1) - p̄(y=0)` at shortest horizon
5. **Three-panel surface PNG** for every dataset × model:
   - Panel 1: predicted `p(t, Δt)` (viridis, 0-1)
   - Panel 2: ground truth `y(t, Δt)` (viridis, 0-1)
   - Panel 3: error `|p - y|` (grayscale, 0-1, with mean |p-y| in title)
   All panels: linear y-axis, same scale. Save to `results/surface_pngs/`.

```python
def report_surface(p_surface, y_surface, horizons, tag=""):
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score

    valid = [i for i in range(len(horizons)) if 0 < y_surface[:, i].mean() < 1]

    for i in valid:
        auroc = roc_auc_score(y_surface[:,i], p_surface[:,i])
        auprc = average_precision_score(y_surface[:,i], p_surface[:,i])
        gap = p_surface[y_surface[:,i]==1, i].mean() - p_surface[y_surface[:,i]==0, i].mean()
        print(f"  dt={horizons[i]:>3}: AUROC={auroc:.3f}  AUPRC={auprc:.3f}  "
              f"gap={gap:+.3f}  pos={y_surface[:,i].mean():.3f}")

    mean_auroc = np.mean([roc_auc_score(y_surface[:,i], p_surface[:,i]) for i in valid])

    base_rates = y_surface.mean(axis=0)
    rng = np.random.RandomState(0)
    p_base = np.tile(base_rates, (y_surface.shape[0], 1)) + rng.normal(0, 1e-6, y_surface.shape)
    pooled = average_precision_score(y_surface.ravel(), p_surface.ravel())
    base_pooled = average_precision_score(y_surface.ravel(), p_base.ravel())

    print(f"  Mean h-AUROC: {mean_auroc:.4f}")
    print(f"  Pooled AUPRC: {pooled:.4f} (base: {base_pooled:.4f}, Δ={pooled-base_pooled:+.4f})")
    print(f"  [{tag}]")
```

Always save surfaces as .npz on the VM. Push only PNGs to git.

---

## Phase 1: Three new datasets (3 h)

### 1a. CHB-MIT Scalp EEG — seizure prediction

- **Domain**: Clinical neurology — pediatric epilepsy
- **Source**: https://physionet.org/content/chbmit/1.0.0/ (no registration)
- **Download**: `wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/`
- **Channels**: 18 EEG channels (10-20 system, use the 18 common across subjects)
- **Rate**: 256 Hz → **downsample to 32 Hz** (`mne.io.read_raw_edf`, then `resample(32)`)
- **P=16 at 32 Hz**: each token covers 0.5s — fine for EEG dynamics
- **Context**: Sliding 512 steps = 16 seconds of EEG
- **norm_mode**: `revin` (per-patient variability is extreme)
- **Event**: Seizure onset. Preictal label: y=1 for 30 min before onset.
  4-hour buffer after seizure offset (exclude from training).
- **Why it matters**: Documented preictal dynamics (spectral shifts, synchrony
  changes). 30-90 min lead time. Genuine prediction task — the precursor is
  visible BEFORE the event.
- **Horizons**: {32, 160, 320, 960, 1920, 3840, 9600} steps
  (= {1s, 5s, 10s, 30s, 60s, 120s, 300s} at 32 Hz)
- **SOTA**: Sensitivity 92.8% at FPR 0.06/h (Ozcan & Bhatt 2021)
- **Protocol**: Per-subject. Pick 3 subjects with ≥5 seizures.
  Leave-one-seizure-out for test, rest for train+val.
- **Loader**: Write `data/chbmit.py`. Parse `chb*/chb*-summary.txt` for
  seizure times. Use `mne` to read .edf files.

### 1b. SKAB (Skoltech Anomaly Benchmark) — hydraulic faults

- **Domain**: Hydraulic test rig
- **Source**: https://github.com/waico/SKAB (direct download, no registration)
- **Channels**: 8 (flow rate, pressure, temperature, vibration, pump RPM,
  motor current)
- **Rate**: 1 Hz
- **P=16 at 1 Hz**: each token covers 16s
- **Context**: Sliding 512 steps = 8.5 minutes
- **norm_mode**: `revin`
- **Event**: Valve blockage/leakage, pump failure. Precursors: pressure
  differential and vibration shift 30-120s before anomaly onset.
- **Horizons**: {1, 5, 10, 20, 50, 100, 150, 200}
- **SOTA**: SKAB leaderboard best NAB-score ~0.7
- **Protocol**: Use the labeled experiments. 60/20/20 split by experiment.
- **Loader**: Write `data/skab.py`. CSV files, column headers are sensor names,
  `anomaly` column is binary label.

### 1c. ETTm1 (Electricity Transformer Temperature) — thermal events

- **Domain**: Power grid / transformer monitoring
- **Source**: https://github.com/zhouhaoyi/ETDataset (direct download)
- **Channels**: 7 (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT)
  OT = oil temperature (target), others = load features
- **Rate**: 1/min (ETTm1, the minute-resolution variant)
- **P=16 at 1/min**: each token covers 16 min
- **Context**: Sliding 512 steps = 8.5 hours
- **norm_mode**: `revin`
- **Event**: Oil temperature exceeding operational threshold. Define event as
  OT > mean(OT_train) + 2*std(OT_train). This is a derived label — document
  the threshold.
- **Horizons**: {1, 5, 10, 20, 50, 100, 150, 200}
- **SOTA**: No event prediction SOTA (dataset is used for forecasting).
  We are the first to frame it as event prediction.
- **Protocol**: Standard ETT train/val/test split (60/20/20 chronological).
- **Loader**: Write `data/ettm.py`. Single CSV file, straightforward.

**For each new dataset:**
1. Write data loader in `data/` following existing patterns
2. Pretrain FAM encoder (self-supervised, no labels)
3. Pred-FT with event labels, 3 seeds
4. Run Chronos-2 baseline (same protocol as v24 `baseline_chronos2.py`)
5. Store surfaces, render PNGs, report full diagnostic
6. Compare to published SOTA on their metric

---

## Phase 2: Predictor architecture ablation (2.5 h)

### Context: how JEPA papers do it

In I-JEPA (Assran+ CVPR 2023) and V-JEPA (Bardes+ 2024), the predictor
is a **narrow transformer operating on ALL encoder tokens** — not an MLP
on a pooled summary. Key design: encoder d=1280, predictor d=384 (narrow
bottleneck). Target positions are communicated via learnable mask tokens
with positional embeddings appended to the encoder output sequence.

Our FAM deviates in two ways:
1. We pass only the last encoder token (h_t) to a 2-layer MLP
2. We KEEP and finetune the predictor (JEPA papers discard it)

Deviation #2 is our contribution (predictor finetuning). Deviation #1
may be losing information. This ablation tests whether aligning with
canonical JEPA predictor design (narrow transformer over all tokens) helps.

### Three variants to test

Run all three on FD001 (norm_mode='none'), FD003 (norm_mode='none'),
and MBA (norm_mode='revin'), 3 seeds each:

**Variant A: MLP on last token (current baseline)**
```python
# No change. h_t = encoder(x)[:, -1, :] → MLP(cat(h_t, Δt)) → ĥ
# This is what we have. 198K params.
```

**Variant B: MLP on mean-pooled tokens (cheap control)**
```python
# Mean-pool all encoder tokens instead of taking last
h_mean = encoder(x, return_all=True).mean(dim=1)  # (B, d)
h_hat = mlp_predictor(cat(h_mean, Δt))             # same MLP as variant A
```
Same 198K params. Tests whether last-token compression is the bottleneck
(if B beats A, the issue is information loss in last-token, not predictor
architecture).

**Variant C: Narrow transformer on all tokens (canonical JEPA design)**
```python
class TransformerPredictor(nn.Module):
    """Narrow transformer over all encoder tokens + Δt query token.
    Follows I-JEPA/V-JEPA predictor design."""

    def __init__(self, d_model=256, n_heads=4, n_layers=1):
        super().__init__()
        self.dt_embed = nn.Sequential(nn.Linear(1, d_model), nn.GELU())
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model, dropout=0.0,
            activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, h_all, delta_t):
        """h_all: (B, N, d) — ALL encoder tokens. delta_t: (B,)."""
        dt_tok = self.dt_embed(delta_t.float().unsqueeze(-1))  # (B, d)
        tokens = torch.cat([h_all, dt_tok.unsqueeze(1)], dim=1)  # (B, N+1, d)
        out = self.transformer(tokens)       # (B, N+1, d)
        return self.out_proj(out[:, -1])     # Δt query attends to all h
```
~200K params (matched). The Δt token appended last acts as a learned
query: "given these N token representations and this horizon, predict
the future." This mirrors I-JEPA's mask tokens with positional embeddings,
but conditioned on Δt instead of spatial position.

**Interface change**: Add `return_all` flag to `CausalEncoder.forward()`:
```python
def forward(self, x, mask=None, return_all=False):
    ...
    if return_all:
        return h  # (B, N, d)
    return h[:, -1, :]  # (B, d)
```
Variant A uses `return_all=False`. Variants B and C use `return_all=True`.

**Note on pretraining**: During pretraining the predictor must also use
the same input (all tokens for C, last token for A). This means variants
B and C need a full pretrain run — they cannot reuse the variant A
checkpoint. Budget accordingly.

### What the results tell us

| If B ≈ A, C ≈ A | Last-token compression is fine. MLP is fine. No change needed. |
|---|---|
| If B > A, C ≈ B | Last-token loses info. Mean-pool recovers it. No need for transformer predictor — just change pooling. |
| If C > B > A | Both compression AND predictor expressivity matter. The transformer over all tokens is worth the complexity. |
| If C > A, B ≈ A | The transformer predictor adds expressivity beyond what mean-pooling gives. The sequence structure matters, not just the average. |

### What to measure

| Metric | A: MLP last | B: MLP mean | C: Transformer all |
|--------|-------------|-------------|---------------------|
| FD001 mean h-AUROC | 0.72 | ? | ? |
| MBA mean h-AUROC | 0.58 | ? | ? |
| FD003 mean h-AUROC | 0.85 | ? | ? |
| Training time | baseline | ? | ? |
| Per-horizon correlation matrix | high corr across horizons? | lower corr (more horizon-specific)? | |

If the transformer predictor produces lower inter-horizon correlation
(more diverse per-horizon representations), that's evidence it conditions
on Δt more effectively than the MLP.

Run on FD001, FD003, MBA — 3 seeds each.

---

## Phase 3: Full benchmark with clean architecture (3 h)

Run ALL datasets with the v27 clean architecture + the better predictor
(MLP or transformer, whichever wins Phase 2):

| Dataset | Type | norm_mode | Context | Horizons | Est. |
|---------|------|-----------|---------|----------|------|
| FD001 | Lifecycle | none | Full history | {1,5,10,20,50,100,150} | 15 min |
| FD002 | Lifecycle | none | Full history | {1,5,10,20,50,100,150} | 15 min |
| FD003 | Lifecycle | none | Full history | {1,5,10,20,50,100,150} | 15 min |
| SMAP | Streaming | revin | Sliding 512 | {1,5,10,20,50,100,150,200} | 20 min |
| MSL | Streaming | revin | Sliding 512 | {1,5,10,20,50,100,150,200} | 20 min |
| PSM | Streaming | revin | Sliding 512 | {1,5,10,20,50,100,150,200} | 15 min |
| SMD | Streaming | revin | Sliding 512 | {1,5,10,20,50,100,150,200} | 30 min |
| MBA | Streaming | revin | Sliding 512 | {1,5,10,20,50,100,150,200} | 15 min |
| GECCO | Streaming | revin | Sliding 512 | {1,5,10,20,50,100,150,200} | 15 min |
| BATADAL | Streaming | revin | Sliding 512 | {1,5,10,20,50,100,150,200} | 15 min |
| CHB-MIT | Streaming | revin | Sliding 512 | {32,160,320,960,1920,3840,9600} | 30 min |
| SKAB | Streaming | revin | Sliding 512 | {1,5,10,20,50,100,150,200} | 15 min |
| ETTm1 | Streaming | revin | Sliding 512 | {1,5,10,20,50,100,150,200} | 15 min |

3 seeds each. Chronos-2 on all new datasets (reuse v24 pattern).

---

## Phase 4: Surface PNGs (1 h)

For EVERY dataset, render two rows:

**Row 1: FAM**
```
[FAM p(t,Δt)] | [ground truth y(t,Δt)] | [|FAM - y| error, grayscale, mean=X.XXX]
```

**Row 2: Chronos-2**
```
[Chronos-2 p(t,Δt)] | [ground truth y(t,Δt)] | [|Chronos-2 - y| error, grayscale, mean=X.XXX]
```

Linear y-axis. Viridis for p and y (0-1). Grayscale for error (0-1).
Mean |p-y| in the error panel title — this is the single most readable
diagnostic (lower = better, interpretable as average probability error).

Save to `experiments/v29/results/surface_pngs/`.
Use **figure-creator agent** for the 3 best examples (paper figures).

---

## Phase 5: Quarto analysis notebook (1.5 h)

Create `notebooks/29_v29_analysis.qmd`. Use **data-curator agent**.

Sections:
1. **New dataset characterization** — event frequency, precursor analysis,
   class imbalance
2. **Transformer vs MLP predictor** — per-horizon AUROC comparison,
   inter-horizon correlation matrices, verdict
3. **Master comparison table** — all 13 datasets × FAM + Chronos-2:
   mean h-AUROC, pooled AUPRC (with base rate), SOTA comparison
4. **Surface gallery** — all triplets with brief annotation
5. **Detection vs prediction** — onset analysis for CHB-MIT (does the
   model detect the preictal state with lead time?)
6. **Honest assessment** — what works, what doesn't, open problems

Render: `quarto render notebooks/29_v29_analysis.qmd`

---

## Phase 6: Self-check + RESULTS.md (30 min)

Launch **ml-researcher agent** to review:
- Are claims supported?
- Are comparisons fair?
- Statistical concerns?

Update RESULTS.md. Commit and push.

---

## Phase priorities

| Phase | What | Est. | Priority |
|-------|------|------|----------|
| 1 | 3 new datasets (data loading + FAM + Chronos-2) | 3 h | Critical |
| 2 | Transformer predictor ablation | 2.5 h | Critical |
| 3 | Full 13-dataset benchmark | 3 h | Critical |
| 4 | Surface PNGs | 1 h | Important |
| 5 | Quarto notebook | 1.5 h | Critical |
| 6 | Self-check + RESULTS.md | 30 min | Always |

**Total**: ~11.5h.

---

## Ground rules

1. **Import from model.py and train.py.** Do NOT copy model code.
2. **finetune_forward returns CDF probabilities.** Do NOT apply sigmoid.
3. **P=16 everywhere.** No exceptions.
4. **norm_mode**: `none` for lifecycle, `revin` for streaming.
5. Store surfaces as .npz ON THE VM. Push only PNGs.
6. **PRIMARY METRIC**: mean per-horizon AUROC.
7. **ALWAYS** render surface PNGs and report full diagnostic for every run.
8. Use agents: **data-curator** (notebook), **figure-creator** (plots),
   **ml-researcher** (self-check).
9. Commit + push after each phase. Update RESULTS.md.
10. **Stay transparent.** Report failures honestly.
