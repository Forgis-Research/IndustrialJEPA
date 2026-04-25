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

**Normalization (the one dataset-dependent choice):**

| Data type | norm_mode | Context | Rationale |
|-----------|-----------|---------|-----------|
| Lifecycle (entity born → fails) | `none` + global z-score | Full history | Drift IS the signal |
| Streaming (continuous monitoring) | `revin` | Sliding 512 | Cross-entity scale varies; shape IS the signal |

This is a structural decision from the data description, not a tuned hyperparameter.

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

## Phase 2: Transformer predictor ablation (2.5 h)

Replace the 2-layer MLP predictor with a small transformer predictor.
Test on FD001 (norm_mode='none') and MBA (norm_mode='revin'), 3 seeds each.

### Architecture: full-sequence transformer predictor

The current MLP predictor sees only h_t (last encoder token) — a single
256-d vector. It throws away 31 of 32 encoder outputs. A transformer
predictor can attend over ALL encoder tokens:

```python
class TransformerPredictor(nn.Module):
    """Transformer predictor: attends over all encoder tokens + Δt query."""

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
        # Append Δt as query token after all encoder tokens
        tokens = torch.cat([h_all, dt_tok.unsqueeze(1)], dim=1)  # (B, N+1, d)
        out = self.transformer(tokens)       # (B, N+1, d)
        return self.out_proj(out[:, -1])     # Δt token attends to all h
```

The Δt token is appended last and attends to all encoder positions — it
acts as a learned query: "given these N representations and this horizon,
what will the future look like?" Output from the Δt position.

**Interface change required**: The encoder currently returns `h_t` (last
token only). For this ablation, it must return `h_all` (all tokens).
Add a flag to `CausalEncoder.forward()`:
```python
def forward(self, x, mask=None, return_all=False):
    ...
    if return_all:
        return h  # (B, N, d) — all tokens
    return h[:, -1, :]  # (B, d) — last token only
```
The MLP predictor uses `return_all=False` (unchanged). The transformer
predictor uses `return_all=True`.

**Param count**: ~200K (1-layer, d=256, H=4, d_ff=256). Matched to MLP.

**Why this might help**: The MLP sees one compressed summary. The
transformer sees the full sequence of encoder states — it can detect
the drift *gradient* across positions (how fast sensors change), local
anomalies in specific positions, and multi-scale patterns. For FD003
(two fault modes), distinct spatial patterns across the token sequence
might be visible to the transformer but collapsed by the last-token
bottleneck.

**Why it might not help**: The causal encoder's last token has already
attended to all earlier tokens via 2 layers of self-attention. In
theory h_t contains everything. Whether 2 layers at d=256 is enough
to losslessly compress 32 tokens is the empirical question this
ablation answers.

### What to measure

| Metric | MLP (v27 baseline) | Transformer predictor | Δ |
|--------|--------------------|-----------------------|---|
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
