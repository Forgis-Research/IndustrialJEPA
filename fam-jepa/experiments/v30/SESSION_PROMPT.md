# V30 Session - Monotone CDF, Fair Ablation, Uniform Benchmark

**Usage**: Paste as opening prompt to a Claude Code session on the GPU VM.
**Working directory**: `IndustrialJEPA/fam-jepa/`
**Duration**: OVERNIGHT (~20 hours on A10G). Use ALL available time.
**Prereqs**: Read CLAUDE.md, `fam-jepa/ARCHITECTURE.md`, `fam-jepa/model.py`,
`fam-jepa/train.py`, `experiments/RESULTS.md`, v28 + v29 SESSION_SUMMARY.md

---

## Architecture (unchanged from v29, except Phase 0 addition)

Use the v29 canonical architecture throughout:

| Component | Value | Justification |
|-----------|-------|---------------|
| Encoder | Causal transformer, d=256, L=2, H=4, 2.16M params | Unchanged since v17 |
| Target encoder | EMA (tau=0.99), bidirectional + attention pool | Default |
| Predictor | 2-layer MLP, hidden=256, 198K params | v29 ablation: transformer tied, MLP wins on parsimony |
| Event head | Discrete hazard CDF (K bins) | v26, zero monotonicity violations |
| Patch size | P=16 | Fixed globally, no dataset tuning |
| Pretraining loss | L1 on L2-normalized reps + variance regularizer | Default |
| Finetuning horizons | Sparse {1,5,10,20,50,100,150} or dataset-specific | Proven sufficient |
| Output | h_t = last token of causal encoder | Standard for causal transformers |

**Two dataset-dependent choices (independent of each other):**

| Dataset | Context | norm_mode | Signal type |
|---------|---------|-----------|-------------|
| C-MAPSS (FD001/FD002/FD003) | Full history | `none` | Slow sensor drift |
| SMAP/MSL/PSM/SMD | Sliding 512 | `revin` | Local anomaly patterns |
| MBA | Sliding 512 | `revin` | Waveform shape |
| GECCO/BATADAL | Sliding 512 | `revin` | Local anomaly patterns |
| SKAB | Sliding 512 | `revin` | Pressure/vibration patterns |
| ETTm1 | Sliding 512 | `revin` | Thermal shape |

**IMPORTANT**: `finetune_forward` returns CDF probabilities. Do NOT apply sigmoid.

---

## Evaluation framework (MANDATORY for every run)

Every experiment MUST report:

1. **Per-horizon AUROC + AUPRC** at every horizon
2. **Mean per-horizon AUROC** (h-AUROC) - the PRIMARY metric
3. **Pooled AUPRC with base-rate baseline** (for context, not headline)
4. **Prediction gap** `p_bar(y=1) - p_bar(y=0)` at shortest horizon
5. **Three-panel surface PNG** for every dataset x model:
   - Panel 1: predicted `p(t, Delta_t)` (viridis, 0-1)
   - Panel 2: ground truth `y(t, Delta_t)` (viridis, 0-1)
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
    print(f"  Pooled AUPRC: {pooled:.4f} (base: {base_pooled:.4f}, D={pooled-base_pooled:+.4f})")
    print(f"  [{tag}]")
```

Always save surfaces as .npz on the VM. Push only PNGs to git.

---

## Phase 0: Dense horizons + MonotoneCDF quick ablation (~1h)

### Default: Dense discrete hazard (K=150)

The sparse K=8 horizon bins from v27-v29 produce banding artifacts. The fix is simple: increase K to 150 (every integer horizon from 1 to 150). This was already proven in v28 dense runs. No architecture change needed.

Implementation: pass `horizons = list(range(1, 151))` to finetune. Everything else unchanged.

Run on FD001, 1 seed (42). Render 3-panel surface PNG. Verify banding is gone. This is the Phase 3 default unless the MonotoneCDF ablation below beats it.

### Quick ablation: MonotoneCDF (30 min max)

Replace the discrete hazard event head with a MonotoneCDF module that takes (predictor_output, Delta_t) and outputs p in [0,1] with architecturally enforced monotonicity in Delta_t (positive weights via softplus). This sits on top of the predictor output (Option A: predictor still runs per-horizon, MonotoneCDF replaces only the event head readout).

Run on FD001, 1 seed (42), same checkpoint. Compare h-AUROC and surface smoothness against dense discrete.

**Decision (spend at most 5 min on this):**
- If MonotoneCDF h-AUROC >= dense discrete AND visually smoother: adopt it, worth a sentence in the paper.
- Otherwise: use dense discrete K=150 for all subsequent phases. Move on.

Do NOT spend more than 1 hour total on Phase 0. The dense discrete baseline is the safe choice.

---

## Phase 1: Apples-to-apples ablation - FAM vs Chronos-2 (~2h)

### Goal

Determine the FAIR comparison before running the full benchmark. The current
comparison (FAM 198K pred-FT vs Chronos-2 769p linear probe) conflates
encoder quality with head capacity. This phase disentangles them.

### Prerequisites

Chronos-2 features must already be cached for FD001, FD003, MBA, and BATADAL.
Check `data/chronos2_features/` or wherever v28/v24 stored them. If missing,
extract features first using `baseline_chronos2.py` from v24.

### Design: 2x2 ablation

Run on FD001, FD003, MBA, BATADAL (3 seeds each):

| Variant | Encoder | Head | Trainable params |
|---------|---------|------|-----------------|
| **FAM-probe** | FAM 2.16M (frozen) | Linear(256, 1) per horizon | 257 per horizon |
| **Chr2-probe** | Chronos-2 120M (frozen) | Linear(768, 1) per horizon | 769 per horizon |
| **FAM-predft** | FAM 2.16M (frozen) | MLP predictor 198K (pretrained init) | 198K |
| **Chr2-mlp** | Chronos-2 120M (frozen) | MLP 198K (random init) | 198K |

Plus one additional comparison:
| **FAM-mlp-rand** | FAM 2.16M (frozen) | MLP 198K (random init) | 198K |

This tests whether FAM's advantage comes from (a) the pretrained predictor
initialization or (b) the encoder representations.

**Implementation of FAM-probe**:

```python
class LinearProbeHead(nn.Module):
    """Linear probe: one linear layer per horizon. Matches Chronos-2 protocol."""
    def __init__(self, d_model: int, n_horizons: int):
        super().__init__()
        self.probes = nn.ModuleList([
            nn.Linear(d_model, 1) for _ in range(n_horizons)
        ])

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        """h_t: (B, d). Returns: (B, K) logits."""
        return torch.cat([probe(h_t) for probe in self.probes], dim=-1)
```

Train with sigmoid + BCE. No hazard CDF - just independent per-horizon logits.
This is the same protocol used for Chronos-2 in v24.

**Implementation of Chr2-mlp**:

```python
# Load cached Chronos-2 features (B, 768) for each sample
# Feed through an MLP matching FAM's predictor architecture:
class Chr2MLP(nn.Module):
    def __init__(self, d_input=768, d_hidden=256, d_output=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input + 1, d_hidden),  # +1 for Delta_t
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),  # direct to logit
        )

    def forward(self, h, delta_t):
        dt = delta_t.float().unsqueeze(-1)
        return self.net(torch.cat([h, dt], dim=-1)).squeeze(-1)
```

### Label efficiency sub-experiment

For FAM-predft and FAM-mlp-rand only, also run at 10% labels on FD001
and MBA. This tests whether the pretrained predictor initialization provides
a sample efficiency advantage.

### Render

For ALL 5 variants on FD001: produce the 3-panel surface PNG (p, GT, |p-y|).
Include all 5 in a single comparison figure and in the quarto notebook.

### Decision gate

Evaluate the 2x2 grid. Save as `results/phase1_decision.json`:

```json
{
  "fam_probe_hauroc": {"FD001": X, "FD003": X, "MBA": X, "BATADAL": X},
  "chr2_probe_hauroc": {"FD001": X, "FD003": X, "MBA": X, "BATADAL": X},
  "fam_predft_hauroc": {"FD001": X, "FD003": X, "MBA": X, "BATADAL": X},
  "chr2_mlp_hauroc": {"FD001": X, "FD003": X, "MBA": X, "BATADAL": X},
  "fam_mlp_rand_hauroc": {"FD001": X, "FD003": X, "MBA": X, "BATADAL": X},
  "interpretation": "...",
  "main_table_variants": ["FAM-predft", "Chr2-probe"]
}
```

**Interpretation guide:**

- **FAM-probe > Chr2-probe**: FAM encoder produces better representations
  even with matched (minimal) head capacity. The encoder is the story.
  Keep pred-FT as the headline but report probe numbers too.

- **FAM-probe < Chr2-probe BUT FAM-predft > Chr2-mlp**: The MLP head
  architecture (not encoder quality) gives FAM the edge. Reframe the
  contribution around predictor finetuning as an inference protocol.

- **Chr2-mlp matches FAM-predft**: The encoders produce equivalent features;
  the original comparison was unfair (linear vs MLP). Add Chr2-mlp to the
  main table and be honest that the gap is smaller than originally claimed.

- **FAM-predft > FAM-mlp-rand** (especially at 10% labels): The pretrained
  predictor initialization is a genuine advantage. This supports the
  "predictor finetuning" contribution.

### Self-check (end of Phase 1)

Before proceeding: verify that all 5 variants were trained with EXACTLY the same data splits, horizons, labels, and loss function. Re-run one variant from a different random init to verify reproducibility within 0.01 h-AUROC. If results are not reproducible, investigate the source of variance before moving on.

---

## Phase 2: Precursor check - skip datasets without signal (~1h)

### Goal

For each dataset where signal has not been confirmed with 3 seeds, quickly
determine whether there is predictable precursor signal. Run a fast 1-seed
pretrain (if no checkpoint exists) + finetune (100% labels) with the chosen
recipe from Phase 0/1.

### Datasets to check

| Dataset | Status from v29 | Action |
|---------|----------------|--------|
| CHB-MIT | NULL confirmed (0.497 +/- 0.003, 3 seeds) | **Re-check with Phase 0 head ONLY if monotone CDF adopted.** Otherwise skip. |
| MSL | 0.438 (n=1), suspicious | Run 3 seeds with chosen recipe. |
| PhysioNet | Never run with h-AUROC metric | Run 1 seed. If > 0.55, run 2 more. |
| SMD | 0.616 (n=1) | Run 2 more seeds to get proper mean +/- std. |

### Datasets with confirmed signal (skip this check)

FD001, FD002, FD003, SMAP, PSM, MBA, SKAB, ETTm1, GECCO, BATADAL.

### Decision rules

- h-AUROC > 0.55: signal present, include in Phase 3
- h-AUROC 0.50-0.55: marginal - render surface, look for structure
- h-AUROC < 0.50: no precursor signal, SKIP. Document the null result.

For marginal cases, render the p-surface. If flat (base-rate predictor), skip.
If it shows any temporal structure (even weak), keep and note as marginal.

Save: `results/phase2_precursor_check.json`
```json
{
  "MSL": {"hauroc": [X, X, X], "mean": X, "decision": "include|skip"},
  "SMD": {"hauroc": [X, X, X], "mean": X, "decision": "include|skip"},
  "PhysioNet": {"hauroc": [X], "mean": X, "decision": "include|skip"},
  "CHBMIT": {"hauroc": [X, X, X], "decision": "skip (null confirmed v29)"}
}
```

---

## Phase 3: Uniform benchmark - all confirmed datasets, 3 seeds (~6h)

### Goal

Fill the main table (Table 4 in paper) with clean, uniform h-AUROC numbers.
This is the v30 highest priority deliverable. The v29 master table was
"best across v27-v29" with heterogeneous hyperparameters. Phase 3 replaces
it with a SINGLE uniform run.

### Protocol

For every dataset that passed the Phase 2 precursor check:
1. Use the chosen recipe from Phase 0 (monotone CDF or dense discrete)
2. Use the chosen comparison from Phase 1
3. 3 seeds (42, 43, 44) for both FAM and Chronos-2
4. Both 100% and 10% label budgets
5. Consistent hyperparameters across all datasets:

| Hyperparameter | Value | Note |
|----------------|-------|------|
| Pretrain epochs | 50 | Same for all |
| Pretrain lr | 1e-3 | AdamW |
| Pretrain weight decay | 0.01 | |
| Finetune epochs | 30 | Same for all |
| Finetune lr | 5e-4 | AdamW |
| Finetune pos_weight | auto (from train set class balance) | |
| Batch size | 256 | Reduce to 128 for CHB-MIT if OOM |
| Pretrain Delta_t_max | dataset-dependent (see table below) | |

**Dataset-specific pretrain Delta_t_max** (the ONLY dataset-specific setting):

| Dataset | Delta_t_max | Rationale |
|---------|-------------|-----------|
| FD001/FD002/FD003 | 150 | Lifecycle, units ~192 cycles |
| SMAP/MSL/PSM/SMD | 200 | Streaming, 512-step context |
| MBA | 200 | Streaming ECG |
| SKAB | 200 | Streaming hydraulic |
| ETTm1 | 200 | Streaming thermal |
| GECCO | 200 | Streaming water quality |
| BATADAL | 200 | Streaming SCADA |

### Re-use existing pretrained checkpoints

If a pretrained encoder checkpoint from v27-v29 exists for a dataset with
the SAME architecture (MLP predictor, same norm_mode), re-use it. Do NOT
re-pretrain from scratch. The only exception: if Phase 0 adopted monotone
CDF, pretraining is unchanged (only the finetuning head changes), so
existing checkpoints are still valid.

List existing checkpoints before starting:
```bash
ls checkpoints/*.pt
```

### Storage per dataset per seed

```
results/phase3/{dataset}/seed{N}/
    p_surface.npz        # (n_test_samples, K) predicted probabilities
    y_surface.npz        # (n_test_samples, K) ground truth labels
    metrics.json         # h-AUROC, AUPRC, per-horizon breakdown
    surface_panel.png    # 3-panel visualization
```

### Chronos-2 runs

For datasets WITH cached Chronos-2 features (FD001, FD002, FD003, SMAP, MSL,
PSM, MBA, GECCO, BATADAL), run the Phase 1-chosen head (probe or MLP) at
both label budgets, 3 seeds.

For datasets WITHOUT cached features (SKAB, ETTm1, SMD), either:
- Extract features first (if time permits, ~1h per dataset)
- Or mark as "Chronos-2: N/A" in the table

### Master table format

```
| Dataset | Domain | h-AUROC 100% FAM | h-AUROC 100% Chr2 | h-AUROC 10% FAM | h-AUROC 10% Chr2 | Legacy FAM | Legacy SOTA |
```

All numbers: `mean +/- std (3s)`. Bold best. Underline second-best.

---

### Self-check (end of Phase 3)

Before proceeding: for 3 randomly selected datasets, load the stored .npz surfaces and recompute h-AUROC from scratch. Verify it matches the reported number to 4 decimal places. Check that no surface has NaN or all-zero columns. If any discrepancy is found, re-run the affected dataset before moving to Phase 4.

---

## Phase 4: SOTA Research and Legacy Metrics (~1.5h)

### Phase 4a: SOTA Deep Research (~1h)

For EACH dataset in the benchmark, conduct deep research to find the current published SOTA result. This is critical for the paper's credibility.

**For each dataset:**
1. Use web search to find the current published SOTA result for the relevant metric (RMSE for C-MAPSS, F1 for anomaly datasets, AUROC for clinical)
2. Understand PRECISELY how the SOTA method computes their metric:
   - What preprocessing do they apply?
   - What evaluation protocol (train/val/test split, cross-validation)?
   - What scoring function (e.g., asymmetric scoring for C-MAPSS)?
   - What thresholding protocol (best-F1, fixed threshold, PA)?
3. Verify we are computing our legacy metric with the EXACT same protocol. If we are not, adapt our computation to match.
4. Self-check: run the SOTA's evaluation protocol on our stored surfaces. If results differ from our naive computation, investigate and fix.
5. Cite the SOTA paper properly (author, year, venue)

**Output**: A table in the quarto notebook (section 9.5) mapping:
```
| Dataset | Legacy Metric | SOTA Method | SOTA Value | Venue/Year | Their Protocol | Our Result (Their Protocol) | Our Result (Naive) | Protocol Match? |
```

If "Protocol Match?" is "No" for any dataset, the "Our Result (Their Protocol)" column is the number that goes in the paper.

### Phase 4b: Compute Legacy Metrics (~30min)

### Goal

Fill the legacy metric columns in Table 4 from stored .npz surfaces, using the protocols verified in Phase 4a.

### C-MAPSS (FD001, FD002, FD003): RMSE

Project the probability surface to a scalar RUL estimate:
```python
def surface_to_rul(p_surface, horizons):
    """Convert p(t, Delta_t) to RUL estimate via expected first-crossing."""
    # For each timestep t, find the expected horizon where p crosses 0.5
    # This is the predicted time-to-event
    rul = np.full(p_surface.shape[0], horizons[-1], dtype=float)
    for i, t in enumerate(range(p_surface.shape[0])):
        crossings = np.where(p_surface[i] >= 0.5)[0]
        if len(crossings) > 0:
            rul[i] = horizons[crossings[0]]
    return rul
```

Compare against actual RUL. Report RMSE. Compare to published SOTA
(typically 12-15 RMSE on FD001).

### Anomaly datasets (SMAP, MSL, PSM, SMD, SKAB, GECCO, BATADAL): F1

From the surface, derive a binary anomaly prediction:
```python
# Use p(t, Delta_t=1) (shortest horizon) as anomaly score
# Threshold at best F1 on validation set
# Report non-PA F1 on test set (NO point-adjust)
```

### Clinical (MBA): AUROC

Already computed as part of h-AUROC. Report the shortest-horizon AUROC
as the clinical AUROC metric.

### ETTm1: no standard legacy metric

Report h-AUROC only. This is a novel event-prediction framing.

### Output

Save: `results/phase4_legacy_metrics.json`

### Self-check (end of Phase 4)

For C-MAPSS FD001, manually verify the RMSE computation against the STAR paper's protocol. Check that we use the same RUL cap (125), the same scoring function, the same test set. If any discrepancy is found, recompute with the correct protocol and update the legacy metrics JSON.

---

## Phase 5: Figures and quarto notebook (~1.5h)

### 5a. Real Figure 4 (paper)

From FD001 Phase 3 surfaces (seed 42), render the publication-quality
figure for `paper-neurips/figures/fig_probability_surface_v2.pdf`:

Layout:
```
Row 1 (FAM):      [p(t,dt) viridis] [y(t,dt) viridis] [|p-y| gray]
Row 2 (Chronos-2): [p(t,dt) viridis] [y(t,dt) viridis] [|p-y| gray]
Panel (g):         [per-horizon AUROC curve, FAM blue, Chr2 orange]
```

Use matplotlib with:
- Figure size: (18, 8) inches
- Font size: 11pt (matches NeurIPS body text)
- Colorbar labels
- Axis labels: "Time step t" (x), "Horizon Delta_t" (y)
- No title (caption in paper handles this)
- Export as PDF (vector) + PNG at 300 DPI

### 5b. Updated bar chart (Figure 5)

h-AUROC across all datasets, FAM (blue) vs Chronos-2 (orange).
Grouped bar chart with error bars (std over 3 seeds).
Sort datasets by FAM h-AUROC descending.

Save as `paper-neurips/figures/fig_benchmark_hauroc.pdf`.

### 5c. Surface gallery

For EVERY dataset in the benchmark, render the 3-panel surface PNG.
These go in the quarto notebook, not the paper.

### 5d. Quarto notebook

Create `notebooks/30_v30_analysis.qmd` with sections:

1. **Executive summary** - key findings, decisions made
2. **Phase 0: Monotone CDF vs Discrete** - comparison table, surface
   comparison, decision rationale
3. **Phase 1: Fair ablation** - 2x2 table, interpretation, surfaces
   for all variants on FD001
4. **Phase 2: Precursor check** - which datasets passed/failed, surfaces
   for marginal cases
5. **Phase 3: Main results** - master table with all numbers
6. **Surface gallery** - every dataset, 3-panel PNGs, sorted by domain
7. **Per-horizon AUROC curves** - one plot per dataset showing AUROC
   as function of Delta_t
8. **Label efficiency** - 100% vs 10% comparison, does FAM degrade
   gracefully?
9. **Honest assessment** - what FAM does well, where it fails, open
   questions for v31
10. **New dataset scouting** - Phase 8 candidate analysis (if completed)

Render to HTML: `quarto render notebooks/30_v30_analysis.qmd`

---

## Phase 6: Theory self-check loop (~2h)

### Goal

Systematically verify and strengthen the theoretical results. Write ALL
findings to: `paper-neurips/theory_findings.tex` (NOT directly into
paper.tex or theory_main.tex, except for correcting errors).

Read `experiments/v30/THEORY_SELF_CHECK.md` for the detailed protocol.
It has three sub-phases:

**Phase 6a: Correctness audit (30 min)**
- Verify every step of Proposition 1 proof (DPI, tower property, Jensen
  gap, Lipschitz variance bound)
- Check each assumption (A1-A4): necessary? sufficient? holds for our data?
- Search for existing bounds on JEPA/predictive coding that overlap

**Phase 6b: Strength and relevance (45 min)**
- For each dataset: does the theory predict the observed h-AUROC?
  Fill the table in THEORY_SELF_CHECK.md mapping I(H*;E) and epsilon
  to observed performance
- Formalize predictor transfer result as a proposition
- Attempt sample complexity bound
- Derive architectural prescriptions from the bound

**Phase 6c: New results (45 min)**
- Attempt horizon-dependent bound (epsilon and L vary with Delta_t)
- Attempt connection between MonotoneCDF and Bayes-optimal predictor
  (IF monotone CDF was adopted in Phase 0)
- Attempt calibration guarantee
- Compile architecture design rules with formal justification

### Self-healing rules

- If a proof step is incorrect: FIX `theory_main.tex` and `theory_appendix.tex`
  immediately. This is the ONE exception to "don't write directly into paper .tex".
- If a stronger result is found: REPLACE the weaker version in theory_findings.tex
- If the theory fails to explain an empirical result: DOCUMENT the gap
- After any change to .tex files: verify compilation with `pdflatex paper.tex`

### Output

`paper-neurips/theory_findings.tex` - standalone file with:
- Audit results (pass/fail for each proof step)
- Assumption verification against empirical data
- New propositions (if any)
- Architecture prescriptions with formal justification
- Gaps between theory and experiment

---

## Phase 7: Commit and summary (~30min)

### Deliverables checklist

Before committing, verify ALL of these exist:

1. `experiments/RESULTS.md` - updated with ALL Phase 3 numbers (uniform benchmark)
2. `results/master_table.json` - machine-readable master table
3. `results/phase0_decision.json` - monotone CDF decision
4. `results/phase1_decision.json` - fair comparison decision
5. `results/phase2_precursor_check.json` - precursor signal check
6. `results/phase3/{dataset}/seed{N}/` - all surfaces and metrics
7. `results/phase4_legacy_metrics.json` - legacy metrics
8. `results/surface_pngs/` - all surface PNGs
9. `paper-neurips/figures/fig_probability_surface_v2.pdf` - real Figure 4
10. `paper-neurips/figures/fig_benchmark_hauroc.pdf` - updated Figure 5
11. `paper-neurips/theory_findings.tex` - theory self-check output
12. `notebooks/30_v30_analysis.qmd` + rendered HTML
13. `experiments/v30/SESSION_SUMMARY.md` - what shipped, what didn't, key decisions

### Commit convention

One commit per phase:
```
git commit -m "v30-p0: monotone CDF experiment on FD001/MBA"
git commit -m "v30-p1: 2x2 ablation FAM vs Chronos-2 (probe + MLP heads)"
git commit -m "v30-p2: precursor check for MSL, SMD, PhysioNet"
git commit -m "v30-p3: uniform benchmark, 13 datasets, 3 seeds"
git commit -m "v30-p4: legacy metrics from stored surfaces"
git commit -m "v30-p5: figures + quarto notebook"
git commit -m "v30-p6: theory self-check findings"
git commit -m "v30-p7: session summary + RESULTS.md update"
```

### SESSION_SUMMARY.md format

```markdown
# V30 Session Summary

**Date**: YYYY-MM-DD
**Duration**: ~Xh on A10G
**Scope**: [one-line]

## One-sentence verdict

[What is the honest headline from this session?]

## Decisions made

- Phase 0: [monotone CDF / discrete hazard] because [reason]
- Phase 1: [fair comparison = ...] because [reason]
- Phase 2: [which datasets passed/failed]

## Main results (Table 4 numbers)

[Paste the master table here]

## What shipped

[Bulleted list with specifics]

## What did not ship

[Bulleted list with honest reasons]

## Open questions for v31

[Bulleted list]
```

---

## Phase 8 (Stretch Goal): New Dataset Scouting (~2h)

### Goal

Identify 4 additional datasets with concrete forecasting events and published
SOTA results that would strengthen the paper's breadth claim. This phase runs
only if the main phases (0-7) finish early. Use remaining compute time here.

### For each candidate dataset, determine:

1. **Dataset name, source, size, sampling rate, number of channels**
2. **The specific EVENT to forecast** - not just anomaly detection, but a concrete forecasting target (e.g., "bearing failure within N cycles", "sepsis onset within 6 hours")
3. **Published SOTA method, metric, and value** - with proper citation (author, year, venue)
4. **Why it fits FAM** - multivariate, has precursors, reasonable context length for P=16
5. **Why it is DIFFERENT from existing datasets** - new domain, new event type, new temporal scale
6. **Quick feasibility check** - is the data publicly available? What preprocessing is needed? How long would a full integration take?

### Good candidates to investigate

- **NASA Bearing Dataset** - vibration, bearing failure prediction
- **FEMTO/PRONOSTIA** - bearing degradation, accelerated life tests
- **Tennessee Eastman Process** - chemical process fault prediction
- **MIMIC-III/IV** - clinical deterioration, sepsis onset, mortality
- **Backblaze Hard Drive** - disk failure prediction from SMART data
- **HAI 22.04** - industrial control system attacks
- **SWaT** - Secure Water Treatment, cyber-physical attacks

### Output

Include brief analysis of each candidate in the quarto notebook section 10.
Save structured results to `results/phase8_dataset_scouting.json`:

```json
{
  "candidates": [
    {
      "name": "...",
      "source": "...",
      "size": "...",
      "channels": N,
      "sampling_rate": "...",
      "event": "...",
      "sota_method": "...",
      "sota_metric": "...",
      "sota_value": X,
      "sota_citation": "Author et al., Venue Year",
      "fits_fam_because": "...",
      "different_because": "...",
      "publicly_available": true,
      "preprocessing_needed": "...",
      "integration_effort_hours": N,
      "recommendation": "include|skip|maybe"
    }
  ],
  "top_4_picks": ["...", "...", "...", "..."]
}
```

---

## Phase 9 (Stretch Goal): Second Foundation Model Baseline (~3h)

Goal: Repeat the entire Chronos-2 comparison protocol on a second open-source time series foundation model. This strengthens the results table by showing FAM's advantage is not specific to one baseline.

#### Step 1: Identify the strongest current open-source model (~15min)

Use web search to determine the current SOTA open-source time series foundation model. Candidates as of April 2026:
- TimesFM 2.5 (Google, ~200M params, point forecasting)
- Moirai 2.0 (Salesforce, universal forecasting)
- MOMENT (CMU, multi-task time series)
- Timer (Peking University, generative time series)
- Lag-Llama (Kashif Rasul et al., probabilistic forecasting)
- UniTS (unified time series model)
- Any newer model released after May 2025

Selection criteria:
1. Open-source with available weights (HuggingFace or similar)
2. Supports multivariate input
3. Large pretrained corpus (the "foundation model" property)
4. Published in a top venue or highly cited preprint
5. Produces embeddings that can be probed (frozen encoder protocol)

#### Step 2: Feature extraction (~1.5h)

Follow the EXACT same protocol used for Chronos-2:
1. Freeze the foundation model encoder
2. Extract embeddings for all test samples across all datasets
3. Cache embeddings to disk (one .pt file per dataset)
4. Document the exact model checkpoint, version, and any preprocessing

Important: match the input format to what the model expects. Some models expect univariate channels separately; others expect multivariate input. Document any adaptations.

#### Step 3: Probe training and evaluation (~1h)

Use the SAME head variants from the Phase 1 ablation:
- Linear probe (d_model -> 1 per horizon)
- MLP head (198K params, random init)

Train with the same labels, horizons, splits, loss, and seeds as FAM and Chronos-2. This ensures a three-way apples-to-apples comparison.

For each dataset with cached features, report h-AUROC at 100% and 10% labels.

#### Step 4: Surface rendering and comparison (~30min)

Produce 3-panel surface PNGs for the second baseline on all evaluated datasets. Add to the quarto notebook as a separate section.

#### Output

An appendix table (for paper-neurips/paper.tex) comparing all three models:

| Dataset | FAM (2.16M) | Chronos-2 (120M) | [Model X] (YM) |
|---------|-------------|-------------------|-----------------|

This goes in the appendix, not the main table. The main table stays FAM vs Chronos-2 (the most recognizable baseline). The three-way comparison strengthens the appendix.

Save features to: `data/[model_name]_features/`
Save results to: `results/phase9_[model_name]/`

#### Self-check

Verify that the chosen model's embeddings are non-trivial: compute the variance of embeddings across samples. If variance is near-zero on any dataset (collapsed representations), flag it and investigate. Do not report results from collapsed embeddings.

---

## Important rules

1. **Decision gates are HARD gates.** Do NOT proceed to Phase 3 before
   Phase 0 and Phase 1 decisions are finalized. The choice of head
   architecture and comparison method determines everything downstream.

2. **Always render surfaces.** Every experiment must produce a 3-panel PNG.
   No result is accepted without visual inspection. If a surface looks wrong
   (flat, inverted, banded unexpectedly), investigate before moving on.

3. **Honest reporting.** If FAM loses to Chronos-2, report it. If the
   probe ablation shows the original comparison was unfair, report it.
   If monotone CDF is worse, report it. The paper's credibility depends
   on honesty. Never cherry-pick seeds or variants.

4. **h-AUROC is primary.** All tables and comparisons use mean per-horizon
   AUROC. AUPRC and legacy metrics are secondary (reported for completeness
   and literature comparison only).

5. **Theory in separate file.** The theory self-check writes to
   `theory_findings.tex`, NOT directly into paper.tex or theory_main.tex.
   The only exception: correcting actual errors in existing proofs.

6. **Uniform benchmark = uniform hyperparameters.** Phase 3 must use the
   SAME hyperparameters across all datasets. The only dataset-specific
   setting is Delta_t_max for pretraining (which determines the temporal
   scale). If a dataset needs special treatment, document why.

7. **3 seeds minimum.** No result enters the paper with fewer than 3 seeds.
   Report mean +/- std. For key comparisons, run a paired t-test.

8. **Surface storage.** Every finetuning run stores p_surface.npz and
   y_surface.npz. All downstream metrics are recomputable from these.

9. **IMPORTANT**: `finetune_forward` returns CDF probabilities already in
   [0, 1]. Do NOT apply sigmoid on top. If using monotone CDF, the output
   is also already in [0, 1].

10. **Time management.** Phases 0-1 are ~4h total. Phase 3 is ~6h. If
    Phase 0/1 decisions are clear early, start Phase 3 immediately. Phase 6
    (theory) can run in parallel with Phase 5 (figures) if you manage
    the files carefully. Do not spend more than 30min on any single
    decision gate.

11. **DO NOT STOP EARLY.** Use ALL available compute time. If you finish
    the main phases early, work on the stretch goal (Phase 8), deepen the
    theory self-check, or re-run marginal datasets with more seeds.

12. **COMMIT HOURLY.** After every hour of work, commit progress with a
    descriptive message. This protects against session crashes and provides
    a clear audit trail. Use the pattern:
    `git commit -m "v30 hourly: [what was done in the last hour]"`.

13. **USE ALL NECESSARY AGENTS.** Launch ml-researcher for SOTA deep
    research (Phase 4a). Launch data-curator for new dataset scouting
    (Phase 8). Use parallel agents when tasks are independent. Do not try
    to do everything sequentially when agents can work in parallel.

14. **PRECURSOR CHECK IS FAST.** If a 1-seed run shows h-AUROC < 0.50
    within the first epoch of finetuning, the dataset has no signal at
    this temporal resolution. Stop the run, document it, move on. Do not
    waste 30 minutes on a null result that is obvious after 2 minutes.

15. **APPLES-TO-APPLES IS NON-NEGOTIABLE.** Every foundation model comparison must use the exact same downstream protocol: same labels, same horizons, same splits, same loss function, same seeds. The ONLY thing that changes is the frozen encoder. If a model requires different preprocessing (e.g. univariate-only input), document the adaptation clearly and flag it as a caveat.
