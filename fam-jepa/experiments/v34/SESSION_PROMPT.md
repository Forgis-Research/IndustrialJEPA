# V34 Overnight Session -- SIGReg, ST-JEPA Fixes, New Datasets, Paper Polish

**Duration**: Full overnight (~10-12 hours). Do NOT stop early.
**Goal**: Five parallel workstreams: (A) Replace EMA with SIGReg from scratch, (B) fix ST-JEPA collapse, (C) add 3 new datasets, (D) paper review+polish loop, (E) figure refinement. If SIGReg produces clean results, rerun ALL datasets.
**Commit cadence**: `git add -A && git commit -m "v34 phase X: <description>" && git push` after each phase. At minimum once per hour.
**Self-check**: After each phase, verify outputs exist and are reasonable. If a result looks wrong, investigate before moving on.
**Codex will review** your work after you finish -- be thorough and document everything.

---

## CRITICAL CONTEXT: What We Know

### SIGReg History in This Repo
- **v15**: SIGReg (bidirectional shared encoder, no EMA) got RMSE 9.16 vs EMA 17.03 on FD001. SIGReg WON.
- **v17**: Curriculum EMA->SIGReg. SIGReg-pred variant (regularize predictor output) got RMSE 15.38 vs EMA 17.81. SIGReg-pred WON.
- **v20**: SIGReg-pred got F1w 0.451 vs EMA 0.391 (+15%). SIGReg-pred WON on legacy metric.
- **v23**: SIGReg (no EMA, VICReg var+cov) got AUPRC 0.878 vs EMA 0.951 on FD001. EMA WON on AUPRC.
- **Paper Table A.8**: SIGReg-pred improves F1w by +0.06 and RMSE by -3.2 over EMA on C-MAPSS.

Mixed results: SIGReg wins on RMSE/F1w but loses on AUPRC. The AUPRC evaluation (v23) used the modern pipeline; the wins used legacy metrics. This session must resolve this using the current AUPRC+h-AUROC evaluation.

### V33 ST-JEPA Collapse -- Root Causes
Per-channel tokenization collapsed within epoch 1 because of three interacting flaws:
1. **Low input dimensionality**: Linear(P=16, d=256) per channel -- 16 values projected to 256 dims, very low rank
2. **Shared projection after RevIN**: channels become statistically near-identical after normalization
3. **Pooled single-vector target**: predictor satisfies loss with a near-constant output

V-JEPA doesn't collapse because: (a) patches are 768-dim not 16-dim, (b) per-position prediction targets not pooled, (c) visual patches are inherently diverse.

### Existing Code Locations
- VICReg var+cov loss: `experiments/archive/v23/pretrain_sigreg.py` (lines 137-158)
- SIGReg moments+EP: `archive/mechanical-jepa-legacy/sigreg.py`
- Current model: `model.py` (FAM class, CausalEncoder, TargetEncoder, Predictor)
- Training: `train.py` (PretrainDataset, pretrain(), finetune(), evaluate())
- Data loaders: `from _runner_v29 import LOADERS`
- v33 runner (reference): `experiments/v33/_runner_v33.py`

---

## Workstream A: SIGReg from Scratch (TOP PRIORITY, ~4-5 hours)

### Goal
Completely replace EMA target encoder with SIGReg. If this produces equal or better h-AUROC across all datasets, it simplifies the architecture (no EMA, no target encoder momentum schedule) and could improve numbers.

### A1: Architecture Design

The target encoder stays but EMA is removed. Stop-gradient replaces EMA:

```python
class FAM_SIGReg(FAM):
    """FAM with SIGReg instead of EMA."""
    
    def pretrain_forward(self, context, target, delta_t,
                         context_mask=None, target_mask=None):
        # Context encoder (causal) -- has gradients
        h_t = self.encoder(context, context_mask)
        h_pred = self.predictor(h_t, delta_t)
        
        # Target encoder -- STOP GRADIENT, no EMA
        # Use current target_encoder weights directly (they were init'd from encoder)
        with torch.no_grad():
            h_target = self.target_encoder(target, target_mask)
        
        h_pred_n = F.normalize(h_pred, dim=-1)
        h_target_n = F.normalize(h_target, dim=-1)
        return h_pred_n, h_target_n, h_pred  # return unnormalized h_pred for SIGReg
    
    def update_ema(self):
        """NO-OP: SIGReg doesn't use EMA."""
        pass
    
    def sync_target(self):
        """Hard-copy encoder weights to target encoder (matching keys only)."""
        self._init_target_encoder()
```

**Key question: how does target encoder get updated without EMA?**

Three options to test:

**Option 1 -- Periodic hard sync**: Every N steps, copy encoder weights to target encoder. This is like EMA with momentum=0 applied periodically. Start with N=100 steps (tune).

**Option 2 -- Target encoder trains with its own gradients**: Remove stop-gradient on target, let both encoders train jointly. SIGReg prevents collapse. This is the most radical change.

**Option 3 -- Target encoder stays frozen at init**: Never update the target encoder after initialization. The context encoder + predictor must learn to map to a fixed target representation space. This is the simplest and most likely to work for the FIRST experiment.

**Start with Option 1** (periodic hard sync). It's closest to EMA behavior but without the smoothing.

### A2: SIGReg Loss

Use the VICReg var+cov formulation from v23 (simpler, proven):

```python
def vicreg_var_cov(h, eps=1e-4):
    B, D = h.shape
    std = h.std(dim=0) + eps
    l_var = F.relu(1.0 - std).mean()
    h_c = h - h.mean(dim=0, keepdim=True)
    cov = (h_c.T @ h_c) / max(B - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    l_cov = (off ** 2).sum() / D
    return l_var, l_cov
```

Apply to **predictor output h_pred** (not encoder output h_t). V17 showed this is more stable.

Full loss:
```python
l_pred = F.l1_loss(h_pred_n, h_target_n.detach())
l_var, l_cov = vicreg_var_cov(h_pred)  # on UNNORMALIZED h_pred
loss = l_pred + lambda_var * l_var + lambda_cov * l_cov
```

Hyperparameters: lambda_var=0.04, lambda_cov=0.02 (from v23).

### A3: Sweep on FD001 First (seed 42 only)

Before running all datasets, sweep on FD001:

| Config | sync_interval | lambda_var | lambda_cov |
|--------|--------------|------------|------------|
| A | 100 steps | 0.04 | 0.02 |
| B | 50 steps | 0.04 | 0.02 |
| C | 200 steps | 0.04 | 0.02 |
| D | 100 steps | 0.10 | 0.05 |
| E | Never (frozen target) | 0.04 | 0.02 |

For each: pretrain 50 epochs, pred-FT, eval h-AUROC.
Compare to v34 EMA baseline (run fresh with same protocol).

### A4: If FD001 Sweep Succeeds, Run ALL Datasets

Pick best config from A3. Run on ALL 11 datasets (FD001, FD002, FD003, SMAP, PSM, SMD, MBA, SKAB, ETTm1, GECCO, BATADAL) + any new datasets from Workstream C. Use 3 seeds each.

This is the big payoff: if SIGReg matches or beats EMA across the board, it becomes the default architecture.

### A5: Curriculum Fallback

If pure SIGReg collapses or regresses, try the v17 curriculum:
- Epochs 1-25: standard EMA (momentum=0.99)
- Epochs 25-40: EMA + SIGReg (ramp lambda from 0 to target)
- Epochs 40-50: SIGReg only (drop EMA, periodic hard sync)

### A6: Collapse Diagnostics

Monitor EVERY epoch:
- `h_pred.std(dim=0).mean()` -- must stay > 0.05
- `h_t.std(dim=0).mean()` -- must stay > 0.05
- Pretrain loss -- should decrease, not collapse to near-zero
- If any collapses: abort that config, try next

Save diagnostics as `results/phaseA/{dataset}_sigreg_diagnostics.json`.

---

## Workstream B: Fix ST-JEPA Collapse (~2 hours, lower priority)

Only attempt if Workstream A finishes early or as a parallel experiment.

### B1: Per-Position Prediction Targets (V-JEPA Style)

The key fix: instead of pooling the target into a single h*, predict each masked position separately.

```python
# Target encoder produces per-position representations
# h_target_grid: (B, N, C, d) -- one repr per (patch, channel)
# Predictor must produce representations at EACH masked position
# Loss: L1 on each masked position separately

# Predictor takes: (h_visible_pooled, delta_t, position_embedding)
# -> predicted representation at one masked position
# Run for each masked position, average the losses
```

This is a much larger architectural change. Implement only if time permits.

### B2: Stronger Variance Regularizer

Quick fix: increase lambda_var from 0.04 to 0.5 in the ST-JEPA pretrain loop. Add VICReg covariance term. This directly fights the collapse.

### B3: Partial Channel Fusion (compromise)

Instead of pure per-channel tokenization, concatenate K=4 neighboring channels per token:
```python
# Group channels into groups of K=4
# Token input: (P * K) = 64 dimensions instead of 16
# Fewer tokens: N_patches * (C // K) instead of N_patches * C
```

This gives each token enough input diversity to avoid collapse while still having some cross-channel token structure.

### B4: Late Fusion (safest)

Keep channel-fusion pretraining (standard FAM). Add a single cross-channel MHA layer during finetuning only:
```python
class FAM_LateFusion(FAM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cross_channel_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True)
        self.cross_channel_norm = nn.LayerNorm(d_model)
    
    def finetune_forward(self, context, horizons, context_mask=None, mode='pred_ft'):
        # Get per-patch representations from frozen encoder
        h_all, patch_mask = self.encoder(context, context_mask, return_all=True)
        # h_all: (B, N, d) -- these are channel-fused patch tokens
        
        # Cross-channel attention would need per-channel tokens here...
        # Actually, late fusion on channel-fused tokens doesn't add
        # cross-channel structure. Skip this variant.
```

**On reflection, late fusion only works if we have per-channel representations.** Since the encoder produces channel-fused tokens, there's nothing for cross-channel attention to operate on. Skip B4.

**Priority**: B2 (strongest variance reg) > B3 (partial fusion) > B1 (per-position targets).

---

## Workstream C: Add 3 New Datasets (~2 hours)

### Datasets to Add

| Dataset | Domain | Channels | Events | Precursor Quality |
|---------|--------|----------|--------|-------------------|
| **TEP** (Tennessee Eastman Process) | Chemical process | 52 | 21 fault types (ramp+step) | Excellent (drift faults have 10-100+ timestep precursors) |
| **SWaT** (Secure Water Treatment) | Cyber-physical ICS | 51 | 41 attacks | Good (multi-stage causal propagation) |
| **Sepsis** (PhysioNet 2019) | Clinical medicine (ICU) | 40 | Sepsis onset | Excellent (3-6 hour physiological precursors) |

### C1: Tennessee Eastman Process

**Download**: Kaggle or GitHub (fully open, no registration).
**Preprocessing**:
- Use the extended dataset (Reinartz 2021) if available, otherwise standard TEP
- Select ramp/drift faults (2, 8, 12, 14, 15) for event prediction -- these have the best precursor structure
- Step faults (1, 6, 7) as secondary (no precursor, but model should still detect them)
- 52 channels, drop near-constant ones
- Standard: 500 timesteps train (normal), 960 timesteps test (fault onset at step 160)
- Multiple simulation runs per fault type -> treat as entities
- Event label: 1 from fault onset onward (or ramp from precursor detection point)

Create `fam-jepa/data/tep.py`:
```python
def load_tep(faults='ramp', normalize=True):
    """Load Tennessee Eastman Process dataset.
    
    Args:
        faults: 'ramp' (2,8,12,14,15), 'step' (1,6,7), or 'all'
    Returns: dict with pretrain_seqs, ft_train, ft_val, ft_test
    """
```

### C2: SWaT

**Download**: iTrust form required. If not pre-downloaded, use the Kaggle mirror for overnight speed.
**Preprocessing**:
- 51 features (continuous sensors + discrete actuator states)
- Train: 7 days normal operation (~946K rows at 1Hz)
- Test: 4 days with 41 attacks (~450K rows)
- Subsample to 1/10 Hz (every 10 seconds) to make pretraining tractable
- Event labels: 1 during attack windows

Create `fam-jepa/data/swat.py`.

### C3: PhysioNet Sepsis

**Download**: PhysioNet (DUA required, may not be pre-downloaded).
**Preprocessing**:
- 40 clinical variables per hour
- Heavy missingness -- forward-fill + zero-fill
- Each ICU stay is an entity
- Event: sepsis onset (Sepsis-3 definition)
- Precursor window: 6 hours before onset
- Only use stays with >= 24 hours of data

Create `fam-jepa/data/sepsis.py`.

**NOTE**: If SWaT or Sepsis data is not available on the VM, skip that dataset and focus on TEP (which is freely downloadable). TEP is the highest-priority new dataset.

### C4: Integrate into LOADERS

Add new datasets to the LOADERS dict in the v34 runner. Run standard FAM pretrain+finetune on each, 3 seeds. Report h-AUROC, h-AUPRC.

### C5: If SIGReg Works, Also Run New Datasets with SIGReg

---

## Workstream D: Paper Review + Polish Loop (~1.5 hours)

### D1: Self-Review as NeurIPS Reviewer

Read `paper-neurips/paper.tex` end-to-end. Score each section 1-10 on:
- Clarity
- Technical correctness
- Completeness
- Novelty claim strength

Write a structured NeurIPS review in `results/phaseD/self_review.md`:
```
Summary: ...
Strengths: ...
Weaknesses: ...
Questions: ...
Suggestions: ...
Overall Score: X/10
Confidence: X/5
```

### D2: Apply MARGINAL Fixes Only

Based on the self-review, fix ONLY:
- Typos, grammar, citation errors
- Unclear sentences (reword, don't restructure)
- Missing uncertainty bars or statistical qualifiers
- Incorrect numbers (cross-reference with latest results)

Do NOT:
- Restructure sections
- Add new content beyond a sentence
- Change the paper's narrative or claims
- Add new figures to the paper

### D3: Identify Major Changes for Quarto Summary

For anything that requires significant changes (new sections, restructured arguments, new figures), do NOT edit paper.tex. Instead, write proposals into `results/phaseD/major_changes_proposal.md`:
```
## Proposed Change 1: ...
Rationale: ...
Current text (line X-Y): ...
Proposed replacement: ...
Impact: ...
```

The user will review these tomorrow and decide.

### D4: Check All Numbers Against Latest Results

Cross-reference every number in paper.tex against:
- `experiments/v30/results/master_table.json`
- `experiments/v31/results/phase1_lf10_master.json`
- `experiments/v32/results/` (legacy metrics, RMSE probe, baselines)
- `experiments/v33/results/` (cross-channel ablation)
- Any v34 results from Workstreams A-C

Flag mismatches in `results/phaseD/number_audit.md`.

---

## Workstream E: Figure Refinement (~1 hour)

### E1: Review All Existing Figures

For each figure in `paper-neurips/figures/`:
1. Check readability at NeurIPS column width (3.25 inches)
2. Check colorblind safety (no red-green only distinctions)
3. Check font sizes (>= 8pt for all text)
4. Check axis labels and legends (complete, no overlaps)

### E2: Marginal Improvements Only

Apply small fixes:
- Tighten spacing
- Fix overlapping labels
- Improve color contrast
- Add missing axis labels
- Ensure consistent font family across figures

Do NOT redesign figures. Do NOT create new figures for the paper.

### E3: Quarto Summary Notebook (v34)

Create `notebooks/34_v34_analysis.qmd` with ALL v34 results:
- SIGReg vs EMA comparison (bar charts, per-dataset)
- SIGReg training dynamics (loss curves, variance diagnostics)
- New dataset results (TEP, SWaT, Sepsis)
- ST-JEPA fix results (if attempted)
- Cross-channel ablation summary (v33+v34)
- Paper review findings

This is where MAJOR figure proposals go -- expressive, publication-quality figures that the user reviews tomorrow.

---

## Execution Order

The workstreams have dependencies:

```
Phase 0: Setup (15 min)
    |
    +-- Workstream A: SIGReg (parallel, highest priority)
    |   A1: Architecture (30 min)
    |   A2: FD001 sweep (1.5 hr)
    |   A3: Pick best config
    |   A4: All datasets if promising (2-3 hr)
    |   A5: Curriculum fallback if needed (1 hr)
    |
    +-- Workstream C: New datasets (parallel with A)
    |   C1: TEP loader + pretrain (1 hr)
    |   C2: SWaT loader + pretrain (45 min)
    |   C3: Sepsis loader + pretrain (45 min)
    |
    +-- Workstream B: ST-JEPA fixes (after A finishes, or skip)
    |   B2: Stronger variance reg (30 min)
    |   B3: Partial fusion (1 hr)
    |
    +-- Workstream D: Paper review (after A4 or during A4 compute)
    |   D1: Self-review (30 min)
    |   D2: Marginal fixes (30 min)
    |   D3: Major change proposals (20 min)
    |   D4: Number audit (20 min)
    |
    +-- Workstream E: Figures + Quarto (last)
        E1: Figure audit (20 min)
        E2: Marginal figure fixes (20 min)
        E3: v34 Quarto notebook (40 min)

Phase Final: Session summary + RESULTS.md update + final commit
```

**Priority if time runs short**: A > C(TEP only) > D > E > B > C(SWaT, Sepsis)

---

## Timing Budget

| Phase | Time | Cumulative |
|-------|------|------------|
| 0. Setup | 15 min | 0:15 |
| A1-A2: SIGReg FD001 sweep | 2:00 | 2:15 |
| C1: TEP dataset + pretrain | 1:00 (parallel with A) | 2:15 |
| A3: Pick best SIGReg config | 0:15 | 2:30 |
| A4: SIGReg all datasets | 3:00 | 5:30 |
| C2-C3: SWaT + Sepsis | 1:30 (parallel with A4) | 5:30 |
| B2-B3: ST-JEPA fixes | 1:30 | 7:00 |
| D1-D4: Paper review | 1:30 | 8:30 |
| E1-E3: Figures + Quarto | 1:30 | 10:00 |
| Final: Summary + commit | 0:30 | 10:30 |

---

## File Locations

- Model: `fam-jepa/model.py` (DO NOT MODIFY for Workstreams A/B -- create v34 variants)
- Training: `fam-jepa/train.py`
- Data loaders: `from _runner_v29 import LOADERS` (existing datasets)
- New data loaders: `fam-jepa/data/tep.py`, `fam-jepa/data/swat.py`, `fam-jepa/data/sepsis.py`
- v23 SIGReg reference: `experiments/archive/v23/pretrain_sigreg.py`
- v33 ST-JEPA reference: `experiments/v33/_runner_v33.py`
- Paper: `paper-neurips/paper.tex`
- Figures: `paper-neurips/figures/`
- This session's code: `fam-jepa/experiments/v34/`

## Output Structure

```
experiments/v34/
  _runner_v34.py
  results/
    phaseA/
      sigreg_sweep_FD001.json
      sigreg_{dataset}.json (for each dataset)
      sigreg_diagnostics_{dataset}.json
      sigreg_vs_ema_summary.json
    phaseB/
      stjepa_fix_{variant}_{dataset}.json
    phaseC/
      tep_results.json
      swat_results.json
      sepsis_results.json
    phaseD/
      self_review.md
      major_changes_proposal.md
      number_audit.md
    phaseE/
      figure_audit.md
    SESSION_SUMMARY.md
  ckpts/
  surfaces/
notebooks/
  34_v34_analysis.qmd
  34_v34_analysis.html
```

## Principles

- **SIGReg is the main bet.** If it works across all datasets, it's a cleaner architecture and potentially better numbers. Invest the most time here.
- **New datasets expand the story.** TEP (chemical process) adds a prestigious domain. Even 1 new dataset with good results strengthens the paper.
- **Paper changes are MARGINAL only.** Typos, number fixes, unclear sentences. Major changes go into the proposal doc and Quarto notebook for the user to review.
- **Commit hourly.** Even partial results are valuable.
- **Don't stop early.** Use the full overnight window.
- **If SIGReg works on FD001 but fails on anomaly datasets**, report honestly. "SIGReg matches EMA on lifecycle datasets but requires curriculum on anomaly datasets" is a valid finding.
