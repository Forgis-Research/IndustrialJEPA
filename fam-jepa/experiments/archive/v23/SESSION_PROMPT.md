# V23 Session — Architecture Cleanup + SIGReg + Sepsis

**Usage**: Paste as opening prompt to a Claude Code session on the GPU VM.
**Working directory**: `IndustrialJEPA/fam-jepa/`
**Duration**: ~3 hours on A10G.
**Prereqs**: Read CLAUDE.md, `fam-jepa/ARCHITECTURE.md` (definitive spec),
experiments/RESULTS.md, v22 SESSION_PROMPT.md

---

## Architecture changes (implement FIRST — all experiments use these)

Read `fam-jepa/ARCHITECTURE.md` before coding anything. Key changes from v22:

### 1. Pretraining target = cumulative interval, no gap

**Old (v22)**: target = x[t+Δt : t+Δt+w], w=10.
Skip Δt steps, then encode a fixed 10-step window.

**New**: target = x(t : t+Δt].
Encode the next Δt steps immediately after the cut. No gap, no fixed w.
The target length IS Δt. This makes pretraining and finetuning use the
same interval: "what happens in the next Δt steps."

### 2. Patch tokenization: P=16 globally

Replace `patch_length=1` with `P=16` for ALL datasets (including C-MAPSS
and Sepsis). One tokenizer, zero dataset-specific hyperparameters.
`SensorProjection` already supports this via the `patch_length` parameter.
Pad last patch with zeros if input length is not divisible by 16.

### 3. Variable-length context (restore from v11)

Drop the fixed-100 window. Use variable-length context:
- C-MAPSS: full engine history (up to ~362 cycles)
- Anomaly datasets: sliding, max 512 steps
- Sepsis: full ICU stay (up to ~336 hours)

The collate function pads + creates a mask. The causal transformer
handles this via `key_padding_mask` (already implemented in v11).

### 4. Horizons are independent of P

The predictor takes Δt as a continuous scalar input — no constraint that
Δt be a multiple of P. The pretraining target x(t:t+Δt] is tokenized
with padding on the last patch if needed.

- C-MAPSS: Δt ∈ {1, 5, 10, 20, 50, 100, 150} cycles
- SMAP/PSM/SMD/MBA: Δt ∈ {1, 5, 10, 20, 50, 100, 150, 200} steps
- Sepsis: Δt ∈ {1, 2, 3, 6, 12, 24, 48} hours

### 5. Stride=1 at evaluation

Stride=1 for test surfaces (every timestep gets a prediction).
Stride>1 at training is fine for compute.

### 6. Terminology

- "context" not "window" or "past"
- "Δt" not "k" — everywhere in code and comments
- h_t not h_past
- h*_(t,t+Δt] not h_future

---

## Four goals

### Goal A: Drop EMA — SIGReg with curriculum (ablation)

Replace EMA target encoder with explicit collapse prevention (SIGReg).
This simplifies the method and removes a hyperparameter.

**Prior work in this repo:**
- V17 phase 3: curriculum EMA→SIGReg on FD001 — worked but never re-evaluated
  under AUPRC.
- V20 phase 3: SIGReg-pred beat EMA on F1w+RMSE (0.451 vs 0.391 F1w),
  never tested under AUPRC or on other datasets.
- V22 already uses variance regularizer (λ_var=0.04) on top of EMA.

**What to run:**

1. **FD001 SIGReg from scratch** (no EMA at any point). Use full VICReg
   triplet (invariance L1 + variance relu(1-std) + covariance off-diag)
   with curriculum on Δt: start Δt∈[1,10] for first 20 epochs, expand to
   Δt∈[1,150] by epoch 40.
   - Loss = L1_normalized + λ_var * var_loss + λ_cov * cov_loss
   - λ_var=0.04, λ_cov=0.02 (start here, adjust if collapse)
   - Target = context_encoder(x(t:t+Δt]).detach() (stop-grad, no EMA)
   - 3 seeds, early stop patience=5, max 50 epochs

2. **Compare to EMA baseline**: rerun EMA pretrain with the new cumulative
   target (not skip-gap). Then freeze both encoders, pred-FT 3 seeds.
   Report AUPRC under identical downstream protocol.

3. **If SIGReg matches baseline on FD001**: repeat on SMAP with entity
   splits. If it doesn't match, document why and move on (<30 min debug).

**Collapse detection**: monitor mean(std(h_t)) per dimension each epoch.
If < 0.01 → collapsed, abort that seed.

### Goal B: PhysioNet 2019 Sepsis — new domain, clear onset event

Add a medical event-prediction dataset with well-defined onset, strong
literature baselines, and a fundamentally different event type (clinical
deterioration, not mechanical/anomaly).

**Dataset**: PhysioNet Computing in Cardiology 2019 Challenge.
- ~40k ICU stays, 40 clinical variables, binary SepsisLabel with clear onset
- Hourly resolution, variable-length stays, missing values
- Literature SOTA: AUROC ~0.78–0.85
- Free download: `wget -r -N -c -np https://physionet.org/files/challenge-2019/1.0.0/`

**What to run:**

1. **Data loader** (`data/sepsis.py` — already created in v23 session, verify
   it matches the new architecture spec):
   - Forward-fill then zero-fill missing values
   - Drop static demographics (Age, Gender, HospAdmTime)
   - Normalize per-feature on training set
   - Split: set A for train/val (80/20 patient-level), set B for test

2. **Pretrain** on set A training split. P=1 (hourly is already coarse).
   Variable-length context (full stay). Δt ∈ [1, 48] hours.
   Target = x(t : t+Δt] — cumulative interval, no gap.

3. **Pred-FT** with patient-level splits. Freeze encoder, finetune
   predictor + sigmoid head. 3 seeds. Horizons Δt ∈ {1,2,3,6,12,24,48} hours.
   Compute AUPRC on probability surface.

4. **Report**: AUROC (for literature comparison) and AUPRC (our primary).

### Goal C: Anomaly datasets with new architecture

Re-pretrain and re-run pred-FT on SMAP with the new architecture
(P=16, cumulative target, variable context up to 512). This replaces the
v22 baseline numbers. 3 seeds, entity splits.

If time: also PSM.

### Goal D: PA-F1 from stored v22 surfaces (quick, first)

Run `experiments/v22/compute_pa_f1_from_surfaces.py`.
Update RESULTS.md. Takes <1 min, do this first.

---

## Phase plan

| Phase | What | Est. time | Priority |
|-------|------|-----------|----------|
| 0 | PA-F1 from v22 surfaces | 5 min | Do first |
| 1 | Implement architecture changes (tokenizer, target, context) | 30 min | Critical |
| 2 | SIGReg pretrain FD001 (3 seeds) | 40 min | Critical |
| 3 | EMA pretrain FD001 with new target (3 seeds) | 40 min | Critical |
| 4 | SIGReg vs EMA: pred-FT comparison | 15 min | Critical |
| 5 | Sepsis: verify loader, pretrain, pred-FT | 40 min | Critical |
| 6 | SMAP: pretrain P=16 + pred-FT (3 seeds) | 30 min | Important |
| 7 | Update RESULTS.md + render notebook | 15 min | Always |

**Total**: ~3.5h.

---

## Ground rules

1. **Read `fam-jepa/ARCHITECTURE.md` first.** All code must match the spec.
2. **Cumulative target**: target = x(t : t+Δt]. No gap. No fixed w.
3. **Collapse monitoring**: log mean(std(h_t)) each epoch. Abort if < 0.01.
4. **SIGReg = no EMA, stop-grad only.** Target = encoder(x(t:t+Δt]).detach().
5. **Curriculum**: ramp Δt range, not λ. Start Δt∈[1,10], expand by epoch 40.
6. Store surfaces as .npz. Compute AUPRC + legacy metrics from surfaces.
7. Reporting: `mean ± std (Ns)`. Decompose F1 into P + R.
8. Commit + push after each phase. Update RESULTS.md after every phase.
9. If something clearly fails, document and move on. Max 30 min debug.
10. **Stride=1 at evaluation.** Stride>1 at training is fine.
