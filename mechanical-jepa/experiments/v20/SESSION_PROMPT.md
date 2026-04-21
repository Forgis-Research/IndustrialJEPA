# V20 Overnight Session — Predictor Finetuning + Multi-Domain Benchmark

**Usage**: Paste as opening prompt to a Claude Code session on the GPU VM.
**Working directory**: `IndustrialJEPA/mechanical-jepa/`
**Duration**: overnight (~10-12 hours on A10G)

---

## Paper Story (read this first)

The paper makes two contributions:

**1. Predictor finetuning.** Every SSL method discards its pretraining head and probes the encoder. We do the opposite: freeze the encoder, finetune the JEPA predictor (198K params, 16% of total) for each event type. The predictor compresses "what does the future look like?" — finetuning steers it toward "does the future contain THIS event?" This should beat frozen probe (too weak) and E2E (overfits at low labels).

**2. One architecture, N datasets, M domains.** Same 1.26M-param model, pretrained per-dataset with no labels, competitive with domain-specific supervised models 100-560x larger.

Everything in this session serves one of these two contributions. If an experiment doesn't produce a cell in the paper's tables, don't run it.

---

## Evaluation Protocol (CRITICAL — read before any experiment)

**Per-window binary F1 at patch resolution.** Same as MTS-JEPA and the anomaly literature, but at our patch granularity.

For each window (= 1 patch-token of P raw samples):
- Label: `y_w = 1` if an event is active during window w, else 0
- Prediction: the predictor is run at horizons k=1..16 beyond context; each h_hat_k feeds a linear head → ŷ_k ∈ {0,1}

**Metrics** (computed for EVERY experiment):
- **F1** (per-window binary, averaged over W=16 horizon windows) — PRIMARY
- Precision, Recall
- AUROC (threshold-free)
- **Legacy**: RMSE + NASA-S for C-MAPSS, PA-F1 for SMAP/MSL/PSM (for literature cross-reference ONLY)

**Reporting**: `mean ± std (Ns, 95% CI [lo, hi])`. Always decompose F1 → P + R.

**Implementation note**: `evaluation/grey_swan_metrics.py` has the building blocks (F1, AUROC, aggregate_seeds). The per-window prediction loop needs to be implemented in the experiment script: run predictor at k=1..16, classify each, compute F1 per window, average. A clean utility for this should be created as Phase 0 infrastructure.

---

## Setup (~15 min)

### A. Sync agents
Copy the `### Reporting Format (NON-NEGOTIABLE)` section from `.claude/agents/ml-researcher.md` into the VM's agent file.

### B. Read CLAUDE.md at repo root
It describes the full repo structure. The paper is `paper-neurips/paper.tex`.

### C. Read context
1. `experiments/RESULTS.md` — master results table
2. `experiments/v19/` — v19 delivered PSM (PA-F1 0.637), MBA ECG (0.551), SMD (0.264), Paderborn (macro-F1 0.781)
3. `experiments/v18/RESULTS.md` — C-MAPSS + SMAP/MSL results
4. `experiments/v11/models.py` — model definitions (TrajectoryJEPA, Predictor)
5. `paper-neurips/paper.tex` — the clean paper draft

### D. Download data if needed
```bash
cd /home/sagemaker-user/IndustrialJEPA
python paper-replications/mts-jepa/download_datasets.py  # PSM
```

---

## Phase 0: Implement Predictor Finetuning (~2 hours)

**This is Contribution #1. Build it right.**

### 0a. Infrastructure: per-window prediction + eval loop

Create a reusable function that:
1. Takes a pretrained checkpoint (encoder + predictor)
2. Freezes the encoder
3. Runs predictor at k=1..16 for each input window
4. Concatenates h_hat_1..h_hat_16
5. Linear head: (16 × d_model) → 16 binary predictions
6. Loss: binary cross-entropy summed over 16 windows
7. Eval: per-window F1, averaged over horizon, + AUROC

Save as `experiments/v20/pred_ft_utils.py` — reused by all subsequent phases.

### 0b. Predictor finetuning on C-MAPSS FD001

Using V2 (or V17) pretrained checkpoint:

| Mode | Encoder | Predictor | Head | Params |
|------|---------|-----------|------|--------|
| Frozen probe (h_past only) | frozen | frozen | linear on h_past | 257 |
| Frozen multi-horizon | frozen | frozen | linear on [h_hat_1;...;h_hat_16] | ~4K |
| **Pred-FT** | **frozen** | **finetuned** | **linear** | **~198K** |
| E2E | finetuned | finetuned | linear | 1.26M |
| Scratch | random | random | linear | 1.26M |

Run at **100% and 5% labels**, 5 seeds each. 10 configs × 5 seeds = 50 runs.

**Key question**: Does Pred-FT beat E2E at 5% labels?

**Save**: `experiments/v20/phase0_pred_ft.json`

---

## Phase 1: Multi-Domain Benchmark Table (~3 hours)

V19 already pretrained on PSM, MBA, SMD, Paderborn. V18 has C-MAPSS, SMAP, MSL.

### 1a. Per-window F1 on all existing datasets

For each dataset where we have a pretrained encoder:
- Run the Phase 0 evaluation loop (predictor at k=1..16, per-window F1)
- Use **Pred-FT** mode (the paper's default) and **Frozen** mode (for comparison)
- 5 seeds
- Also compute legacy metric

### 1b. Pretrain on SMAP/MSL (if not done in v19)

V18 pretrained a C-MAPSS encoder and scored SMAP via Mahalanobis. For the "per-dataset pretrain" story, we need a SMAP-pretrained encoder evaluated with Pred-FT on SMAP event labels. ~45 min.

### 1c. Fill benchmark table

| Dataset | Domain | Event | FAM F1 | FAM AUROC | Legacy | SOTA | SOTA Legacy | SOTA params |
|---------|--------|-------|--------|-----------|--------|------|-------------|-------------|

**Save**: `experiments/v20/phase1_benchmark.json`

---

## Phase 2: Foundation Model Baselines (~1 hour)

V18 has Chronos on C-MAPSS (RMSE only). Re-evaluate with per-window F1. Add to benchmark table.

If time: Chronos on SMAP or PSM.

**Save**: `experiments/v20/phase2_chronos.json`

---

## Phase 3: Ablations (~2 hours)

Three ablations that serve the paper:

### 3a. Causal vs bidirectional
5 seeds, 200ep, FD001 at 100% and 5%. Use Pred-FT mode. Per-window F1.

### 3b. EMA vs SIGReg vs curriculum
Same checkpoint comparison. Pred-FT mode at 100% and 5%.

### 3c. Does predictor help at inference without finetuning?
Frozen probe on h_past vs frozen probe on [h_hat_1;...;h_hat_16]. Quick, no retraining.

**Save**: `experiments/v20/phase3_ablations.json`

---

## Phase 4: Paper + Review (~1 hour)

### 4a. Fill paper.tex tables
- Benchmark table (Tab 1) from Phase 1
- Pred-FT comparison table from Phase 0
- Chronos comparison from Phase 2
- Ablation tables from Phase 3

### 4b. Run neurips-reviewer
Focus: (1) Is predictor finetuning convincing? (2) Is multi-domain breadth sufficient?

---

## Phase 5: Stretch (~2 hours)

In priority order:
1. Label-efficiency sweep at 50%/20%/10% (extends Phase 0)
2. Pred-FT on Paderborn classification (bearing faults, not just anomaly)
3. Window sensitivity: vary W ∈ {4, 8, 16, 32} on FD001
4. SMAP-pretrained encoder (if not done in Phase 1b)

---

## Ground Rules

1. **Per-window binary F1 for every experiment.** W=16 horizon windows at patch resolution. No ad-hoc metrics.
2. **Reporting**: `mean ± std (Ns, 95% CI [lo, hi])`. Decompose F1 → P + R. Include AUROC.
3. **5 seeds** for headline results. 3 for screening.
4. **Update `experiments/RESULTS.md`** after every phase.
5. **Commit + push every 1-2 phases.** Tag: `v20 phase N: ...`
6. **Paper is `paper-neurips/paper.tex`.**
7. **Phase 0 is 50% of the session budget.** If predictor finetuning doesn't work, the paper doesn't work.
8. **Memory guard.** Before loading any dataset, check its size. If >2GB, stream or chunk. V19 likely OOM'd on large dataset load.

---

## Success Criteria (ordered by paper impact)

- [ ] Pred-FT infrastructure: reusable per-window prediction + eval loop
- [ ] Pred-FT vs Frozen vs E2E at 100% and 5% on FD001 (5 seeds) — **the headline**
- [ ] Benchmark table: all datasets with per-window F1 + legacy + 95% CI
- [ ] Pred-FT on at least one non-C-MAPSS dataset (SMAP or PSM)
- [ ] Chronos comparison with per-window F1
- [ ] Causal vs bidirectional ablation
- [ ] EMA vs SIGReg ablation
- [ ] Paper tables filled
- [ ] One reviewer round
