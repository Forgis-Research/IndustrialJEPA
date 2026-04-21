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

## Setup (~15 min)

### A. Sync agents
Copy the `### Reporting Format (NON-NEGOTIABLE)` section from `.claude/agents/ml-researcher.md` into the VM's agent file. All results: `mean ± std (Ns, 95% CI [lo, hi])`.

### B. Verify infrastructure
```bash
cd /home/sagemaker-user/IndustrialJEPA/mechanical-jepa
python -c "
from evaluation.grey_swan_metrics import evaluate_event_prediction, aggregate_seeds, format_result
import numpy as np
tte = np.random.uniform(5, 125, 50); pred = tte + np.random.randn(50)*10
r = evaluate_event_prediction(tte, pred, window_size=30, n_windows=4, legacy_rul_cap=125)
print(f'Det F1@30={r[\"detection\"][\"h30\"][\"f1\"]:.3f}, Timing={r[\"timing\"][\"macro_f1\"]:.3f}, RMSE={r[\"legacy\"][\"rmse\"]:.1f}')
print('OK')
"
```

### C. Read context
1. `experiments/RESULTS.md` — master results table
2. `experiments/v19/` — v19 already delivered PSM, MBA, SMD, Paderborn results. **Incorporate these.**
3. `paper-neurips/paper_v20_draft.tex` — the clean paper draft. This is the target.
4. `experiments/v18/RESULTS.md` — v18 results
5. `experiments/v11/models.py` — model definitions

### D. Download data if needed
```bash
cd /home/sagemaker-user/IndustrialJEPA
python paper-replications/mts-jepa/download_datasets.py  # PSM
```

---

## Phase 0: Predictor Finetuning — The Core Experiment (~2.5 hours)

**This is Contribution #1. Run it first and run it thoroughly.**

### 0a. Implement predictor finetuning

Using the existing V2 (or V17) pretrained checkpoint on C-MAPSS FD001:

1. **Freeze** the context encoder entirely (no gradients through $f_\theta$)
2. **Finetune** the predictor $g_\phi$ (198K params) + a linear head
3. The predictor is run at multiple horizons k ∈ {30, 60, 90, 120} (matching the 4 evaluation windows)
4. The concatenated $[\hat{h}_{k=30}; \hat{h}_{k=60}; \hat{h}_{k=90}; \hat{h}_{k=120}]$ feeds the linear head
5. For detection (Stage 1): binary cross-entropy, "will fail within H cycles?"
6. For timing (Stage 2): cross-entropy over 5 classes (4 windows + no-event)

**Comparison at each label budget (100%, 50%, 20%, 10%, 5%):**

| Mode | Encoder | Predictor | Head | Params |
|------|---------|-----------|------|--------|
| Frozen probe (h_past only) | frozen | frozen | linear on h_past | 257 |
| Frozen probe (multi-horizon) | frozen | frozen | linear on [h_past; h_hat_k1; ...] | ~1K |
| **Pred-FT** | **frozen** | **finetuned** | **linear** | **~198K** |
| E2E | finetuned | finetuned | linear | 1.26M |
| Scratch | random | random | linear | 1.26M |

5 seeds per cell. Use `evaluate_event_prediction(tte, pred, window_size=30, n_windows=4, legacy_rul_cap=125)` for ALL.

**This is ~25 runs × 5 seeds × ~10 min = ~20 hours if sequential.** Parallelise where possible. At minimum, run the 5-row comparison at 100% and 5% labels (10 rows × 5 seeds = 50 runs × ~10 min = ~8 hours). 50%, 20%, 10% are stretch.

**Save**: `experiments/v20/phase0_pred_ft.json`

**Key question**: Does Pred-FT beat E2E at 5% labels? If yes, that's the headline result.

---

## Phase 1: Multi-Domain Benchmark Table (~3 hours)

**This is Contribution #2.** V19 already delivered pretrained encoders + Mahalanobis results for PSM, MBA, SMD, Paderborn. V18 has C-MAPSS and SMAP/MSL.

### 1a. Recompute all v18/v19 results with unified eval

For each dataset where we have predictions or can quickly re-run:
- Call `evaluate_event_prediction()` with appropriate Δ and W=4
- Also compute legacy metric for SOTA comparison column

Window conventions:
| Dataset | Δ | W | Total horizon | Legacy metric |
|---------|---|---|---------------|---------------|
| C-MAPSS | 30 cycles | 4 | 120 cycles | RMSE, NASA-S |
| SMAP | 50 steps | 4 | 200 steps | PA-F1 |
| MSL | 50 steps | 4 | 200 steps | PA-F1 |
| PSM | 50 steps | 4 | 200 steps | PA-F1 |
| SMD | 50 steps | 4 | 200 steps | PA-F1 |
| MBA | 25 beats | 4 | 100 beats | PA-F1 |
| Paderborn | N/A (classification) | N/A | N/A | macro-F1 |

For anomaly datasets: time_to_event = distance from each timestep to next anomaly segment start. Timesteps far from any future anomaly → TTE = ∞.

### 1b. Predictor finetuning on non-C-MAPSS datasets

Where v19 used only Mahalanobis/frozen probe, **also run Pred-FT** on at least SMAP and PSM. This tests whether predictor finetuning generalises beyond C-MAPSS.

### 1c. Fill the benchmark table

The paper's Table 1 needs:

| Dataset | Domain | Event | FAM Det-F1 | FAM Time-F1 | Legacy | SOTA | SOTA Legacy | SOTA params |
|---------|--------|-------|-----------|-------------|--------|------|-------------|-------------|

Every cell filled, every number with 95% CI.

**Save**: `experiments/v20/phase1_benchmark.json`

---

## Phase 2: Foundation Model Baselines (~1 hour)

V18 already has Chronos on C-MAPSS. Recompute with unified eval. Add to benchmark table.

If time: run Chronos on one anomaly dataset (SMAP or PSM) for a second comparison point.

**Save**: `experiments/v20/phase2_chronos.json`

---

## Phase 3: Targeted Ablations (~2 hours)

Only ablations that serve the paper. Three questions:

### 3a. Causal vs bidirectional (supports architecture choice)
- 5 seeds, 200ep, FD001 at 100% and 5%
- Also on FD003 transfer
- Use Pred-FT mode (not just frozen probe)

### 3b. EMA vs SIGReg vs curriculum (supports §3.2)
- Same checkpoint comparison
- V2 (EMA-only) vs V17 (curriculum EMA→SIGReg)
- Pred-FT mode at 100% and 5%

### 3c. Does the predictor help at inference even WITHOUT finetuning? (supports §6.2)
- Frozen probe on h_past vs frozen probe on [h_past; h_hat_k1; ...; h_hat_k4]
- Quick, no retraining needed. Uses existing checkpoint.

**Save**: `experiments/v20/phase3_ablations.json`

---

## Phase 4: Paper + Review (~1 hour)

### 4a. Fill paper_v20_draft.tex
- Fill benchmark table (Tab 1) from Phase 1
- Fill Pred-FT table from Phase 0
- Fill Chronos comparison from Phase 2
- Fill ablation tables from Phase 3

### 4b. Run neurips-reviewer
Focus: (1) Is predictor finetuning convincing? (2) Is multi-domain breadth sufficient? (3) Any missing baselines?

**Save**: commit to `paper-neurips/paper_v20_draft.tex`

---

## Phase 5: Stretch (~2 hours)

In priority order, if time remains:
1. Label-efficiency sweep at 50%/20%/10% (extends Phase 0)
2. Window sensitivity study (Δ ∈ {10, 20, 30, 50} on FD001)
3. Pred-FT on Paderborn classification
4. V2 E2E honest (the 13.80 mystery — interesting but not paper-critical)

---

## Ground Rules

1. **`evaluate_event_prediction()` for EVERY experiment.** No ad-hoc metrics.
2. **Reporting**: `mean ± std (Ns, 95% CI [lo, hi])`. Always.
3. **5 seeds** for headline results. 3 for screening.
4. **Update `experiments/RESULTS.md`** after every phase.
5. **Commit + push every 1-2 phases.** Tag: `v20 phase N: ...`
6. **Paper is `paper_v20_draft.tex`** (not the old `paper.tex`).
7. **Phase 0 is 50% of the session budget.** If predictor finetuning doesn't work, the paper doesn't work. Invest the time.
8. **Memory guard.** Before loading any dataset, check its size: `arr = np.load(f); print(f'{f}: {arr.nbytes/1e9:.1f} GB')`. If >2GB, stream or chunk. V19 likely OOM'd loading a large dataset into memory. Never load a full high-frequency dataset (Paderborn, Hydraulic) without downsampling first.

---

## Success Criteria (ordered by paper impact)

- [ ] Pred-FT vs Frozen vs E2E at 100% and 5% labels on FD001 (5 seeds each) — **the headline result**
- [ ] Benchmark table: all datasets with two-stage F1 + legacy metric + 95% CI
- [ ] Pred-FT on at least one non-C-MAPSS dataset (SMAP or PSM)
- [ ] Chronos comparison with unified eval
- [ ] Causal vs bidirectional ablation
- [ ] EMA vs SIGReg ablation
- [ ] Paper_v20_draft.tex tables filled
- [ ] One reviewer round
