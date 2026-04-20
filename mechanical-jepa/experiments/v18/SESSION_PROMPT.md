# V18 Overnight Session — E2E, Honest Probing, Reviewer Loop, MTS-JEPA Comparison

**Usage**: Paste this as opening prompt to a Claude Code session on the GPU VM.
**Working directory**: `IndustrialJEPA/mechanical-jepa/`
**Duration**: overnight (~10-12 hours on A10G)
**Agent**: use `ml-researcher` sub-agent for experiment design validation, literature checks, and result interpretation. Use `neurips-reviewer` for paper reviews. Use `paper-writer` for revisions. Use `Explore` for codebase navigation.

---

## Mission

Execute the V18 experiment plan (`experiments/v18/PLAN.md`) in 7 phases. The three goals are:
1. **Beat v11's 13.80 E2E RMSE** with v17's architectural improvements
2. **Establish honest baselines** by re-probing v2 with fixed protocol
3. **Harden the paper** via reviewer-writer loop and rigorous MTS-JEPA comparison

**Commit every hour** with descriptive messages. Tag each commit with the phase.

---

## Read first (mandatory)

1. `experiments/v18/PLAN.md` — full plan with all phases, success criteria, hyperparameters
2. `experiments/v17/RESULTS.md` — v17 results (our current best)
3. `experiments/v17/PLAN.md` — v17 architecture explanation (LogU k, fixed-window, trajectory probing, curriculum SIGReg)
4. `experiments/v11/models.py` — V2 model definition (TrajectoryJEPA, ContextEncoder, TargetEncoder, Predictor)
5. `experiments/v11/data_utils.py` — C-MAPSS data loading
6. `experiments/v17/phase2_trajectory_probe.py` — honest probe protocol (WD=1e-2, val n_cuts=10)
7. `experiments/v17/phase3_curriculum_sigreg.py` — curriculum SIGReg implementation
8. `experiments/v15/phase1_sigreg.py` — SIGRegEP class (lines 178-257)
9. `experiments/v14/RESULTS.md` — cross-sensor attention results, full-sequence results, multi-subset table
10. `evaluation/grey_swan_metrics.py` — anomaly_metrics signature: `(scores, y_true, threshold)` — NOT `(labels, scores)`
11. `paper-neurips/paper.tex` — current paper draft (for reviewer loop)

---

## Phase 0: Honest re-probe of v2 baseline (~30 min)

**Why this matters**: V17's "improvement" from 17.81 to 15.38 may be partially or entirely a probe fix, not an architecture win. We need ground truth.

**Steps**:
1. Check `checkpoints/` directory for v2-compatible weights. Load each .pt file into a TrajectoryJEPA(n_sensors=14, d_model=256, n_heads=4, n_layers=2) and check if it loads without error. The files are from March 30-31 and may be v2 checkpoints.
2. If compatible v2 checkpoint found:
   - Probe with **old protocol**: `CMAPSSFinetuneDataset(val_engines, use_last_only=True)`, no WD, Adam LR=1e-3
   - Probe with **new protocol**: `CMAPSSFinetuneDataset(val_engines, n_cuts_per_engine=10)`, WD=1e-2, AdamW
   - 5 seeds for each protocol
   - Report both side-by-side
3. If no compatible checkpoint: pretrain v2 from scratch with original config (k~U[5,30], 200 epochs). Then probe with both protocols.

**Key config for v2 pretrain** (if needed):
```python
# Original v2: target = ema_encoder(x_{t+1:t+k}), k ~ U[5,30]
# Everything else same: d=256, 2L, EMA 0.99, L1, cosine LR 3e-4, 200 epochs
```

**Save to**: `experiments/v18/phase0_honest_reprobe.json`
```json
{
  "v2_old_protocol_rmse": {"mean": ..., "std": ..., "per_seed": [...]},
  "v2_new_protocol_rmse": {"mean": ..., "std": ..., "per_seed": [...]},
  "source": "existing_ckpt" or "retrained",
  "interpretation": "..."
}
```

---

## Phase 1: V17 pretrain + E2E fine-tuning (~2.5 hours)

This is the main event. Pretrain v17 architecture, then E2E fine-tune.

### Step 1: Pretrain (~90 min)

Write `experiments/v18/phase1_pretrain.py`. Reuse v17 Phase 1 code but with honest probing built in.

**Config** (same as v17):
- LogU k ∈ [1, 150], w=10, EMA 0.99
- 200 epochs, cosine LR from 3e-4, batch=64
- 3 seeds (42, 123, 456)
- Save checkpoints at ep50, ep75, ep100, ep150, ep200

**Probe protocol** (fixed from v17 Phase 1):
- Every 10 epochs: frozen linear probe on h_past
- WD=1e-2, AdamW, val n_cuts_per_engine=10
- Report RMSE + F1 at k=30

### Step 2: E2E fine-tuning (~60 min)

Write `experiments/v18/phase1_e2e.py`.

From best pretrained checkpoint per seed:
- Unfreeze entire model (encoder + predictor)
- Add linear head: h_past → RUL (same as v11)
- LR=1e-4 (10x lower than pretrain), cosine schedule, 50 epochs
- MSE loss on RUL labels (capped at 125)

**Label efficiency sweep**: 100%, 20%, 10%, 5% labels. 5 seeds per budget.

**Reference numbers to beat**:

| Budget | V2 frozen | V2 E2E | V17 frozen (honest) | V18 E2E (target) |
|--------|-----------|--------|---------------------|-------------------|
| 100% | 17.81 | 13.80 | 15.38 | < 13.80 |
| 20% | 19.83 | 16.54 | — | < 16.54 |
| 10% | 19.93 | 18.66 | — | < 18.66 |
| 5% | 21.53 | 25.33 | — | < 21.53 |

### Step 3: F1 evaluation

For EVERY configuration (frozen, E2E, each budget):
- Binary task: "fails within k cycles?" for k ∈ {10, 20, 30, 50}
- Report: F1, precision, recall, AUC-PR at each k
- Also report RMSE for C-MAPSS backward compatibility

**Save to**: `experiments/v18/phase1_results.json`

---

## Phase 2: Accelerated curriculum SIGReg (~1.5 hours)

Write `experiments/v18/phase2_curriculum.py`.

**Three schedules from Phase 1 ep75/ep100 checkpoints**:

**Schedule A** (v17 original, control):
- EMA 0-100, SIGReg ramp 100-150, SIGReg-only 150-200. Total: 200 epochs.

**Schedule B** (compressed):
- EMA 0-75, SIGReg ramp 75-100, SIGReg-only 100-150. Total: 150 epochs.

**Schedule C** (gradual EMA fade — the novel one):
- EMA 0-75, SIGReg ramp 75-100, EMA momentum fade 0.99→1.0 over 100-120, SIGReg-only 120-150. Total: 150 epochs.

```python
# Schedule C implementation:
if epoch <= 75:
    use_ema = True; momentum = 0.99; lam_sig = 0
elif epoch <= 100:
    use_ema = True; momentum = 0.99
    lam_sig = LAMBDA_SIG_MAX * (epoch - 75) / 25
elif epoch <= 120:
    use_ema = True
    # Gradual fade: momentum 0.99 → 1.0 (frozen target = effectively stop-grad)
    momentum = 0.99 + 0.01 * (epoch - 100) / 20
    lam_sig = LAMBDA_SIG_MAX
    model.ema_momentum = momentum  # update the model's EMA momentum
else:
    use_ema = False  # stop-grad on live encoder
    lam_sig = LAMBDA_SIG_MAX
```

**SIGReg placement**: predictor only (v17 showed encoder placement destroys RUL signal).

**Evaluation per schedule**:
- Honest frozen probe at final epoch
- E2E fine-tune (100% labels, 3 seeds)
- Track loss curve — specifically measure spike at EMA→SIGReg transition
- Track PC1 explained variance trajectory

**Save to**: `experiments/v18/phase2_curriculum_results.json`

---

## Phase 3: Reviewer-Writer loop (~2 hours)

This is a meta-experiment on the paper itself.

### Round 1: Review

Launch 4 independent `neurips-reviewer` agents on `paper-neurips/paper.tex`. Each should produce a structured NeurIPS review with:
- Overall score (1-10)
- Summary, strengths, weaknesses
- Questions for authors
- Suggestions for improvement

### Round 2: Synthesize

Read all 4 reviews. Extract:
1. **Consensus weaknesses** (mentioned by ≥2 reviewers)
2. **Critical experiments needed** (missing baselines, missing datasets, missing ablations)
3. **Writing issues** (clarity, flow, overclaiming)

Write synthesis to `experiments/v18/reviewer_synthesis.md`.

### Round 3: Respond

For each consensus weakness:
- If it's a writing issue: launch `paper-writer` to revise the relevant section
- If it's a missing experiment: design and run it (budget: 30 min compute max per experiment)
- If it's a fundamental limitation: add honest discussion to the paper

**Known weaknesses to watch for** (flag these if reviewers miss them):
- SMAP results inconsistency: paper may claim 62.5% PA-F1 (v15) but v17 got 21.9%. Paper MUST use the most recent, honest number.
- Only FD001 as primary RUL benchmark — need FD003/FD004 (Phase 6 addresses this)
- Frozen probe protocol was broken in v11-v16 — acknowledge this, show corrected numbers
- 3 seeds is minimal for CI — run 5 seeds where possible
- MTS-JEPA comparison is apples-to-oranges (anomaly detection vs prognostics)

### Round 4: Re-review (if time permits)

Launch 2 more `neurips-reviewer` agents on the revised paper. Check if the score improves.

**Save to**: `experiments/v18/reviewer_synthesis.md`, `experiments/v18/reviewer_round2.md`

---

## Phase 4: MTS-JEPA head-to-head comparison (~1.5 hours)

Write `experiments/v18/phase4_mtsjepa_comparison.py` and `experiments/v18/mtsjepa_comparison.md`.

### 4a. Comparison table (no compute needed)

Build a comprehensive comparison table. Source numbers from:
- Our results: v17 RESULTS.md, v18 Phase 1 results, v14 RESULTS.md
- MTS-JEPA: their paper (He et al. 2026), our replication (see `experiments/v14/mtsjepa_comparison.md`)

| Dimension | FAM (ours) | MTS-JEPA |
|-----------|-----------|----------|
| **C-MAPSS FD001 frozen RMSE** | 15.38 (v17) / Phase 1 result | N/A (not evaluated) |
| **C-MAPSS FD001 E2E RMSE** | 13.80 (v11) / Phase 1 result | N/A |
| **C-MAPSS FD001 F1@k=30** | 0.919 (v17) / Phase 1 result | N/A |
| **SMAP non-PA F1** | 0.038 (v17) | ~0.33 (their paper) |
| **SMAP PA-F1** | 0.219 (v17) | 0.336 (their paper) |
| **MSL PA-F1** | 0.433 (v15) | reported in paper |
| **Model size** | 1.26M | ~5-8M |
| **Loss terms** | 1 (L1) | 7+ |
| **Training** | 200 epochs, ~45 min A10G | ? |
| **Prediction horizon** | Stochastic LogU[1,150] | Fixed (1 window) |
| **Multi-scale** | Implicit via LogU k | Explicit dual-resolution |

### 4b. SMAP re-evaluation (compute needed, ~45 min)

Re-run SMAP with our best understanding:
- 50 epochs (v17 showed more training = worse)
- LogU k, w=10
- **New scoring approach**: instead of raw prediction error (which anti-correlates with anomalies because SMAP anomalies are predictable), try:
  1. **Representation shift score**: ||h_past(t) - h_past(t-1)||, detecting when the encoder's internal state changes rapidly
  2. **Trajectory divergence score**: ||γ(k_short) - γ(k_long)||, detecting when short and long horizon predictions disagree
  3. **Mahalanobis distance**: distance of h_past from the training distribution mean (in PCA space)
- Report all scoring methods with non-PA F1 and PA-F1

### 4c. Lead-time decomposition

For whatever SMAP method works best:
- Decompose detections into "continuation" (anomaly already active) vs "prediction" (detected before onset)
- Report the fraction — MTS-JEPA's replication showed 89% continuation on SMAP
- Even if our overall F1 is lower, a higher prediction fraction would be a meaningful win

### 4d. Honest framing for paper

Based on 4a-4c, write the comparison narrative:
- If SMAP is still bad: "FAM targets prognostics (RUL, TTE), not anomaly detection. MTS-JEPA excels at anomaly detection; FAM excels at remaining useful life prediction. Complementary methods for different safety-critical tasks."
- If SMAP improves with new scoring: report honestly, compare fairly

**Save to**: `experiments/v18/mtsjepa_comparison.md`, `experiments/v18/phase4_smap_results.json`

---

## Phase 5: Cross-sensor attention + v17 improvements (~1.5 hours, stretch)

Write `experiments/v18/phase5_cross_sensor.py`.

**Combine v14 cross-sensor with v17 improvements**:
- Sensor-as-token: 14 tokens/cycle, learnable sensor-ID embeddings
- Alternating temporal causal + cross-sensor attention, 2 layer pairs, d=128, 4 heads
- LogU k ∈ [1, 150], w=10 (new from v17)
- Curriculum SIGReg on predictor (Schedule B or C from Phase 2)
- 150 epochs

**Hypothesis**: LogU k + SIGReg curriculum may stabilize the cross-sensor architecture that was brittle in v14 (std=10.19 at 20% labels). The broader horizon range gives the cross-sensor attention more diverse training signal.

**Evaluation**: frozen probe + E2E at 100% and 20% labels, 3 seeds. Report F1 alongside RMSE.

**Reference** (v14 cross-sensor): 14.98 ± 0.22 frozen at 100%, but 25.02 ± 10.19 at 20%.

**Save to**: `experiments/v18/phase5_cross_sensor_results.json`

---

## Phase 6: Multi-subset evaluation — FD003 + FD004 (~1 hour, stretch)

Write `experiments/v18/phase6_multisubset.py`.

Using best Phase 1 checkpoint, evaluate on FD003 and FD004:
- In-domain pretrain on each subset (reuse Phase 1 code, change data source)
- Honest frozen probe + E2E at 100% labels, 3 seeds
- F1 at k=30
- Compare to v14 full-sequence numbers

**Per-condition normalization for FD002/FD004**: use the same KMeans(6) approach from v11/data_utils.py.

**Save to**: `experiments/v18/phase6_multisubset_results.json`

---

## Phase 7: Final results compilation + paper update (~30 min)

1. Write `experiments/v18/RESULTS.md` — comprehensive results table across all phases
2. Update `paper-neurips/paper.tex` with:
   - New E2E numbers (if they beat 13.80)
   - Honest re-probed baselines
   - MTS-JEPA comparison table
   - F1 metrics throughout
   - Corrected SMAP numbers
   - FD003/FD004 numbers (if Phase 6 completed)
3. Regenerate any LaTeX tables

---

## Ground rules

- **Hourly commits.** Use descriptive messages: `v18 phase N (HH:MM): <what was done>`
- **Every number in a results JSON.** No hardcoded claims.
- **Do not modify experiments/v11/ through v17/.** V18 is self-contained.
- **Reuse existing code by importing**, not copy-pasting:
  - Models: `sys.path.insert(0, v11_dir); from models import TrajectoryJEPA, RULProbe`
  - SIGReg: `sys.path.insert(0, v15_dir); from phase1_sigreg import SIGRegEP`
  - Data: `from data_utils import load_cmapss_subset, CMAPSSFinetuneDataset, ...`
  - Metrics: `from evaluation.grey_swan_metrics import anomaly_metrics`
- **Honest probe protocol everywhere**: WD=1e-2, AdamW, val n_cuts_per_engine=10. No exceptions.
- **F1 reported for every config**: alongside RMSE. Binary at k=30 minimum, ideally k ∈ {10, 20, 30, 50}.
- **If a phase fails or takes too long (>2.5h), skip it.** Note the failure in RESULTS.md and move on.
- **Use `ml-researcher` sub-agent** for: validating experiment design, interpreting surprising results, literature context.
- **Use `neurips-reviewer` sub-agent** for Phase 3 reviews. Launch 4 in parallel.
- **Use `paper-writer` sub-agent** for Phase 3 revisions and Phase 7 paper updates.
- **Save all plots** to `analysis/plots/v18/` as PNG (notebook) and PDF (paper figures/).

---

## Priority ordering (if time is short)

**Must-do** (4-5 hours):
1. Phase 0 — honest re-probe (30 min) — this validates whether v17's improvement is real
2. Phase 1 — pretrain + E2E (2.5 hours) — the main result
3. Phase 4a — MTS-JEPA comparison table (30 min, no compute)
4. Phase 7 — results compilation (30 min)

**Should-do** (3 hours):
5. Phase 3 — reviewer-writer loop (2 hours) — hardens the paper
6. Phase 2 — accelerated curriculum (1 hour, can run just Schedule C)

**Stretch** (3 hours):
7. Phase 4b-d — SMAP re-evaluation with new scoring (1 hour)
8. Phase 6 — FD003/FD004 multi-subset (1 hour)
9. Phase 5 — cross-sensor + v17 (1.5 hours)

---

## Success criteria (end of session)

**Minimum**:
- [ ] Phase 0 complete: honest v2 re-probe, side-by-side with old protocol
- [ ] Phase 1 complete: v17 pretrain + E2E at 100% labels, frozen + E2E RMSE + F1
- [ ] Phase 4a complete: MTS-JEPA comparison table in markdown
- [ ] `experiments/v18/RESULTS.md` exists with Phase 0 + Phase 1 numbers

**Target**:
- [ ] Phase 1 full label sweep (100/20/10/5%)
- [ ] Phase 3 complete: 4 reviews + synthesis + at least 1 paper revision
- [ ] Phase 2 complete: at least Schedule C tested

**Stretch**:
- [ ] Phase 4b-d: SMAP with new scoring methods
- [ ] Phase 5: cross-sensor + v17
- [ ] Phase 6: FD003/FD004 numbers
- [ ] Phase 7: paper updated with all new numbers

---

## Key questions to answer by end of session

1. **Is v17's improvement real?** (Phase 0: re-probe v2 honestly)
2. **Does v17 + E2E beat 13.80?** (Phase 1: the headline number)
3. **Can the curriculum be faster?** (Phase 2: gradual EMA fade)
4. **What would reviewers say?** (Phase 3: simulated review)
5. **How do we compare to MTS-JEPA fairly?** (Phase 4: honest comparison)
