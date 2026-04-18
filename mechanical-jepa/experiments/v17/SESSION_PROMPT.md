# V17 Overnight Session — Autonomous Experiment Run

**Usage**: Paste this as opening prompt to a Claude Code session on the GPU VM.
**Working directory**: `IndustrialJEPA/mechanical-jepa/`
**Duration**: overnight (~8 hours on A10G)
**Agent**: use `ml-researcher` sub-agent for experiment design validation, literature checks, and result interpretation. Use `Explore` for codebase navigation.

---

## Mission

Execute the V17 experiment plan (`experiments/v17/PLAN.md`) in 6 phases:
1. V17 baseline (LogUniform k, fixed-window target)
2. Trajectory probing (predictor at inference)
3. Curriculum EMA → SIGReg (two variants)
4. TTE via trajectory sweep
5. SMAP anomaly detection
6. Quarto notebook summary

**Commit every hour** with descriptive messages. Tag each commit with the phase.

---

## Read first (mandatory)

1. `experiments/v17/PLAN.md` — full plan with architecture, phases, success criteria, F1 evaluation protocol
2. `experiments/v17/SMAP_FIX.md` — SMAP/MSL bug diagnosis (argument order, encoder choice, overtraining)
3. `experiments/v11/models.py` — V2 model definition (ContextEncoder, TargetEncoder, Predictor, TrajectoryJEPA)
4. `experiments/v11/train_utils.py` — training utilities (data loading, C-MAPSS preprocessing)
5. `experiments/v15/phase1_sigreg.py` — SIGReg implementation (SIGRegEP class, lines 82-257)
6. `experiments/v16/RESULTS.md` — V16 results for context
7. `evaluation/grey_swan_metrics.py` — anomaly_metrics signature: `(scores, y_true, threshold)` — NOT (labels, scores)

---

## Phase 1: V17 Baseline (target: ~90 min)

Write `experiments/v17/phase1_v17_baseline.py`. This is the core architectural change.

**What to change from V2 (v11/models.py):**

1. **Target window**: instead of `target = ema_encoder(x_{t+1:t+k})`, use `target = ema_encoder(x_{t+k:t+k+w})` with w=10 fixed.

2. **LogUniform k sampling**: replace `k ~ U[5, 30]` with:
```python
k = int(torch.exp(torch.empty(1).uniform_(0, math.log(K_max))).clamp(min=1).item())
k = min(k, available_future - w)  # ensure target window fits
```
K_max = 150 for C-MAPSS.

3. **Everything else unchanged**: same encoder (causal, 2L, d=256), predictor (MLP), EMA (0.99), L1 loss, 200 epochs, cosine LR from 3e-4, batch=64.

**Data handling**: the context is always x_{1:t} (full history from cycle 1). The target is x_{t+k:t+k+w}. Must ensure t+k+w ≤ T (engine length). Sample t uniformly from valid range.

**Evaluation**: every 10 epochs, run frozen linear probe on h_past (same protocol as V2). Save best checkpoint. 3 seeds (42, 123, 456).

**F1 evaluation** (in addition to RMSE): for each test engine at each cycle t, compute binary label y_k = 1 if RUL(t) ≤ k (for k=30). Probe output thresholded → binary → compute F1, precision, recall, AUC-PR. This makes C-MAPSS results directly comparable to anomaly detection metrics.

**Save results to** `experiments/v17/phase1_results.json`:
```json
{
  "config": "v17_baseline",
  "w": 10, "K_max": 150,
  "seeds": [42, 123, 456],
  "probe_rmse_per_seed": [...],
  "probe_rmse_mean": ...,
  "probe_rmse_std": ...,
  "f1_at_k30": ...,
  "auc_pr_at_k30": ...,
  "v2_baseline_rmse": 17.81
}
```

**Success**: mean frozen probe RMSE ≤ 17.81.

---

## Phase 2: Trajectory Probing (target: ~45 min)

Write `experiments/v17/phase2_trajectory_probe.py`.

**Using Phase 1 best checkpoints** (encoder + predictor frozen):

1. For each test sample at time t:
   - h = encoder(x_{1:t})  — frozen, 256-dim
   - γ(k) = predictor(h, k) for k ∈ {5, 10, 20, 50, 100}  — frozen, 256-dim each
   - features = concat([h, γ(5), γ(10), γ(20), γ(50), γ(100)])  — 1536-dim

2. Train a linear probe (1536 → 1) with MSE loss on RUL labels. Same protocol as V2 frozen probe (LR=1e-3, same val/test split).

3. Compare to Phase 1 (h_past only, 256-dim) and V2 (17.81).

4. Report RMSE + F1/AUC-PR at k=30 (same binary task as Phase 1).

**Also test**: probe on γ(k) alone for each k (which horizon is most informative for RUL? for F1?).

**Save results to** `experiments/v17/phase2_trajectory_probe_results.json`.

**Success**: trajectory probe < 15.0 RMSE.

---

## Phase 3: Curriculum EMA → SIGReg (target: ~2.5 hours)

Write `experiments/v17/phase3_curriculum_sigreg.py`.

**Starting from Phase 1 checkpoint at epoch 100** (not epoch 200 — we want the model mid-training):

**Schedule:**
```
Epoch 100-150:  EMA (0.99) + SIGReg λ ramps linearly 0 → 0.05
Epoch 150-200:  No EMA (stop-grad on SAME encoder) + SIGReg λ = 0.05
```

When EMA is dropped (epoch 150+): the target becomes `stop_grad(encoder(x_{t+k:t+k+w}))` — same encoder weights, just detached. The prediction loss + SIGReg prevent collapse.

**Two variants to run:**

**(a) SIGReg after encoder**: `L_sig = SIGReg(encoder(x_{1:t}))`
- Reuse `SIGRegEP` class from `experiments/v15/phase1_sigreg.py` (lines 178-257)
- Applied to h_past (encoder output)

**(b) SIGReg after predictor**: `L_sig = SIGReg(predictor(h, k))`
- Same SIGRegEP class, applied to γ(k) (predictor output)
- Novel: regularizes the space where probes operate

**Total loss** (epoch 100-200):
```python
L = L_prediction + lambda_sig * L_sig  # lambda_sig ramps 0→0.05 over epochs 100-150
```

**Evaluation**: frozen probe + trajectory probe every 10 epochs. Track PC1 explained variance (isotropy diagnostic) every 10 epochs.

Run both variants with 3 seeds each (6 runs total, can parallelize if 2 GPUs).

**Save results to** `experiments/v17/phase3_curriculum_results.json`.

**Success**: matches Phase 1 frozen quality (≤ 17.81) without target network. Variant (b) preferred if trajectory probe is better.

---

## Phase 4: TTE via Trajectory Sweep (target: ~45 min)

Write `experiments/v17/phase4_tte_sweep.py`.

**Using best checkpoint from Phases 1-3:**

1. Define TTE ground truth for C-MAPSS FD001:
   - Sensor: s14 (corrected fan speed, index 10 in 14-sensor subset)
   - Threshold: μ - 3σ computed from cycles 1-50 (healthy baseline) per engine
   - TTE(t) = first cycle after t where s14 crosses threshold (or ∞ if never)

2. Train event-boundary probe:
   - Input: γ(k) = predictor(h, k), 256-dim
   - Output: p(s14 crosses threshold within k cycles), sigmoid
   - Train with BCE loss on (γ(k), y_k) pairs where y_k = 1 if TTE ≤ k
   - Sample k from same LogUniform distribution as pretraining

3. Inference:
   - For each test sample: sweep k = 1, 2, ..., 150
   - TTE_hat = min { k : probe(γ(k)) > 0.5 }
   - Compare to ground-truth TTE

4. Evaluate: **F1 (primary)**, precision, recall, AUC-PR. Also RMSE for backward compat.

**Save results to** `experiments/v17/phase4_tte_results.json`.

**Success**: TTE F1 and AUC-PR reported. Paper section 5.4 stops being future work.

---

## Phase 5: SMAP Anomaly Detection (target: ~90 min)

Write `experiments/v17/phase5_smap_anomaly.py`.

**CRITICAL: Read `experiments/v17/SMAP_FIX.md` first.** V16 Phase 3 crashed due to 3 bugs.

**Pretrain V17 on SMAP:**
- Use data adapter from `data/smap_msl.py`
- k ~ LogU[1, 500], w=10
- Same encoder/predictor architecture (adjust n_sensors=25 for SMAP)
- **Use EMA mode, NOT SIGReg** (V16 used SIGReg which overfits anomaly patterns)
- **50 epochs** (not 100 — V16 showed overtraining inverts the anomaly signal)

**Bug fixes to apply:**
1. `anomaly_metrics(scores, labels, threshold)` — NOT `(labels, scores)`. Check signature.
2. For scoring: use `target_encoder` (EMA copy), NOT `context_encoder`.
3. Monitor anomaly-vs-normal score gap. If anomalies score LOWER, stop early.

**Anomaly scoring:**
- score(t) = ||predictor(encoder(x_{1:t}), k) - **ema_target_encoder**(x_{t+k:t+k+w})||_1
- Test with k ∈ {5, 10, 20, 50} and average scores across k values
- Also test: use trajectory probe score instead of raw prediction error

**Evaluate**: **non-PA F1 (primary)**, PA-F1 (MTS-JEPA comparability), AUC-PR.
MTS-JEPA references: SMAP PA-F1=33.6%, SWaT PA-F1=72.9%.
Use `evaluation/grey_swan_metrics.py` — `anomaly_metrics(scores, y_true, threshold)`.

**Save results to** `experiments/v17/phase5_smap_results.json`.

**Success**: non-PA F1 > 0.10. PA-F1 reported alongside for MTS-JEPA comparison.

---

## Phase 6: Quarto Notebook (target: ~45 min)

Write `notebooks/17_v17_analysis.qmd`.

**Requirements:**
- Concise educational walkthrough — a collaborator should understand V17 in one read
- Load all results JSONs from experiments/v17/
- Key sections:
  1. **Motivation** (2 paragraphs): why LogUniform k, why trajectory probing, why curriculum SIGReg
  2. **Architecture diagram**: pretrain (encoder + predictor + EMA) → probe (frozen encoder + frozen predictor + linear probe) → infer (sweep k, find event)
  3. **Phase 1 results**: LogUniform vs V2 baseline, effect of wider k range
  4. **Phase 2 results**: trajectory probe vs h_past probe, which k values are most informative
  5. **Phase 3 results**: curriculum SIGReg transition, PC1 trajectory, encoder vs predictor placement
  6. **Phase 4 results**: TTE sweep visualization (predicted p(event) vs k for example engines)
  7. **Phase 5 results**: SMAP anomaly timeline with multi-k scoring
  8. **Summary table**: all variants, frozen probe, trajectory probe, TTE, anomaly

Use the V15 analysis notebook (`notebooks/15_v15_analysis.qmd`) as a style reference.

---

## Ground rules

- **Hourly commits.** Use descriptive messages:
  ```
  v17 phase N (HH:MM): <what was done>
  ```
- **Every number in a results JSON.** No hardcoded claims.
- **Do not modify experiments/v11/ or any other version.** V17 is self-contained.
- **Reuse existing code by importing**, not copy-pasting:
  - Models: `sys.path.insert(0, v11_dir); from models import TrajectoryJEPA, ...`
  - SIGReg: `sys.path.insert(0, v15_dir); from phase1_sigreg import SIGRegEP`
  - Data: `from data.smap_msl import SMAPDataset`
  - Metrics: `from evaluation.grey_swan_metrics import ...`
- **If a phase fails or takes too long (>2h), skip it.** Note the failure in `experiments/v17/RESULTS.md` and move on.
- **Use `ml-researcher` sub-agent** for: validating experiment design before running, interpreting surprising results, literature context for curriculum SIGReg.
- **Save all plots** to `analysis/plots/v17/` as PNG (for notebook) and PDF (for paper figures/).

---

## Success criteria (end of session)

At minimum:
- [ ] Phase 1 complete: V17 baseline trained, frozen probe measured, results JSON saved
- [ ] Phase 2 complete: trajectory probe results
- [ ] `experiments/v17/RESULTS.md` exists with at least Phase 1-2 results
- [ ] At least 4 hourly commits

Stretch:
- [ ] Phase 3: curriculum SIGReg results for both variants
- [ ] Phase 4: TTE numbers (paper section 5.4 stops being future work)
- [ ] Phase 5: SMAP results
- [ ] Phase 6: Quarto notebook rendered

---

## Fallback ordering (if time is short)

If you can only do 3 phases: do 1, 2, 6 (baseline + trajectory probe + notebook).
If you can only do 4: add Phase 4 (TTE — fills a paper gap).
Phase 3 (curriculum SIGReg) and Phase 5 (SMAP) are the most experimental and can be deferred.
