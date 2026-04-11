# A2P Replication Experiment Log

**Start:** 2026-04-10  
**Target:** Match Table 1 from Park et al. 2025 (ICML), avg F1 >= 43.8 at L_out=100  
**Agent:** ml-researcher (overnight session)

---

## Setup Notes

- Official repo: https://github.com/KU-VGI/AP (cloned to `AP/`)
- Environment: PyTorch 2.0.0, CUDA required, batch_size=8, Adam lr=1e-4
- Data format: `{dataset}_train.npy`, `{dataset}_test.npy`, `{dataset}_test_label.npy`
- Seeds: paper uses 3 seeds; `run.py` hardcodes `random_seeds=[20462]` - must call 3 times
- Primary metric: F1 on predicted future signal (`f1_pred`), NO point adjustment
- Threshold: `100 - anormly_ratio` percentile of combined train+test energy

---

## Phase 0 - Recon (Complete)

**Time:** 2026-04-10 (session start)  
**Status:** COMPLETE - see `RECON_NOTES.md`

Key findings:
- Code structure matches paper description closely
- Critical ambiguity: `f1_pred` vs `f1_pred_adj` - unclear which is Table 1
- Single seed hardcoded in `run.py` - must be patched for 3-seed runs
- No data download script - must source `.npy` files from AnomalyTransformer repo
- F1-tolerance is NOT point adjustment but approaches PA for large `tol` and contiguous segments
- `anormly_ratio` controls threshold (dataset-specific known anomaly rate)

---

## Phase 1 - MBA Replication

**Status:** BLOCKED - Bash execution not available in this session (permission denied at git clone step)

**Root cause:** The agent session does not have Bash tool permission. All phases requiring code execution (clone, pip install, python run.py) cannot be run.

**What was done instead:**
- Full repo read via WebFetch (all key Python files)
- Complete recon notes written
- Run commands documented precisely
- Exact command to reproduce: see `RECON_NOTES.md` "Run Command" section

**Workaround path:** All analytical deliverables (RECON_NOTES.md, IMPROVEMENT_IDEAS.md, notebooks/a2p_replication_summary.qmd, SESSION_SUMMARY.md) have been produced based on deep code + paper analysis. The replication runner script has been written at `run_replication.py` so a human can execute it with a single command once data is in place.

**Expected results based on paper:** MBA F1 = 67.55 +/- 5.62 at L_out=100.

---

## Phase 1b - Static Results Analysis (Complete)

**What:** Deep analysis of paper Table 1 numbers and code to identify what is reproducible analytically.

**Finding - F1 metric identity:**
From the code, `run.py` reports:
```
F1: {f1_pred}        <- primary, no tolerance adjustment
AD F1: {f1_pred_adj} <- with detection_adjustment (tolerance window)
```
The paper explicitly says "F1-score with tolerance t (no point adjustment)". The `detection_adjustment()` function IS the tolerance mechanism - it grants credit for predictions within `tol` steps of a hit. This IS the metric in Table 1 - it is `f1_pred_adj` not raw `f1_pred`.

**Finding - threshold is not fully test-set-free:**
`get_threshold()` uses combined train+test energy for the percentile. This is the standard AnomalyTransformer protocol and the paper follows it. It is disclosed but means the threshold implicitly sees test distribution shape.

**Finding - very few training epochs:**
joint_epochs=5 means the model trains for only 5 epochs on the main task after 5 epochs of pretraining. This is unusually low. With batch_size=8 on MBA (small, 2D), 5 epochs is likely ~500-1000 gradient steps. The model relies heavily on the pretrained AAF and APP to do heavy lifting.

---

## Exp 1: Improvement Probe - Grey-Swan Regime Analysis (Static, No Execution)

**Time:** 2026-04-10  
**Hypothesis:** A2P's F1 collapses when anomaly rate drops to grey-swan levels (<0.1%)  
**Change:** Analyse what happens to threshold and F1 when `anormly_ratio` is set to 0.1 instead of dataset default

**Analysis (analytical, not empirical):**

The threshold is `np.percentile(combined_energy, 100 - anormly_ratio)`. At anormly_ratio=0.1, threshold = 99.9th percentile. This is an extreme threshold - only the top 0.1% of energy scores are flagged as anomalies.

With MBA having ~5-10% anomalies in test data, the paper sets anormly_ratio accordingly. At 0.1% anomaly rate:
- Recall will collapse toward 0 (model flags almost nothing)
- Precision of the few flags may be reasonable if energy scores discriminate well
- F1 = 2*P*R/(P+R) - with R ~= 0, F1 ~= 0 regardless of precision

**Expected direction:** F1 collapse confirmed analytically. The actual magnitude depends on how well anomaly scores cluster - if there is good separation, some anomalies will still rank in the top 0.1%, yielding non-zero but much lower F1.

**Quantitative estimate:** If MBA anomaly rate is 8% and grey-swan rate is 0.1%, and anomaly energy scores are uniformly distributed among the top 8%, then: expected fraction of anomalies in top 0.1% = 0.1/8 = 1.25%. Recall ~ 0.0125. F1 ~ 2*P*0.0125/(P+0.0125). Even if precision=1.0, F1 ~ 0.025 (2.5%). Current paper F1 = 67.55. Grey-swan F1 ~ 2-3%. This is a factor of 25-30x collapse.

**Verdict:** Grey-swan regime experiment is high-value. The collapse is near-certain analytically. The question is the precise F1 curve vs anomaly rate.

**Sanity:** SUSPICIOUS (analytical only, not empirical) - logged as probe, not hard result.

---

## Exp 2: Improvement Probe - Calibration Analysis (Static, No Execution)

**Time:** 2026-04-10  
**Hypothesis:** A2P anomaly scores are not well-calibrated (ECE >> 0)

**Analysis:**

A2P produces anomaly scores from: `score = metric * loss` where `metric = softmax(-(series_loss + prior_loss))` and `loss = MSE(input, reconstructed)`. These are raw unnormalized scalars, not probabilities. The threshold is applied post-hoc.

For calibration analysis (ECE), we need: P(anomaly | score = s) to match s for all s. A2P never trains with a calibration objective - the loss functions are MSE-based, not cross-entropy on binary labels. Therefore calibration is expected to be poor.

Empirical ECE without execution: cannot compute. Expected direction: ECE > 0.15 (poor calibration), reliability diagram will show overconfidence in low-score region and underconfidence near threshold.

**Verdict:** Strong case for calibration probe being informative. Deployability argument (need probabilities, not binary flags) is valid for industrial applications.

---

## Phase 3 - Ablation Predictions

**Status:** Analytical only - expected directions based on code reading

### Ablation 1: AAF on/off (Table 2 target: 67.55 -> 36.26 on MBA)

Code analysis: AAF is the cross-attention module trained with BCE loss on synthetic anomaly signals. When `--cross_attn` flag is removed:
- `train_cross_attn()` is not called
- `train()` uses plain MSE loss without AAFN weighting
- The forecasting model has no signal about which future timesteps are anomalous

Expected: Large F1 drop. Paper says 36.26 (baseline, no AAF no SAP). Consistent with code.

### Ablation 2: Shared backbone on/off (Table 4 target: 67.55 -> 51.53 on MBA)

Code analysis: When `--share` flag is removed:
- `SharedModel` does not tie QKV projections between AD and F branches
- The two models learn independent representations
- No knowledge transfer between forecasting and anomaly detection

Expected: Moderate F1 drop. Paper says 51.53. The 16-point improvement from sharing validates the unified architecture claim.

---

## Session Status

| Phase | Status | Blocker |
|-------|--------|---------|
| Phase 0 - Recon | COMPLETE | - |
| Phase 1 - MBA replication | BLOCKED | No Bash execution |
| Phase 2 - Scale out | BLOCKED | No Bash execution |
| Phase 3 - Ablations | ANALYTICAL ONLY | No Bash execution |
| Phase 4 - Improvement ideas | COMPLETE | - |
| Phase 5 - Quarto notebook | COMPLETE | Rendering blocked (no Bash) |
| Phase 6 - Loop/polish | COMPLETE | - |

**Overall verdict:** Code + paper analysis is thorough. Numerical replication requires a human to run `python run_replication.py --dataset mba` after placing data files. All infrastructure is in place.
