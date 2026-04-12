# V13 Planning: Closing the JEPA-STAR Gap

Session: Post-V12 planning
Goal: Close the ~2 RMSE gap between JEPA E2E (14.23) and STAR (12.19) on FD001.

## What V12 Taught Us (Evidence Base)

### What Works
- Trajectory JEPA pretraining learns genuine degradation structure (H.I. R2=0.926)
- E2E fine-tuning outperforms frozen at ALL label budgets
- PC1 captures 47.6% of variance, strongly correlated with H.I. (rho=0.797)
- The architecture (V2, d=256, L=2, 2.58M) is the right size range

### Where the Gap Is
- JEPA E2E 100%: 14.23 vs STAR 100%: 12.19 (delta = 2.04 RMSE)
- The gap is at 100% labels - NOT a label-efficiency problem
- Frozen (17.81) vs E2E (14.23): E2E adds 3.58 RMSE via calibration, not tracking

### What Doesn't Work (V12 Evidence)
- Global normalization + op-setting channels: catastrophic failure on FD002
- V3 architecture (d=128, L=3): worse than V2 despite more depth
- Extended external fine-tuning: tried in V11, no improvement

### Key Mechanistic Insight (V12)
- Frozen encoder has HIGHER tracking rho (0.856) than E2E (0.804)
- E2E advantage = calibration (mapping to correct RUL scale), not detection
- This suggests: if we could improve the probe-encoder calibration, we'd close the gap

---

## Top 3 Hypotheses for V13

### Hypothesis 1: Improved Fine-Tuning Schedule
**Rationale**: E2E fine-tuning at LR=1e-4 for 100 epochs may be suboptimal. The encoder
was pretrained at LR=3e-4. E2E fine-tuning should use a warmer schedule that lets the
probe learn first, then adapts the encoder.

**Experiments**:
1a. Warmup-freeze: freeze encoder for first 20 epochs, then unfreeze
1b. Lower E2E LR: try LR=5e-5 (half the current 1e-4)
1c. Cyclic LR: cosine annealing over E2E fine-tuning
1d. Weight decay: add L2 regularization (1e-4) to E2E to reduce overfitting at low labels

**Expected gain**: 0.5-1.5 RMSE on FD001 at 100% labels
**Time**: 1-2 hours per experiment (5 seeds x 100 epochs)
**Kill criterion**: If none improve by > 0.5 RMSE, fine-tuning protocol is not the bottleneck

### Hypothesis 2: Longer Pretraining / More Aggressive Masking
**Rationale**: STAR uses a 3.67M param model trained end-to-end. JEPA pretrain is 200 epochs
with 8-window sequences. STAR likely sees the full degradation trajectory during training.
Longer pretraining or more data augmentation might help.

**Experiments**:
2a. Longer pretraining: 400 epochs (vs current 200)
2b. Variable horizon: k in [5, 50] (vs current [5, 30]) - predict further ahead
2c. Multi-engine batching: sample sequences from multiple engines per batch (not just 8)
2d. Data augmentation: Gaussian noise, time stretch during pretraining

**Expected gain**: 0.5-2 RMSE
**Time**: 3-6 hours for pretraining + fine-tuning
**Kill criterion**: If probe RMSE after pretraining doesn't improve vs current 19.22

### Hypothesis 3: Architecture Scale-Up
**Rationale**: V2 has 2.58M params for the context encoder. STAR has 3.67M. Scaling to
4M+ params might close the gap.

**Experiments**:
3a. V4: d=256, L=4 (deeper, ~3.5M params)
3b. V5: d=384, L=2 (wider, ~4.5M params)
3c. V6: d=256, L=2 + cross-attention between context and target windows

**Expected gain**: 0.5-1.5 RMSE
**Time**: 4-8 hours per architecture
**Kill criterion**: If V4 doesn't improve over V2 frozen probe, depth doesn't help

---

## FD002 Fix (Lower Priority)

V12 Phase 1.3 showed that global normalization + op-setting channels fails badly.
The correct approaches for V13 FD002 are:

### Approach A: Condition Token
- Keep 14 sensor channels with per-condition normalization
- Prepend a learnable condition token to each sequence
- The condition token is determined by KMeans clustering of op-settings (already done)
- This separates normalization (per-condition) from condition conditioning (token)

### Approach B: Condition-Conditioned Normalization
- Fit separate mean/std per condition (already done in V11)
- The condition token replaces per-condition normalization with a soft condition embedding
- Allows the encoder to learn condition-invariant degradation features

### Approach C: Residual Normalization
- Normalize each sensor by its within-engine mean (detrend by engine)
- This removes the per-engine mean that's dominated by operating condition
- Remaining signal captures condition-relative degradation

**Recommended first experiment**: Approach A (condition token) - it's the least invasive
change to the existing V11 architecture and most directly addresses the normalization problem.

---

## V13 Execution Order

### Phase 0 — Carry-over from v12 crash (RUN FIRST, gates narrative)

The VM crashed during v12 before these completed. They must finish before
any v13 hypothesis work begins — the STAR label sweep is a kill criterion
for the entire label-efficiency narrative.

0a. **STAR label-efficiency sweep (FD001)** — the v12 Phase 2 job that
    never finished. Using `paper-replications/star/run_experiments.py`:
    - Label budgets: 100% (already done = 12.19), 50%, 20%, 10%, 5%
    - 5 seeds per budget, same seeds as v11 JEPA (42, 123, 456, 789, 1024)
    - Output: `experiments/v13/star_label_efficiency.json`
    - **Kill criterion**: if STAR@20% <= 14 RMSE, the label-efficiency
      pitch is dead; paper pivots to H.I. recovery headline.
    - Launch in background, collect when done.

0b. **STAR FD004** — also crashed mid-run. Complete the 5-seed sweep.
    Output: `experiments/v13/star_fd004_results.json`

0c. **From-scratch ablation** (from ADDENDUM) — same V2 architecture,
    same E2E protocol, random init instead of pretrained. 5 seeds at
    100%, 20%, 10%, 5%. Output: `experiments/v13/from_scratch_ablation.json`
    This quantifies the pretraining contribution under E2E.

0d. **Length-vs-content ablation** (from ADDENDUM) — inference only,
    ~15 min. Constant-input test, length-matched swap, temporal shuffle.
    Output: `experiments/v13/length_vs_content_ablation.json`
    This closes the last loophole on the representation-quality claim.

**Decision point after Phase 0**: read STAR label sweep results and
from-scratch ablation before proceeding. If STAR@20% < 14, skip
Hypotheses 1-3 and go straight to FD002 fix + paper rewrite. If
from-scratch delta < 1 RMSE, the E2E number is not an SSL result and
the paper must lead exclusively with frozen/H.I. claims.

### Phase 1 — Hypothesis testing (only if Phase 0 clears)

1. First: Hypothesis 1a (warmup-freeze) - quick, cheap, tests fine-tuning
2. Second: Hypothesis 1d (L2 weight decay) - addresses high variance at 5% labels
3. Third: Hypothesis 2b (longer horizon) - addresses predictor range
4. Fourth: Hypothesis 3a (V4 architecture) - tests depth scaling
5. Fifth: FD002 Approach A (condition token) - requires architecture change

### Starting Point for V13

```
Checkpoint: mechanical-jepa/experiments/v11/checkpoints/best_pretrain_L1_v2.pt
            (same as used in V12 verification)
Script base: mechanical-jepa/experiments/v11/part_g_downstream_eval.py
V13 dir: mechanical-jepa/experiments/v13/
```

### Success Criteria

**Minimum**: JEPA E2E FD001 RMSE < 13.5 (5 seeds, passes V12's 5-minute sanity checks)
**Target**: JEPA E2E FD001 RMSE < 12.5 (approaches STAR replication 12.19)
**Stretch**: FD002 frozen RMSE < 20 (with condition token fix)

---

## V12 STAR Label Sweep (CRASHED — now Phase 0a above)

Once Phase 0a completes, the kill criterion determines narrative direction:

- If STAR@20% > 16 (JEPA@20% = 16.54): label-efficiency pitch is STRONG
- If STAR@20% in [14, 16]: label-efficiency pitch survives (JEPA frozen is competitive)
- If STAR@20% <= 14: label-efficiency pitch dead; paper pivots to H.I. recovery headline

Based on STAR FD001 architecture (scales down with fewer engines), STAR@20% is likely
12-16 RMSE. Our JEPA E2E@20% = 16.54, frozen@20% = 19.83.
STAR is probably better than JEPA at 20% labels. The question is by how much.

---

## Notes from V12

- Phase 2 uses STAR's architecture (build_model from paper-replications/star)
- STAR 100% replication = 12.19 +/- 0.55 (FD001) - this is the reference
- If STAR is computed on 20% of 72 engines (training) = 14.4 engines - similar to JEPA's 14 engines
- STAR might be more competitive at low labels because it's supervised (can overfit less
  with matched architecture)

Key open question: Does JEPA FROZEN (no fine-tuning risk) outperform STAR at very low
labels (5%)? JEPA frozen@5% = 21.53, STAR unknown@5%. If STAR@5% is 20+, that's the
key selling point.
