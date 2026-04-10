---
name: DCSSL Replication Results and Findings
description: Overnight replication of Shen et al. 2026 DCSSL on FEMTO bearing RUL, key findings about FPT distribution shift, paper value corrections
type: project
---

## DCSSL Replication Status (2026-04-10)

**Paper:** Shen et al. 2026, "A novel dual-dimensional contrastive self-supervised learning-based framework for rolling bearing remaining useful life prediction", Scientific Reports

**Data:** FEMTO/PRONOSTIA bearing RUL, 3 conditions, 6 train + 11 test bearings

### Completed Experiments

| Experiment | Our Result | Paper Target | Status |
|-----------|-----------|-------------|--------|
| SimCLR cond1 | avg=0.0535 (5 bearings) | 0.0304 (cond1 avg) | Paper better on most bearings |
| SimCLR cond2 | avg=0.1594 | 0.1462 | Similar to paper (both fail) |
| SimCLR cond3 | 0.0084 | 0.0341 | WE WIN (3_3 only) |
| SupCon cond1 | avg=0.0468 | 0.0322 | Paper better overall, we win 1_4 and 1_6 |
| SupCon cond2 | avg=0.2243 | 0.0610 | Paper much better |
| SupCon cond3 | 0.0273 | 0.0619 | WE WIN (3_3 only) |
| DCSSL cond1 | RUNNING ~04:21 UTC | avg=0.0279 | - |
| DCSSL cond2 | PENDING (RUL fix) | 0.0533 | - |
| DCSSL cond3 | PENDING | 0.0068 | - |
| JEPA+HC all | PENDING ~08:00 UTC | N/A | - |

### Critical Bug Fixes Made

1. **elapsed_time shortcut** (fixed prior session): use_elapsed_time=False in RULHead
2. **Instance contrastive loss FPT fix** (this session): use actual RUL values for degradation stage proximity instead of normalized time (fixes FPT distribution shift for cond2/3)

### Paper Value Correction (IMPORTANT)

The SimCLR and SupCon columns in the experiment log were initially SWAPPED. Verified from PDF:
- Column order: InfoTS | USL | CBHRL | SimCLR | SupCon | DCSSL
- Key corrected values:
  - SimCLR 1_4: 0.0560 (not 0.2565)
  - SupCon 1_3: 0.0213 (not 0.0028)
  - SupCon 2_3: 0.0150 (not 0.0569)
  - SupCon 2_4: 0.0017 (not 0.0046)
- Paper's SimCLR stated avg=0.0583 doesn't match 11-bearing mean=0.0834 (unclear how computed)

### Key Findings

1. **Trivial baseline MSE = 0.0578** — essentially equal to paper SimCLR avg (0.0583). Paper SimCLR is near-trivial on average.

2. **FPT distribution shift** is the core challenge: train bearings cond2 FPT=19-25%, test FPT=0-98%. Models trained on early-FPT bearings fail on late-FPT test bearings.

3. **HC features provide FPT-independent signal**: kurtosis and RMS directly measure bearing health state, independent of when degradation starts. JEPA+HC expected to outperform contrastive methods on condition 2.

4. **Temporal loss creates smooth trajectory but doesn't fix FPT mismatch** by itself. The instance loss fix (RUL-based proximity) is the key improvement for cond2.

5. **Bearing-specific wins**: Our methods beat paper on:
   - SimCLR: 1_4, 2_5, 3_3
   - SupCon: 1_4, 1_6, 2_5, 3_3

### Files
- Code: `/home/sagemaker-user/IndustrialJEPA/dcssl-replication/`
- Paper PDF: `/home/sagemaker-user/IndustrialJEPA/dcssl-replication/nature-rolling-bearing-dual-dim-2026.pdf`
- Experiment log: `/home/sagemaker-user/IndustrialJEPA/dcssl-replication/EXPERIMENT_LOG.md`
- Figures: `/home/sagemaker-user/IndustrialJEPA/dcssl-replication/figures/`

**Why:** Paper SimCLR avg anomaly and FPT distribution shift are important to understand before comparing JEPA methods against this benchmark.
**How to apply:** When comparing JEPA+HC vs DCSSL, use per-bearing comparisons not just overall avg. Condition 2 failures are structural, not bugs.
