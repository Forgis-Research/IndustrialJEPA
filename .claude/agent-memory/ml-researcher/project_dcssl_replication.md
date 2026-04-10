---
name: DCSSL Replication Results and Findings
description: Complete overnight replication of Shen et al. 2026 DCSSL on FEMTO bearing RUL, all 10 experiments done
type: project
---

## DCSSL Replication Status (COMPLETE — 2026-04-10)

**Paper:** Shen et al. 2026, "A novel dual-dimensional contrastive self-supervised learning-based framework for rolling bearing remaining useful life prediction", Scientific Reports

**Data:** FEMTO/PRONOSTIA bearing RUL, 3 conditions, 6 train + 11 test bearings

### FINAL RESULTS (All 10 Experiments)

| Method | Avg MSE (11 bearings) | vs Paper DCSSL | Wins vs Paper DCSSL |
|--------|---------------------|----------------|---------------------|
| Paper DCSSL | 0.0375 | baseline | — |
| **Our DCSSL (RUL fix)** | **0.0807** | +115% | **3/11** |
| Our JEPA+HC | 0.0874 | +133% | 1/11 |
| Our SimCLR | 0.0975 | +160% | 3/11 |
| Our SupCon | 0.1257 | +235% | 3/11 |
| Trivial baseline | 0.0578 | — | — |

**None beat the trivial baseline overall — condition 2 FPT shift dominates.**

### Per-Condition Best Results

| Condition | Our Best Method | Avg MSE | Paper DCSSL |
|-----------|----------------|---------|-------------|
| Cond 1 | DCSSL | 0.0441 | 0.0279 |
| Cond 2 | DCSSL | 0.1308 | 0.0533 |
| Cond 3 | SimCLR | 0.0084 | 0.0068 |

### Wins vs Paper DCSSL (our best per bearing)

| Bearing | Our Best | Paper DCSSL | Method |
|---------|----------|-------------|--------|
| 1_4 | **0.0304** | 0.0476 | SupCon |
| 1_6 | **0.0707** | 0.0892 | SupCon |
| 2_5 | **0.0512** | 0.2538 | SimCLR (FPT=0%) |
| 2_7 | **0.0034** | 0.0075 | DCSSL |

### Critical Bug Fixes Made

1. **elapsed_time shortcut** (prior session): use_elapsed_time=False in RULHead
2. **Instance contrastive loss FPT fix** (this session): use actual RUL values for degradation stage proximity instead of normalized time (fixes FPT distribution shift for cond2/3)
3. **Paper Table 3 column swap correction**: SimCLR and SupCon columns were swapped in original recording

### Key Technical Findings

1. **FPT distribution shift** is the dominant failure mode:
   - Train bearings cond2: FPT=19-25%
   - Test bearings cond2: FPT=0-98%
   - Models learned "halfway = degrading" but test bearings degrade at different times
   - This causes inverted predictions on some bearings (e.g., SupCon 2_4: pred_healthy=0.236, pred_deg=0.710)

2. **HC features solve late-FPT bearings**: JEPA+HC wins on Bearing2_6 (FPT=98%):
   - JEPA+HC: 0.0135 vs DCSSL: 0.1807 vs SimCLR: 0.3305
   - Kurtosis/RMS stay near baseline for 98% of life → correct health signal

3. **JEPA encoder collapsed**: loss_var ≈ 0.97-0.98 throughout all 300 epochs (all 3 conditions)
   - Results driven entirely by HC features, not encoder representations
   - Variance regularization (relu(1-std)) is too weak relative to prediction loss

4. **Paper's architectural gap**: Paper uses 20-timestamp sliding windows (we used single snapshots)
   - This is the primary reason for the performance gap
   - Temporal context within window provides degradation rate information independent of FPT

5. **Trivial baseline context**:
   - Trivial avg MSE = 0.0578
   - Paper SimCLR stated avg = 0.0583 (essentially trivial-level!)
   - Our methods beat trivial on individual bearings but not overall (cond2 failures)

6. **Paper DCSSL suspicious results**: Claims 5-14x better than trivial on bearings with FPT=97-98%
   - These bearings have only 7-51 degradation points (2-14 seconds of data)
   - Possible: paper evaluates only degradation phase, different normalization, or data leakage

### Paper Value Correction (IMPORTANT)

The SimCLR and SupCon columns in the experiment log were initially SWAPPED. Verified from PDF:
- Column order: InfoTS | USL | CBHRL | SimCLR | SupCon | DCSSL
- Key corrected values:
  - SimCLR: 1_4=0.0560, 2_3=0.1849, 2_4=0.2577, 2_5=0.2782
  - SupCon: 1_3=0.0213, 2_3=0.0150, 2_4=0.0017, 2_5=0.2752
- Paper's SimCLR stated avg=0.0583 doesn't match 11-bearing mean=0.0834 (inconsistency)

### Files

- Code: `/home/sagemaker-user/IndustrialJEPA/dcssl-replication/`
- Paper PDF: `/home/sagemaker-user/IndustrialJEPA/dcssl-replication/nature-rolling-bearing-dual-dim-2026.pdf`
- Experiment log: `/home/sagemaker-user/IndustrialJEPA/dcssl-replication/EXPERIMENT_LOG.md`
- Figures: `/home/sagemaker-user/IndustrialJEPA/dcssl-replication/figures/`
- All results: `/home/sagemaker-user/IndustrialJEPA/dcssl-replication/results/`
- Post-analysis: `/home/sagemaker-user/IndustrialJEPA/dcssl-replication/post_experiment_analysis.py`

**Why:** Understanding the full failure modes and wins of each method is essential for the IndustrialJEPA paper comparing JEPA-based pretraining to contrastive SSL on bearing RUL.
**How to apply:** When citing DCSSL comparison results, use per-bearing comparisons not just overall avg. Note the architectural gap (single snapshot vs 20-timestamp windows). The JEPA encoder collapse is a key weakness to address in future work.
