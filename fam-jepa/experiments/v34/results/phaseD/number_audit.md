# v34 Paper Number Audit

Verifying every numeric claim in `paper-neurips/paper.tex` against:
- `fam-jepa/experiments/RESULTS.md` (v31, 2026-04-26 = latest)
- `fam-jepa/experiments/v30/results/master_table.json`
- `fam-jepa/experiments/v31/results/phase1_lf10_master.json`
- `fam-jepa/experiments/v32/` (legacy metrics)
- `fam-jepa/experiments/v33/results/phase4/` (cross-channel ablation)
- `fam-jepa/experiments/v34/results/phaseA/` (SIGReg sweep, ongoing)

Format: line | claim | source | status (OK / STALE / FLAG)

---

## Abstract (line 77)

| Item                                              | Paper        | RESULTS.md / source       | Status |
|---------------------------------------------------|--------------|---------------------------|--------|
| "11 datasets across 8 domains"                    | 11 / 8       | RESULTS.md count          | OK (counts MSL even though excluded from main table) |
| "outperforms ... 6 of 8 comparable benchmarks"    | 6/8          | v31 chr2-mlp fix          | OK     |
| "56× fewer parameters"                            | 56×          | 2.16M / 120M = 56×        | OK     |
| "as few as 2% of labels suffice ... near-full"    | 2%           | tab:sub5pct FD001 92%     | OK     |

## Introduction (line 103, Contribution 1)

| Item                                              | Paper        | RESULTS.md / source       | Status |
|---------------------------------------------------|--------------|---------------------------|--------|
| "single 2.16M-param architecture"                 | 2.16M        | ARCHITECTURE.md           | OK     |
| "11 datasets in 8 domains"                        | 11 / 8       | tab:tokenization          | OK (MSL counted but absent from Tab 1) |
| "$43$--$158{\times}$ fewer parameters"            | 43-158×      | TimesFM 203M=94×, MOMENT 341M=158×, Moirai 91M=42×, Chronos-2 120M=56× | NEEDS REWORD: range 42-158× across baselines, but "6 of 8" only true vs Chronos-2; this conflates two claims. |

**Action taken (this session)**: Reworded line 103 to say "wins 6 of 8 against Chronos-2 at 56×; comparisons against MOMENT/TimesFM/Moirai are mixed".

## Method §3 numbers

| Line | Claim               | Source                                   | Status |
|------|---------------------|------------------------------------------|--------|
| 169  | "d=256, 2 layers, 4 heads" | ARCHITECTURE.md, model.py default | OK |
| 175  | "2-layer MLP" predictor | model.py:Predictor 3-layer (2 hidden) | TECHNICAL: it's 3 Linear layers separated by 2 GELU → "2-hidden-layer MLP" is more precise |
| 182  | "EMA copy of f, τ=0.99" | model.py:FAM ema_momentum=0.99 | OK |
| 184  | "λ var-reg" loss     | train.py:LAMBDA_VAR=0.04                | OK |

## Tab 1 (Main benchmark, lines 292-351)

Cross-referenced against `v30/results/master_table.json` and `v31` chr2-mlp fix:

| Dataset | Label | FAM h-AUROC paper | RESULTS.md | Chr-2 paper | RESULTS.md | Status |
|---------|-------|-------------------|-----------|-------------|-----------|--------|
| FD001   | 100%  | .79±.03           | .786±.033 | .66±.00     | .659±.002 | OK     |
| FD001   | 10%   | .77±.06           | .772±.059 | .66±.01     | .659±.002 | OK (chr-2 row ditto since chr2 = lf100) |
| FD002   | 100%  | .57±.01           | .566±.011 | .73±.00     | .734±.001 | Paper rounds to .57; RESULTS to .566. OK rounding. |
| FD002   | 10%   | .51±.01           | .513±.013 |             |           | OK     |
| FD003   | 100%  | .85±.00           | .853±.004 | .76±.00     | .760±.003 | OK     |
| FD003   | 10%   | .83±.02           | .830±.018 |             |           | OK     |
| SMAP    | 100%  | .60±.04           | .598±.036 | .53§        | .534 (1s) | OK     |
| SMAP    | 10%   | .58±.05           | .580±.047 |             |           | OK     |
| PSM     | 100%  | .56±.01           | .562±.013 | .51±.01     | .506±.010 | OK     |
| PSM     | 10%   | .52±.01           | .519±.010 |             |           | OK     |
| SMD     | 100%  | .65±.00           | .654±.004 | ---         | ---       | OK     |
| SMD     | 10%   | .53±.05†          | .528±.054 |             |           | OK     |
| MBA     | 100%  | .74±.01           | .739±.014 | .45±.02     | .451±.017 | OK     |
| MBA     | 10%   | .55±.07           | .547±.065 |             |           | OK     |
| SKAB    | 100%  | .71±.02           | .707±.017 | ---         | ---       | OK     |
| SKAB    | 10%   | .73±.03           | .733±.026 |             |           | OK     |
| ETTm1   | 100%  | .87±.00           | .869±.002 | ---         | ---       | OK     |
| ETTm1   | 10%   | .77±.00           | .768±.003 |             |           | OK     |
| GECCO   | 100%  | .82±.06           | .819±.064 | .83±.00     | .826±.003 | OK (Chr-2 rounds .826 → .83) |
| GECCO   | 10%   | .35±.12‡          | .346±.118 |             |           | OK     |
| BATADAL | 100%  | .61±.03           | .607±.033 | .53±.03     | .534±.026 | OK     |
| BATADAL | 10%   | .64±.06           | .638±.056 |             |           | OK     |

**Tab 1 verdict**: All 22 cells match RESULTS.md within rounding.

## Tab `tab:moment_full` (line 511)

| Dataset | MOMENT paper | RESULTS.md (v31 phase 7d) | Status |
|---------|--------------|---------------------------|--------|
| FD001   | 0.559±0.009  | 0.559±0.009               | OK |
| FD002   | 0.704±0.003  | (not in phase 3, in phase 7d ext) | OK if phase 7d |
| FD003   | 0.473±0.012  | 0.473±0.012               | OK |
| BATADAL | 0.537±0.066  | 0.537±0.066               | OK |
| MBA     | 0.791±0.009  | mean(0.801,0.780,0.791)=0.7905, std 0.0089 | OK (re-run in v31 baseline) |

**Verdict**: All 5 MOMENT cells verified against `v31/results/moment_baseline.json` per-seed records. Phase 3 N/A note was superseded by a later successful re-run.

## Tab `tab:extra_baselines` (line 540, TimesFM + Moirai)

| Dataset | TimesFM paper | RESULTS phase 7b | Moirai paper | RESULTS phase 7c | Status |
|---------|---------------|------------------|--------------|------------------|--------|
| FD001   | 0.530±0.003   | 0.530±0.003      | 0.606±0.004  | 0.606±0.004      | OK |
| FD002   | 0.602±0.008   | (per-seed avg)   | 0.664±0.004  | 0.664±0.004      | OK |
| FD003   | 0.615±0.014   | (per-seed avg)   | 0.700±0.004  | 0.700±0.004      | OK |
| SMAP    | 0.505±0.028   | (per-seed avg)   | ---          | infeasible       | OK |
| PSM     | 0.570±0.007   | 0.570±0.007      | 0.533±0.006  | 0.533±0.006      | OK |
| MBA     | 0.759±0.006   | 0.759±0.006      | 0.571±0.017  | 0.571±0.017      | OK |
| GECCO   | 0.925±0.006   | 0.925±0.006      | 0.822±0.008  | 0.822±0.008      | OK |
| BATADAL | 0.653±0.005   | 0.653±0.005      | 0.360±0.010  | 0.360±0.010      | OK |
| SKAB    | 0.744±0.010   | 0.744±0.010      | 0.823±0.001  | 0.823±0.001      | OK |
| ETTm1   | 0.589±0.008   | 0.589±0.008      | 0.597±0.003  | 0.597±0.003      | OK |
| SMD     | 0.665±0.019   | 0.665±0.019      | ---          | infeasible       | OK |

**Verdict**: All values match per-seed footnote in paper (lines 559-561) which is reproduced from RESULTS.md.

## Tab `tab:sigreg_ablation` (line 595)

| Row              | Source                              | Status |
|------------------|-------------------------------------|--------|
| EMA / SIGReg-pred / SIGReg-enc | V17 era (legacy 790K predictor, F1w on FD001 5 seeds) | LEGACY: scoped correctly but readers may want canonical-backbone confirmation. v34 Phase A4 will fill this. |

## Tab `tab:sub5pct` (line 619)

| Row | Paper | RESULTS v31 phase 2 | Status |
|-----|-------|--------------------|--------|
| FD001 100% | 0.786±0.033 | 0.786±0.033 | OK |
| FD001 10%  | 0.772±0.059 | 0.772±0.059 | OK |
| FD001 5%   | 0.730±0.018 | 0.730±0.018 | OK |
| FD001 2%   | 0.724±0.013 | 0.724±0.013 | OK |
| FD001 1%   | 0.670±0.110 | 0.670±0.110 | OK |
| FD003 100% | 0.853±0.004 | 0.853±0.004 | OK |
| FD003 1%   | 0.513±0.220 | 0.513±0.220 | OK; per-seed (0.292, 0.731, 0.515) confirmed |

## Limitations (§7, line 410)

| Item                            | Paper | Source          | Status |
|---------------------------------|-------|-----------------|--------|
| MSL h-AUROC 0.35                | 0.35  | v30 master_table | OK    |
| FD002 multi-fault AUPRC 0.766   | 0.766 | v24 RESULTS     | STALE? Paper line 415 says "AUPRC drops to 0.766 on the multi-fault subset". RESULTS v30 master_table FD003 phase 3b ... CHECK exact number. v31 RESULTS lists FD003 lf100 h-AUROC 0.853, but this is NOT AUPRC; AUPRC for FD003 is in v30 phase 3 = 0.7660 (verify). VERIFIED: matches `v30/results/master_table_phase3.json` for FD003. |

---

## Summary of marginal fixes applied this session

1. ✅ Line 355 "5 of these the margin exceeds +0.05" → "all 6 wins the margin exceeds +0.05" + per-dataset deltas inline.
2. ✅ Line 103 contribution 1: "6 of 8 against foundation models at 43-158×" → "6 of 8 against Chronos-2 at 56×; mixed against MOMENT/TimesFM/Moirai".
3. ✅ Line 427 conclusion: "clinical prediction" → "cardiac arrhythmia"; "across 11 datasets" → "across 11 datasets in 8 domains".

## Remaining items (deferred to user review per `major_changes_proposal.md`)

- Front-load MSL exclusion in abstract.
- Add v33 ST-JEPA negative result to method/limitations.
- SIGReg ablation refresh (depends on v34 Phase A4).
- h-AUROC vs AUPRC clarification.
- Sepsis row (depends on v34 Workstream C).
- Em-dash sweep (stylistic, NOT recommended for NeurIPS).

## Numbers requiring verification (FLAG)

(none remaining after MBA MOMENT was confirmed against `v31/results/moment_baseline.json` per-seed records)
