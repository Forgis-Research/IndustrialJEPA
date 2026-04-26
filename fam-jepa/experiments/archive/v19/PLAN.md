# V19 Plan: Multi-Domain Breadth Sweep

## Narrative

V18 landed the paper at 7/10 weak accept. The round-5 reviewer's single blocker to 8/10 was "a real generalization story." V19 delivers it by applying the SAME FAM recipe (v17 architecture, Mahalanobis(PCA-k) scoring with label-free variance-retention k-selection) across 6 additional benchmarks spanning 4 domains:

| Domain | Benchmark | Task | Prior Best (published) | V18 coverage |
|---|---|---|---|---|
| Space | SMAP | anomaly segment | MTS-JEPA 0.336 PA-F1 | **v18 ✓** |
| Space | MSL | anomaly segment | MTS-JEPA 0.336 PA-F1 | **v18 ✓** |
| Server | **PSM** | anomaly segment | MTS-JEPA 0.616 PA-F1 | V19 |
| Server | **SMD** | anomaly segment | (OmniAnomaly) | V19 |
| Medical | **MBA (MIT-BIH)** | ECG arrhythmia | (various) | V19 |
| Mixed | **NAB** | univariate anomaly | (Numenta scoreboard) | V19 |
| Mechanical | **Paderborn** | bearing fault class | (CNN SOTA ~98%) | V19 |
| Mechanical | **Hydraulic** | component degradation | (UCI baselines) | V19 stretch |

Target narrative for paper: **"One pretraining recipe, eight benchmarks, four domains."** Sharp, simple, honest.

## Sharp Contributions

1. **First-class medical time-series coverage** (MBA MIT-BIH ECG). Makes the abstract's "onset of sepsis" motivation concrete rather than aspirational.
2. **Complete MTS-JEPA head-to-head** by adding PSM (SMAP/MSL done; SWaT blocked on data access, explicitly flagged). Beats MTS-JEPA on 3 of 4 if FAM + Mahalanobis holds.
3. **Domain-breadth table**: single SAME recipe applied to 6-8 benchmarks, report per-domain PA-F1 / macro-F1 / AUC-PR.
4. **Variance-retention k-selection transfers across all domains** - label-free, principled, zero hyperparameter tuning per dataset.

## Execution Order (by impact/cost ratio)

### Phase 0 (15 min, no GPU): domain readiness audit
 - Load each of PSM, MBA, NAB, SMD, Paderborn, Hydraulic into standardised numpy arrays.
 - Document per-dataset stats: n_channels, train/test shape, anomaly rate, sampling freq, licence.
 - One script per dataset that outputs a shared `{train, test, labels, n_channels, anomaly_rate}` dict (matching `smap_msl.py`).
 - Saves `experiments/v19/datasets_summary.json`.

### Phase 1 (45 min, GPU): PSM (completes MTS-JEPA coverage)
 - Pretrain FAM on PSM train set with v17 recipe (50 ep for anomaly, or 150 if loss doesn't converge). 3 seeds.
 - Mahalanobis(PCA-k) scoring with k selected via variance ≥ 0.99 (label-free).
 - Report non-PA F1, PA-F1, AUC-PR vs MTS-JEPA PSM reference.

### Phase 2 (45 min, GPU): MBA (medical ECG)
 - MIT-BIH Arrhythmia. Train FAM on normal-rhythm segments; score on arrhythmia segments.
 - Same recipe. Report same metrics.
 - Medical-domain positive finding would be a headline addition.

### Phase 3 (30 min, GPU): NAB (univariate)
 - Run on 5-7 NAB series as a demonstration of univariate-input FAM.
 - Aggregate score across series.

### Phase 4 (45 min, GPU): SMD (server machines)
 - Third MTS-JEPA-adjacent server-metrics benchmark.

### Phase 5 (45 min, GPU): Paderborn (mechanical vibration)
 - Real bearing data: K001 healthy, KA01 outer-race, KI01 inner-race.
 - FAM pretrained on healthy vibration windows, linear probe for 3-class classification.
 - Compare against Chronos-T5 frozen probe + supervised CNN baseline.

### Phase 6 (30 min, GPU, stretch): Hydraulic
 - UCI hydraulic condition monitoring. Multi-component degradation (4 components × 4 levels each).
 - Same recipe.

### Phase 7 (45 min, GPU, stretch): Cross-domain zero-shot
 - FD001-pretrained FAM applied directly to Paderborn vibration (no further pretraining).
 - If > random accuracy, that's a true foundation-model finding.

### Phase 8 (30 min, CPU): paper + summary
 - Write `experiments/v19/RESULTS.md` with per-domain tables.
 - Render `experiments/v19/summary.qmd` (Quarto, transparent per-seed numbers).
 - Add a new section to `paper-neurips/paper.tex` titled "Multi-domain evaluation: four domains, one recipe."
 - Update abstract with the domain-breadth claim.

### Phase 9 (10 min, CPU): round-6 review
 - Launch `neurips-reviewer` agent on the final paper. Target: 8/10 via generalization evidence.

## Ground rules (from user's updated feedback memory, non-negotiable)

1. **Commit hourly, push after each commit.** Not batched.
2. **Verify completion before stopping.** Grep paper for `\todo{}`, check PLAN vs actual, re-evaluate stopping conditions at every wakeup.
3. **Don't self-terminate with "out-of-scope"** if runway remains.
4. **Every session ends with a transparent summary.qmd**: TL;DR, headline table, phase-by-phase with per-seed numbers, explicit "did NOT run" list, commit log, artifact index. Forensic record, not marketing.
5. **All notebooks .qmd, never .ipynb.**
6. **Paper prose uses " - " not " — " (em-dash rule).**

## Agent use guidance

 - **`Explore`** subagent (thoroughness="medium") for dataset directory scouting at the start of each new domain. Don't spend your own context on `ls` and file-inspection.
 - **`ml-researcher`** for any ambiguous experiment-design call (e.g., "what's the right pretraining length for MBA given the heartbeat periodicity?").
 - **`paper-writer`** for all paper edits. Brief it with: (a) which section, (b) the new numbers, (c) the specific reviewer concern being closed, (d) what NOT to change.
 - **`neurips-reviewer`** for end-of-session review round. Tell it which prior rounds said what, and which specific item we closed in v19.
 - **`Plan`** if the next architectural decision is non-trivial (e.g., "how to handle MBA's variable heartbeat cycle count without padding artifacts").
 - Spawn in **parallel** when independent (e.g., 3 reviewers in one message; multiple datasets scouted in parallel).
 - **Do NOT** spawn an agent for trivial lookups - just use `Read`/`Grep` yourself.
 - **Do NOT** re-run a subagent with the same prompt - use `SendMessage` with the agent ID to continue.

## Hyperparameters (inherited from V17/V18, do NOT retune per-dataset unless forced)

| Parameter | Value | Source |
|---|---|---|
| Pretrain epochs | 50 for anomaly (as v17 Phase 5); 150 if loss doesn't converge after 50 | v17 Phase 5 + v18 Phase 4g/4i |
| LogU k range | `[1, K_max]` where K_max = 500 for anomaly, 150 for RUL | v17 |
| Fixed target window w | 10 | v17 |
| EMA momentum | 0.99 | v17 |
| `d_model` | 256 | v17 |
| `n_layers` | 2 | v17 |
| PCA rank selection | k s.t. cum_var >= 0.99 (label-free) | v18 Phase 4j |
| Probe: AdamW WD=1e-2 | | v18 |
| Probe val cuts | 10 per engine (RUL) / standard last 10% of test (anomaly) | v18 |
| Threshold | 95th percentile of first 10% test scores | v17 |
| Seeds | 3 (42, 123, 456) for pretrain; 3 for probe | v18 |

## What we're NOT doing

 - Retuning FAM per-domain. Same recipe everywhere.
 - SWaT - data registration blocked.
 - New architectural variants. V17 arch frozen.
 - Multi-dataset pretraining (pretrain on union) - V20 stretch.
 - Fine-tuning Chronos for comparison. Deferred.
 - FEMTO / XJTU-SY RUL regression - data requires registration.

## Success criteria

**Must-do** (end of session):
 - Phase 0 complete (datasets loadable).
 - Phase 1 complete (PSM) - completes MTS-JEPA head-to-head to 3 of 4.
 - Phase 2 complete (MBA) - first medical-domain number in the paper.
 - `summary.qmd` with per-seed numbers, rendered, pushed.
 - 0 new `\todo{}` markers in paper.

**Target** (full overnight):
 - All 4 new MTS-JEPA-family benchmarks (PSM, MBA, NAB, SMD).
 - Paderborn + one additional domain.
 - 8-benchmark table in the paper.
 - Round-6 review run on final paper.

**Stretch**:
 - Zero-shot cross-domain (Phase 7).
 - Paper moves 7 -> 8 in round-6 review.
