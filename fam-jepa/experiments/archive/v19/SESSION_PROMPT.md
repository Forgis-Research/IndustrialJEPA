# V19 Overnight Session - Multi-Domain Breadth Sweep

**Usage**: Paste this as opening prompt to a new Claude Code session on the GPU VM.
**Working directory**: `/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/`
**Duration**: full night (10+ hours target, per user memory)
**Narrative**: V18 landed the paper at 7/10 weak accept. V19 delivers the "real generalization story" the round-5 reviewer said would push to 8/10, by running the SAME FAM recipe across 6+ benchmarks across 4 domains.

---

## Mission

Execute `experiments/v19/PLAN.md` end-to-end. The three durable goals, in priority order:

1. **Complete MTS-JEPA head-to-head**: add PSM (SMAP/MSL done in v18). SWaT blocked on data access; flag honestly.
2. **Add medical domain**: MIT-BIH Arrhythmia (MBA) via the exact same FAM + Mahalanobis recipe. Makes the paper's "onset of sepsis" motivation concrete.
3. **Add real mechanical domain**: Paderborn bearing vibration. Tests if "domain-specific beats scale" holds on real (not simulated) data.

**Commit hourly with descriptive messages**, push after each commit. This is a durable user rule. Do not batch commits.

---

## Read first (mandatory, before any experiments)

 1. `experiments/v19/PLAN.md` - this session's concrete experiments and hyperparameters.
 2. `experiments/v18/RESULTS.md` - all the numbers you're extending, including the Chronos-2 comparison, Mahalanobis multi-seed SMAP/MSL, and label-free k-selection.
 3. `experiments/v18/summary.qmd` - final forensic record of v18 including Chronos adapter fairness discussion.
 4. `paper-neurips/paper.tex` - current paper at 7/10.
 5. `mechanical-jepa/data/smap_msl.py` - reference `load_smap/load_msl` interface that all new-dataset loaders should match.
 6. `experiments/v18/phase4f_smap_msl_seeds.py` - anomaly pretraining + Mahalanobis reference implementation. New datasets get the same recipe.
 7. `experiments/v18/phase4i_msl_multiseed.py` - longer-pretraining MSL (150 ep) reference for when 50 ep loss doesn't converge.
 8. `experiments/v18/phase4j_principled_k.py` - label-free k-selection via variance retention.
 9. `paper-replications/mts-jepa/data/PSM/` - PSM data, already in the repo (train.npy, test.npy, test_labels.npy).
 10. `paper-replications/mts-jepa/data/tranad_repo/data/MBA/` - MBA ECG data.
 11. `datasets/data/paderborn/` - K001, KA01, KI01 directories with .mat vibration files.

---

## Phase ordering

Run Phases 0 -> 9 in order from `PLAN.md`. If GPU is tight, prioritise:
 1. Phase 1 (PSM) - highest impact, easiest (data ready, pipeline known).
 2. Phase 2 (MBA) - headline new-domain result.
 3. Phase 5 (Paderborn) - real-data mechanical.
 Then fill in Phase 3/4/6/7 as time allows.

At each phase boundary:
 - Commit with `v19 phase N (HH:MM): <what was done>` message.
 - Push.
 - Update `experiments/v19/RESULTS.md` incrementally.
 - Re-evaluate stopping conditions (see below).

---

## Durable rules (from user's updated feedback memory, DO NOT VIOLATE)

 1. **Run the full night (10+ hours).** Do not self-terminate early. If a stretch item looks infeasible, say so once and attempt it anyway if there's runway.
 2. **Commit hourly, push after each commit.** Do not batch.
 3. **Verify completion before stopping.** Grep paper.tex for `\todo{}`, re-check PLAN vs actual, re-evaluate stopping conditions at each wakeup.
 4. **Every session ends with a transparent `summary.qmd`.** Required structure: TL;DR (5-8 bullets), headline table (pandas DataFrame), key figures, phase-by-phase with per-seed numbers, explicit "did NOT run" list, commit log, artifact index. Forensic record, not marketing.
 5. **All notebooks are Quarto `.qmd`, never Jupyter `.ipynb`.**
 6. **Paper prose uses " - ", not " — " (em-dash rule).** Applies to .tex, .md, .qmd.
 7. **Save per-seed predictions in result JSONs**, not just aggregates. The user needs forensic access.
 8. **Scan the paper with a round-6 reviewer agent at the end.** Goal: 7 -> 8. If it's still 7, honestly report why.

---

## Agent use guidance

Spawn subagents for tasks that would burn context on routine work. **Do NOT** spawn for trivial lookups (use `Read`/`Grep`/`Glob` directly).

| Subagent | When to use | How to brief |
|---|---|---|
| `Explore` | Scouting a new dataset directory or locating existing code patterns. Thoroughness: "medium" usually; "very thorough" only if you've gotten conflicting results. | Give it the specific question ("Where is the Paderborn loader that already exists, and does it already normalise?") rather than "look around." |
| `ml-researcher` | Ambiguous experiment-design calls (e.g., "is 50 ep enough for MBA given heartbeat periodicity of ~800 samples?"). Also for reading MTS-JEPA paper to pull their exact evaluation protocol. | Tell it what you already ruled out. Ask for a concrete decision with reasoning. |
| `paper-writer` | Any paper edit longer than one line. | Give it (a) section, (b) new numbers, (c) which reviewer concern it closes, (d) what NOT to change. Budget "20 min." |
| `neurips-reviewer` | End-of-session review round. Also occasional round-2 after major additions. | Tell it: prior rounds + scores, what was added between rounds, what specific item we claim to have closed. |
| `Plan` | Non-trivial architectural decisions (e.g., handling MBA's variable-length heartbeats). | Rare. Usually ml-researcher is the better choice. |

**Parallel spawns** when independent (multiple reviewers; multiple dataset loader stubs). **`SendMessage`** to continue an existing agent rather than spawning a fresh one with the same prompt.

**Context hygiene**: don't `Read` or `tail` an agent's output file - that's the JSONL transcript and will overflow your context. Use the return value (summary) instead.

---

## Stopping conditions (all must be met before stopping)

 - [ ] `experiments/v19/RESULTS.md` exists and is complete.
 - [ ] `experiments/v19/summary.qmd` exists, renders, includes TL;DR + tables + commit log.
 - [ ] Paper `\todo{}` count: 0 (or each remaining `\todo` has an honest in-text reason).
 - [ ] All v19 commits pushed to `origin/main`.
 - [ ] Round-6 review run on final paper. Score reported.
 - [ ] At least 3 of the 6 new benchmarks have numbers (PSM + MBA + Paderborn as the floor).
 - [ ] Wall-clock time used: >= 8 hours (target; soft).

If any stopping condition is not met: keep going. Do NOT stop "because it's morning" or "because the obvious next step is out of scope."

---

## Key narrative touchstone for the paper

The current paper's title ends at "...Anomaly Segment Scoring in Multivariate Sensor Streams." After v19, the story is:

> **Same recipe, four domains, eight benchmarks.** FAM (1.26M params, causal JEPA) pretrained with identical v17 hyperparameters - cycle-level or sample-level inputs, no per-dataset tuning - achieves frozen-probe parity or win over supervised and foundation-model baselines across turbofan RUL (FD001-4), spacecraft telemetry (SMAP, MSL), server anomaly (PSM, SMD), medical ECG (MBA), and mechanical vibration (Paderborn). The matched-$k$ label-free PCA rule (variance retention >= 0.99) applies unchanged across all eight.

If the round-6 reviewer sees this breadth story, 8/10 is plausible. Without it, we stay at 7/10.

---

## Priority ordering if time is short

**Must-do** (first 3 hours): Phase 0 + Phase 1 (PSM) + Phase 2 (MBA) + hourly commits pushed + partial `summary.qmd`.

**Should-do** (3-6 hours in): Phase 5 (Paderborn), Phase 3 (NAB), Phase 4 (SMD). Fill the 8-benchmark table.

**Stretch** (6+ hours in): Phase 6 (Hydraulic), Phase 7 (zero-shot FD001 -> Paderborn), Phase 9 (round-6 review).

**At the end**: render `summary.qmd`, update paper, commit+push, round-6 review, final summary to user.

---

## Questions you should be able to answer at end-of-session

1. Did FAM + Mahalanobis win on PSM? (4th MTS-JEPA benchmark)
2. Does the label-free k-selection heuristic (variance >= 0.99) transfer across domains?
3. How does FAM compare to Chronos-T5 on medical ECG and on bearing vibration? (multi-domain Chronos)
4. Did zero-shot FD001 -> Paderborn give non-random accuracy? (true foundation-model test)
5. Did the round-6 reviewer move the score from 7 to 8?
