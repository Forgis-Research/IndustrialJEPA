# Open Questions - V16 Overnight Session

Items requiring user decision or unresolved during the overnight run.

## Phase 3 SMAP/MSL 100-epoch retries crashed

Both `experiments/v16/phase3_smap_100epochs.py` and
`experiments/v16/phase3b_msl_100epochs.py` finished pretraining (100 epochs each: loss
0.0057 on SMAP, 0.0069 on MSL) but crashed at the evaluation step with `KeyError:
'non_pa_f1'`. The evaluator printed diagnostic means before crashing:

- SMAP: anomaly-window score mean = 0.809 vs normal-window mean = 0.869 (near-null signal;
  anomalies score slightly LOWER than normal windows).
- MSL: anomaly mean = 1.201 vs normal mean = 1.200 (essentially no separation).

No `phase3_smap_results.json` or `phase3_msl_results.json` was produced for V16. Per the
orchestrator instruction, V15 20-epoch numbers (SMAP PA-F1 62.5 / MSL PA-F1 43.3) remain
the real quoted values; the 100-epoch results are referred to only in a forward-looking
`\plannedc{...}` sentence. User will rerun Phase 3 separately after fixing the evaluator
key bug and the argument order in `evaluate_*_anomaly(model, data)` calls.

## No sub-agent harness available

Spec calls for spawning 4 parallel `neurips-reviewer` sub-agents per iteration. This
orchestrator session had no Task/Agent tool loaded. Reviews were performed by the
orchestrator under four distinct personas (Empirical rigor / Story / Figures / Related
work). Recorded in `review_history.md`.

## Page budget: currently above NeurIPS 9-page main-text limit

The paper uses `\usepackage[preprint]{neurips_2026}`, which (per the style guide) allows
extended length for arXiv/preprint drafts. After iter 3, main body runs ~12 pages (intro
through conclusion), appendix starts around page 15. For a NeurIPS submission the
`[preprint]` option would be replaced with the final style and text tightened to 9 pages.
Sections that would trim first under a hard 9-page limit:

- Related work (~1.3 pp) to 1 pp.
- §5.5 Ablations: SIGReg speculative block (blue) can move to appendix.
- §6.2 "Why trajectory prediction learns degradation": three bullets to one paragraph.
- §7 Limitations: each item one sentence shorter.

All figures are currently essential; no figure cuts recommended.

## Broader impact reliant on appendix cross-refs

§7 "Broader impact" references `app:fd002` and `sec:verification`. Make sure these labels
stay valid after any future restructuring.

