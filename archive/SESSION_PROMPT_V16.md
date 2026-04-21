# V16 Overnight Paper Improvement — Autonomous Orchestration Prompt

**Usage**: Paste this entire file as the opening prompt to a Claude Code session on the VM. The agent has until morning to iterate the paper to NeurIPS acceptance quality.

**Working directory**: repo root (`IndustrialJEPA/`)
**Primary artifact**: `paper-neurips/paper.tex` (NeurIPS 2026 submission)
**Mode**: Multi-iteration reviewer → figure-agent → writer loop, fully autonomous.

---

## Mission

Improve `paper-neurips/paper.tex` to NeurIPS acceptance quality by running a multi-iteration improvement loop. Each iteration: spawn 4 parallel NeurIPS reviewers, then do figure audit/rework, then apply writer fixes, then compile and commit.

You are the orchestrator. You spawn sub-agents (`neurips-reviewer`, `Explore`, `general-purpose`) and apply their outputs yourself using the Edit/Write tools.

Plan for **3 iterations** (hard stop: 6). If reviewer panel converges to mean score ≥ 7 earlier, stop early and polish.

---

## State at Session Start (2026-04-16 evening)

### Completed experiments — numbers MUST match paper (traceable to JSON)

| Result | Value | Source JSON |
|---|---|---|
| V2 frozen (causal + EMA + future-only, 5 seeds) | 17.81 ± 1.7 | V14 phase results |
| V2 E2E | 14.23 ± 0.4 | V14 phase results |
| V14 full-seq target frozen | 15.70 ± 0.2 | `experiments/v14/` |
| V14 full-seq target E2E | 14.32 ± 0.6 | `experiments/v14/` |
| Cross-sensor (V14 sensor-as-token) frozen | 14.98 ± 0.22 | `experiments/v14/` |
| Cross-sensor E2E | 14.35 ± 0.9 | `experiments/v14/` |
| V16b frozen TEST (bidi + VICReg + EMA, 3 seeds) | 25.72 ± 1.59 | `experiments/v16/phase7_frozen_probe_test_rmse.json` |
| V16b E2E (3 seeds) | 15.06 ± 1.15 | `experiments/v16/phase1_v16b_e2e_results.json` |
| V15-SIGReg frozen VAL (last-window, biased) | 9.2 ± 1.5 | `experiments/v15/phase1_v15_sigreg_results.json` |
| V15-SIGReg verified re-run (same seed 42) | best 11.61, mean 11.13 ± 0.79 | `experiments/v16/v15sigreg_seed42_results.json` |
| Phase 2 cross-sensor without sensor_id_embed | 21.0 ± 6.4 (seeds: 14.22, 27.01, 21.81) | `experiments/v16/phase2_cross_sensor_results.json` |
| Phase 4 cross-machine V2 | FD002 27.68, FD003 31.45, FD004 38.32 | `experiments/v16/phase4_cross_machine_results.json` |
| Phase 4 cross-machine V16a | FD002 32.62, FD003 40.02, FD004 43.60 | same |
| Phase 4 cross-machine V16b | FD002 38.04, FD003 37.76, FD004 49.66 | same |
| Phase 6b feature regressor FD001 test | 19.07 | `experiments/v16/phase6b_fd3_fd4_regressor_results.json` |
| Phase 6b FD003 / FD004 test | 19.74 / 32.09 | same |
| Phase 8 V16b label efficiency (100/50/20/10/5%) | 15.36 / 22.43 / 27.01 / 29.27 / 40.65 | `experiments/v16/phase8_label_efficiency.json` |
| Phase 5 shuffle test | +20.83 shuffled, +28.45 random | `experiments/v16/phase5_shuffle_test_results.json` |
| NASA-S (5 seeds) | 395.7 ± 69.4 | `experiments/v16/nasa_s_results.json` |
| SMAP V15 (20 epochs) | PA-F1 62.5%, non-PA 6.9% | `experiments/v15/phase3_smap_results.json` |
| MSL V15 (20 epochs) | PA-F1 43.3%, non-PA 7.9% | `experiments/v15/phase3_msl_results.json` |

**Every number in the paper must be traceable to one of these JSONs.** If you can't find it in a JSON, flag it in `paper-neurips/open_questions.md` — do not guess or invent.

### Still running (may finish mid-session)

- **Phase 3 V16 SMAP 100-epoch** — will supersede V15 20-epoch numbers. Before each iteration, check `experiments/v16/phase3_smap_results.json` (if exists) or `experiments/v16/phase3_stdout.log`. If new numbers are available, remove the corresponding `\plannedc{}` blue and fill in real numbers.

---

## Blue `\plannedc{}` Content — Allowed List

These are the **ONLY** sections that may be written in blue. Every other claim must be grounded in an actual JSON result.

Per user instruction: write the planned sections as if completed — present tense, confident framing, full narrative — with `\placeholder{X.XX}` for specific numbers.

1. **SWaT anomaly evaluation** (water-treatment domain, 3rd domain group). Benchmark table shows empty column; fill narrative + placeholders. Describe the protocol in detail.
2. **C-MAPSS TTE numerical results** (§5.4). Protocol is written; numerical results are future work. Fill `\plannedc{}` results narrative + table rows with `\placeholder{--}`.
3. **V16 SMAP 100-epoch results** — until Phase 3 finishes, keep V15 20-epoch numbers in real text and add a `\plannedc{}` forward-looking sentence about the 100-epoch extension. When Phase 3 finishes, replace both.
4. **SIGReg on V2 architecture** (speculative future direction ablation — "causal context + future-only target + SIGReg regularizer, hypothesized to provide the isotropy benefit without EMA instability"). Write as a one-paragraph forward-looking ablation note, not a full experiment.

**Do NOT** blue-wash anything that contradicts a known result. E.g., don't write "V16b beats V2" — we know it doesn't. Blue is for genuinely unfinished work written narratively, not for rewriting outcomes.

`\draftmodetrue` stays ON throughout the overnight session. Do not toggle it.

---

## Tools You Will Use

- **`neurips-reviewer`** sub-agent: independent NeurIPS reviews. Spawn 4 in parallel per iteration.
- **`Explore` or `general-purpose`** sub-agent: figure audits, figure creation tasks, literature checks.
- **Edit / Write**: your own paper edits.
- **Bash**: `pdflatex`, `git`, `bash paper-neurips/figure-pipeline/compile_figure.sh`, `python paper-neurips/figure-pipeline/validate_figure.py`.
- **Read / Grep / Glob**: inspecting results JSONs, existing figures, reviewer outputs.

---

## Iteration Protocol

### Stage N.1 — Parallel Review (4 reviewers)

Spawn 4 `neurips-reviewer` agents in a single message (parallel). Each gets the path to the current `paper.tex` and a distinct emphasis:

- **Reviewer A — Empirical rigor**: statistical validity, seed counts, reproducibility, protocol blindspots, claim-evidence gaps, numerical consistency across abstract/body/tables.
- **Reviewer B — Story & framing**: does "one encoder, any event type" actually land? Is the grey-swan framing oversold? Does the paper flow? Abstract → intro → method → experiments → conclusion continuity.
- **Reviewer C — Figures & presentation**: figure legibility, table design, caption quality, section balance, missing visuals, reference consistency.
- **Reviewer D — Related work & positioning**: coverage of 2023-2026 SSL-for-time-series literature, JEPA variants (I-JEPA, V-JEPA, TS-JEPA, MTS-JEPA, A2P, DCSSL), RUL SOTA (STAR, DCNN, AE-LSTM, TTSNet), anomaly detection baselines (DCdetector, AnomalyTransformer), limitations honesty.

Each reviewer must produce this structured output:
```
SUMMARY: (2-3 sentences)
STRENGTHS: (3-5 bullets)
WEAKNESSES: (3-5 bullets, each with paper.tex line number)
QUESTIONS: (3-5 for authors)
ACTIONABLE SUGGESTIONS: (3-5, prioritized)
SCORE: X/10
RECOMMENDATION: {strong reject | reject | weak reject | borderline | weak accept | accept | strong accept}
```

Tell reviewers: "Blue `\plannedc{}` text is planned work written narratively as if complete. Flag if the framing is misleading (claims not matching likely outcome), but do not attack for missing numerical details in those specific sections."

### Stage N.2 — Figure Audit / Rework (in parallel with N.1 or after)

Spawn 1 `Explore` agent with this task:

> Audit paper figures against `paper-neurips/figure-pipeline/figure_prompt.md` (the design bible, Sections §2–§10 mandatory).
>
> Figures currently used in `paper.tex`:
> - `figures/fig_tokenization.pdf` (§3.2)
> - `figures/fig_architecture_ema.pdf` + `figures/fig_architecture_sigreg.pdf` (side-by-side in §3.3)
> - `figures/fig_domain_overview.pdf` (§5.1)
> - `figures/v14_label_efficiency_from_scratch.pdf` (§5.2)
>
> Existing PDFs available but not currently referenced:
> - `trajectory_jepa_architecture.pdf`, `v12_cmapss_main_results.pdf`, `v12_cmapss_tracking.pdf`, `v12_fd002_diagnosis.pdf`, `v8_rul_comparison.pdf`
>
> Produce a report with:
> 1. Per-figure verdict (keep as-is / rework / reject) against design bible.
> 2. List of figures to CREATE to reach NeurIPS-acceptance figure density (target 5–7 figures).
> 3. For each figure-to-create: purpose, layout sketch, data source, priority.

After audit, for each figure to rework or create:
1. Work inside `paper-neurips/figure-pipeline/` — draft `.tex` following the design bible (boilerplate §2, typography §3, layout §5, colors §6 from `color_schema.json`).
2. Compile: `bash paper-neurips/figure-pipeline/compile_figure.sh <name>.tex`
3. Validate: `python paper-neurips/figure-pipeline/validate_figure.py <name>.pdf`
4. Run the §10 Self-Check Protocol (14 items). If any FAIL, fix and re-check. Do not ship with any FAIL.
5. Copy PDF to `paper-neurips/figures/`.
6. Reference the new figure in `paper.tex` at the natural point.

**High-impact candidate figures** (pick 1–2 per iteration, based on reviewer feedback):
- **Degradation clock PCA** — convert `analysis/plots/v15/degradation_clock_v15_sigreg_pca.png` to a clean publication-grade figure (matplotlib script with agg backend → PDF, or a TikZ scatter if simple enough). Shows PC1 tracking RUL for SIGReg vs EMA configs. Evidence the encoder learns a degradation axis.
- **Cross-machine transfer bar chart** — from `phase4_cross_machine_results.json`. Grouped bars: FD002/FD003/FD004 × (V2 / V16a / V16b). Shows causal V2 wins on transfer.
- **SMAP anomaly-score timeline** — one channel showing prediction-error score spikes aligned with ground-truth anomaly windows. Qualitative evidence for the zero-label anomaly claim.
- **Label-efficiency V16b vs V2** — from `phase8_label_efficiency.json`. Two curves (V2 and V16b) on log-x. Shows the bidirectional tradeoff.
- **Architecture ablation family tree** — V2 → V14 → V15-EMA → V15-SIGReg → V16a → V16b with arrows annotated by the single architectural change and the resulting RMSE delta. Motivates the causal-inductive-bias finding.

**Do not inflate figure count** — if reviewers don't flag a gap, don't add.

### Stage N.3 — Writer Application (you, serial)

With 4 reviews + figure audit in hand, apply fixes to `paper.tex` directly. Priorities in order:
1. Every numerical inconsistency or wrong number (check the JSON table above).
2. Every consensus issue raised by ≥ 2 reviewers.
3. Missing figures flagged by the figure audit (integrate the new ones).
4. Fill in `\plannedc{}` sections from the allowed-list if reviewers flag gaps in the SWaT/TTE/SIGReg-on-V2 narrative.
5. Tighten prose where reviewers flag bloat.
6. Fix captions, cross-refs, and typos.

Known issues from my pre-session audit (the writer should address these proactively if reviewers miss them):

| # | Issue | Location | Fix |
|---|---|---|---|
| 1 | Abstract "+15.6 at 10%" claim disagrees with table arithmetic (35.59 − 18.66 = +16.93) | abstract, intro | verify against table and pick one |
| 2 | SMAP 62.5% PA-F1 in abstract without the fixed-threshold caveat | abstract line 77 | add caveat |
| 3 | V15-SIGReg row in ablation table is empty (`---/---`) | tab:ablation | fill with caveated numbers, or drop row |
| 4 | Benchmark table has empty SWaT and TTE columns | tab:benchmark | blue-fill with `\plannedc{}` narrative |
| 5 | MSL number missing from benchmark table (have 43.3 in anomaly table) | tab:benchmark | add |
| 6 | "Supervised SOTA" in abstract at 5% is really "STAR at matched budget" | abstract | tighten language |
| 7 | V2 frozen 17.81 protocol footnote missing (test vs val clarification) | tab:ablation or near 17.81 | add footnote |
| 8 | Cross-sensor paragraph appears in §5.5 and appendix E (duplication) | both | tighten |
| 9 | `\draftmodetrue` still on (intentional for overnight, confirm and move on) | line 27 | leave ON |

After editing, `git diff paper-neurips/paper.tex | head -200` to sanity-check.

### Stage N.4 — Compile + Commit

From repo root:
```
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=paper-neurips paper-neurips/paper.tex
```
If compile fails, read `paper-neurips/paper.log`, fix the LaTeX errors, retry (up to 3 attempts).
If compile succeeds, run a second pass for bibliography/references:
```
bibtex paper-neurips/paper
pdflatex -interaction=nonstopmode -output-directory=paper-neurips paper-neurips/paper.tex
pdflatex -interaction=nonstopmode -output-directory=paper-neurips paper-neurips/paper.tex
```

**Commit** (via Bash; DO NOT add `-A` or stage temp files like `.aux`, `.log`, `.out`, `.synctex.gz`):
```bash
git add paper-neurips/paper.tex paper-neurips/figures/ paper-neurips/figure-pipeline/
git add paper-neurips/review_history.md paper-neurips/open_questions.md 2>/dev/null
git commit -m "$(cat <<'EOF'
paper v16 iter N: <one-line summary of main changes>

- bullet per substantive change (reviewer issue addressed, figure added, etc.)
- cite which reviewer(s) flagged if applicable

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

Do NOT push individual iteration commits — only push at the end (user preference: one final push when overnight run is done; see Final Deliverables).

---

## Between Iterations

Before starting iteration N+1:
1. **Check Phase 3 SMAP**: if `experiments/v16/phase3_smap_results.json` exists and is newer than session start, the 100-epoch numbers are in. Promote them from `\plannedc{}` to real text, remove placeholders, fill in real values.
2. **Convergence check**: if all 4 reviewers in iteration N scored ≥ 7/10, declare completion. Skip remaining iterations and go to final polish.
3. **Divergence check**: if scores are unchanged or worsening between iterations, stop spawning new reviewers and do one diagnostic pass yourself — maybe the feedback is contradictory.

---

## Progress Logs

### `paper-neurips/review_history.md` — append one line per iteration
```
## Iter N (YYYY-MM-DD HH:MM)
- Reviewer A: score=X/10, key issues: <top-3>
- Reviewer B: score=Y/10, key issues: <top-3>
- Reviewer C: score=Z/10, key issues: <top-3>
- Reviewer D: score=W/10, key issues: <top-3>
- Mean: M
- Figures: kept=N, reworked=M, added=K
- Writer actions: <bullet summary>
- Compile: PASS/FAIL
```

### `paper-neurips/open_questions.md` — append any question where you couldn't decide
Things like "reviewer B wants a new 'why causal wins' section but reviewer A says cut — needs user" or "cannot find source JSON for claim X on line Y".

---

## Final Deliverables (what the user expects in the morning)

Before you stop:
1. `paper-neurips/paper.tex` — clean-compiling, all issues addressed within your judgment.
2. `paper-neurips/paper.pdf` — final build.
3. `paper-neurips/figures/` — every used figure passes the design bible.
4. `paper-neurips/review_history.md` — full iteration record.
5. `paper-neurips/open_questions.md` — any unresolved items for user.
6. Git history — one commit per iteration, clean messages.
7. **Final push to remote** (`git push origin main` — the repo IS the source of truth for the paper).
8. A concluding summary message (printed to your own final output) with:
   - Iterations completed
   - Final reviewer score mean vs. initial
   - List of figures (kept / reworked / added)
   - Any remaining `open_questions.md` items
   - Any `\plannedc{}` sections still live

---

## Hard Rules

- **Do not invent numbers.** If a claim lacks a JSON backing, flag it in `open_questions.md`, do not fabricate.
- **Do not toggle `\draftmodetrue`.** Stays on.
- **Every figure used in paper.pdf must pass the Figure Design Bible §10 Self-Check.**
- **Do not push after every iteration.** Push only at the very end.
- **Do not edit the `experiments/` directory or `mechanical-jepa/` results.** Those are immutable ground truth.
- **Do not delete existing figures or figure `.tex` sources** — only add or rework.
- **If you hit a git conflict on push, DO NOT force-push.** Stop, leave a note in `open_questions.md`, return control to user.
- **If a sub-agent takes > 30 minutes, move on with partial output.**

---

## Starting Checklist

1. Read `paper-neurips/paper.tex` in full.
2. Read `paper-neurips/figure-pipeline/figure_prompt.md` (design bible).
3. Read `mechanical-jepa/experiments/v16/RESULTS.md` and `mechanical-jepa/experiments/v15/RESULTS.md` for full context.
4. Verify baseline compile: `pdflatex -interaction=nonstopmode -halt-on-error -output-directory=paper-neurips paper-neurips/paper.tex` → expect success.
5. Start iteration 1, stage N.1 (spawn 4 reviewers in parallel).

Begin now. Be rigorous. NeurIPS standard. Good luck.
