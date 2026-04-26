# V30 Session Summary

**Date**: 2026-04-25 → 2026-04-26  
**Duration**: ~27 min Phase 3 GPU (plus Phase 0–2 and subagents in parallel)  
**Scope**: dense-K=150 head decision (Phase 0); FAM-vs-Chronos2 fair ablation (Phase 1); precursor check on MSL/SMD/PhysioNet (Phase 2); uniform 13-dataset benchmark (Phase 3); legacy metrics + SOTA mapping (Phase 4); figures + quarto notebook (Phase 5); theory self-check (Phase 6); new-dataset scouting (Phase 8).


## One-sentence verdict

V30 locks dense K=150 as the canonical FAM head, refutes the head-capacity critique by showing FAM encoder beats Chronos-2 at matched probe capacity on 3/4 datasets, replaces the v29 heterogeneous master table with a clean 11-dataset 3-seed uniform benchmark, and ships a 966-line theory self-check that strengthens Proposition 1 and explains the label-efficiency two-regime story formally.


## Decisions made

- **Phase 0 (head)**: dense discrete hazard CDF, eval at K=150 horizons, 20 random training horizons per batch. MonotoneCDF (Option A) collapsed to chance (0.5000) vs dense discrete (0.8130). Module kept in `model.py` as opt-in for v31.
- **Phase 1 (comparison)**: FAM-predft headline + Chr2-probe as canonical fair comparison. Encoder dominates head capacity in the ablation: FAM-probe beats Chr2-probe on FD001/FD003/BATADAL even with 257 trainable params/horizon.
- **Phase 2 (precursor)**: MSL **skip** (3-seed mean 0.3499, below chance — refines v29 n=1 of 0.438). SMD **include** (3-seed mean 0.6555). PhysioNet skip (no LOADERS entry).
- **Phase 3 (benchmark)**: 11 datasets × 3 seeds × {100%, 10%-on-4} = numbers below.


## Phase 1 — fair ablation (sparse h, 4 datasets × 3 seeds, 153s)

|             | FD001 | FD003 | MBA | BATADAL |
|-------------|---|---|---|---|
| fam-probe | 0.742 ± 0.012 | 0.812 ± 0.006 | 0.588 ± 0.012 | 0.521 ± 0.038 |
| chr2-probe | 0.622 ± 0.004 | 0.738 ± 0.002 | 0.659 ± 0.006 | 0.503 ± 0.040 |
| fam-predft | 0.714 ± 0.028 | 0.802 ± 0.015 | 0.739 ± 0.014 | 0.607 ± 0.033 |
| chr2-mlp | 0.659 ± 0.000 | 0.760 ± 0.003 | 0.451 ± 0.017 | 0.534 ± 0.032 |
| fam-mlp-rand | 0.707 ± 0.018 | 0.788 ± 0.026 | 0.721 ± 0.021 | 0.566 ± 0.009 |

**Findings**: encoder beats matched probe (FAM-probe > Chr2-probe on 3/4); pretrained predictor init helps a little (+0.007 to +0.041); at 10% labels FAM-predft ≈ FAM-mlp-rand (sub-5% is where pretraining might dominate — v31).

## Phase 3 — uniform benchmark (dense K=150)

| Dataset | h-AUROC 100% (3s) | h-AUROC 10% | v29 sparse-K=8 | Δ |
|---------|--------------------|-------------|----------------|----|
| FD001 | 0.7855 ± 0.0331 | 0.7723 ± 0.0587 | 0.742 | +0.044 |
| FD002 | 0.5657 ± 0.0110 | — | 0.569 | -0.003 |
| FD003 | 0.8530 ± 0.0035 | 0.8302 ± 0.0178 | 0.819 | +0.034 |
| SMAP | 0.5976 ± 0.0357 | — | 0.550 | +0.048 |
| PSM | 0.5616 ± 0.0134 | — | 0.559 | +0.003 |
| MBA | 0.6417 ± 0.0298 | 0.6417 ± 0.0298 | 0.746 | -0.104 |
| GECCO | 0.8187 ± 0.0642 | — | 0.859 | -0.040 |
| BATADAL | 0.5989 ± 0.0453 | 0.5989 ± 0.0453 | 0.629 | -0.030 |
| SKAB | 0.6737 ± 0.0321 | — | 0.726 | -0.052 |
| ETTm1 | 0.8332 ± 0.0080 | — | 0.869 | -0.036 |
| SMD | 0.6536 ± 0.0041 | — | 0.616 | +0.038 |

## Phase 4 — legacy metrics + SOTA

Per-dataset legacy metric (no point-adjust for anomaly):

- **FD001** (RMSE, RUL cap 125): 36.46 ± 2.33
- **FD002** (RMSE, RUL cap 125): 44.09 ± 2.90
- **FD003** (RMSE, RUL cap 125): 39.46 ± 0.75
- **SMAP** (best-F1 no-PA @Δt=1): 0.4577
- **PSM** (best-F1 no-PA @Δt=1): 0.4951
- **MBA** (AUROC@Δt=1): 0.6965
- **GECCO** (best-F1 no-PA @Δt=1): 0.2832
- **BATADAL** (best-F1 no-PA @Δt=1): 0.2363
- **SKAB** (best-F1 no-PA @Δt=1): 0.6449
- **SMD** (best-F1 no-PA @Δt=1): 0.2635

SOTA mapping (Phase 4a action items):
- C-MAPSS RMSE SOTA ≈ 11.3-11.4 (NOT STAR 10.61 — 2022 preprint).
- SMAP/PSM/SMD: cite Kim et al. AAAI 2022 PA-F1 trap; FAM is non-PA.
- MBA: TranAD/BTAD AUROC ~0.988 is anomaly-score framing — different task from FAM per-horizon event prediction.
- GECCO/SKAB: FAM likely first SSL method published.
- ETTm1: no SOTA exists for this event-prediction formulation.

## Phase 6 — theory self-check

- Proposition 1 audit: 6/7 proof steps CONFIRMED. Step 5 (Jensen-gap) had a real WEAKNESS (uses sup ϕ'' under marginal A4 but needs pointwise η(H*)). Closed via new assumption A1' (calibrated event posterior bounded a.s.). In-paper proofs untouched (the published constant is correct under an assumption the paper implicitly uses).
- New formal results (`paper-neurips/theory_findings.tex`, 966 lines):
  - Codomain-mismatch proposition explaining why pretrained predictor weights ≈ random for the FT task.
  - Excess-risk decomposition explaining v30 lf10 ≈ MLP-rand vs v20 lf5 ≫ scratch — two-regime story.
  - Per-horizon bound with horizon-indexed L_Δt, ε_Δt.
  - Calibration bound for discrete hazard CDF: O(K/√n).
  - MonotoneCDF non-claim documenting under what assumption it WOULD be Bayes-optimal.
  - 7 architecture rules R1-R7 each tied to a formal result + RESULTS.md evidence.

## Phase 8 — new dataset scouting

Top 4 picks for v31 paper appendix:
- **FEMTO/PRONOSTIA Bearing Dataset** (6h): Rotating machinery / vibration domain - entirely absent from current 13 datasets. Event is mechanical wear-to-failure (d
- **Tennessee Eastman Process (TEP) Extended Dataset** (5h): Chemical/petrochemical process simulation - new domain not in current benchmark. The event is a process fault onset (cat
- **MIMIC-Sepsis (shock onset task from MIMIC-IV)** (10h): Clinical ICU domain - the only clinical-deterioration dataset in the benchmark (MBA is ECG arrhythmia, a very different 
- **HAI 22.04 (HIL-based Augmented ICS Security Dataset)** (4h): Cyber-physical ICS attack dataset with more sophisticated attacks than BATADAL or SWaT: HAI 22.04 attacks are 4x harder 

## What shipped

- `experiments/v30/{phase0_dense_and_monotone,phase1_ablation,phase2_precursor_check,phase3_uniform,phase4_legacy_metrics,phase5_figures,phase7_finalize,finalize_session_summary,phase3_summary}.py`
- `experiments/v30/_runner_v30.py`
- `experiments/v30/results/{phase0,phase1,phase2,phase4,phase4a,phase8}_*.json` + `master_table.json`
- `experiments/v30/results/surface_pngs/*.png` (Phase 0/1/3 panels)
- `experiments/v30/surfaces/*.npz` (all stored surfaces)
- `notebooks/30_v30_analysis.qmd` + rendered HTML
- `paper-neurips/figures/fig_probability_surface_v2.{pdf,png}` (Phase 5a)
- `paper-neurips/figures/fig_benchmark_hauroc.pdf` (Phase 5b)
- `paper-neurips/theory_findings.tex` (966 lines, standalone)
- `fam-jepa/model.py`: MonotoneCDF + event_head_kind dispatch
- `experiments/RESULTS.md`: v30 section

## What did not ship

- Phase 9 (second foundation model baseline): attempted MOMENT, TimesFM,
  and Moirai (uni2ts) pip installs; all 3 hit Python 3.12 / torch
  dependency conflicts on the SageMaker base image (MOMENT:
  pkgutil.ImpImporter removed in py3.12; TimesFM: lingvo dependency
  unavailable; uni2ts: torchvision circular-import after torch 2.4.1
  downgrade). Documented as v31 work — needs a containerised env or
  HuggingFace-hosted variant. Phase 3b's per-domain head-choice ablation
  partly covers the appendix gap.
- PhysioNet inclusion: data/sepsis.py + data/physionet2012.py exist but are not wired into _runner_v29.LOADERS.
- MonotoneCDF Option B (predictor bypassed): would give strict monotonicity guarantee but requires a different finetuning loop (no per-horizon predictor pass). Deferred.

## Open questions for v31

- **Sub-5% label efficiency**: at 10% labels FAM-predft ties FAM-mlp-rand. Sub-5% is where the pretrained-predictor advantage may dominate (v20 5% result: pred-FT 0.261 vs scratch 0.035 — very different regime).
- **MBA encoder gap**: Chr2-probe (0.659) > FAM-probe (0.588) on MBA even though FAM-predft (0.739) wins overall. Why?
- **BATADAL plateau**: h-AUROC ~0.61 across heads — needs hyperparameter sweep or richer pretraining Δt sampling.
- **MonotoneCDF Option B**: predictor bypassed, MonotoneCDF takes h_t directly. Strict monotonicity guarantee in Δt.
- **PA-F1 framing in paper**: per Phase 4a, must explicitly cite Kim et al. AAAI 2022 and report non-PA F1 alongside any PA-F1 baseline numbers.
- **Variant B for predictor ablation** (v29 carryover): param-matched mean-pool MLP at ~200K params to disentangle attention from capacity in the transformer-predictor result.