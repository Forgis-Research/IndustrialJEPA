"""Compose the final v30 SESSION_SUMMARY.md from results JSON files.

Pulls:
  - results/master_table.json     (Phase 3 main numbers)
  - results/phase0_decision.json  (head choice)
  - results/phase1_decision.json  (5-variant ablation)
  - results/phase2_precursor_check.json
  - results/phase4_legacy_metrics.json
  - results/phase4a_sota.json
  - results/phase8_dataset_scouting.json

Writes:
  - SESSION_SUMMARY.md (overwrites the placeholder)
"""
import json
from datetime import date
from pathlib import Path

V30 = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v30')
RES = V30 / 'results'


def load(p):
    return json.load(open(p)) if p.exists() else None


def fmt(v, decimals=4):
    return f"{v:.{decimals}f}" if v is not None else '—'


def fmt_ms(mean, std, decimals=4):
    if mean is None: return '—'
    if std is None: return fmt(mean, decimals)
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def main():
    p0 = load(RES / 'phase0_decision.json') or {}
    p1 = load(RES / 'phase1_decision.json') or {}
    p2 = load(RES / 'phase2_precursor_check.json') or {}
    mt = load(RES / 'master_table.json') or {}
    p4 = load(RES / 'phase4_legacy_metrics.json') or {}
    p4a = load(RES / 'phase4a_sota.json') or {}
    p8 = load(RES / 'phase8_dataset_scouting.json') or {}

    md = []
    md.append("# V30 Session Summary\n")
    md.append(f"**Date**: 2026-04-25 → 2026-04-26  ")
    md.append(f"**Duration**: ~{int((mt.get('time_total_s') or 0) / 60)} min Phase 3 GPU "
              f"(plus Phase 0–2 and subagents in parallel)  ")
    md.append("**Scope**: dense-K=150 head decision (Phase 0); FAM-vs-Chronos2 "
              "fair ablation (Phase 1); precursor check on MSL/SMD/PhysioNet "
              "(Phase 2); uniform 13-dataset benchmark (Phase 3); legacy "
              "metrics + SOTA mapping (Phase 4); figures + quarto notebook "
              "(Phase 5); theory self-check (Phase 6); new-dataset scouting "
              "(Phase 8).\n")

    md.append("\n## One-sentence verdict\n")
    md.append("V30 locks dense K=150 as the canonical FAM head, refutes the "
              "head-capacity critique by showing FAM encoder beats Chronos-2 "
              "at matched probe capacity on 3/4 datasets, replaces the v29 "
              "heterogeneous master table with a clean 11-dataset 3-seed "
              "uniform benchmark, and ships a 966-line theory self-check "
              "that strengthens Proposition 1 and explains the "
              "label-efficiency two-regime story formally.\n")

    md.append("\n## Decisions made\n")
    md.append("- **Phase 0 (head)**: dense discrete hazard CDF, eval at K=150 "
              "horizons, 20 random training horizons per batch. MonotoneCDF "
              f"(Option A) collapsed to chance ({fmt(p0.get('variant_b_monotone_cdf', {}).get('mean_h_auroc'))}) "
              f"vs dense discrete ({fmt(p0.get('variant_a_dense_discrete', {}).get('mean_h_auroc'))}). "
              "Module kept in `model.py` as opt-in for v31.")
    md.append("- **Phase 1 (comparison)**: FAM-predft headline + Chr2-probe "
              "as canonical fair comparison. Encoder dominates head capacity "
              "in the ablation: FAM-probe beats Chr2-probe on FD001/FD003/"
              "BATADAL even with 257 trainable params/horizon.")
    msl = p2.get('datasets', {}).get('MSL', {})
    smd = p2.get('datasets', {}).get('SMD', {})
    md.append(f"- **Phase 2 (precursor)**: MSL **skip** (3-seed mean "
              f"{fmt(msl.get('mean'))}, below chance — refines v29 n=1 "
              f"of 0.438). SMD **include** (3-seed mean {fmt(smd.get('mean'))}). "
              f"PhysioNet skip (no LOADERS entry).")
    md.append("- **Phase 3 (benchmark)**: 11 datasets × 3 seeds × {100%, "
              "10%-on-4} = numbers below.\n")

    md.append("\n## Phase 1 — fair ablation (sparse h, 4 datasets × 3 seeds, 153s)\n")
    if p1:
        variants = ['fam-probe', 'chr2-probe', 'fam-predft', 'chr2-mlp', 'fam-mlp-rand']
        md.append("|             | " + " | ".join(p1.get('datasets', [])) + " |")
        md.append("|-------------|" + "|".join(['---'] * len(p1.get('datasets', []))) + "|")
        for v in variants:
            row = [v]
            for ds in p1.get('datasets', []):
                d = p1.get(v + '_hauroc', {}).get(ds, {})
                row.append(fmt_ms(d.get('mean'), d.get('std'), 3))
            md.append("| " + " | ".join(row) + " |")
        md.append("")
        md.append("**Findings**: encoder beats matched probe (FAM-probe > "
                  "Chr2-probe on 3/4); pretrained predictor init helps a "
                  "little (+0.007 to +0.041); at 10% labels FAM-predft ≈ "
                  "FAM-mlp-rand (sub-5% is where pretraining might dominate "
                  "— v31).")

    md.append("\n## Phase 3 — uniform benchmark (dense K=150)\n")
    if mt and 'datasets' in mt:
        v29_baselines = {
            'FD001': 0.742, 'FD002': 0.569, 'FD003': 0.819,
            'SMAP': 0.550, 'PSM': 0.559, 'MBA': 0.746,
            'GECCO': 0.859, 'BATADAL': 0.629, 'SKAB': 0.726,
            'ETTm1': 0.869, 'SMD': 0.616,
        }
        md.append("| Dataset | h-AUROC 100% (3s) | h-AUROC 10% | v29 sparse-K=8 | Δ |")
        md.append("|---------|--------------------|-------------|----------------|----|")
        for ds, row in mt['datasets'].items():
            f100 = row.get('lf100', {})
            f10 = row.get('lf10', {})
            f = fmt_ms(f100.get('mean_h_auroc'), f100.get('std_h_auroc'))
            t = fmt_ms(f10.get('mean_h_auroc'), f10.get('std_h_auroc')) if f10 else '—'
            v29 = v29_baselines.get(ds)
            v29s = fmt(v29, 3) if v29 else '—'
            d = (f100.get('mean_h_auroc') - v29) if (f100.get('mean_h_auroc') and v29) else None
            ds_str = f"{d:+.3f}" if d is not None else '—'
            md.append(f"| {ds} | {f} | {t} | {v29s} | {ds_str} |")

    md.append("\n## Phase 4 — legacy metrics + SOTA\n")
    if p4:
        md.append("Per-dataset legacy metric (no point-adjust for anomaly):\n")
        for ds, info in p4.items():
            if info.get('mean_rmse'):
                md.append(f"- **{ds}** (RMSE, RUL cap 125): {info['mean_rmse']:.2f}"
                          + (f" ± {info['std_rmse']:.2f}" if info.get('std_rmse') else ''))
            elif info.get('mean_auroc'):
                md.append(f"- **{ds}** (AUROC@Δt=1): {info['mean_auroc']:.4f}")
            elif info.get('mean_f1'):
                md.append(f"- **{ds}** (best-F1 no-PA @Δt=1): {info['mean_f1']:.4f}")

    if p4a:
        md.append("\nSOTA mapping (Phase 4a action items):")
        md.append("- C-MAPSS RMSE SOTA ≈ 11.3-11.4 (NOT STAR 10.61 — 2022 preprint).")
        md.append("- SMAP/PSM/SMD: cite Kim et al. AAAI 2022 PA-F1 trap; FAM is non-PA.")
        md.append("- MBA: TranAD/BTAD AUROC ~0.988 is anomaly-score framing — different task from FAM per-horizon event prediction.")
        md.append("- GECCO/SKAB: FAM likely first SSL method published.")
        md.append("- ETTm1: no SOTA exists for this event-prediction formulation.")

    md.append("\n## Phase 6 — theory self-check\n")
    md.append("- Proposition 1 audit: 6/7 proof steps CONFIRMED. Step 5 "
              "(Jensen-gap) had a real WEAKNESS (uses sup ϕ'' under marginal "
              "A4 but needs pointwise η(H*)). Closed via new assumption A1' "
              "(calibrated event posterior bounded a.s.). In-paper proofs "
              "untouched (the published constant is correct under an "
              "assumption the paper implicitly uses).")
    md.append("- New formal results (`paper-neurips/theory_findings.tex`, 966 lines):")
    md.append("  - Codomain-mismatch proposition explaining why pretrained predictor weights ≈ random for the FT task.")
    md.append("  - Excess-risk decomposition explaining v30 lf10 ≈ MLP-rand vs v20 lf5 ≫ scratch — two-regime story.")
    md.append("  - Per-horizon bound with horizon-indexed L_Δt, ε_Δt.")
    md.append("  - Calibration bound for discrete hazard CDF: O(K/√n).")
    md.append("  - MonotoneCDF non-claim documenting under what assumption it WOULD be Bayes-optimal.")
    md.append("  - 7 architecture rules R1-R7 each tied to a formal result + RESULTS.md evidence.")

    md.append("\n## Phase 8 — new dataset scouting\n")
    if p8 and p8.get('top_4_picks'):
        md.append(f"Top 4 picks for v31 paper appendix:")
        for c in p8.get('candidates', []):
            if c.get('name') in p8.get('top_4_picks', []):
                md.append(f"- **{c['name']}** ({c.get('integration_effort_hours')}h): "
                          f"{c.get('different_because', '')[:120]}")

    md.append("\n## What shipped\n")
    shipped = [
        '`experiments/v30/{phase0_dense_and_monotone,phase1_ablation,phase2_precursor_check,phase3_uniform,phase4_legacy_metrics,phase5_figures,phase7_finalize,finalize_session_summary,phase3_summary}.py`',
        '`experiments/v30/_runner_v30.py`',
        '`experiments/v30/results/{phase0,phase1,phase2,phase4,phase4a,phase8}_*.json` + `master_table.json`',
        '`experiments/v30/results/surface_pngs/*.png` (Phase 0/1/3 panels)',
        '`experiments/v30/surfaces/*.npz` (all stored surfaces)',
        '`notebooks/30_v30_analysis.qmd` + rendered HTML',
        '`paper-neurips/figures/fig_probability_surface_v2.{pdf,png}` (Phase 5a)',
        '`paper-neurips/figures/fig_benchmark_hauroc.pdf` (Phase 5b)',
        '`paper-neurips/theory_findings.tex` (966 lines, standalone)',
        '`fam-jepa/model.py`: MonotoneCDF + event_head_kind dispatch',
        '`experiments/RESULTS.md`: v30 section',
    ]
    for s in shipped:
        md.append(f"- {s}")

    md.append("\n## What did not ship\n")
    md.append("- Phase 9 (second foundation model baseline): stretch goal not "
              "executed — TimesFM 2.5 / Moirai 2.0 feature extraction would "
              "take ~3h on top of Phase 3.")
    md.append("- PhysioNet inclusion: data/sepsis.py + data/physionet2012.py "
              "exist but are not wired into _runner_v29.LOADERS.")
    md.append("- MonotoneCDF Option B (predictor bypassed): would give strict "
              "monotonicity guarantee but requires a different finetuning loop "
              "(no per-horizon predictor pass). Deferred.")

    md.append("\n## Open questions for v31\n")
    md.append("- **Sub-5% label efficiency**: at 10% labels FAM-predft ties "
              "FAM-mlp-rand. Sub-5% is where the pretrained-predictor "
              "advantage may dominate (v20 5% result: pred-FT 0.261 vs "
              "scratch 0.035 — very different regime).")
    md.append("- **MBA encoder gap**: Chr2-probe (0.659) > FAM-probe (0.588) "
              "on MBA even though FAM-predft (0.739) wins overall. Why?")
    md.append("- **BATADAL plateau**: h-AUROC ~0.61 across heads — needs "
              "hyperparameter sweep or richer pretraining Δt sampling.")
    md.append("- **MonotoneCDF Option B**: predictor bypassed, MonotoneCDF "
              "takes h_t directly. Strict monotonicity guarantee in Δt.")
    md.append("- **PA-F1 framing in paper**: per Phase 4a, must explicitly "
              "cite Kim et al. AAAI 2022 and report non-PA F1 alongside any "
              "PA-F1 baseline numbers.")
    md.append("- **Variant B for predictor ablation** (v29 carryover): "
              "param-matched mean-pool MLP at ~200K params to disentangle "
              "attention from capacity in the transformer-predictor result.")

    out = V30 / 'SESSION_SUMMARY.md'
    out.write_text('\n'.join(md))
    print(f"wrote {out}")
    print('\n--- preview ---')
    print('\n'.join(md[:30]))


if __name__ == '__main__':
    main()
