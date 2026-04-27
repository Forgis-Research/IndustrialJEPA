"""V32 Phase 6: cross-check, generate Table 4 LaTeX, update RESULTS.md,
write SESSION_SUMMARY.md.

Reads:
  - results/sota_research.md  (Phase 1)
  - results/rmse_probe.json   (Phase 2)
  - results/legacy_metrics_full.json (Phase 3)
  - results/baseline_lf10.json (Phase 4)

Writes:
  - results/table4_latex.txt
  - results/SESSION_SUMMARY.md
  - appends to /home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/RESULTS.md
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

V32 = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v32')
RES = V32 / 'results'
RESULTS_MD = V32.parent / 'RESULTS.md'


def load_json(name: str) -> Dict[str, Any]:
    p = RES / name
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def fmt_pm(mean: float, std: float, fmt: str = '.2f') -> str:
    return f"${mean:{fmt}}${{\\scriptsize$\\pm {std:{fmt}}$}}"


def gen_rmse_row(rmse_data: Dict, ds: str, lf: int) -> str:
    """Format C-MAPSS RMSE for a (dataset, lf) cell."""
    key = f'{ds}_lf{lf}'
    agg = rmse_data.get('agg', {}).get(key, {})
    if not agg:
        return r'\placeholder{--}'
    return f"${agg['rmse_mean']:.2f}${{\\scriptsize$\\pm{agg['rmse_std']:.2f}$}}"


def gen_legacy_row(legacy_data: Dict, ds: str, lf: int, kind: str = 'auto') -> str:
    """Format legacy metric for a (dataset, lf) cell.

    kind:
      'auto' = pick PA-F1 for SMAP/PSM/SMD; non-PA F1 for SKAB/GECCO/BATADAL;
              AUROC for MBA. (Driven by SOTA reference protocol.)
      'pa', 'non_pa', 'auroc', 'auprc' = explicit.
    """
    key = f'{ds}_lf{lf}'
    agg = legacy_data.get(key, {}).get('agg', {})
    if not agg:
        return r'\placeholder{--}'

    if kind == 'auto':
        if ds in ('SMAP', 'PSM', 'SMD', 'MSL'):
            target = 'best_pa_f1'
        elif ds in ('SKAB', 'GECCO', 'BATADAL', 'ETTm1'):
            target = 'best_non_pa_f1'
        elif ds in ('MBA',):
            target = 'best_auroc'
        else:
            target = 'best_non_pa_f1'
    else:
        target = {'pa': 'best_pa_f1', 'non_pa': 'best_non_pa_f1',
                  'auroc': 'best_auroc'}.get(kind, 'best_non_pa_f1')

    mean = agg.get(f'{target}_mean')
    std = agg.get(f'{target}_std', 0.0)
    if mean is None:
        return r'\placeholder{--}'
    label = {'best_pa_f1': 'PA-F1', 'best_non_pa_f1': 'F1',
             'best_auroc': 'AUROC'}.get(target, '')
    return f"{label} ${mean:.2f}${{\\scriptsize$\\pm{std:.2f}$}}"


def gen_chr2_lf10_row(chr2_data: Dict, ds: str) -> str:
    """Chr-2 h-AUROC at 10% labels."""
    agg = chr2_data.get('agg', {}).get('chr2', {}).get(f'{ds}_lf10', {})
    if not agg:
        return '---'
    return f"${agg['mean_h_auroc']:.2f}${{\\scriptsize$\\pm{agg['std_h_auroc']:.2f}$}}"


def main():
    rmse = load_json('rmse_probe.json')
    legacy = load_json('legacy_metrics_full.json')
    chr2 = load_json('baseline_lf10.json')

    # Build LaTeX snippets to drop into Table 4 cells
    snippets = []
    snippets.append('=== Table 4 cell replacements ===\n')
    snippets.append('# C-MAPSS RMSE (FAM column)')
    snippets.append('FD001 100%: ' + gen_rmse_row(rmse, 'FD001', 100))
    snippets.append('FD001 10%:  ' + gen_rmse_row(rmse, 'FD001', 10))
    snippets.append('FD002 100%: ' + gen_rmse_row(rmse, 'FD002', 100))
    snippets.append('FD002 10%:  ' + gen_rmse_row(rmse, 'FD002', 10))
    snippets.append('FD003 100%: ' + gen_rmse_row(rmse, 'FD003', 100))
    snippets.append('FD003 10%:  ' + gen_rmse_row(rmse, 'FD003', 10))
    snippets.append('')
    snippets.append('# Legacy metric — FAM column')
    for ds in ['SMAP', 'PSM', 'SMD', 'MBA', 'SKAB', 'GECCO', 'BATADAL']:
        snippets.append(f'{ds} 100%: ' + gen_legacy_row(legacy, ds, 100))
        snippets.append(f'{ds} 10%:  ' + gen_legacy_row(legacy, ds, 10))
    snippets.append('')
    snippets.append('# Chronos-2 h-AUROC at 10% labels')
    for ds in ['FD001', 'FD002', 'FD003', 'SMAP', 'PSM', 'SMD',
               'MBA', 'GECCO', 'BATADAL', 'MSL']:
        v = gen_chr2_lf10_row(chr2, ds)
        snippets.append(f'{ds}: {v}')
    snippets.append('')

    out = '\n'.join(snippets)
    (RES / 'table4_latex.txt').write_text(out)
    print(out)
    print(f'\nWrote {RES / "table4_latex.txt"}')

    # Build SESSION_SUMMARY
    summary = build_summary(rmse, legacy, chr2)
    (RES / 'SESSION_SUMMARY.md').write_text(summary)
    print(f'Wrote {RES / "SESSION_SUMMARY.md"}')

    # Append to RESULTS.md
    resmd_addition = build_resultsmd_section(rmse, legacy, chr2)
    if RESULTS_MD.exists():
        existing = RESULTS_MD.read_text()
        if '## v32' not in existing:
            with open(RESULTS_MD, 'a') as f:
                f.write('\n\n' + resmd_addition)
            print(f'Appended v32 section to {RESULTS_MD}')
        else:
            print(f'NOTE: v32 section already exists in {RESULTS_MD}')


def build_summary(rmse: Dict, legacy: Dict, chr2: Dict) -> str:
    out = ['# V32 Session Summary',
           '',
           'NeurIPS 2026 Table 4 SOTA-grade results: rigorous baselines, label efficiency, and protocol-aligned legacy metrics.',
           '',
           '## What was computed',
           '',
           '- **Phase 1**: SOTA literature research per dataset. Output `results/sota_research.md`.',
           '- **Phase 2**: MSE-RUL probe on the frozen v30 encoder for FD001/FD002/FD003 (3 seeds, hidden-dim sweep, MSE & Huber loss, 1- and 2-layer MLP). 100% and 10% labels. Output `results/rmse_probe.json`.',
           '- **Phase 3**: Legacy metric recomputation (AUROC / AUPRC / PA-F1 / non-PA F1) over horizon sweep {1, 3, 5, 10, 20, 50, 100, 150} for all anomaly datasets at 100% and 10% labels. Deep investigation of GECCO/BATADAL. Output `results/legacy_metrics_full.json`.',
           '- **Phase 4**: Chronos-2 (cached features) and MOMENT (cached features) baselines at 10% labels using identical Chr2MLP head. Output `results/baseline_lf10.json`.',
           '- **Phase 5**: Real dense-K=150 surface comparison figure (FD001, 4 engines). Output `paper-neurips/figures/fig_probability_surface_v2.{pdf,png}`.',
           '',
           '## Phase 2 RMSE results (frozen-encoder MSE probe)',
           '',
           '| Dataset | Labels | RMSE | NASA-Score | STAR SOTA |',
           '|---------|--------|------|-----------|-----------|',
    ]
    for ds, sota in [('FD001', 10.61), ('FD002', 13.47), ('FD003', 10.71)]:
        for lf in [100, 10]:
            agg = rmse.get('agg', {}).get(f'{ds}_lf{lf}', {})
            if agg:
                out.append(f'| {ds} | {lf}% | {agg["rmse_mean"]:.2f} ± {agg["rmse_std"]:.2f} | {agg["nasa_mean"]:.0f} | {sota if lf==100 else "-"} |')
    out.append('')

    out.append('## Phase 3 Legacy metrics (best across horizon sweep)')
    out.append('')
    out.append('| Dataset | Labels | Best non-PA F1 | Best PA-F1 | Best AUROC |')
    out.append('|---------|--------|----------------|-----------|-----------|')
    for ds in ['SMAP', 'PSM', 'SMD', 'MBA', 'SKAB', 'GECCO', 'BATADAL']:
        for lf in [100, 10]:
            agg = legacy.get(f'{ds}_lf{lf}', {}).get('agg', {})
            if agg:
                npa = agg.get('best_non_pa_f1_mean')
                pa = agg.get('best_pa_f1_mean')
                auc = agg.get('best_auroc_mean')
                out.append(f'| {ds} | {lf}% | '
                           f'{npa:.3f} | ' if npa is not None else '— | '
                           f'{pa:.3f} | ' if pa is not None else '— | '
                           f'{auc:.3f} |' if auc is not None else '— |')
    out.append('')

    out.append('## Phase 4 Chronos-2 / MOMENT lf10 baselines')
    out.append('')
    out.append('| Model | Dataset | Labels | h-AUROC | h-AUPRC |')
    out.append('|-------|---------|--------|---------|---------|')
    for model in ['chr2', 'moment']:
        agg = chr2.get('agg', {}).get(model, {})
        for k, v in sorted(agg.items()):
            out.append(f'| {model} | {k} | {v["mean_h_auroc"]:.3f}±{v["std_h_auroc"]:.3f} | {v["mean_h_auprc"]:.3f}±{v["std_h_auprc"]:.3f} |')
    out.append('')

    out.append('## Key findings')
    out.append('')
    out.append('TODO: fill after consolidation.')
    out.append('')
    return '\n'.join(out)


def build_resultsmd_section(rmse: Dict, legacy: Dict, chr2: Dict) -> str:
    out = ['## v32: Table 4 final pass (2026-04-26)',
           '',
           '**Goal**: complete every \\placeholder{--} in Table 4 with defensible numbers.',
           '',
           '### C-MAPSS MSE-probe RMSE (frozen v30 encoder)']
    for ds in ['FD001', 'FD002', 'FD003']:
        for lf in [100, 10]:
            agg = rmse.get('agg', {}).get(f'{ds}_lf{lf}', {})
            if agg:
                out.append(f'- {ds} lf{lf}%: RMSE {agg["rmse_mean"]:.2f}±{agg["rmse_std"]:.2f}, NASA {agg["nasa_mean"]:.0f}')
    out.append('')
    return '\n'.join(out)


if __name__ == '__main__':
    main()
