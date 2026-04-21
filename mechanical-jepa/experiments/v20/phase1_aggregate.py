"""V20 Phase 1: Multi-domain benchmark aggregation.

Collates results from v18/v19/v20 JSONs into a single benchmark table:

| Dataset | Domain | FAM Primary | FAM Legacy | SOTA | SOTA Params |

Primary = per-window F1 (where available) or domain-standard metric.
Legacy = RMSE / PA-F1 / macro-F1 / etc. for cross-paper comparability.

Produces:
  - phase1_benchmark.json (structured)
  - phase1_benchmark.md (paper-ready table)
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')
V18 = ROOT / 'experiments' / 'v18'
V19 = ROOT / 'experiments' / 'v19'
V20 = ROOT / 'experiments' / 'v20'


def load_json(p):
    with open(p) as f:
        return json.load(f)


def fmt(mean, std, n):
    if mean is None:
        return '-'
    if std is None or n is None or n <= 1:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f} ({n}s)"


def collect_cmapss():
    """C-MAPSS FD001 from v20 phase 0."""
    row = {'dataset': 'C-MAPSS FD001', 'domain': 'Turbofan RUL',
           'event': 'Failure', 'fam_params': 2_373_632}
    p = V20 / 'phase0_pred_ft.json'
    if not p.exists():
        row['status'] = 'missing'
        return row
    d = load_json(p)
    results = d.get('results', {})
    pred_ft_100 = results.get('pred_ft@1.0', {})
    pred_ft_5   = results.get('pred_ft@0.05', {})
    e2e_100     = results.get('e2e@1.0', {})
    e2e_5       = results.get('e2e@0.05', {})
    probe_100   = results.get('probe_h@1.0', {})
    for mode, agg in [('pred_ft_100', pred_ft_100),
                      ('pred_ft_5', pred_ft_5),
                      ('e2e_100', e2e_100),
                      ('e2e_5', e2e_5),
                      ('probe_h_100', probe_100)]:
        if not agg:
            continue
        row[f'{mode}_f1w'] = agg.get('per_window_f1_mean_mean')
        row[f'{mode}_f1w_std'] = agg.get('per_window_f1_mean_std')
        row[f'{mode}_auroc'] = agg.get('per_window_auroc_mean_mean')
        row[f'{mode}_rmse'] = agg.get('test_rmse_mean')
        row[f'{mode}_rmse_std'] = agg.get('test_rmse_std')
        row[f'{mode}_n_seeds'] = agg.get('n_seeds')

    # SOTA: STAR RMSE 10.61 (paper), supervised ~100M params (PatchTST scale)
    row['sota_method'] = 'STAR (Fan 2024)'
    row['sota_rmse'] = 10.61
    row['sota_rmse_note'] = '1 seed (paper)'
    row['sota_params'] = None            # STAR paper doesn't break it out cleanly
    return row


def collect_psm():
    row = {'dataset': 'PSM', 'domain': 'Server metrics',
           'event': 'Anomaly', 'fam_params': 2_373_632}
    # v19 PA-F1 and non-PA F1
    p = V19 / 'phase1_psm_results.json'
    if p.exists():
        d = load_json(p)
        agg = d.get('ep50_aggregate', {})
        k100 = agg.get('k100', {})
        k_auto = agg.get('k_auto', {})
        row['fam_mahal_pa_f1_k100']    = k100.get('pa_mean')
        row['fam_mahal_pa_f1_k100_std']= k100.get('pa_std')
        row['fam_mahal_pa_f1_auto']    = k_auto.get('pa_mean')
        row['fam_mahal_pa_f1_auto_std']= k_auto.get('pa_std')
        row['fam_mahal_nonpa_k100']    = k100.get('non_pa_mean')
    # v20 pred-FT if exists
    p2 = V20 / 'phase1_pred_ft_psm.json'
    if p2.exists():
        d2 = load_json(p2)
        res = d2.get('results', {})
        for m in ['probe_h', 'frozen_multi', 'pred_ft']:
            if m not in res: continue
            runs = res[m]
            if not runs: continue
            f1s = [r['f1_mean'] for r in runs]
            auc = [r['auroc_mean'] for r in runs]
            row[f'fam_{m}_f1w']       = float(np.mean(f1s))
            row[f'fam_{m}_f1w_std']   = float(np.std(f1s, ddof=1))
            row[f'fam_{m}_auroc_w']   = float(np.mean(auc))
            row[f'fam_{m}_n_seeds']   = len(runs)

    # SOTA refs
    row['sota_method']   = 'MTS-JEPA (He 2026)'
    row['sota_pa_f1']    = 0.616
    row['sota_pa_f1_note'] = '1 seed (paper)'
    row['sota_params']   = None
    return row


def collect_smd():
    row = {'dataset': 'SMD', 'domain': 'Server machine',
           'event': 'Anomaly', 'fam_params': 2_373_632}
    p = V19 / 'phase4_smd_results.json'
    if p.exists():
        d = load_json(p)
        agg = d.get('aggregate', {})
        k100 = agg.get('k100', {})
        row['fam_mahal_pa_f1_k100']    = k100.get('pa_mean')
        row['fam_mahal_pa_f1_k100_std']= k100.get('pa_std')
        row['fam_mahal_nonpa_k100']    = k100.get('non_pa_mean')
    row['sota_method'] = 'Anomaly Transformer (Xu 2022)'
    row['sota_pa_f1']  = 0.925
    row['sota_pa_f1_note'] = '1 seed (paper, PA inflates)'
    return row


def collect_mba():
    row = {'dataset': 'MBA ECG', 'domain': 'Cardiac',
           'event': 'Arrhythmia', 'fam_params': 2_373_632}
    p = V19 / 'phase2_mba_results.json'
    if p.exists():
        d = load_json(p)
        agg = d.get('aggregate', {})
        k50 = agg.get('k50', {})
        row['fam_mahal_pa_f1_k50']     = k50.get('pa_mean')
        row['fam_mahal_pa_f1_k50_std'] = k50.get('pa_std')
        row['fam_mahal_nonpa_k50']     = k50.get('non_pa_mean')
    row['sota_method'] = 'Schmidt & Simic (2020)'
    row['sota_pa_f1']  = None
    row['sota_note']   = 'Few SSL papers target MBA explicitly'
    return row


def collect_paderborn():
    row = {'dataset': 'Paderborn bearing', 'domain': 'Bearing fault',
           'event': 'Bearing class (K001/KA01/KI01)',
           'fam_params': 2_373_632}
    p = V19 / 'phase5_paderborn_results.json'
    if p.exists():
        d = load_json(p)
        agg = d.get('aggregate', {})
        row['fam_macro_f1']     = agg.get('macro_f1_mean')
        row['fam_macro_f1_std'] = agg.get('macro_f1_std')
        row['fam_test_acc']     = agg.get('test_acc_mean')
    row['sota_method'] = 'STAR (Fan 2024)'        # or RmGPT
    row['sota_macro_f1'] = None
    return row


def collect_smap():
    row = {'dataset': 'SMAP', 'domain': 'Spacecraft telemetry',
           'event': 'Anomaly', 'fam_params': 2_373_632}
    # SMAP Mahalanobis k=100 from v18 RESULTS.md (phase 4)
    row['fam_mahal_pa_f1_k100']     = 0.793
    row['fam_mahal_pa_f1_k100_std'] = 0.014
    row['fam_mahal_nonpa_k100']     = 0.038
    row['fam_mahal_source']         = 'v18 phase 4 (3 seeds)'
    row['sota_method']  = 'MTS-JEPA (He 2026)'
    row['sota_pa_f1']   = 0.336
    row['sota_pa_f1_note'] = '1 seed (paper)'
    return row


def collect_msl():
    row = {'dataset': 'MSL', 'domain': 'Spacecraft telemetry',
           'event': 'Anomaly', 'fam_params': 2_373_632}
    # MSL Mahalanobis from v18 RESULTS.md (phase 4)
    row['fam_mahal_pa_f1_k100']     = 0.707
    row['fam_mahal_pa_f1_k100_std'] = 0.050
    row['fam_mahal_source']         = 'v18 phase 4 (3 seeds)'
    row['sota_method']  = 'MTS-JEPA (He 2026)'
    row['sota_pa_f1']   = 0.336
    row['sota_pa_f1_note'] = '1 seed (paper)'
    return row


def build_benchmark_md(rows):
    lines = []
    lines.append("# V20 Multi-Domain Benchmark (FAM across N datasets)\n")
    lines.append("All FAM results use the 1.26M-parameter (V17 arch, d_model=256, 2L, 4H) "
                 "model. Legacy metrics (PA-F1, RMSE, macro-F1) reported for literature "
                 "comparability. Per-window F1 (W=16) reported where pred-FT is run.\n")
    lines.append("\n## FAM primary results\n")
    lines.append("| Dataset | Domain | FAM primary | FAM legacy | SOTA (primary) | SOTA ref |")
    lines.append("|---------|--------|-------------|------------|----------------|----------|")
    for r in rows:
        ds   = r['dataset']
        dom  = r['domain']
        # Primary metric pick
        if 'C-MAPSS' in ds:
            f1 = r.get('pred_ft_100_f1w')
            f1_std = r.get('pred_ft_100_f1w_std')
            rmse = r.get('pred_ft_100_rmse')
            rmse_std = r.get('pred_ft_100_rmse_std')
            primary = f"F1w {fmt(f1, f1_std, r.get('pred_ft_100_n_seeds'))} (pred-FT)"
            legacy  = f"RMSE {fmt(rmse, rmse_std, r.get('pred_ft_100_n_seeds'))}"
            sota    = f"RMSE {r.get('sota_rmse', '-')}"
            sota_ref = r.get('sota_method', '-')
        elif ds == 'PSM':
            if r.get('fam_pred_ft_f1w') is not None:
                primary = f"F1w {fmt(r.get('fam_pred_ft_f1w'), r.get('fam_pred_ft_f1w_std'), r.get('fam_pred_ft_n_seeds'))} (pred-FT)"
            else:
                primary = "(pred-FT pending)"
            legacy = (f"PA-F1 {fmt(r.get('fam_mahal_pa_f1_k100'), r.get('fam_mahal_pa_f1_k100_std'), 3)} "
                      f"(Mahal k=100)")
            sota = f"PA-F1 {r.get('sota_pa_f1', '-')}"
            sota_ref = r.get('sota_method', '-')
        elif ds in ('SMAP', 'MSL', 'SMD'):
            primary = (f"PA-F1 {fmt(r.get('fam_mahal_pa_f1_k100'), r.get('fam_mahal_pa_f1_k100_std'), 3)}")
            legacy  = f"non-PA F1 {r.get('fam_mahal_nonpa_k100', '-')}"
            sota    = f"PA-F1 {r.get('sota_pa_f1', '-')}"
            sota_ref = r.get('sota_method', '-')
        elif ds == 'MBA ECG':
            primary = (f"PA-F1 {fmt(r.get('fam_mahal_pa_f1_k50'), r.get('fam_mahal_pa_f1_k50_std'), 3)}")
            legacy  = f"non-PA F1 {r.get('fam_mahal_nonpa_k50', '-')}"
            sota = '-'
            sota_ref = r.get('sota_method', '-')
        elif 'Paderborn' in ds:
            primary = (f"macro-F1 {fmt(r.get('fam_macro_f1'), r.get('fam_macro_f1_std'), 3)}")
            legacy  = f"acc {r.get('fam_test_acc', '-'):.3f}"
            sota = '-'
            sota_ref = r.get('sota_method', '-')
        else:
            primary = '-'; legacy = '-'; sota = '-'; sota_ref = '-'
        lines.append(f"| {ds} | {dom} | {primary} | {legacy} | {sota} | {sota_ref} |")

    lines.append("\n## Pred-FT (FD001) vs baselines detailed (Phase 0)\n")
    cm = next((r for r in rows if 'C-MAPSS' in r['dataset']), None)
    if cm:
        lines.append("| Mode | Labels | F1w | RMSE |")
        lines.append("|------|--------|-----|------|")
        for m, lb in [('probe_h', '100%'), ('pred_ft', '100%'), ('pred_ft', '5%'),
                      ('e2e', '100%'), ('e2e', '5%')]:
            pct = '100' if lb == '100%' else '5'
            f1 = cm.get(f'{m}_{pct}_f1w')
            f1s = cm.get(f'{m}_{pct}_f1w_std')
            rmse = cm.get(f'{m}_{pct}_rmse')
            rmse_std = cm.get(f'{m}_{pct}_rmse_std')
            ns = cm.get(f'{m}_{pct}_n_seeds')
            lines.append(f"| {m} | {lb} | {fmt(f1, f1s, ns)} | {fmt(rmse, rmse_std, ns)} |")

    return "\n".join(lines) + "\n"


def main():
    rows = [collect_cmapss(), collect_psm(), collect_smap(),
            collect_msl(), collect_smd(), collect_mba(),
            collect_paderborn()]
    out_json = V20 / 'phase1_benchmark.json'
    out_md = V20 / 'phase1_benchmark.md'
    with open(out_json, 'w') as f:
        json.dump({'config': 'v20_phase1_benchmark', 'rows': rows},
                  f, indent=2, default=float)
    md = build_benchmark_md(rows)
    with open(out_md, 'w') as f:
        f.write(md)
    print(md)


if __name__ == '__main__':
    main()
