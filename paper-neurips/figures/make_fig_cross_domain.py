"""Generate a cross-domain bar chart: FAM vs Chronos-2 vs legacy-SOTA.

Shows pooled AUPRC (primary metric) across all v24 datasets. Groups by
domain. SOTA-legacy numbers are on different metrics per dataset; shown
as annotated horizontal markers (e.g. "AUROC 0.87") for reference rather
than direct comparison.

Reads from:
  fam-jepa/experiments/v24/results/phase2_FD00*.json      (turbofan)
  fam-jepa/experiments/v24/results/phase4_{SMAP,MSL,PSM,SMD,MBA}.json
  fam-jepa/experiments/v24/results/phase6_sepsis.json
  fam-jepa/experiments/v24/results/phase11_{GECCO,BATADAL,physionet2012}.json
  fam-jepa/experiments/v24/results/baseline_chronos2_agg.json
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'figure.dpi': 150,
})

FAM = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
RES = FAM / 'experiments/v24/results'


def _load(p):
    if not p.exists():
        return None
    return json.loads(p.read_text())


# Dataset (display name, loader path, domain, sota_metric, sota_value)
ROWS = [
    ('FD001',  'phase2_FD001.json',         'Turbofan',     'RMSE',  10.61),
    ('FD002',  'phase2_FD002.json',         'Turbofan',     'RMSE',  13.47),
    ('FD003',  'phase2_FD003.json',         'Turbofan',     'RMSE',  10.71),
    ('SMAP',   'phase4_SMAP.json',          'Spacecraft',   'PA-F1', 0.336),
    ('MSL',    'phase4_MSL.json',           'Spacecraft',   'PA-F1', 0.336),
    ('PSM',    'phase4_PSM.json',           'Server',       'PA-F1', 0.616),
    ('SMD',    'phase4_SMD.json',           'Server',       'PA-F1', 0.925),
    ('MBA',    'phase4_MBA.json',           'Cardiac',      '-',     None),
    ('Sepsis', 'phase6_sepsis.json',        'ICU',          'AUROC', 0.85),
    ('GECCO',  'phase11_GECCO.json',        'Water-IoT',    'AUROC', 0.88),
    ('BATADAL', 'phase11_BATADAL.json',     'Water-Cyber',  'AUC',   0.972),
    ('PN2012', 'phase11_physionet2012.json', 'ICU-Mort.',   'AUROC', 0.868),
]
DOMAIN_ORDER = ['Turbofan', 'Spacecraft', 'Server', 'Cardiac', 'ICU',
                'Water-IoT', 'Water-Cyber', 'ICU-Mort.']


def main():
    chronos = _load(RES / 'baseline_chronos2_agg.json') or {}

    # Map chronos keys to display names
    chronos_lookup = {
        'FD001': chronos.get('FD001'), 'FD002': chronos.get('FD002'),
        'FD003': chronos.get('FD003'), 'SMAP': chronos.get('SMAP'),
        'MSL': chronos.get('MSL'),    'PSM': chronos.get('PSM'),
        'SMD': chronos.get('SMD'),    'MBA': chronos.get('MBA'),
        'Sepsis': None,
        'GECCO': chronos.get('GECCO'),
        'BATADAL': chronos.get('BATADAL'),
        'PN2012': None,
    }

    names, fam_m, fam_s, chr_m, chr_s, domains, sota_labels = [], [], [], [], [], [], []
    for nm, fname, dom, sota_metric, sota_val in ROWS:
        fam = _load(RES / fname)
        if fam is None or 'auprc_mean' not in fam:
            continue
        names.append(nm); domains.append(dom)
        fam_m.append(fam['auprc_mean']); fam_s.append(fam['auprc_std'])
        c = chronos_lookup.get(nm)
        if c:
            chr_m.append(c['auprc_mean']); chr_s.append(c['auprc_std'])
        else:
            chr_m.append(np.nan); chr_s.append(np.nan)
        if sota_val is not None:
            sota_labels.append(f'{sota_metric}\n{sota_val:g}')
        else:
            sota_labels.append('-')

    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    x = np.arange(len(names))
    w = 0.35
    fam_arr = np.array(fam_m); chr_arr = np.array(chr_m)
    fam_err = np.array(fam_s); chr_err = np.array(chr_s)

    b1 = ax.bar(x - w/2, fam_arr, w, yerr=fam_err,
                color='#2E5FAB', label='FAM (v24, pred-FT)',
                capsize=2, edgecolor='none')
    ok = ~np.isnan(chr_arr)
    b2 = ax.bar(x[ok] + w/2, chr_arr[ok], w, yerr=chr_err[ok],
                color='#E28F41',
                label=r'Chronos-2 (frozen) + linear probe',
                capsize=2, edgecolor='none')

    # Annotate SOTA-legacy under each bar pair
    for i, lbl in enumerate(sota_labels):
        if lbl == '-':
            continue
        ax.text(x[i], -0.09, lbl, ha='center', va='top', fontsize=6,
                color='#666', transform=ax.get_xaxis_transform())

    # Domain labels as minor group marks at top
    last_dom = None
    for i, (nm, dom) in enumerate(zip(names, domains)):
        if dom != last_dom:
            ax.text(x[i], 1.08, dom, ha='left', va='bottom', fontsize=7,
                    color='#444', transform=ax.get_xaxis_transform(),
                    fontstyle='italic')
            last_dom = dom

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=0, fontsize=7)
    ax.set_ylabel('pooled AUPRC (primary metric)')
    ax.set_ylim(0, 1.05)
    ax.set_title('FAM vs Chronos-2 across 12 datasets / 8 domains (v24)',
                 fontsize=9, pad=20)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20),
              ncol=2, frameon=False)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout(pad=0.6)
    out = Path(__file__).parent / 'fig_cross_domain.pdf'
    plt.savefig(out, bbox_inches='tight', pad_inches=0.05)
    png = Path(__file__).parent / 'fig_cross_domain.png'
    plt.savefig(png, bbox_inches='tight', pad_inches=0.05, dpi=300)
    print(f'wrote {out}')
    print(f'wrote {png}')


if __name__ == '__main__':
    main()
