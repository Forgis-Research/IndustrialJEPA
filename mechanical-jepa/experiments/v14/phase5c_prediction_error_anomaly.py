"""
V14 Phase 5c.4: Prediction-error anomaly score diagnostic.

Inspired by MTS-JEPA comparison: use per-cycle prediction error as an
zero-shot anomaly / health signal. If the JEPA predictor is worse on
degraded cycles than healthy ones, prediction error IS a zero-label
anomaly indicator.

Protocol:
- Use the pretrained V2 checkpoint (experiments/v11/best_pretrain_L1_v2.pt)
  so results are directly comparable to the reported numbers in v12/v13.
- For N representative test engines at every cycle t > 10:
    - Take past = x_{1:t}, future = x_{t+1:t+k} with k=15 (mid-range).
    - Compute predictor(h_past, k) vs target_encoder(future).
    - L1 norm = prediction error at cycle t.
- Plot: prediction error vs cycle, colored by RUL stage (healthy vs degraded).
- Metric: Spearman rho(prediction_error, t / T_engine) across cycles.

Output:
- experiments/v14/prediction_error_analysis.json
- analysis/plots/v14/prediction_error_vs_degradation.png + pdf
"""

import sys, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V14_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v14')
PLOT_PNG = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/analysis/plots/v14')
PLOT_PDF = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/notebooks/plots')
sys.path.insert(0, str(V11_DIR))

from data_utils import load_cmapss_subset, N_SENSORS, RUL_CAP, compute_rul_labels
from models import TrajectoryJEPA

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = V11_DIR / 'best_pretrain_L1_v2.pt'

plt.rcParams.update({
    'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def main():
    print(f"V14 Phase 5c.4: prediction-error anomaly score")
    print(f"Checkpoint: {CKPT}")
    t0 = time.time()

    data = load_cmapss_subset('FD001')
    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1, d_model=256, n_heads=4,
        n_layers=2, d_ff=512, dropout=0.1,
    ).to(DEVICE)
    model.load_state_dict(torch.load(str(CKPT), map_location=DEVICE))
    model.eval()

    # Pick engines spanning short/medium/long lifetimes
    engines = data['train_engines']
    lengths = [(eid, len(seq)) for eid, seq in engines.items()]
    lengths.sort(key=lambda kv: kv[1])
    picks = [lengths[5], lengths[len(lengths) // 4], lengths[len(lengths) // 2],
             lengths[3 * len(lengths) // 4], lengths[-5]]
    pick_ids = [eid for eid, _ in picks]
    print(f"Picked engines: {pick_ids} with lengths {[l for _,l in picks]}")

    K = 15  # fixed horizon for diagnostic
    MIN_T = 10

    results_by_engine = {}
    rhos = []
    for eid, T in picks:
        seq = engines[eid]
        rul = compute_rul_labels(T, RUL_CAP)
        # RUL fraction: 1 = start (healthy), 0 = failure
        rul_frac = rul / RUL_CAP

        pred_errors, cycles_used = [], []
        for t in range(MIN_T, T - K + 1):
            past = torch.from_numpy(seq[:t]).unsqueeze(0).to(DEVICE)
            future = torch.from_numpy(seq[t:t + K]).unsqueeze(0).to(DEVICE)
            k_tensor = torch.tensor([K], dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                h_past = model.context_encoder(past)
                h_future = model.target_encoder(future)
                pred = model.predictor(h_past, k_tensor)
                err = F.l1_loss(pred, h_future).item()
            pred_errors.append(err)
            cycles_used.append(t)
        pe = np.array(pred_errors)
        cc = np.array(cycles_used)
        ri = rul[cc - 1]  # RUL at cut point t (0-indexed)
        rfrac = rul_frac[cc - 1]

        rho, pval = spearmanr(pe, -rfrac)  # -rfrac so rho > 0 means error grows with degradation
        rhos.append(float(rho))
        print(f"  engine {eid} (len {T}): rho(pred_err, degradation) = {rho:+.3f} (p={pval:.2e})")
        results_by_engine[str(eid)] = {
            'length': int(T), 'cycles': cc.tolist(),
            'pred_error': pe.tolist(), 'rul': ri.tolist(), 'rul_frac': rfrac.tolist(),
            'spearman_rho_vs_degradation': float(rho), 'spearman_p': float(pval),
        }

    mean_rho = float(np.mean(rhos))
    print(f"\nMean Spearman rho across {len(picks)} engines: {mean_rho:+.3f}")
    verdict = (
        'POSITIVE - prediction error tracks degradation (zero-label signal)'
        if mean_rho > 0.3
        else 'WEAK - prediction error does not reliably track degradation'
    )
    print(f"Verdict: {verdict}")

    # ============================================================
    # Plot
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=False)
    axes = axes.ravel()
    for i, (eid, r) in enumerate(zip(pick_ids, rhos)):
        if i >= len(axes): break
        ax = axes[i]
        info = results_by_engine[str(eid)]
        cycles = np.array(info['cycles'])
        err = np.array(info['pred_error'])
        rfrac = np.array(info['rul_frac'])
        # Two-axis plot: error on left, RUL (capped-normalized) on right
        ax2 = ax.twinx()
        ax.plot(cycles, err, color='#d62728', lw=1.4, label='pred error (L1)')
        ax2.plot(cycles, rfrac, color='#4477AA', lw=1.4, linestyle='--', alpha=0.6,
                 label='RUL / R_max')
        ax.set_title(f'engine {eid} (len {info["length"]}), ρ={r:+.2f}', fontsize=9)
        ax.set_xlabel('cycle t', fontsize=8)
        ax.set_ylabel('pred error', fontsize=8, color='#d62728')
        ax2.set_ylabel('RUL / R_max', fontsize=8, color='#4477AA')
        ax.tick_params(axis='y', colors='#d62728')
        ax2.tick_params(axis='y', colors='#4477AA')
        ax.grid(False)

    # Summary panel: prediction error vs RUL fraction across all engines
    ax = axes[-1]
    for eid in pick_ids:
        info = results_by_engine[str(eid)]
        ax.scatter(info['rul_frac'], info['pred_error'], s=6, alpha=0.4,
                   label=f'eng {eid}')
    ax.set_xlabel('RUL / R_max  (1 = healthy, 0 = failure)', fontsize=8)
    ax.set_ylabel('prediction error (L1)', fontsize=8)
    ax.set_title(f'pooled across {len(picks)} engines\nmean ρ(err, degradation) = {mean_rho:+.3f}',
                 fontsize=9)
    ax.legend(loc='upper right', frameon=False, fontsize=6)
    ax.invert_xaxis()  # so "more degraded" is rightward

    fig.suptitle('V14 Phase 5c.4: predictor prediction error vs. degradation\n'
                 '(V2 pretrained encoder, horizon k=15; zero-label diagnostic)',
                 fontsize=10, y=1.00)
    fig.tight_layout()
    png = PLOT_PNG / 'prediction_error_vs_degradation.png'
    pdf = PLOT_PDF / 'prediction_error_vs_degradation.pdf'
    fig.savefig(png, dpi=200, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")

    out = {
        'engines': results_by_engine,
        'mean_spearman_rho': mean_rho,
        'engine_ids': [int(e) for e in pick_ids],
        'horizon_k': K,
        'checkpoint': str(CKPT),
        'verdict': verdict,
        'wall_time_s': time.time() - t0,
    }
    with open(V14_DIR / 'prediction_error_analysis.json', 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f"Saved: {V14_DIR / 'prediction_error_analysis.json'}")
    print(f"Total wall time: {(time.time()-t0):.1f} s")


if __name__ == '__main__':
    main()
