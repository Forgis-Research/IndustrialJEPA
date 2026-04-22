"""
Phase 1b: Collapse Diagnostic - EMA vs SIGReg isotropy tracking.

Quick 50-epoch diagnostic to verify:
1. V15-EMA collapses (PC1 grows)
2. V15-SIGReg maintains isotropy (PC1 stays low)

Uses seed=42, 50 epochs. Plots PC1 trajectory for both configs.
Runs in ~5 min total.

Output: phase1b_collapse_diagnostic.json + plot
"""

import sys, json, time, warnings
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V15_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v15')
PLOT_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/analysis/plots/v15')
sys.path.insert(0, str(V11_DIR))

warnings.filterwarnings('ignore')
PLOT_DIR.mkdir(exist_ok=True)

from data_utils import load_cmapss_subset, N_SENSORS, RUL_CAP
from torch.utils.data import DataLoader

# Import V15 architecture
sys.path.insert(0, str(V15_DIR))
from phase1_sigreg import (
    V15JEPA, V15PretrainDataset, collate_v15_pretrain,
    D_MODEL, N_HEADS, N_LAYERS, LAMBDA_SIG, LAMBDA_VAR, EMA_MOMENTUM,
    M_SLICES, BATCH_SIZE, LR, N_CUTS, DEVICE
)

N_EPOCHS_DIAG = 50
SEED = 42


@torch.no_grad()
def compute_pc1(model):
    """Compute PC1 explained variance of context encoder outputs on train data."""
    model.eval()
    data = load_cmapss_subset('FD001')
    all_h = []
    for eid, arr in data['train_engines'].items():
        T = len(arr)
        if T < 20:
            continue
        # Sample 5 timesteps
        for t in [T//4, T//2, 3*T//4, T-1]:
            past = arr[:t+1]
            mu = past.mean(0, keepdims=True)
            std = past.std(0, keepdims=True) + 1e-6
            past_norm = (past - mu) / std
            x = torch.from_numpy(past_norm).float().unsqueeze(0).to(DEVICE)
            h = model.encode_context(x)  # (1, D)
            all_h.append(h.squeeze(0).cpu())
    H = torch.stack(all_h)  # (N, D)
    H_c = H - H.mean(0, keepdim=True)
    _, s, _ = torch.pca_lowrank(H_c, q=min(10, H_c.shape[1]))
    ev = (s**2) / (s**2).sum()
    return float(ev[0].item())


def run_diagnostic(config_name, n_epochs=N_EPOCHS_DIAG):
    """Run N epochs tracking PC1 and probe RMSE every 10 epochs."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    mode = 'ema' if 'ema' in config_name else 'sigreg'
    model = V15JEPA(
        n_sensors=N_SENSORS, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, mode=mode,
        lambda_sig=LAMBDA_SIG, lambda_var=LAMBDA_VAR,
        ema_momentum=EMA_MOMENTUM, sigreg_m=M_SLICES,
    ).to(DEVICE)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n_epochs)

    data = load_cmapss_subset('FD001')

    pc1_history = []
    loss_history = []

    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        ds = V15PretrainDataset(data['train_engines'], n_cuts_per_engine=N_CUTS,
                                 min_past=10, min_horizon=5, max_horizon=30,
                                 seed=epoch + SEED)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_v15_pretrain, num_workers=0)

        model.train()
        total_loss, n = 0.0, 0
        for x_past, past_mask, x_full, full_mask, k in loader:
            x_past = x_past.to(DEVICE); past_mask = past_mask.to(DEVICE)
            x_full = x_full.to(DEVICE); full_mask = full_mask.to(DEVICE)
            k = k.to(DEVICE)
            optim.zero_grad()
            loss, _, _ = model.forward_pretrain(x_past, past_mask, x_full, full_mask, k)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            if mode == 'ema':
                model.update_ema()
            total_loss += loss.item() * len(x_past)
            n += len(x_past)
        avg_loss = total_loss / n
        loss_history.append(avg_loss)
        sched.step()

        if epoch % 5 == 0 or epoch == 1:
            pc1 = compute_pc1(model)
            pc1_history.append((epoch, pc1))
            print(f"  [{config_name}] Ep {epoch:3d} | loss={avg_loss:.4f} | PC1={pc1:.4f}")

    elapsed = time.time() - t0
    print(f"  [{config_name}] done in {elapsed/60:.1f} min")
    return {
        'config': config_name,
        'pc1_history': pc1_history,
        'loss_history': loss_history,
        'final_pc1': pc1_history[-1][1] if pc1_history else None,
        'final_loss': loss_history[-1] if loss_history else None,
        'runtime_min': elapsed / 60,
    }


def main():
    print("=" * 60)
    print("Phase 1b: Collapse Diagnostic (EMA vs SIGReg)")
    print(f"N_epochs: {N_EPOCHS_DIAG}, Seed: {SEED}")
    print("=" * 60)

    results = {}
    for config in ['v15_ema', 'v15_sigreg']:
        print(f"\n--- {config} ---")
        results[config] = run_diagnostic(config, N_EPOCHS_DIAG)

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for config, color in [('v15_ema', 'red'), ('v15_sigreg', 'blue')]:
        r = results[config]
        epochs_pc1 = [ep for ep, _ in r['pc1_history']]
        pc1_vals = [pc1 for _, pc1 in r['pc1_history']]
        axes[0].plot(epochs_pc1, pc1_vals, '-o', color=color, label=config, ms=4)

        axes[1].plot(range(1, len(r['loss_history'])+1), r['loss_history'],
                     color=color, label=config)

    axes[0].axhline(0.30, color='gray', linestyle='--', label='isotropy target', linewidth=1)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('PC1 Explained Variance')
    axes[0].set_title('Encoder Isotropy: PC1 (lower = more isotropic)')
    axes[0].legend(); axes[0].set_ylim(0, 1)

    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Training Loss')
    axes[1].set_title('Training Loss')
    axes[1].legend()

    plt.tight_layout()
    plot_path = PLOT_DIR / 'phase1b_collapse_diagnostic.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nPlot saved: {plot_path}")

    # Summary
    print("\n" + "=" * 55)
    print("COLLAPSE DIAGNOSTIC RESULTS")
    print("=" * 55)
    for config, r in results.items():
        pc1_final = r.get('final_pc1', 'N/A')
        loss_final = r.get('final_loss', 'N/A')
        collapsed = pc1_final > 0.5 if isinstance(pc1_final, float) else None
        print(f"\n{config}:")
        print(f"  Final PC1 = {pc1_final:.4f} ({'COLLAPSED' if collapsed else 'ISOTROPIC'})")
        print(f"  Final loss = {loss_final:.4f}")
        print(f"  PC1 trajectory: {[f'{v:.3f}' for _, v in r['pc1_history'][:5]]}")

    # Save results
    out = {}
    for config, r in results.items():
        out[config] = {k: v for k, v in r.items() if k != 'loss_history'}
    with open(V15_DIR / 'phase1b_collapse_diagnostic.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {V15_DIR / 'phase1b_collapse_diagnostic.json'}")


if __name__ == '__main__':
    main()
