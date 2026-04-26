"""
Phase 0d: Length-vs-Content Ablation

Disentangles whether the V2 encoder reads sensor degradation or just
encodes sequence length via positional encoding.

Three inference-only tests on the frozen V2 encoder:
  Test 1: Constant input (repeat first cycle's sensors t times)
  Test 2: Length-matched cross-engine swap
  Test 3: Temporal shuffle (strongest test)

Output: experiments/v13/length_vs_content_ablation.json
"""

import sys
import json
import time
import numpy as np
import torch
from pathlib import Path
from scipy.stats import spearmanr

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V13_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v13')
sys.path.insert(0, str(V11_DIR))

from data_utils import load_cmapss_subset, N_SENSORS, RUL_CAP
from models import TrajectoryJEPA, RULProbe

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRETRAIN_CKPT = V11_DIR / 'best_pretrain_L1_v2.pt'

print(f"Phase 0d: Length-vs-Content Ablation")
print(f"Device: {DEVICE}")
t0 = time.time()

# Load model and data
data = load_cmapss_subset('FD001')
train_engines = data['train_engines']
val_engines = data['val_engines']
test_engines = data['test_engines']
test_rul = data['test_rul']

model = TrajectoryJEPA(
    n_sensors=N_SENSORS, patch_length=1, d_model=256, n_heads=4, n_layers=2,
    d_ff=512, dropout=0.1, ema_momentum=0.996, predictor_hidden=256
).to(DEVICE)
model.load_state_dict(torch.load(str(PRETRAIN_CKPT), map_location=DEVICE))
model.eval()

# Train a quick frozen probe for evaluation
from data_utils import CMAPSSFinetuneDataset, CMAPSSTestDataset, collate_finetune, collate_test
from torch.utils.data import DataLoader
import torch.nn.functional as F
import copy

print("\n--- Training frozen probe for evaluation ---")
probe = RULProbe(256).to(DEVICE)
optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
train_ds = CMAPSSFinetuneDataset(train_engines, n_cuts_per_engine=5, seed=42)
val_ds = CMAPSSFinetuneDataset(val_engines, use_last_only=True)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_finetune)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_finetune)

best_val, best_state, patience_count = float('inf'), None, 0
for epoch in range(100):
    probe.train()
    for past, mask, rul in train_loader:
        past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
        optimizer.zero_grad()
        with torch.no_grad():
            h = model.encode_past(past, mask)
        pred = probe(h)
        loss = F.mse_loss(pred, rul)
        loss.backward()
        optimizer.step()
    # Val
    probe.eval()
    pv, tv = [], []
    with torch.no_grad():
        for past, mask, rul in val_loader:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = model.encode_past(past, mask)
            pv.append(probe(h).cpu().numpy())
            tv.append(rul.numpy())
    val_rmse = float(np.sqrt(np.mean((np.concatenate(pv)*RUL_CAP - np.concatenate(tv)*RUL_CAP)**2)))
    if val_rmse < best_val:
        best_val = val_rmse
        best_state = copy.deepcopy(probe.state_dict())
        patience_count = 0
    else:
        patience_count += 1
        if patience_count >= 20:
            break

probe.load_state_dict(best_state)
probe.eval()
print(f"Probe trained: val RMSE = {best_val:.2f}")

# ============================================================
# Test 1: Constant Input
# ============================================================
print("\n=== Test 1: Constant Input ===")
print("Feed sequences where every row is the first cycle repeated t times.")
print("If predictions change with t, PE is doing the work.")

test_lengths = [30, 50, 80, 110, 140, 170, 200]
# Pick 10 test engines to use as "donors" for the first-cycle sensor values
test_engine_ids = sorted(test_engines.keys())[:10]

constant_results = []
for t in test_lengths:
    preds_at_t = []
    for eid in test_engine_ids:
        seq = test_engines[eid]
        first_row = seq[0:1]  # (1, 14)
        constant_seq = np.tile(first_row, (t, 1))  # (t, 14)
        x = torch.tensor(constant_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            h = model.encode_past(x, None)
            pred = probe(h).cpu().item() * RUL_CAP
        preds_at_t.append(pred)
    mean_pred = float(np.mean(preds_at_t))
    std_pred = float(np.std(preds_at_t))
    constant_results.append({
        'length': t,
        'mean_pred': mean_pred,
        'std_pred': std_pred,
        'preds': preds_at_t,
    })
    print(f"  t={t:3d}: mean_pred={mean_pred:.2f} +/- {std_pred:.2f}")

# Check if predictions vary with t
pred_means = [r['mean_pred'] for r in constant_results]
pred_range = max(pred_means) - min(pred_means)
print(f"\n  Prediction range across lengths: {pred_range:.2f} cycles")
print(f"  {'PE dominates' if pred_range > 20 else 'Content dominates (PE weak)' if pred_range < 5 else 'Mixed signal'}")


# ============================================================
# Test 2: Length-Matched Cross-Engine Swap
# ============================================================
print("\n=== Test 2: Length-Matched Cross-Engine Swap ===")
print("Pick engine pairs with similar total length. Same cut point, different sensors.")

# Find pairs of test engines with similar length
all_ids = sorted(test_engines.keys())
engine_lengths = {eid: len(test_engines[eid]) for eid in all_ids}

# Sort by length and pair consecutive engines
sorted_ids = sorted(all_ids, key=lambda x: engine_lengths[x])
pairs = []
for i in range(0, len(sorted_ids)-1, 2):
    e1, e2 = sorted_ids[i], sorted_ids[i+1]
    l1, l2 = engine_lengths[e1], engine_lengths[e2]
    if abs(l1 - l2) <= 10:  # within 10 cycles
        pairs.append((e1, e2, l1, l2))
    if len(pairs) >= 10:
        break

print(f"  Found {len(pairs)} engine pairs within 10 cycles of each other")

swap_results = []
for e1, e2, l1, l2 in pairs:
    # Use the shorter engine's length as cut point
    cut = min(l1, l2)
    seq1 = test_engines[e1][:cut]
    seq2 = test_engines[e2][:cut]

    x1 = torch.tensor(seq1, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    x2 = torch.tensor(seq2, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        h1 = model.encode_past(x1, None)
        h2 = model.encode_past(x2, None)
        pred1 = probe(h1).cpu().item() * RUL_CAP
        pred2 = probe(h2).cpu().item() * RUL_CAP

    # Cosine similarity
    cos_sim = float(F.cosine_similarity(h1, h2, dim=-1).cpu().item())

    swap_results.append({
        'engine1': int(e1), 'engine2': int(e2),
        'length1': l1, 'length2': l2,
        'cut_point': cut,
        'pred1': pred1, 'pred2': pred2,
        'pred_diff': abs(pred1 - pred2),
        'cosine_similarity': cos_sim,
    })
    print(f"  Engines {e1} vs {e2} (len {l1},{l2}, cut={cut}): "
          f"pred_diff={abs(pred1-pred2):.2f}, cos_sim={cos_sim:.4f}")

mean_cos = float(np.mean([r['cosine_similarity'] for r in swap_results]))
mean_diff = float(np.mean([r['pred_diff'] for r in swap_results]))
print(f"\n  Mean cosine similarity: {mean_cos:.4f}")
print(f"  Mean prediction difference: {mean_diff:.2f} cycles")
print(f"  {'Length dominates' if mean_cos > 0.95 else 'Content matters' if mean_cos < 0.8 else 'Mixed'}")


# ============================================================
# Test 3: Temporal Shuffle (Strongest Test)
# ============================================================
print("\n=== Test 3: Temporal Shuffle ===")
print("Randomly permute temporal order of sensor rows. If rho collapses, encoder reads temporal patterns.")

# For each test engine, compute original predictions at multiple cut points,
# then do the same with shuffled sequences
np.random.seed(42)
test_engine_ids = sorted(test_engines.keys())

original_preds_all = []
shuffled_preds_all = []
original_trues_all = []

rho_original_list = []
rho_shuffled_list = []

for idx, eid in enumerate(test_engine_ids):
    seq = test_engines[eid]
    T = seq.shape[0]
    oracle_rul = float(test_rul[idx])

    min_cycle = min(30, T)
    cycles = list(range(min_cycle, T + 1, max(1, (T - min_cycle) // 10)))
    if T not in cycles:
        cycles.append(T)

    orig_preds, shuf_preds, trues = [], [], []
    for c in cycles:
        prefix = seq[:c]
        true_rul = min(oracle_rul + (T - c), RUL_CAP)

        # Original
        x = torch.tensor(prefix, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            h = model.encode_past(x, None)
            pred = probe(h).cpu().item() * RUL_CAP
        orig_preds.append(pred)

        # Shuffled (permute rows)
        perm = np.random.permutation(c)
        prefix_shuf = prefix[perm]
        x_shuf = torch.tensor(prefix_shuf, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            h_shuf = model.encode_past(x_shuf, None)
            pred_shuf = probe(h_shuf).cpu().item() * RUL_CAP
        shuf_preds.append(pred_shuf)

        trues.append(true_rul)

    orig_preds = np.array(orig_preds)
    shuf_preds = np.array(shuf_preds)
    trues = np.array(trues)

    # Compute per-engine Spearman rho
    if len(orig_preds) >= 3 and orig_preds.std() > 0 and trues.std() > 0:
        rho_orig, _ = spearmanr(orig_preds, trues)
        rho_original_list.append(float(rho_orig) if not np.isnan(rho_orig) else 0.0)
    else:
        rho_original_list.append(0.0)

    if len(shuf_preds) >= 3 and shuf_preds.std() > 0 and trues.std() > 0:
        rho_shuf, _ = spearmanr(shuf_preds, trues)
        rho_shuffled_list.append(float(rho_shuf) if not np.isnan(rho_shuf) else 0.0)
    else:
        rho_shuffled_list.append(0.0)

    # Last-window predictions for RMSE
    original_preds_all.append(orig_preds[-1])
    shuffled_preds_all.append(shuf_preds[-1])
    original_trues_all.append(float(test_rul[idx]))

original_preds_all = np.array(original_preds_all)
shuffled_preds_all = np.array(shuffled_preds_all)
original_trues_all = np.array(original_trues_all)

rmse_original = float(np.sqrt(np.mean((original_preds_all - original_trues_all)**2)))
rmse_shuffled = float(np.sqrt(np.mean((shuffled_preds_all - original_trues_all)**2)))
rho_orig_median = float(np.median(rho_original_list))
rho_shuf_median = float(np.median(rho_shuffled_list))
rho_orig_mean = float(np.mean(rho_original_list))
rho_shuf_mean = float(np.mean(rho_shuffled_list))

print(f"\n  Original:  RMSE={rmse_original:.2f}, rho_median={rho_orig_median:.3f}, rho_mean={rho_orig_mean:.3f}")
print(f"  Shuffled:  RMSE={rmse_shuffled:.2f}, rho_median={rho_shuf_median:.3f}, rho_mean={rho_shuf_mean:.3f}")
print(f"  RMSE delta (shuffle - orig): {rmse_shuffled - rmse_original:+.2f}")
print(f"  Rho delta (shuffle - orig): {rho_shuf_median - rho_orig_median:+.3f}")

# Interpretation
rho_drop = rho_orig_median - rho_shuf_median
if rho_drop > 0.3:
    interpretation = "STRONG: Encoder reads temporal degradation patterns. Shuffle destroys tracking. Representation claims HOLD."
elif rho_drop > 0.1:
    interpretation = "MODERATE: Encoder uses temporal structure, but also relies on length/summary stats. Mixed signal."
else:
    interpretation = "WEAK/FAIL: Encoder primarily encodes length or bag-of-sensors, NOT temporal patterns. H.I. R2=0.926 may be a length artifact."

print(f"\n  INTERPRETATION: {interpretation}")

# ============================================================
# Compile results
# ============================================================
wall_time = time.time() - t0

results = {
    "test1_constant_input": {
        "description": "Feed constant sequences (first cycle repeated t times). If predictions change with t, PE dominates.",
        "results": constant_results,
        "prediction_range": pred_range,
        "verdict": "PE dominates" if pred_range > 20 else "Content dominates" if pred_range < 5 else "Mixed",
    },
    "test2_length_matched_swap": {
        "description": "Same-length engine pairs, different sensors. High cos_sim = length dominates.",
        "results": swap_results,
        "mean_cosine_similarity": mean_cos,
        "mean_prediction_difference": mean_diff,
        "n_pairs": len(pairs),
        "verdict": "Length dominates" if mean_cos > 0.95 else "Content matters" if mean_cos < 0.8 else "Mixed",
    },
    "test3_temporal_shuffle": {
        "description": "Permute temporal order of rows. If rho collapses, encoder reads temporal patterns.",
        "original_rmse": rmse_original,
        "shuffled_rmse": rmse_shuffled,
        "rmse_delta": rmse_shuffled - rmse_original,
        "original_rho_median": rho_orig_median,
        "shuffled_rho_median": rho_shuf_median,
        "original_rho_mean": rho_orig_mean,
        "shuffled_rho_mean": rho_shuf_mean,
        "rho_drop": rho_drop,
        "verdict": "Encoder reads temporal patterns" if rho_drop > 0.3 else "Mixed" if rho_drop > 0.1 else "Encoder primarily encodes length",
    },
    "overall_interpretation": interpretation,
    "gate_passed": rho_drop > 0.1,  # True if encoder reads more than just length
    "wall_time_s": wall_time,
}

out_path = V13_DIR / 'length_vs_content_ablation.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
print(f"Wall time: {wall_time:.1f}s")
print(f"\n{'='*60}")
print(f"GATE: {'PASSED' if results['gate_passed'] else 'FAILED'}")
print(f"{'='*60}")
