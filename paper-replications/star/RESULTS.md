# STAR Replication Results

Paper: Fan et al., "A Two-Stage Attention-Based Hierarchical Transformer for Turbofan Engine
Remaining Useful Life Prediction", Sensors 2024.

Replication date: 2026-04-11
Hardware: NVIDIA A10G (22 GB), CUDA 12.x, PyTorch 2.6.0

---

## Main Results Table

| Subset | Paper RMSE | Ours RMSE (mean +/- std) | Paper Score | Ours Score (mean +/- std) | RMSE Gap | Assessment |
|--------|-----------|--------------------------|-------------|---------------------------|----------|------------|
| FD001  | 10.61     | TBD                      | 169         | TBD                       | TBD      | TBD        |
| FD002  | 13.47     | TBD                      | 784         | TBD                       | TBD      | TBD        |
| FD003  | 10.71     | TBD                      | 202         | TBD                       | TBD      | TBD        |
| FD004  | 15.87     | TBD                      | 1449        | TBD                       | TBD      | TBD        |

Assessment labels: EXACT (<=10% gap), GOOD (<=20% gap), MARGINAL (<=30% gap), FAILED (>30% gap)

---

## Ablations

(To be filled after main results are complete)

---

## Honest Assessment

(To be filled after all experiments complete)

---

## Deviations from Paper

1. Per-scale prediction head uses mean pooling over K and D dims before MLP (paper does not
   specify exact flattening strategy; full flatten would create billions of parameters for
   FD004 with d_model=256).
2. PatchMerging with odd K: we truncate the last patch (paper not explicit about this).
3. Sinusoidal PE for first decoder input: Vaswani-style, not learned.
4. Train/val split: 15% of training engines held out for validation (paper may use a different
   fraction; not specified).
5. Model checkpoints saved per seed (not averaged across seeds before evaluation).
