# A2P Replication Experiment Log

**Start:** 2026-04-10
**Target:** Match Table 1 from Park et al. ICML 2025 (A2P)
**Key metric:** F1 with tolerance t=50 (no point adjustment)
**Paper targets:** MBA L100=67.55, SMD L100=52.07, avg L100=46.84

---

## Setup Notes

- GPU: NVIDIA A10G (23GB)
- Python: 3.12, PyTorch 2.6.0
- Official code: AP/ (cloned from KU-VGI/AP)
- MBA data: TranAD xlsx converted to npy, 7680x2, 3.12% anomaly rate
- SMD data: HuggingFace thuml/Time-Series-Library, 708K x 38, 4.16% anomaly rate
- Special deps installed: finch-clust, arch, openpyxl
- Quarto installed: /tmp/quarto-1.5.57/bin/quarto

---

## Exp 0: Pipeline smoke test (1 epoch, MBA, seed=42)

**Time:** 2026-04-10 ~23:00
**Hypothesis:** Pipeline runs end-to-end with 1 epoch
**Change:** Default run.sh params but joint_epochs=1, cross_attn_epochs=1
**Result:** F1=0.0 (expected - 1 epoch insufficient, threshold at 96.6% = almost all flagged)
**Sanity checks:** ✓ Loss decreasing (FE: 0.617 -> 0.065 over 10 epochs; pretrain loss: 1.72; main: decreasing) ✓ No NaN ✓ Predictions generated
**Verdict:** PIPELINE WORKS
**Insight:** With 1 joint epoch, the AD model fires on almost everything (96.6% flagged vs 3.12% actual) - model needs more training to sharpen discriminability
**Next:** Run full 5 epoch version with correct anormly_ratio

---

## Exp 1: MBA Full Training, L_out=100, seed=0

**Time:** 2026-04-10 ~23:10
**Hypothesis:** Full training (5 pretraining epochs + 5 main) should produce F1 close to paper's 67.55
**Change:** joint_epochs=5, cross_attn_epochs=5, anormly_ratio=1.0 (paper default)
**Seeds run:** [0]
**Sanity checks:** Pending - waiting for result
**Result:** TBD
**Verdict:** TBD

---
