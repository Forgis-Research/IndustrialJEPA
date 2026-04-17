# NeurIPS Review: Self-Supervised Learning for Mechanical Grey Swan Prediction

**Date**: 2026-04-09
**Paper**: `paper-neurips/paper.tex`
**4 independent reviews + quality audit**

---

## Score Summary

| Reviewer | Expertise | Soundness | Significance | Novelty | Clarity | Reproducibility | Overall | Confidence |
|----------|-----------|-----------|-------------|---------|---------|----------------|---------|------------|
| R1 (SSL expert) | SSL, contrastive, JEPA | 5 | 5 | 4 | 7 | 7 | **4/10** | 4/5 |
| R2 (PHM domain) | Bearing RUL, benchmarks | 5 | 5 | 6 | 7 | 7 | **4/10** | 4/5 |
| R3 (Methods/theory) | Representation learning | 4 | 5 | 5 | 7 | 6 | **4/10** | 4/5 |
| R4 (Generalist) | Broad ML | 6 | 4 | 5 | 7 | 7 | **4/10** | 3/5 |
| **Mean** | | **5.0** | **4.75** | **5.0** | **7.0** | **6.75** | **4.0** | **3.75** |

**Consensus: Weak Reject (4/10)**

All four reviewers independently converged on the same score. The paper is well-written and honest, but has critical gaps that prevent acceptance.

---

## Consensus Strengths (cited by 3+ reviewers)

1. **Honest, well-structured ablation with proper statistical testing** (all 4 reviewers). The 11-method comparison, random encoder control, paired t-tests, and 5-10 seed evaluations are above-average rigor.

2. **Genuinely interesting mechanistic insight** (all 4 reviewers). The finding that JEPA captures waveform texture (PC1 correlation 0.071 with spectral centroid) while contrastive captures degradation dynamics (0.856) provides actionable insight, not just benchmark numbers.

3. **Refreshingly honest limitations section** (all 4 reviewers). The paper builds trust by acknowledging small dataset, single channel, JEPA instability, and limited transfer pairs.

4. **Well-written related work** (3/4 reviewers). Thorough, recent (2024-2026), organized by theme, and every paragraph ends with positioning.

---

## Consensus Weaknesses (cited by 3+ reviewers)

### W1: Headline result conflates JEPA with architecture contribution (R1, R3, R4)

The hybrid JEPA+HC Transformer (0.055) is presented as the main result, but most of the gain comes from handcrafted features + Transformer architecture, not JEPA. Transformer+HC alone achieves 0.070; JEPA adds only 0.015 RMSE.

**Missing ablation**: Random encoder + HC Transformer (to isolate JEPA contribution from dimensionality effect).

### W2: Evaluation too narrow for NeurIPS (all 4 reviewers)

- Only 23 bearing episodes across 2 datasets
- Only 2 transfer pairs (FEMTO↔XJTU-SY)
- C-MAPSS validation promised but not delivered
- No non-bearing domain evaluation

### W3: "Grey swan" framing not experimentally validated (R2, R4)

The paper frames bearing failures as grey swans but evaluates on standard RUL benchmarks with standard splits. No experiment tests prediction of genuinely novel/rare failure modes. The title promises more than the experiments deliver.

### W4: "Mechanistic analysis" is descriptive correlation, not mechanism (R3)

The paper claims to explain WHY JEPA and contrastive learn different features. In reality, it observes WHAT they learn (via PC1 correlations) and offers plausible intuition. A true mechanistic explanation would require loss-landscape analysis or gradient-based probing.

### W5: JEPA pretraining instability unresolved (R1, R3, R4)

Best checkpoint at epoch 2 of 100 is an admission the method isn't working as designed. Modern collapse prevention (SIGReg/LeJEPA) exists and isn't used. No analysis of what degrades after epoch 2.

### W6: Closest SSL competitor (DCSSL) not compared quantitatively (R2)

DCSSL (Shen et al., 2026) is the only other SSL method for bearing RUL on FEMTO. It's cited but not compared empirically — a serious gap given it's the single most relevant baseline.

### W7: Non-standard evaluation protocol (R2)

The 75/25 random split on mixed FEMTO+XJTU-SY is not the standard PHM 2012 evaluation protocol. The 0.055 RMSE cannot be directly compared to any published benchmark number.

---

## Critical Missing Experiments (ranked by reviewer urgency)

| Experiment | Reviewers | Impact |
|-----------|-----------|--------|
| Random encoder + HC Transformer ablation | R1, R3, R4 | Isolates JEPA contribution from architecture |
| Fine-tuned JEPA (not frozen) | R1, R3 | Tests whether frozen protocol is the bottleneck |
| C-MAPSS evaluation | R1, R2, R4 | Demonstrates cross-domain generalization |
| Standard FEMTO split evaluation | R2 | Enables comparison with published benchmarks |
| Leave-one-episode-out CV | R2 | Addresses small-N stability concerns |
| DCSSL reproduction/comparison | R2 | Compares against closest competitor |
| CKA/mutual information between JEPA and HC | R3 | Validates complementarity claim rigorously |

---

## Critical Missing References (from reviewers + audit)

| Paper | Why it matters |
|-------|---------------|
| **T-JEPA** (ICLR 2025) | Augmentation-free JEPA for time series — directly relevant baseline |
| **Connecting JEPA with Contrastive Learning** (NeurIPS 2024) | Theoretical connection between the two objectives we compare |
| **LeWorldModel/LeWM** (LeCun group, March 2026) | Addresses JEPA collapse without EMA — relevant to our instability |
| **MAE** (He et al., CVPR 2022) | Foundational masked autoencoder paper — we compare against MAE-style methods |
| **VICReg** (Bardes et al., ICLR 2022) | Our variance regularization descends from this work |
| **CKA** (Kornblith et al., 2019) | Standard tool for representation complementarity analysis |

---

## Quality Audit Findings

### Figures: ZERO (Critical Gap)

The paper has no figures. Existing plots available in `mechanical-jepa/notebooks/plots/`:
- `v8_rul_comparison.pdf` — main results bar chart
- `v8_cross_dataset.pdf` — transfer results
- `v8_encoder_analysis.pdf` — mechanistic analysis (key figure)
- `v8_latent_trajectories.pdf` — latent space evolution
- `v8_pretrain_history.pdf` — training instability evidence

**Must create**: Architecture diagram (Figure 1) — no existing version.

### Citations: 14 entries with `author={various}` (Must Fix)

All need real author names: `tsjepa2024`, `mtsjepa2026`, `zhang2022tcntransformer`, `chen2024cnugrumha`, `li2020fewshot`, `openMAE2025`, `ttsnet2025`, `mdsct2024`, `nomi2024`, `greyswan_weather2024`, `greyswan_factory2026`, `greyswan_supplychain2025`, `physics_augment_rul2024`, `physics_rul_review2024`.

### Numerical Accuracy

All RMSE values, standard deviations, and p-values verified against `v8/RESULTS.md`. One discrepancy:
- **21.4% vs 20.7%**: Paper uses rounded RMSE (0.070→0.055 = 21.4%), ground truth JSON uses exact (0.0697→0.0553 = 20.7%). Internally consistent but a reviewer checking the JSON will find the discrepancy.

---

## Reviewer-Specific Highlights

### Reviewer 1 (SSL Expert) — Score: 4/10
> "The honest reading is: handcrafted features do most of the heavy lifting, and JEPA provides a modest boost on top."

> "Temporal contrastive learning uses RUL-correlated supervision... the comparison between JEPA and contrastive is therefore confounded."

Key unique point: The contrastive encoder has a supervisory advantage (temporal position ≈ RUL), making the JEPA vs contrastive comparison unfair.

### Reviewer 2 (PHM Domain) — Score: 4/10
> "The standard FEMTO/PRONOSTIA evaluation protocol uses the IEEE PHM 2012 challenge split... The paper instead uses a custom 75%/25% random episode split over a MIXED FEMTO + XJTU-SY dataset."

> "The 0.055 RMSE cannot be compared to any published number."

Key unique point: Protocol mismatch makes the headline number unverifiable. The PHM community will immediately notice this.

### Reviewer 3 (Methods/Theory) — Score: 4/10
> "The paper claims to explain 'WHY' JEPA captures waveform texture... But the evidence is post-hoc correlation analysis. This is descriptive, not mechanistic."

> "The hybrid architecture is unprincipled concatenation, not a designed fusion."

Key unique point: The complementarity claim needs CKA analysis or mutual information measurement, not just PC1 correlations.

### Reviewer 4 (Skeptical Generalist) — Score: 4/10
> "The paper oscillates between being an engineering contribution and a methods contribution. For NeurIPS, the methods angle is stronger, but the evaluation is too domain-specific."

> "If I strip away the domain framing, is there an interesting ML contribution here?"

Key unique point: The paper needs to commit to being either a domain-specific engineering contribution or a general ML insights paper.

---

## Path to Acceptance: Prioritized Action Items

### Tier 1: Required for Resubmission (would move from 4→6)

1. **Add the missing ablation**: Random encoder + HC Transformer. This single experiment resolves W1.
2. **Add C-MAPSS experiments**. This resolves W2 (narrow evaluation) and partially resolves W3 (cross-domain generalization).
3. **Add figures**. At minimum: architecture diagram, results bar chart, encoder analysis visualization.
4. **Fix all 14 `author={various}` bib entries** with real author names.
5. **Run standard FEMTO split evaluation** for comparability with published benchmarks.
6. **Remove or complete \plannedc{} content** — reviewers evaluate delivered results only.

### Tier 2: Strongly Recommended (would move from 6→7)

7. **Fine-tune JEPA ablation** (not frozen). Tests whether frozen protocol is the bottleneck.
8. **Compute CKA between JEPA and HC representations** to substantiate complementarity.
9. **Soften "mechanistic analysis" to "empirical representation analysis"** throughout.
10. **Acknowledge contrastive supervision asymmetry** (temporal position ≈ RUL).
11. **Compare with DCSSL** on a common protocol.

### Tier 3: Would Strengthen (would move from 7→8)

12. **Implement SIGReg** or modern collapse prevention to fix training instability.
13. **Design a genuine grey swan experiment** (e.g., held-out fault types).
14. **Add leave-one-episode-out CV** for small-N robustness.
15. **Resolve title: either validate grey swan framing or change title** to match delivered scope.

---

## Meta-Assessment

The reviewers agree this is a **well-executed, honest paper with a genuinely interesting core finding** (JEPA vs contrastive representations) wrapped in framing that overpromises relative to what's delivered. The path to acceptance is clear and achievable:

- The missing experiments (ablations, C-MAPSS, standard FEMTO split) are all straightforward to run
- The figures already exist as PDFs — just need to be included
- The framing adjustments are editorial, not scientific

**Estimated effort to reach borderline accept (score 5-6)**: 2-3 weeks of experiments + 1 week of writing.
**Estimated effort to reach solid accept (score 7+)**: 1-2 months (SIGReg implementation, grey swan experiment design, CKA analysis).
