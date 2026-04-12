# Brainstorming: NeurIPS Extension Ideas

## First Round: 15 Ideas

### Safe Bets (Workshop-Level)

**1. VICReg + Codebook JEPA**
- *Pitch*: C-JEPA showed EMA alone causes collapse; add VICReg regularization to MTS-JEPA's encoder
- *Approach*: Add variance-invariance-covariance loss on encoder outputs alongside codebook losses
- *Novel*: No one has combined VICReg with a codebook JEPA
- *Risk*: Low — both components are validated separately
- *Effort*: 1 day

**2. Lead-Time-Aware Anomaly Prediction (LTAP) Metric**
- *Pitch*: Proper evaluation metric that separates prediction from continuation detection
- *Approach*: Define TRUE_PREDICTION AUC, weight by lead time, report breakdown
- *Novel*: No existing paper quantifies the prediction/detection conflation
- *Risk*: Low — it's a metric, not a method
- *Effort*: 1 day

**3. Codebook Utilization via VQBridge**
- *Pitch*: Replace MTS-JEPA's entropy regularization with VQBridge for guaranteed 100% codebook usage
- *Approach*: Soft→hard quantization annealing from the "Scalable VQ" paper
- *Novel*: First application of VQBridge to JEPA
- *Risk*: Low
- *Effort*: 1 day

### Medium Swings (NeurIPS Poster Territory)

**4. Degradation Regime Codebook (DRC)**
- *Pitch*: Codebook entries naturally discover degradation regimes when trained on run-to-failure data
- *Approach*: Train MTS-JEPA-style codebook on C-MAPSS; analyze which codes activate at different RUL stages; show codes form a degradation taxonomy
- *Novel*: No one has shown codebook = degradation regimes with physical interpretation
- *Risk*: Medium — depends on whether codes are interpretable
- *Effort*: 3 days
- **WHY THIS COULD BE NEURIPS**: If codebook entries map to known degradation physics (healthy → degraded → near-failure), it bridges ML and domain science

**5. Causal Codebook JEPA (CC-JEPA)**
- *Pitch*: Merge Trajectory JEPA's causal encoder with MTS-JEPA's codebook for genuine future prediction
- *Approach*: Causal Transformer encoder → soft codebook → horizon-aware predictor in code space
- *Novel*: Trajectory prediction in discrete code space (not continuous embeddings)
- *Risk*: Medium — new architecture, may need tuning
- *Effort*: 3 days
- **WHY THIS COULD BE NEURIPS**: Addresses both the collapse problem (codebook) and the prediction credibility (causal masking)

**6. Multi-Resolution Codebook with Scale-Specific Codes**
- *Pitch*: Separate codebooks for fine and coarse scales, with cross-scale consistency loss
- *Approach*: K_fine codes for patch-level patterns, K_coarse codes for global trends; enforce that coarse codes aggregate fine codes
- *Novel*: No multi-scale codebook with structural constraints
- *Risk*: Medium
- *Effort*: 2 days

**7. Anomaly Prediction as RUL Regression**
- *Pitch*: Reframe window-level anomaly prediction as time-to-anomaly regression (RUL-style)
- *Approach*: Instead of binary y_{t+1}, predict continuous time-to-next-anomaly; use MTS-JEPA codebook features as input
- *Novel*: Bridges the anomaly prediction and prognostics communities
- *Risk*: Medium — requires labeled time-to-anomaly which may not exist in all datasets
- *Effort*: 2 days

**8. Information Bottleneck Analysis of the Codebook**
- *Pitch*: Formally characterize the codebook as an information bottleneck; show it achieves optimal compression-prediction tradeoff
- *Approach*: Compute I(X;Z) and I(Z;Y) at different codebook sizes K; plot the IB curve; show MTS-JEPA operates near the optimal frontier
- *Novel*: No IB analysis of codebook JEPA
- *Risk*: Medium — IB estimation in high dimensions is noisy
- *Effort*: 2 days

### Moonshots (NeurIPS Oral If It Works)

**9. Self-Discovering Fault Taxonomy via Codebook Composition**
- *Pitch*: The composition of active codes across time forms a "fault trajectory" that uniquely identifies fault types — without any labels
- *Approach*: Track code activation sequences across run-to-failure trajectories on C-MAPSS; cluster sequences; show clusters correspond to fault modes (FD001 vs FD003)
- *Novel*: Self-supervised fault taxonomy discovery
- *Risk*: High — depends on codebook actually learning meaningful patterns
- *Effort*: 4 days

**10. Predictive Codebook Transfer**
- *Pitch*: Pre-train codebook on one industrial process, transfer to a different process; show codes transfer because degradation patterns are universal
- *Approach*: Pre-train on C-MAPSS → transfer to bearing data (FEMTO/XJTU); show codebook entries map to analogous degradation stages
- *Novel*: Cross-domain codebook transfer for prognostics
- *Risk*: High — degradation patterns may be domain-specific
- *Effort*: 5 days

**11. Continuous-Time Codebook JEPA**
- *Pitch*: Replace discrete windows with continuous-time encoding; codebook operates on continuous trajectories
- *Approach*: Neural ODE encoder → codebook quantization → ODE predictor
- *Novel*: First continuous-time codebook JEPA
- *Risk*: Very high — Neural ODEs are finicky
- *Effort*: 5+ days

**12. Codebook as a Language: Anomaly Description Generation**
- *Pitch*: Codebook sequences form a "language" that describes system state; use this for anomaly explanation
- *Approach*: Train code-to-text decoder that maps code sequences to natural language descriptions of system state
- *Novel*: Bridging codebook representations with explainability
- *Risk*: Very high — needs external knowledge or carefully crafted templates
- *Effort*: 5+ days

**13. Adversarial Codebook Robustness**
- *Pitch*: Show codebook JEPA is more robust to adversarial perturbations than continuous JEPA
- *Approach*: Discretization provides natural robustness; measure adversarial accuracy vs continuous baselines
- *Novel*: Robustness analysis of discrete vs continuous SSL for time series
- *Risk*: Medium-high
- *Effort*: 3 days

**14. Optimal Codebook Size Selection via Minimum Description Length**
- *Pitch*: Use MDL principle to automatically select K — neither too few (underfitting) nor too many (overfitting)
- *Approach*: MDL = description length of codes + description length of data given codes; sweep K and find minimum
- *Novel*: Principled codebook size selection for time series
- *Risk*: Low-medium
- *Effort*: 2 days

**15. Frequency-Aware Codebook with Wavelet Views**
- *Pitch*: Replace fine/coarse temporal views with multi-resolution wavelet decomposition
- *Approach*: Wavelet transform → codebook per frequency band → cross-frequency prediction
- *Novel*: Wavelet + codebook JEPA
- *Risk*: Medium
- *Effort*: 3 days

---

## Filtering

### Filter 1: Does it address a gap in the gap map?
- ✅ Ideas 4, 5, 6, 7, 8, 9, 10 — all address open gaps
- ⚠️ Ideas 1, 2, 3 — improve existing methods but don't fill major gaps
- ❌ Ideas 11, 12 — too disconnected from identified gaps

### Filter 2: Can we prototype in <2 days?
- ✅ Ideas 1, 2, 3, 6, 7, 14
- ⚠️ Ideas 4, 5, 8, 13 — 2-3 days
- ❌ Ideas 9, 10, 11, 12, 15 — 4+ days

### Filter 3: Would a NeurIPS reviewer say "that's clever"?
- ✅ Ideas 4, 5, 9 — genuinely novel insights
- ⚠️ Ideas 6, 7, 8, 10 — novel combinations
- ❌ Ideas 1, 2, 3, 14 — solid engineering but not surprising

### Filter 4: Clear falsifiable experiment?
- ✅ All ideas except 11, 12

### Top 3 Selection

**#1: Causal Codebook JEPA (CC-JEPA) — Idea #5**
- Addresses two critical gaps: prediction credibility (causal) + collapse prevention (codebook)
- Natural fusion of our Trajectory JEPA V11 + MTS-JEPA's codebook
- We have all the infrastructure
- Clear experiment: compare CC-JEPA vs MTS-JEPA vs Trajectory JEPA on both anomaly prediction (PSM/MSL) and RUL (C-MAPSS)
- *Rationale*: This IS the NeurIPS paper — it's the fusion the prompt describes

**#2: Degradation Regime Codebook (DRC) — Idea #4**
- If it works, it's the most visually compelling result (codebook visualization showing degradation stages)
- C-MAPSS has known physics — we can validate
- Natural extension of CC-JEPA
- *Rationale*: The interpretability story makes the paper memorable

**#3: Information Bottleneck Analysis — Idea #8**
- Provides theoretical grounding that MTS-JEPA lacks empirically
- IB analysis + codebook size selection via MDL (merges #8 and #14)
- Strengthens the theory contribution
- *Rationale*: Reviewers want theory; this gives it with empirical backing

---

## Second Round: Strengthening Top 3

### Making CC-JEPA (#5) Stronger
- **Theory**: Prove that causal masking + codebook quantization gives tighter prediction bounds than either alone
- **Combination with DRC (#4)**: The same codebook that enables causal prediction also discovers degradation regimes — two contributions from one architecture
- **Adversarial angle**: What would a reviewer attack? "You just concatenated two methods." Preempt by showing the interaction effect: CC-JEPA > Causal JEPA + Codebook JEPA (the combination is more than the sum)

### Making DRC (#4) Stronger
- **Quantitative validation**: Compute adjusted Rand index between codebook clusters and known fault modes
- **Temporal ordering**: Show that code activation sequences follow a consistent ordering across engines — evidence of learned degradation trajectory
- **Surprising combination**: Merge with #9 — full fault taxonomy from unsupervised codebook, validated against known C-MAPSS fault modes

### Making IB Analysis (#8) Stronger
- **Merge with #14 (MDL)**: Automatic K selection + IB analysis gives both a theoretical framework and a practical tool
- **Surprising insight**: Show that the optimal K is much smaller than MTS-JEPA's K=128 — suggest that over-parameterized codebooks waste capacity
- **Adversarial preemption**: Reviewer would say "IB estimation is unreliable." Preempt with multiple estimators (KSG, MINE, variational bounds) and show they agree

---

## Final Selection Rationale

The NeurIPS contribution is **CC-JEPA + DRC + IB**: a causal codebook JEPA that (a) genuinely predicts future anomalies (causal architecture), (b) discovers interpretable degradation regimes (codebook visualization), and (c) achieves provably optimal compression-prediction tradeoff (IB analysis). The three components are synergistic:
- CC-JEPA provides the architecture
- DRC provides the interpretability and physical validation
- IB provides the theoretical grounding

This addresses all major weaknesses in the MTS-JEPA NeurIPS review: lead-time credibility (causal), theory validation (IB), and practical impact (degradation regimes).
