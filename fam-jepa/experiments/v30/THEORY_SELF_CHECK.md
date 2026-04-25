# Theory Self-Check and Self-Healing Loop (v30)

**Duration**: ~2h autonomous run
**Agent**: theoretical-mathematician (via general-purpose with full prompt)
**Input**: `paper-neurips/theory_main.tex`, `paper-neurips/theory_appendix.tex`, `paper-neurips/paper.tex`
**Output**: `paper-neurips/theory_findings.tex` (new file, standalone, \input{} later if findings are strong)

---

## Objective

Systematically verify, stress-test, and strengthen the theoretical results in the FAM paper. The loop has three phases, each producing concrete findings that either (a) support empirical claims, (b) expose weaknesses to fix, or (c) inform architecture improvements.

---

## Phase 1: Correctness Audit (30 min)

### 1.1 Proof verification
For each step in the proof of Proposition 1:
- [ ] Verify the DPI step: is the Markov chain W - H_t - Hhat correct? (Yes: Hhat is a deterministic function of H_t for fixed Delta_t.)
- [ ] Verify the tower property under A1: P(E=1|Hhat=h) = E[eta(H*)|Hhat=h]. Write out the full derivation. Does it require E ⊥ X_{≤t} | H*, or E ⊥ Hhat | H*? Are these equivalent?
- [ ] Verify the Jensen gap bound: D_KL(Ber(q)||Ber(p)) is convex in q. Compute d²/dq² explicitly. Is the bound 1/(q(1-q)) correct? Where is the supremum taken, and is the use of A4 (bounded event rate) sufficient?
- [ ] Verify the conditional variance bound: Var(eta(H*)|Hhat) ≤ L² E[||H* - Hhat||² | Hhat]. This uses A3 (Lipschitz) and the identity Var(Y|Z) ≤ E[(Y-c)²|Z] for Z-measurable c. Is eta(Hhat) Z-measurable? (Yes: Hhat is Z-measurable trivially.)
- [ ] Verify the final assembly: does chaining give I(H_t; E) ≥ I(H*; E) - C_p L² epsilon with C_p = (2 p_min (1-p_max))^{-1}?

### 1.2 Assumption necessity
For each assumption (A1-A4):
- [ ] Can it be dropped? What happens to the bound?
- [ ] Is there a weaker version that still gives a useful bound?
- [ ] Does it hold in practice for our datasets? Cite specific evidence.

### 1.3 Known results check
- [ ] Search for existing information-theoretic bounds on JEPA or predictive coding that our result either extends or is a special case of.
- [ ] Check: is our Proposition 1 a novel result, or a known result applied to a new setting?
- [ ] If known, cite the original and clarify our contribution.

---

## Phase 2: Strength and Relevance (45 min)

### 2.1 Does the theory explain ALL empirical results?

For each dataset result, check whether Proposition 1 + Corollary 2 predicts the correct outcome:

| Dataset | h-AUROC | Theory prediction | Matches? | Notes |
|---------|---------|-------------------|----------|-------|
| FD001   | 0.74    | High I(H*;E), low eps → strong | ? | |
| FD002   | 0.57    | Higher eps (mixed modes) → weaker | ? | |
| FD003   | 0.82    | High I(H*;E), low eps → strong | ? | |
| SMAP    | 0.55    | Moderate I(H*;E)? | ? | |
| MSL     | 0.44    | Low I(H*;E)? Or high eps? | ? | |
| PSM     | 0.56    | ? | ? | |
| SMD     | 0.62    | ? | ? | |
| MBA     | 0.75    | High I(H*;E), clear precursors → strong | ? | |
| SKAB    | 0.73    | ? | ? | |
| ETTm1   | 0.87    | Very high I(H*;E) → strong | ? | |
| GECCO   | 0.86    | Very high I(H*;E) → strong | ? | |
| BATADAL | 0.63    | Moderate | ? | |
| CHB-MIT | 0.50    | I(H*;E) ≈ 0 → null | ? | |

### 2.2 Does the theory explain the predictor transfer result?
- [ ] The appendix argues codomain mismatch. Is this formal enough? Can we make it a proposition?
- [ ] Specifically: can we bound the correlation between optimal pretraining weights and optimal finetuning weights?

### 2.3 Does the theory explain label efficiency?
- [ ] At 5% labels: pred-FT 0.261 vs scratch 0.035. The argument is that I(H_t; E) is established during pretraining and independent of label count.
- [ ] Can we formalize this as a sample complexity bound? E.g., "with a pretrained encoder providing I(H_t; E) ≥ alpha bits, the downstream finetuning requires O(1/alpha) labels to achieve error epsilon."
- [ ] This would be a strong new result connecting representation quality to label efficiency.

### 2.4 What does the theory say about architecture design?

Derive concrete architectural recommendations from the bound:

1. **Encoder capacity**: A1 requires sufficient capacity. What is the minimum d for a given I(H*; E)?
2. **Target encoder design**: A1 requires target sufficiency. Does bidirectional attention pool over the full interval guarantee this? Under what conditions?
3. **Prediction horizon range**: The bound holds per-horizon. For which Delta_t values is epsilon smallest? Largest? This informs the horizon sampling strategy during pretraining.
4. **Representation dimension d**: The bottleneck d controls how much noise is discarded. Is there an optimal d* that maximizes I(H_t; E) by trading off compression (smaller d = less noise) against capacity (larger d = more signal)?
5. **Normalization**: RevIN erases drift signal. Does this violate A1 for lifecycle datasets? (Yes: if drift IS the precursor, removing it removes I(H*; E).)

---

## Phase 3: New Results and Architecture Implications (45 min)

### 3.1 Attempt: sample complexity bound
Try to prove: if the pretrained encoder satisfies I(H_t; E) ≥ alpha, then a linear probe on H_t achieves downstream error at most epsilon with O(2^{H(E)} / alpha) labeled samples. This connects Proposition 1 to the label efficiency story.

### 3.2 Attempt: optimal representation dimension
Try to derive the optimal d* that maximizes the bound I(H_t; E) ≥ I(H*; E) - C_p L² epsilon. As d increases: I(H*; E) increases (more capacity) but epsilon may also increase (harder to predict in higher dimensions). Is there a closed-form optimum?

### 3.3 Attempt: horizon-dependent bound
Extend Proposition 1 to be horizon-specific: I(H_t; E_{t+Delta_t}) ≥ I(H*_{Delta_t}; E_{t+Delta_t}) - C_p L_{Delta_t}² epsilon_{Delta_t}. The Lipschitz constant and prediction error are both horizon-dependent. What does this imply for the per-horizon AUROC curve shape?

### 3.4 Architecture prescriptions
Based on all findings, compile a concrete list of architecture design rules that are GUARANTEED (by the theory) to improve performance:

1. **Rule**: ... **Justification**: from Proposition 1, ... **Implementation**: ...
2. ...

---

## Output Format

Write all findings to `paper-neurips/theory_findings.tex` as a LaTeX document with:
- Section for each phase
- For each finding: statement, evidence, implication
- Clearly mark: CONFIRMED / REFUTED / OPEN for each check
- Any new propositions with full proofs
- Architecture recommendations with formal justification

Also update `theory_main.tex` and `theory_appendix.tex` if any corrections are needed.

---

## Self-Healing Rules

1. If a proof step is found to be incorrect, FIX IT immediately and re-verify the downstream consequences.
2. If an assumption is found to be unnecessary, REMOVE IT and tighten the bound.
3. If a stronger result is found, REPLACE the weaker version.
4. If the theory fails to explain an empirical result, DOCUMENT the gap and propose what additional assumption or analysis would close it.
5. Every change to theory_main.tex or theory_appendix.tex must be followed by re-checking that the paper still compiles (run pdflatex).
