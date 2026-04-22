# Section 6: Theoretical Rationale for Trajectory JEPA as a Degradation Representation Learner

This section provides a principled rationale for why trajectory prediction in latent space tends to learn representations that are strongly correlated with health state. We present three complementary arguments: a connection to Slow Feature Analysis (SFA), an information-theoretic argument based on signal decomposition, and an explanation for the empirical counterintuitive finding that frozen representations track degradation better than end-to-end fine-tuned representations. We treat these as a theoretical sketch - the arguments are principled and internally consistent, but stop short of formal theorems. The key assumptions are identified and their plausibility on C-MAPSS is discussed.

---

## 6.1 Connection to Slow Feature Analysis

### 6.1.1 Informal Intuition

A trajectory JEPA predictor is asked to forecast $\hat{h}_{t+1:t+k}$ from $h_{1:t}$. Representations that vary rapidly - due to operating condition oscillations, sensor noise, or transient load spikes - are intrinsically harder to predict several steps ahead. Representations that vary slowly - because they track a physical quantity like rotor degradation that changes over hundreds of cycles, not dozens of samples - provide a stable "anchor" that the predictor can extrapolate reliably. The training signal therefore systematically favors representations with low temporal innovation, which is precisely the criterion formalized by Slow Feature Analysis (SFA).

### 6.1.2 SFA Background

Wiskott and Sejnowski (2002) introduced SFA as a method for extracting slowly-varying signals from rapidly-varying input streams. Given a time-varying input $\mathbf{x}(t)$, SFA learns a mapping $g$ that produces output signals $y_j(t) = g_j(\mathbf{x}(t))$ that minimize temporal variation:

$$\Delta(y_j) = \langle \dot{y}_j^2 \rangle_t$$

subject to the constraints $\langle y_j \rangle_t = 0$ (zero mean), $\langle y_j^2 \rangle_t = 1$ (unit variance), and $\langle y_i y_j \rangle_t = 0$ for $i \neq j$ (decorrelation). The discrete-time analog replaces $\dot{y}_j$ with the finite difference $y_j(t+1) - y_j(t)$, so the objective becomes:

$$\Delta(y_j) = \langle (y_j(t+1) - y_j(t))^2 \rangle_t = 2(1 - \langle y_j(t) y_j(t+1) \rangle_t)$$

SFA has been applied to neural representations of temporal data, and its solutions are known to recover latent state variables in generative models where the latent state evolves slowly (Wiskott 2003, Franzius et al. 2007).

### 6.1.3 Formal Connection

**Setup.** Let $h_t \in \mathbb{R}^d$ be the encoder output at time $t$. Let $g_\phi$ be the predictor. The trajectory JEPA L1 loss over a horizon of $k$ steps is:

$$\mathcal{L}_{\text{JEPA}} = \mathbb{E}\left[ \sum_{s=1}^{k} \| \hat{h}_{t+s} - \bar{h}_{t+s} \|_1 \right]$$

where $\hat{h}_{t+s} = g_\phi(h_{1:t}, s)$ is the predictor output and $\bar{h}_{t+s}$ is the stop-gradient EMA target encoding of $x_{t+1:t+s}$.

**Decomposition by feature.** Consider a single scalar feature dimension $j$ and write $h_t^{(j)}$ for its value at time $t$. The $k$-step L1 prediction error for that dimension satisfies:

$$\mathbb{E}[|\hat{h}_{t+s}^{(j)} - h_{t+s}^{(j)}|] \geq |\mathbb{E}[\hat{h}_{t+s}^{(j)}] - \mathbb{E}[h_{t+s}^{(j)}]| \cdot \mathbf{1}_{\text{bias}} + \text{Var}^{1/2}(h_{t+s}^{(j)} - \hat{h}_{t+s}^{(j)})$$

For a fixed predictor architecture, the residual variance of the prediction at horizon $s$ is bounded below by the innovation variance of the feature:

$$\text{Var}(h_{t+s}^{(j)} - \hat{h}_{t+s}^{(j)}) \geq \text{Var}(\xi_{t,s}^{(j)})$$

where $\xi_{t,s}^{(j)} = h_{t+s}^{(j)} - \mathbb{E}[h_{t+s}^{(j)} | h_{1:t}]$ is the conditional innovation - the component of $h_{t+s}^{(j)}$ that is unpredictable given the past. This lower bound is tight for any predictor that captures all predictable structure.

**Minimizing loss requires minimizing innovation.** The JEPA loss is minimized by the encoder that (a) produces target encodings that are maximally predictable from the past, and (b) minimizes the innovation variance $\text{Var}(\xi_{t,s}^{(j)})$. This is because the predictor can perfectly match the predictable component but cannot reduce the innovation. The total loss therefore decomposes as:

$$\mathcal{L}_{\text{JEPA}} \geq C \cdot \mathbb{E}\left[ \sum_{s=1}^{k} \sum_{j=1}^{d} \text{Var}^{1/2}(\xi_{t,s}^{(j)}) \right]$$

where $C$ is a constant depending on the noise distribution. This lower bound is minimized when each feature dimension has minimal innovation - i.e., each dimension varies as slowly as possible while retaining enough variation to avoid representational collapse (prevented in practice by the EMA target encoder and L1 loss).

**SFA equivalence (informal).** The constraint that features vary slowly while retaining discriminative power is exactly the SFA objective. The trajectory JEPA encoder is therefore performing an implicit version of SFA in latent space: it is rewarded for encoding features whose future is predictable from their past, which means encoding slowly-varying components of the signal. Mechanical degradation state - the health index - is precisely the kind of slowly-evolving latent variable that SFA is designed to extract.

**Caveat.** The formal SFA objective operates on raw features with explicit covariance constraints. Our L1 loss in latent space is not equivalent to SFA in a strict sense: the constraints (zero mean, unit variance, decorrelation) are not explicitly enforced, and the L1 norm behaves differently from the squared-error norm used in classical SFA. What we claim is that the inductive bias is the same: both objectives reward slowly-varying representations.

---

## 6.2 Information-Theoretic Argument

### 6.2.1 Assumptions

We formalize three assumptions about the data-generating process for industrial sensor streams under progressing degradation.

**A1 (Signal Decomposition).** Sensor readings at cycle $t$ decompose as:

$$x^{(t)} = f(\text{HI}(t), \varepsilon_t)$$

where $\text{HI}(t) \in [0, 1]$ is a scalar health index evolving deterministically (or near-deterministically) from 1 (healthy) toward 0 (fault), and $\varepsilon_t$ is a zero-mean noise process capturing operating condition variations, measurement noise, and transient disturbances.

**A2 (Fast Noise).** The noise process $\varepsilon_t$ is approximately i.i.d. across cycles: $\varepsilon_t \perp \varepsilon_s$ for $t \neq s$, and $\varepsilon_t \perp \text{HI}(t)$.

**A3 (Smooth Degradation).** The health index evolves smoothly: $\text{HI}(t+k) | \text{HI}(t)$ is highly concentrated around a deterministic function $\phi_k(\text{HI}(t))$ for moderate horizons $k$. More formally, the conditional entropy $H(\text{HI}(t+k) | \text{HI}(t))$ is small relative to $H(\text{HI}(t+k))$.

These assumptions are plausible on C-MAPSS: the dataset is generated by a physical simulation with a deterministic degradation trajectory per engine, operating conditions vary across cycles but are independent of wear state, and degradation evolves over hundreds of cycles.

### 6.2.2 Proposition

**Proposition 1 (Informal).** Under assumptions A1-A3, any representation $h_{\text{past}} = f_\theta(x_{1:t})$ that maximizes predictive accuracy for $h_{\text{future}} = f_\theta(x_{t+1:t+k})$ must concentrate mutual information on the slow component $\text{HI}(t)$.

**Sketch of argument.**

The mutual information between past and future observations decomposes under A1-A3. Write the joint distribution as:

$$p(x_{1:t}, x_{t+1:t+k}) = \int p(x_{1:t} | \text{HI}_{1:t}) \, p(x_{t+1:t+k} | \text{HI}_{t+1:t+k}) \, p(\text{HI}_{1:t+k}) \, d\text{HI}$$

Under A1, the only shared information between past and future observations passes through the latent state trajectory $\text{HI}_{1:t+k}$. By the data processing inequality, for any Markov chain $X_{1:t} \to \text{HI}_{1:t+k} \to X_{t+1:t+k}$:

$$I(X_{1:t}; X_{t+1:t+k}) \leq I(\text{HI}_{1:t}; \text{HI}_{t+1:t+k})$$

Under A3, $\text{HI}(t+k)$ is nearly determined by $\text{HI}(t)$, so:

$$I(\text{HI}_{1:t}; \text{HI}_{t+1:t+k}) \approx H(\text{HI}(t))$$

which is the entropy of the scalar health index at time $t$ - a low-dimensional quantity. In contrast, the noise component $\varepsilon_t$ contributes zero mutual information to the future under A2, because $\varepsilon_{t+s} \perp \varepsilon_t$ for all $s > 0$. Therefore:

$$I(X_{1:t}; X_{t+1:t+k}) = I(\text{HI}(t); X_{t+1:t+k}) + I(\varepsilon_{1:t}; X_{t+1:t+k})$$

$$= I(\text{HI}(t); X_{t+1:t+k}) + \underbrace{I(\varepsilon_{1:t}; X_{t+1:t+k})}_{=0 \text{ under A2}}$$

$$= I(\text{HI}(t); X_{t+1:t+k})$$

This shows that essentially all predictable information in the future is carried by $\text{HI}(t)$, not by $\varepsilon_{1:t}$.

**Applying the data processing inequality to the representation.** For any encoder $f_\theta$, the representation satisfies:

$$I(h_{\text{past}}; h_{\text{future}}) \leq I(X_{1:t}; X_{t+1:t+k}) = I(\text{HI}(t); X_{t+1:t+k})$$

An encoder that maximizes $I(h_{\text{past}}; h_{\text{future}})$ therefore pushes toward an upper bound determined entirely by the slow component HI. The fast noise $\varepsilon$ contributes nothing to this bound; including $\varepsilon$-dominated features in $h_{\text{past}}$ wastes representational capacity without improving the mutual information bound.

**Connection to JEPA L1 loss.** The JEPA loss is not a literal MI estimator. However, Oord et al. (2018) showed that InfoNCE lower-bounds $I(X; Y)$ and that maximizing InfoNCE maximizes a lower bound on mutual information between representations. The L1 prediction loss, while not equivalent to InfoNCE, shares the same inductive bias: reducing $\mathcal{L}_{\text{JEPA}}$ requires the predictor to capture as much of $h_{\text{future}}$ as possible from $h_{\text{past}}$, which under assumptions A1-A3 means capturing $\text{HI}(t)$. The L1 loss penalizes prediction errors on every dimension equally, so it does not explicitly maximize MI, but it creates gradient signal that flows most strongly through features that are both variable and predictable - which are exactly the slow degradation features.

**Caveat.** This argument depends on A1-A3 holding approximately. C-MAPSS is a simulation, so A1 holds by construction. A2 is reasonable but not perfect: operating conditions in FD002/FD004 can exhibit regime shifts that are correlated with engine state. A3 holds well for C-MAPSS's piecewise-linear degradation, but might fail for real-world bearings with sudden-onset faults.

---

## 6.3 Why Frozen Representations Track Degradation Better than End-to-End Fine-Tuning

### 6.3.1 The Empirical Finding

On the C-MAPSS full dataset, we observe:

| Encoder mode | Within-engine Spearman $\rho$ | Test RMSE |
|---|---|---|
| Frozen (no fine-tuning) | **0.856** $\pm$ 0.023 | 14.23 $\pm$ 0.39 |
| End-to-end fine-tuned | 0.804 $\pm$ 0.031 | 13.80 $\pm$ 0.28 |

This is initially counterintuitive: the E2E model sees RUL labels and should learn to track degradation better. Yet the frozen encoder produces representations with higher rank correlation with the health trajectory inside each engine. The E2E encoder achieves a lower RMSE on the test metric but loses tracking fidelity. This section explains why.

### 6.3.2 The Capped RUL Gradient Bias

C-MAPSS uses a piecewise-constant RUL target: the true RUL is capped at $R_{\max} = 125$ cycles during the healthy phase. Formally, the supervised target is:

$$y_t = \min(R_{\max},\ T_{\text{total}} - t)$$

where $T_{\text{total}}$ is the total engine life. For an engine of total life $T$, the fraction of cycles spent at the plateau (where $y_t = R_{\max}$) is approximately:

$$\pi_{\text{plateau}} = \frac{\max(T - R_{\max}, 0)}{T}$$

In C-MAPSS FD001, mean engine life is approximately 206 cycles and $R_{\max} = 125$, giving $\pi_{\text{plateau}} \approx (206 - 125) / 206 \approx 0.39$, so roughly 40% of all training labels are at the plateau value.

**Gradient decomposition.** The MSE loss on capped RUL is:

$$\mathcal{L}_{\text{RUL}} = \mathbb{E}[(f_\theta(x_{1:t}) - y_t)^2]$$

The gradient with respect to the encoder output $h_t$ decomposes into two terms by splitting the expectation over plateau and degradation phases:

$$\nabla_{h_t} \mathcal{L}_{\text{RUL}} = \pi_{\text{plateau}} \cdot \nabla_{h_t} \mathcal{L}_{\text{plateau}} + (1 - \pi_{\text{plateau}}) \cdot \nabla_{h_t} \mathcal{L}_{\text{degrad}}$$

In the plateau phase, the target is the constant $R_{\max} = 125$, so the gradient signal is:

$$\nabla_{h_t} \mathcal{L}_{\text{plateau}} = -2 \cdot (R_{\max} - \hat{y}_t) \cdot \nabla_{h_t} \hat{y}_t$$

This gradient pushes $h_t$ toward a representation that predicts a constant value $R_{\max}$. The encoder learns to map plateau observations to a region of representation space where the readout head outputs $\approx 125$. This is a collapsing signal: the optimal plateau representation can be a fixed point in $\mathbb{R}^d$, discarding all information about within-plateau variation.

**Effect on rank fidelity.** Within-engine Spearman $\rho$ measures whether the encoder's representation is monotonically ordered along the degradation trajectory. During the plateau phase, the engine is healthy and its health index is near 1 across all plateau cycles. An encoder trained with the capped RUL signal has no incentive to order plateau observations - the gradient reward is identical for any ordering that puts plateau observations in the constant-output region. The 40% plateau gradient therefore actively competes with the degradation-phase gradient, which does reward ordering.

Let $\text{ord}(h_{1:T})$ denote the ordinal fidelity of the representation (measured by Spearman $\rho$). The effective gradient signal for ordinal fidelity is:

$$\frac{\partial \text{ord}}{\partial \theta} \propto (1 - \pi_{\text{plateau}}) \cdot G_{\text{degrad}} - \pi_{\text{plateau}} \cdot G_{\text{flatten}}$$

where $G_{\text{degrad}} > 0$ rewards monotone ordering and $G_{\text{flatten}} > 0$ penalizes variation in the plateau region. When $\pi_{\text{plateau}} \approx 0.4$, the flattening signal is substantial.

**Why frozen wins on tracking.** The JEPA pretraining objective has no plateau - it never sees RUL labels. Every cycle, including plateau cycles, contributes a prediction task: predict the future sensor trajectory from the past. In the plateau phase, the engine is still evolving (slowly), and the trajectory predictor must track this slow evolution to reduce prediction error. The frozen encoder therefore has no incentive to collapse the plateau region and retains ordinal structure across the full trajectory.

When we then attach a thin readout head on top of the frozen encoder, the head must map from an already well-ordered representation to RUL values. The head absorbs the calibration mismatch (plateau vs. degradation) without distorting the encoder's representation. The result is slightly worse RMSE (the head has limited capacity to solve the calibration problem) but better within-engine tracking.

**Summary.** E2E fine-tuning optimizes for the composite objective "predict RUL value correctly," which under capped RUL includes a $\pi_{\text{plateau}} \approx 40\%$ signal that flattens the plateau region. This improves RMSE by calibrating the absolute scale but degrades tracking fidelity. The frozen encoder optimizes for temporal predictability across the full trajectory, retaining ordinal structure at the cost of needing an external calibration step.

This result has a practical implication: if the downstream use case is anomaly detection or threshold-crossing prediction (where rank matters more than absolute scale), frozen representations are preferable. If the use case is point estimation of RUL (where absolute scale matters), E2E fine-tuning is preferable despite the fidelity loss.

---

## 6.4 Summary of Theoretical Arguments

**Argument 1 (SFA connection).** Trajectory prediction in latent space rewards features with low conditional innovation - features whose future is predictable given the past. This is equivalent in inductive bias to the SFA objective, which explicitly minimizes temporal variation. Mechanical degradation state is the archetypal slowly-varying latent variable in turbofan and bearing data; the JEPA encoder is therefore implicitly performing SFA, which explains the strong health index correlation ($|ρ| = 0.797$ on PC1, $R^2 = 0.926$ from Ridge probe).

**Argument 2 (Information-theoretic).** Under the decomposition $x^{(t)} = f(\text{HI}(t), \varepsilon_t)$ with i.i.d. fast noise, all predictable information in future observations is carried by the slow component HI. The data processing inequality implies that any predictor maximizing mutual information between past and future representations must concentrate on HI. The L1 JEPA loss shares this inductive bias even though it does not explicitly maximize MI.

**Argument 3 (Frozen vs. E2E).** The capped RUL target introduces a $\approx 40\%$ gradient mass from the plateau phase that pushes the encoder to collapse plateau observations to a constant. This improves absolute RUL calibration (lower RMSE) but degrades ordinal fidelity (lower within-engine Spearman $\rho$). The frozen JEPA encoder, never exposed to this collapsing signal, retains ordinal structure. This explains the empirical finding that frozen $\rho = 0.856$ beats E2E fine-tuned $\rho = 0.804$.

---

## 6.5 Limitations and Honest Assessment

We list the limitations of this theoretical treatment explicitly.

1. **SFA connection is by analogy, not equivalence.** Classical SFA operates on raw signals with explicit covariance normalization constraints. Our loss is L1 in latent space without explicit decorrelation. The inductive bias is similar but the formal equivalence does not hold. A rigorous equivalence would require additional constraints (e.g., whitening of the latent space) that we do not impose.

2. **Information-theoretic argument depends on A1-A3.** These assumptions are plausible for C-MAPSS simulation data but may fail for real-world machines. In particular, A2 (i.i.d. noise) fails when operating conditions are temporally correlated, and A3 (smooth degradation) fails for sudden-onset faults. The argument is therefore more useful as design intuition than as a guarantee.

3. **We have not proven that L1 JEPA maximizes MI.** The connection to InfoNCE/CPC (Oord et al. 2018) is informal. L1 prediction loss and MI maximization are related by inductive bias but are not equivalent objectives. A formal connection would require specifying a distributional family and deriving a lower bound.

4. **The capped RUL gradient argument is a sketch.** We have not measured the actual gradient variance attributable to plateau vs. degradation cycles in E2E training. The argument is directionally correct and consistent with the empirical finding, but alternative explanations (e.g., overfitting, optimizer dynamics, learning rate interactions) are not ruled out.

5. **The empirical observations are on a single dataset family.** C-MAPSS is a simulation dataset with specific properties (smooth degradation, controlled noise). The theoretical arguments may not generalize to datasets with different degradation morphology (e.g., FEMTO-ST bearings, which have exponential rundown and sharp failure onset).

Despite these limitations, we believe the theoretical sketch provides genuine insight: the three arguments are mutually consistent, they make predictions that can be tested (e.g., the SFA connection predicts that the JEPA encoder should outperform autoencoders on slowly-varying health indices - which we verify empirically), and they explain the otherwise puzzling finding that frozen representations outperform supervised representations on tracking.

---

## References (for this section)

- Wiskott, L., and Sejnowski, T. J. (2002). "Slow feature analysis: Unsupervised learning of invariances." Neural computation, 14(4), 715-770.
- Franzius, M., Sprekeler, H., and Wiskott, L. (2007). "Slowness and sparseness lead to place, head-direction, and spatial-view cells." PLoS Computational Biology, 3(8), e166.
- Oord, A. v. d., Li, Y., and Vinyals, O. (2018). "Representation learning with contrastive predictive coding." arXiv:1807.03748.
- Assran, M., Duval, Q., Misra, I., et al. (2023). "Self-supervised learning from images with a joint-embedding predictive architecture." CVPR 2023.
