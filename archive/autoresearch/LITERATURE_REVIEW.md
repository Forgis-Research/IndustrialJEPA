# Literature Review: Three Paths to Breakthrough

**Date**: 2026-03-23
**Papers reviewed**: 35+
**Scope**: Sparse graph learning, latent concepts, and JEPA/world models for industrial time series

---

## Direction 1: Learned Sparse Graph for Channel Dependence

### Key Papers

| Paper | Venue | Citations | Sparse? | Transfer? | Validated vs Physics? |
|-------|-------|-----------|---------|-----------|----------------------|
| NRI (Kipf) | ICML 2018 | ~900 | Discrete types | No | Toy physics only |
| Graph WaveNet | IJCAI 2019 | ~1500 | No (softmax) | No | No |
| MTGNN | KDD 2020 | ~800 | Top-k | No | No |
| AGCRN | NeurIPS 2020 | ~700 | No | No | No |
| StemGNN | NeurIPS 2020 | ~600 | No | No | No |
| GTS (Shang) | ICLR 2021 | ~400 | Yes (binary) | No | No |
| Cini et al. | JMLR 2023 | ~100 | Yes (budget) | No | No |
| iTransformer | ICLR 2024 | ~500 | No (dense attn) | Partial | No |
| FourierGNN | NeurIPS 2023 | ~100 | No | No | No |

### Taxonomy of Graph Learning Methods

1. **Embedding similarity** (Graph WaveNet, MTGNN, AGCRN): A = f(E1 * E2^T). Dense.
2. **Gumbel-softmax** (GTS, NRI): Binary edges via differentiable sampling. Sparse.
3. **Score-based probabilistic** (Cini et al.): MC gradient estimation with sparsity budget. Most principled.
4. **Attention-based** (iTransformer): Implicit soft graph. Dense.
5. **Spectral** (StemGNN, FourierGNN): Fourier domain. Not interpretable.

### Critical Gaps

1. **No physics validation**: No paper validates that learned sensor graphs match known physical coupling. NRI does it for toy springs; nobody for real engineering systems.
2. **No transfer**: All methods learn dataset-specific graphs tied to specific N sensors. No graph transfer across machines.
3. **Channel-independence paradox**: PatchTST/DLinear (channel-independent) often beat graph methods, suggesting learned graphs capture noise, not physics.
4. **Static graphs**: Most methods learn one fixed graph. Real systems have time-varying dependencies.

### Our Opportunity

A method that learns sparse, discrete graphs; validates them against known physics (C-MAPSS component structure); demonstrates graph transfer across operating conditions; and resolves the channel-independence paradox by showing sparse dependence beats both extremes.

**Connection to Role-Trans**: Our Role-Transformer is essentially a fixed physics-informed graph. Learned sparse graphs could be more flexible while maintaining the transfer advantage.

---

## Direction 2: Learned Latent Concepts

### Key Papers

| Paper | Venue | Citations | Unsupervised? | Time Series? | Industrial? |
|-------|-------|-----------|---------------|-------------|------------|
| CBM (Koh) | ICML 2020 | ~1200 | No (needs labels) | No | No |
| CBM for Prognostics (EPFL) | Info Fusion 2025 | New | No (expert concepts) | Yes | Yes (C-MAPSS!) |
| UCBMs (Schrodi) | arXiv 2024 | New | Yes (via CLIP) | No | No |
| Slot Attention (Locatello) | NeurIPS 2020 | ~1500 | Yes | No (images) | No |
| SlotFormer | ICLR 2023 | ~200 | Yes | Video | No |
| SlotPi | KDD 2025 | New | Yes | Video (physics) | No |
| SlotFM | arXiv 2025 | New | Yes | Yes (accel only) | Partial |
| CoST | ICLR 2022 | ~300 | Yes | Yes | No |
| TS2Vec | AAAI 2022 | ~700 | Yes | Yes | No |
| beta-VAE | ICLR 2017 | ~5000 | Yes | No | No |
| Locatello impossibility | ICML 2019 BP | ~3000 | N/A (theory) | N/A | N/A |

### Critical Insights

1. **CBMs for C-MAPSS exist** (EPFL 2025): Binary degradation concepts per component → RUL prediction. But concepts are expert-defined.
2. **Slot attention for sensors exists** (SlotFM 2025): First slot attention on accelerometer data, discovers frequency-band components. But not physical components.
3. **Locatello's impossibility result**: Unsupervised disentanglement requires inductive bias. For us, sensor groupings provide this naturally.
4. **SlotPi** (KDD 2025): Slot attention + Hamiltonian mechanics for physics-informed concepts. But vision-only.

### Our Opportunity

Nobody has combined slot attention + physics-informed structure + industrial sensor data. A method that:
- Uses slot attention to decompose multivariate sensors into latent concepts
- Uses physical structure (sensor groupings) as inductive bias
- Learns via self-supervised prediction
- Produces interpretable, transferable concepts

This is genuinely novel. The closest work (SlotFM) does frequency decomposition, not physical component discovery.

---

## Direction 3: Mechanical-JEPA

### Key Papers

| Paper | Venue | Citations | Transfer? | Multi-scale? | Collapse Fix? |
|-------|-------|-----------|-----------|-------------|--------------|
| I-JEPA | CVPR 2023 | ~1000 | No | Block masking | EMA only |
| V-JEPA | TMLR 2024 | ~200 | Frozen eval | Spatiotemporal | EMA + 90% mask |
| TS-JEPA | NeurIPS WS 2024 | New | No | No | EMA |
| MTS-JEPA | arXiv 2026 | New | Limited | Yes (dual-res) | Codebook! |
| Brain-JEPA | NeurIPS 2024 SP | New | Cross-ethnic | Domain masking | EMA |
| C-JEPA | NeurIPS 2024 | New | No | No | VICReg + EMA |
| DreamerV3 | ICLR 2023 | ~1000 | No | No | Discrete latent |
| TF-C | NeurIPS 2022 | ~300 | Yes (15%+) | Time+Freq | Contrastive |

### Why Our JEPA Failed (Root Cause Analysis)

| Issue | Our Implementation | What Literature Says |
|-------|-------------------|---------------------|
| Mask ratio | 40% of 5 components (2 masked) | V-JEPA: 90%, TS-JEPA: 70% |
| Collapse prevention | EMA only | Need codebook (MTS-JEPA) or VICReg (C-JEPA) |
| Encoder depth | 2 layers | Apple NeurIPS 2024: JEPA needs deep encoders |
| Loss function | MSE | TS-JEPA: L1 more robust |
| Multi-scale | No | MTS-JEPA: critical for time series |
| Masking strategy | Component-level | Brain-JEPA: temporal + spatial + double-cross |
| Condition invariance | None | Need explicit invariance mechanism |

### What Mechanical-JEPA Needs

1. **Codebook bottleneck** (from MTS-JEPA): Prevent collapse, force discrete states
2. **Multi-resolution** (from V-JEPA, MTS-JEPA): Fine dynamics + coarse trends
3. **High temporal masking** (70-90%): Force dynamics modeling, not interpolation
4. **Deep encoder** (4-6 layers): Activate JEPA's implicit bias toward influential features
5. **Domain-adapted masking** (from Brain-JEPA): Cross-time + cross-sensor + double-cross
6. **Condition-invariant objective**: VICReg or adversarial domain confusion
7. **L1 loss**: More robust than MSE

### Our Opportunity

No paper applies JEPA to cross-condition/cross-machine industrial transfer. A "Mechanical-JEPA" paper that:
1. Documents failure modes of standard JEPA (we have the evidence)
2. Diagnoses root causes (we now understand them)
3. Proposes fixes (codebook + multi-scale + domain masking)
4. Demonstrates successful transfer

This "Why JEPA Fails and How to Fix It" narrative is compelling for NeurIPS.

---

## Gap Analysis: Cross-Direction

| Capability | Direction 1 (Graph) | Direction 2 (Concepts) | Direction 3 (JEPA) |
|-----------|--------------------|-----------------------|-------------------|
| Novelty | Medium (graph learning exists, but no physics validation or transfer) | High (slot attention for industrial sensors is new) | High (JEPA for industrial transfer is new, plus failure analysis) |
| Risk | Low (incremental over existing methods) | Medium (slot attention on sensors is unproven) | High (JEPA has failed 3 times; fixing it is uncertain) |
| Paper narrative | "Sparse physics-validated graphs transfer" | "Learned concepts replace expert knowledge" | "Why JEPA fails and how to fix it" |
| Connection to prior work | Extension of Role-Trans: learned vs fixed grouping | New direction | Fix of our main thesis |
| Potential impact | Medium (niche: graph learning for TS) | High (interpretable + transferable concepts) | High (JEPA is hot topic, failure analysis valuable) |

## Recommendation

**Direction 2 (Learned Latent Concepts) is the most promising breakthrough**, combining:
- Genuine novelty (no prior work on slot attention for industrial sensors)
- Strong connection to our existing results (Role-Trans ≈ fixed concepts; learned concepts generalize this)
- Interpretability (concepts are inspectable and interventionable)
- Transfer story (concepts should transfer better than raw features)
- Ties to existing hot areas (concept bottleneck models, object-centric learning)

**Direction 3 (Mechanical-JEPA) is the strongest paper narrative**, because:
- We already have extensive negative results
- Root cause analysis is now clear from literature
- "Why X fails and how to fix it" is a compelling NeurIPS format
- But higher risk: JEPA might still fail even with fixes

**Direction 1 (Sparse Graph) is the safest but least impactful**, being an incremental extension of existing graph learning methods.

### Hybrid Approach

The strongest paper might combine Directions 2 + 3: **Slot-JEPA** — use slot attention to discover physical components from sensor data, then apply JEPA in the slot space with a codebook bottleneck. This gives:
- Unsupervised concept discovery (Direction 2)
- Latent prediction for dynamics (Direction 3)
- Transfer via shared concept structure
- The codebook prevents collapse (fixing our JEPA failures)

---

## Sources

35+ papers cited across all three directions. Full references in research agent outputs.
