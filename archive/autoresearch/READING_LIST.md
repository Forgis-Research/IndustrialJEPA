# Paper Reading List

## General Background (Read First)

| Paper | Venue | Link | Why |
|-------|-------|------|-----|
| **PatchTST** | ICLR 2023 | [arXiv](https://arxiv.org/abs/2211.14730) | Current paradigm — channel-independence wins forecasting |
| **iTransformer** | ICLR 2024 Spotlight | [arXiv](https://arxiv.org/abs/2310.06625) | SOTA channel-mixing — inverted attention on channels |
| **TabPFN** | ICLR 2023 Oral | [arXiv](https://arxiv.org/abs/2207.01848) | The breakthrough — prior-data fitted network, zero-shot tabular |
| **TimesFM** | ICML 2024 | [arXiv](https://arxiv.org/abs/2310.10688) | Foundation model — decoder-only, pretrained on 100B points |
| **Chronos-2** | arXiv Oct 2025 | [arXiv](https://arxiv.org/abs/2510.15821) | Universal forecasting — multivariate + covariates, 120M params |
| **Lagrangian NN** | ICLR 2020 Workshop | [arXiv](https://arxiv.org/abs/2003.04630) | Learning physics — double pendulum, energy conservation |

---

## Direction 1: Learned Sparse Graph

| Priority | Paper | Venue | Link | One-liner |
|----------|-------|-------|------|-----------|
| 🔴 | **NRI** | ICML 2018 | [arXiv](https://arxiv.org/abs/1802.04687) | Foundational — learns interaction graphs from trajectories |
| 🔴 | **MTGNN** | KDD 2020 | [arXiv](https://arxiv.org/abs/2005.11650) | Graph learning + temporal conv for forecasting |
| 🟡 | **GTS** | ICLR 2021 | [arXiv](https://arxiv.org/abs/2101.06861) | Discrete graph structure via Gumbel-softmax |

---

## Direction 2: Learned Latent Concepts

| Priority | Paper | Venue | Link | One-liner |
|----------|-------|-------|------|-----------|
| 🔴 | **Slot Attention** | NeurIPS 2020 | [arXiv](https://arxiv.org/abs/2006.15055) | Discovers object-centric representations unsupervised |
| 🟡 | **Concept Bottleneck** | ICML 2020 | [arXiv](https://arxiv.org/abs/2007.04612) | Interpretable concepts as intermediate layer |
| 🟡 | **TS2Vec** | AAAI 2022 | [arXiv](https://arxiv.org/abs/2106.10466) | Contrastive learning for time series representations |

---

## Direction 3: Mechanical-JEPA

| Priority | Paper | Venue | Link | One-liner |
|----------|-------|-------|------|-----------|
| 🔴 | **I-JEPA** | CVPR 2023 | [arXiv](https://arxiv.org/abs/2301.08243) | Core idea — predict in latent space, not pixel space |
| 🔴 | **Brain-JEPA** | NeurIPS 2024 | [OpenReview](https://openreview.net/forum?id=gtU2eLSAmO) | JEPA for multivariate brain signals — closest to our setting |
| 🟡 | **Dreamer** | ICLR 2020 | [arXiv](https://arxiv.org/abs/1912.01603) | World models — learn dynamics for planning |

---

**Legend:** 🔴 Must read · 🟡 Skim abstract + method
