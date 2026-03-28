# Unify

**Cross-Morphological Latent Alignment for Universal Physics Forecasting**

> *Multiple related datasets are stronger than the sum of their parts?!*

A foundation model that maps heterogeneous mechanical systems into a shared latent space, enabling zero-shot transfer across system morphologies.

## Project Structure

```
unify/
├── RESEARCH_IDEA.md    # Detailed research proposal
├── README.md           # This file
├── data/               # Data generation and loading
├── models/             # Architecture implementations
├── experiments/        # Training scripts and configs
└── notebooks/          # Exploration and visualization
```

## Quick Links

- [Full Research Proposal](./RESEARCH_IDEA.md)

## Core Concept

```
Heterogeneous Inputs  →  Set Encoder  →  Shared Latent  →  Dynamics Model  →  Decoder
   (N sensors)            (any N)         (fixed D)         (universal)      (N outputs)
```

## Getting Started

```bash
# Generate synthetic data
python data/generate_pendulums.py

# Train the model
python experiments/train.py

# Evaluate zero-shot transfer
python experiments/evaluate.py --test-n 5
```
