# Signal Tokenization for FactoryNet

Karim,

**Problem**: Different machines have different sensors. Padding to fixed schema doesn't scale.

**Idea**: Embed signal names semantically (like word embeddings). Model learns "torque" signals behave like torques, regardless of which machine.

```
"effort_torque_0" → SentenceTransformer → embedding → shared latent
```

This is validated—see [CHARM](https://arxiv.org/abs/2505.14543) (C3 AI, May 2025), SOTA with this approach.

**What we need**: Metadata per signal (description, unit, category). LLM can generate this from FactoryNet column names.

**Validation**: Zero-shot AURSAD → voraus-AD. Beat current AUC (0.538) without retraining.

You know the data better than I do—any other approaches you see? Maybe something simpler?

Jonas
