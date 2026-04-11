# Signal Tokenization: Research & Proof of Concept

## The Idea (Validated by Recent Research)

Treat sensor signals like language tokens: embed signal names semantically, enabling one model to handle any machine with any sensors.

**This is not speculative.** C3 AI's [CHARM](https://arxiv.org/abs/2505.14543) (May 2025) does exactly this and achieves SOTA on forecasting, classification, and anomaly detection.

## Key Papers

| Paper | Core Idea | Result |
|-------|-----------|--------|
| [CHARM](https://arxiv.org/abs/2505.14543) | Channel descriptions → embeddings via text encoder | SOTA on 7M params |
| [UniTS](https://zitniklab.hms.harvard.edu/projects/UniTS/) | Unified model for 38 datasets, zero-shot transfer | 10.5% better than baselines |
| [SensorLM](https://arxiv.org/abs/2506.09108) | Google's sensor-language model, 60M hours | Zero-shot activity recognition |
| [TF-C](https://zitniklab.hms.harvard.edu/projects/TF-C/) | Time-frequency consistency for cross-domain | Strong transfer across sensor types |

## How CHARM Works

```
Channel: "effort_torque_0"
Description: "Joint 0 torque in Nm, measures motor effort"
           ↓
Text Encoder (frozen, e.g., SentenceTransformer)
           ↓
Channel Embedding (384-dim)
           ↓
Modulates TCN kernels + attention
           ↓
Shared latent representation
```

Key insight: **channel descriptions guide the convolutions**, not just concatenated as features.

## Minimal Metadata Required

For FactoryNet signals, we need:

```json
{
  "effort_torque_0": {
    "description": "Joint 0 motor torque",
    "unit": "Nm",
    "category": "effort",
    "physical_type": "torque"
  },
  "setpoint_pos_0": {
    "description": "Joint 0 commanded position",
    "unit": "rad",
    "category": "setpoint",
    "physical_type": "angle"
  }
}
```

**Can LLM generate this?** Yes. Signal names in FactoryNet are already descriptive:
- `effort_torque_0` → "Joint 0 torque effort signal"
- `setpoint_vel_cartesian_2` → "Cartesian Z velocity setpoint"
- `ctx_state_tightening` → "Boolean indicating tightening phase"

A simple prompt to GPT-4 with the column names would generate 90%+ correct metadata.

## Proof of Concept Plan

### Phase 1: Validate on FactoryNet (1 week)

1. **Generate metadata** for all FactoryNet signals using LLM
2. **Embed signal names** using SentenceTransformer (`all-MiniLM-L6-v2`)
3. **Simple baseline**: Train on AURSAD, test on voraus-AD
   - Compare: fixed padding vs signal tokenization
   - Metric: zero-shot anomaly detection AUC

### Phase 2: Minimal CHARM Implementation (1 week)

```python
class SignalTokenizer:
    def __init__(self, signal_metadata):
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = {
            name: self.text_encoder.encode(meta['description'])
            for name, meta in signal_metadata.items()
        }

    def tokenize(self, signals: dict) -> Tensor:
        """Convert {signal_name: value} to sequence of tokens."""
        tokens = []
        for name, value in signals.items():
            signal_embed = self.embeddings[name]  # (384,)
            value_embed = self.value_encoder(value)  # (64,)
            tokens.append(concat([signal_embed, value_embed]))
        return stack(tokens)  # (num_signals, 448)
```

### Phase 3: Cross-Machine Experiment (1 week)

| Train On | Test On | Expected Result |
|----------|---------|-----------------|
| AURSAD | voraus-AD | Zero-shot works if tokenization correct |
| AURSAD + voraus | RH20T | Multi-source improves transfer |
| All 4 robots | NASA Milling | Hardest: robot → CNC |

Success = zero-shot AUC > 0.6 (better than our current 0.538 with padding)

## Why This Matters for FactoryNet

Current state: 5 datasets, hardcoded schema, padding hack.

With signal tokenization:
- **Any new machine** with named signals works immediately
- **Model learns physics** ("torque" signals behave like torques)
- **Composable** — add tool signals, context signals, whatever
- **Foundation model potential** — train once, deploy everywhere

## Message to Karim (Updated)

See `message_to_karim.md` — now with research backing.

## References

- [CHARM: Time to Embed](https://arxiv.org/abs/2505.14543) - Channel descriptions for time series
- [UniTS](https://zitniklab.hms.harvard.edu/projects/UniTS/) - Unified multi-task time series
- [SensorLM](https://arxiv.org/abs/2506.09108) - Google's sensor-language foundation model
- [TabTransformer](https://arxiv.org/abs/2012.06678) - Column embeddings for tabular data
- [TF-C](https://zitniklab.hms.harvard.edu/projects/TF-C/) - Cross-domain time series transfer
