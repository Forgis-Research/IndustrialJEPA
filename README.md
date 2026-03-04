# IndustrialJEPA

**JEPA-based fault detection and cross-machine transfer for industrial robotics.**

> By predicting Effort from Setpoint in latent space, JEPA learns the physics of machine behavior rather than hardware-specific statistics, enabling fault detection that transfers across different robot types.

## Key Idea

Traditional industrial fault detection learns sensor statistics specific to one machine. When deployed on a different machine, it fails because it learned hardware fingerprints, not physics.

IndustrialJEPA exploits **causal structure** in industrial data:
- **Setpoint**: What the controller commanded (target position, velocity)
- **Effort**: What the machine expended (motor current, torque)
- **Feedback**: What actually happened (measured position)

Under healthy operation, Effort is a predictable function of Setpoint. Faults break this relationship. By learning to predict Effort from Setpoint in latent space (JEPA), the model learns transferable physics.

```
Traditional: Sensors → Fault Label (hardware-specific)
Ours:        Setpoint → Effort prediction (physics-based, transferable)
```

## Installation

```bash
git clone https://github.com/ForgisX/IndustrialJEPA.git
cd IndustrialJEPA
pip install -e .
```

## Quick Start

```python
from industrialjepa.training import Trainer, TrainingConfig
from industrialjepa.model import ModelConfig

# Load FactoryNet dataset
# TODO: Add FactoryNet dataloader

# Train JEPA
config = TrainingConfig(...)
trainer = Trainer(config)
trainer.train()
```

## Project Structure

```
IndustrialJEPA/
├── src/industrialjepa/
│   ├── model/          # JEPA architecture
│   │   ├── backbone/   # Mamba-Transformer hybrid
│   │   └── config.py   # Model configuration
│   ├── data/           # FactoryNet dataloader (TODO)
│   ├── training/       # Training loop, JEPA loss
│   └── evaluation/     # Fault detection benchmarks
├── scripts/
│   ├── train_jepa.py   # Main training script
│   └── evaluate_jepa.py
├── configs/            # YAML configurations
├── paper/              # Paper draft and figures
└── tests/
```

## Datasets

We use [FactoryNet](https://huggingface.co/datasets/Forgis/factorynet-hackathon), a causally-structured dataset with:

| Dataset | Robot | Task | Rows |
|---------|-------|------|------|
| AURSAD | UR3e (6-DOF) | Screwdriving | 6.2M |
| voraus-AD | Yu-Cobot (6-DOF) | Pick-and-place | 2.3M |
| NASA Milling | CNC (3-axis) | Milling | 1.5M |
| RH20T | Franka (7-DOF) | Manipulation | 4.1M |
| REASSEMBLE | Franka (7-DOF) | Assembly | 4.1M |

## Experiments

See [`paper/EXECUTION_PLAN.md`](paper/EXECUTION_PLAN.md) for the full experiment plan:

1. **JEPA vs Baselines**: Prove JEPA > MAE on fault detection
2. **Causal Ablation**: Prove Setpoint matters
3. **Cross-Machine Transfer**: Train on UR3e, test on Yu-Cobot (zero-shot)
4. **Multi-Dataset Pretraining**: Does combining datasets help?
5. **Q&A Evaluation**: Test machine understanding (Tiers 1-2)

## Citation

```bibtex
@article{industrialjepa2026,
  title={JEPA Learns Physics, Not Hardware: Causal Structure Enables Cross-Machine Transfer in Industrial Time Series},
  author={Petersen, Jonas},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
