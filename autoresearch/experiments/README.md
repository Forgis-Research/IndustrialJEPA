# Experiments Folder

Store novel approaches here as separate scripts.

## Naming Convention

```
{number}_{approach_name}.py
```

Examples:
- `01_baseline_transfer.py`
- `02_revin_normalization.py`
- `03_domain_adversarial.py`
- `04_semantic_signals.py`

## Template

```python
#!/usr/bin/env python
"""
Experiment: [Name]

Hypothesis: [What you expect]
Approach: [What you're doing]
Expected Result: [What metric you expect]
"""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Your experiment code
    pass

if __name__ == "__main__":
    main()
```

## Rules

1. Each experiment is self-contained
2. Log results to stdout AND to EXPERIMENT_LOG.md
3. Include clear success/failure criteria
4. Save models/results to `results/experiments/`
