"""Quick status check for training progress."""
from pathlib import Path
import sys

checkpoint_dir = Path('checkpoints')
if checkpoint_dir.exists():
    checkpoints = list(checkpoint_dir.glob('*.pt'))
    if checkpoints:
        print(f"Training complete! Found {len(checkpoints)} checkpoint(s):")
        for cp in sorted(checkpoints):
            print(f"  - {cp.name}")
        sys.exit(0)
    else:
        print("Training in progress (no checkpoints yet)")
        sys.exit(1)
else:
    print("Training not started (no checkpoint directory)")
    sys.exit(1)
