"""
Wait for first training to complete, then launch multi-seed experiments.
"""
import time
from pathlib import Path
import subprocess
import sys

print("Waiting for initial training to complete...")
print("Checking for checkpoints every 60 seconds...")

checkpoint_dir = Path('checkpoints')
max_wait = 60 * 60  # 1 hour max
elapsed = 0

while elapsed < max_wait:
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob('*.pt'))
        if checkpoints:
            print(f"\n✓ Training complete! Found checkpoint: {checkpoints[0].name}")

            # Now launch multi-seed experiments
            print("\nLaunching multi-seed validation...")
            seeds = [123, 456]  # Already have 42

            for seed in seeds:
                print(f"\nRunning seed {seed}...")
                cmd = [
                    sys.executable, 'train.py',
                    '--epochs', '30',
                    '--seed', str(seed),
                    '--no-wandb'
                ]
                result = subprocess.run(cmd)
                if result.returncode != 0:
                    print(f"Warning: Training with seed {seed} failed")

            print("\n✓ Multi-seed validation complete!")
            break

    time.sleep(60)
    elapsed += 60
    print(f"  Still waiting... ({elapsed//60} min elapsed)")

if elapsed >= max_wait:
    print("\n✗ Timeout waiting for training to complete")
    sys.exit(1)
