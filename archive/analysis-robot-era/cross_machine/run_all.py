#!/usr/bin/env python
"""
Run all cross-machine transfer analysis scripts.

Usage:
    python analysis/cross_machine/run_all.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

SCRIPTS = [
    "01_distribution_analysis.py",
    "02_correlation_fingerprints.py",
    "03_entropy_profiles.py",
    "04_linear_probe.py",
]


def main():
    print("="*60)
    print("CROSS-MACHINE TRANSFER ANALYSIS")
    print("="*60)

    for script in SCRIPTS:
        script_path = SCRIPT_DIR / script
        if not script_path.exists():
            print(f"WARNING: {script} not found, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Running: {script}")
        print("="*60 + "\n")

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(SCRIPT_DIR.parent.parent),  # Run from repo root
        )

        if result.returncode != 0:
            print(f"ERROR: {script} failed with return code {result.returncode}")
            # Continue anyway

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nOutputs saved to: {SCRIPT_DIR / 'outputs'}")


if __name__ == "__main__":
    main()
