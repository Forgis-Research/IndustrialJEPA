#!/usr/bin/env python3
"""
Exp 52 (v2): Mechanical-JEPA on Voraus-AD — standalone run.
Reuses all code from aursad_jepa_v2 but just runs Voraus.
"""

import sys
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA')

from autoresearch.experiments.aursad_jepa_v2 import run_dataset
import json
from pathlib import Path

PROJECT_ROOT = Path('/home/sagemaker-user/IndustrialJEPA')

if __name__ == '__main__':
    voraus_summary = run_dataset('voraus', exp_num=52)
    out_path = PROJECT_ROOT / "datasets" / "data" / "voraus_jepa_results.json"
    with open(out_path, 'w') as f:
        json.dump(voraus_summary, f, indent=2)
    print(f"Voraus results saved: {out_path}")
