"""
Central path configuration for all dataset loaders.

Set INDUSTRIAL_JEPA_DATA environment variable to override the default root.
Default: /home/sagemaker-user/IndustrialJEPA (SageMaker VM).
"""

import os
from pathlib import Path

_ROOT = Path(os.environ.get('INDUSTRIAL_JEPA_DATA',
             '/home/sagemaker-user/IndustrialJEPA'))

CMAPSS_DIR = _ROOT / 'datasets' / 'data' / 'cmapss' / '6. Turbofan Engine Degradation Simulation Data Set'
SMAP_DIR   = _ROOT / 'paper-replications' / 'mts-jepa' / 'data' / 'SMAP'
MSL_DIR    = _ROOT / 'paper-replications' / 'mts-jepa' / 'data' / 'MSL'
PSM_DIR    = _ROOT / 'paper-replications' / 'mts-jepa' / 'data' / 'PSM'
MBA_DIR    = _ROOT / 'paper-replications' / 'mts-jepa' / 'data' / 'tranad_repo' / 'data' / 'MBA'
SMD_DIR    = _ROOT / 'paper-replications' / 'mts-jepa' / 'data' / 'SMD'
SWAT_DIR   = _ROOT / 'datasets' / 'data' / 'swat'
