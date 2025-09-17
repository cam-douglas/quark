"""
Registration Configuration for Brainstem Segmentation

Sets up registration pipeline parameters for aligning embryonic brain datasets.
"""

import json
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def create_registration_config(data_dir: Path) -> Dict[str, Any]:
    """Create registration pipeline configuration.
    
    Args:
        data_dir: Base directory for data storage
        
    Returns:
        Registration configuration dictionary
    """
    config = {
        "template": "DevCCF_E13.5",  # Use E13.5 as middle timepoint
        "registration_method": "ANTs",  # Advanced Normalization Tools
        "parameters": {
            "transforms": ["Rigid", "Affine", "SyN"],
            "metric": "MI",  # Mutual Information
            "convergence": "[1000x500x250x100,1e-6,10]",
            "smoothing_sigmas": "3x2x1x0vox",
            "shrink_factors": "8x4x2x1"
        },
        "target_resolution_um": 25,
        "output_format": "nifti"
    }
    
    config_file = data_dir / "metadata" / "registration_config.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Created registration configuration at {config_file}")
    return config
