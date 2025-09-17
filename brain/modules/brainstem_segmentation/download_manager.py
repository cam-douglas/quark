"""
Download Manager for Brainstem Segmentation Datasets

Handles generation of download scripts and instructions for dataset acquisition.
"""

from pathlib import Path
from typing import List
import logging

from .dataset_catalog import DatasetEntry

logger = logging.getLogger(__name__)


def generate_download_scripts(data_dir: Path, datasets: List[DatasetEntry]) -> None:
    """Generate download scripts for each data source.
    
    Args:
        data_dir: Base directory for data storage
        datasets: List of dataset entries to process
    """
    scripts_dir = data_dir / "metadata" / "download_scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    
    _create_allen_script(scripts_dir)
    _create_download_readme(scripts_dir)
    
    logger.info(f"Generated download scripts in {scripts_dir}")


def _create_allen_script(scripts_dir: Path) -> None:
    """Create Allen Brain Atlas download script.
    
    Args:
        scripts_dir: Directory to save scripts
    """
    allen_script = scripts_dir / "download_allen.py"
    with open(allen_script, 'w') as f:
        f.write('''#!/usr/bin/env python3
"""Download Allen Brain Atlas developmental mouse data."""

import requests
import json
from pathlib import Path


def download_allen_developmental():
    """Download developmental mouse brain data from Allen Brain Atlas."""
    base_url = "https://developingmouse.brain-map.org/api/v2"
    
    # Get available experiments
    experiments_url = f"{base_url}/data/query.json?criteria=model::SectionDataSet"
    
    # Add download logic here
    print("Allen Brain Atlas download script placeholder")
    print("Visit: https://developingmouse.brain-map.org/")
    print("API docs: https://help.brain-map.org/display/api/")


if __name__ == "__main__":
    download_allen_developmental()
''')


def _create_download_readme(scripts_dir: Path) -> None:
    """Create README with manual download instructions.
    
    Args:
        scripts_dir: Directory to save README
    """
    readme = scripts_dir / "README.md"
    with open(readme, 'w') as f:
        f.write("""# Dataset Download Instructions

## Automated Downloads
- Allen Brain Atlas: Run `python download_allen.py`

## Manual Downloads Required

### GUDMAP
1. Visit https://www.gudmap.org/
2. Search for "mouse brain E11.5" 
3. Download MRI volumes in NIFTI format

### EBRAINS
1. Register at https://ebrains.eu/
2. Navigate to atlas datasets
3. Filter for developmental mouse brain
4. Download histology sections

### DevCCF (Developmental Common Coordinate Framework)
1. Visit https://community.brain-map.org/t/developmental-common-coordinate-framework/
2. Download the 3D reference atlases for E11.5, E13.5, E15.5
3. These include anatomical segmentations perfect for registration templates

## Notes
- Total estimated storage: ~20-50 GB for full dataset
- Prioritize DevCCF for registration templates
- Allen ISH data provides gene expression patterns
""")
