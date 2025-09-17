"""
Brainstem Data Collection Orchestrator

Main module for Step 1.F2: Collect open embryonic brainstem MRI/histology datasets.
Coordinates dataset identification, cataloging, and registration setup.
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .dataset_catalog import DatasetEntry, get_public_datasets
from .download_manager import generate_download_scripts
from .registration_config import create_registration_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BrainstemDataCollector:
    """Manages collection of embryonic brainstem datasets.
    
    Attributes:
        data_dir: Base directory for all dataset storage
        metadata_file: Path to dataset catalog JSON
        datasets: List of identified datasets
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize data collector.
        
        Args:
            data_dir: Base directory for data storage (defaults to Quark data path)
        """
        if data_dir is None:
            data_dir = Path("/Users/camdouglas/quark/data/datasets/brainstem_segmentation")
        self.data_dir = data_dir
        self.metadata_file = self.data_dir / "metadata" / "dataset_catalog.json"
        self.datasets: List[DatasetEntry] = []
        
    def identify_datasets(self) -> List[DatasetEntry]:
        """Identify available public embryonic brain datasets.
        
        Returns:
            List of identified dataset entries
        """
        self.datasets = get_public_datasets()
        logger.info(f"Identified {len(self.datasets)} public datasets")
        return self.datasets
    
    def save_catalog(self) -> None:
        """Save dataset catalog to JSON file."""
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        
        catalog = {
            "generated": datetime.now().isoformat(),
            "total_datasets": len(self.datasets),
            "datasets": [asdict(d) for d in self.datasets]
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(catalog, f, indent=2)
        
        logger.info(f"Saved catalog to {self.metadata_file}")
    
    def generate_summary(self) -> None:
        """Generate human-readable summary report."""
        report_file = self.data_dir / "metadata" / "dataset_summary.md"
        
        with open(report_file, 'w') as f:
            f.write("# Embryonic Brainstem Dataset Catalog\n")
            f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
            
            f.write("## Summary\n")
            f.write(f"- Total datasets: {len(self.datasets)}\n")
            f.write("- Stages: E11.5 to E18.5/P0\n")
            f.write("- Resolution: 1-100 µm\n\n")
            
            f.write("## Priority Datasets\n\n")
            f.write("### 1. DevCCF\n")
            f.write("- Gold standard 3D reference\n")
            f.write("- E11.5, E13.5, E15.5\n\n")
            
            f.write("### 2. Allen Developing Mouse\n")
            f.write("- Gene expression patterns\n")
            f.write("- E11.5-E18.5\n\n")
            
            f.write("## All Datasets\n\n")
            for d in self.datasets:
                f.write(f"**{d.name}**\n")
                f.write(f"- {d.source} | {d.modality} | ")
                f.write(f"{d.developmental_stage} | {d.resolution_um}µm\n\n")
        
        logger.info(f"Generated summary at {report_file}")


def execute_step1() -> None:
    """Execute Step 1.F2: Collect and catalog brainstem datasets."""
    collector = BrainstemDataCollector()
    
    # Identify datasets
    datasets = collector.identify_datasets()
    
    # Save catalog
    collector.save_catalog()
    
    # Generate download scripts
    generate_download_scripts(collector.data_dir, datasets)
    
    # Create registration config
    create_registration_config(collector.data_dir)
    
    # Generate summary
    collector.generate_summary()
    
    print("\n✅ Step 1.F2 Complete!")
    print(f"📁 Data: {collector.data_dir}")
    print(f"📊 Datasets: {len(datasets)}")
    print("\nNext: Download priority datasets")


if __name__ == "__main__":
    execute_step1()