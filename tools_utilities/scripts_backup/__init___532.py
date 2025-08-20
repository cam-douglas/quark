"""
Small-Mind Neurodata Integration Module

This module provides unified access to truly open neuroscience data sources:
- Open Neurophysiology Databases (CRCNS, NeuroMorpho, ModelDB, OSB, NeuroElectro)
- Open Brain Imaging Databases (OpenNeuro, Brainlife, NITRC, INDI)
- CommonCrawl Web Data (WARC/ARC format, S3 public access)
- All sources are publicly accessible without API keys or signups

Author: OmniNode Team
"""

__version__ = "0.2.0"
__author__ = "Small-Mind Development Team"

from ................................................open_neurophysiology import OpenNeurophysiologyInterface
from ................................................open_brain_imaging import OpenBrainImagingInterface
from ................................................commoncrawl_interface import CommonCrawlInterface
from ................................................human_brain_development import SmallMindBrainDevTrainer, create_smallmind_brain_dev_trainer
from ................................................neurodata_manager import NeurodataManager

__all__ = [
    "OpenNeurophysiologyInterface",
    "OpenBrainImagingInterface", 
    "CommonCrawlInterface",
    "SmallMindBrainDevTrainer",
    "create_smallmind_brain_dev_trainer",
    "NeurodataManager"
]
