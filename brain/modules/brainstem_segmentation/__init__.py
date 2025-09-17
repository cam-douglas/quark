"""
Brainstem Segmentation Module

Implements embryonic brainstem subdivision segmentation pipeline for Stage 1 roadmap.
Segments midbrain, pons, and medulla with sensorimotor/autonomic labels.

Modules:
    - dataset_catalog: Dataset metadata definitions
    - data_collector: Main collection orchestrator
    - download_manager: Dataset download utilities
    - registration_config: Registration pipeline setup
    - inference_engine: Automatic segmentation inference engine
    - segmentation_hook: Brain simulator integration hook
"""

from .dataset_catalog import DatasetEntry
from .data_collector import BrainstemDataCollector

__all__ = ["DatasetEntry", "BrainstemDataCollector"]

# Try to import new components (may not be available if training not complete)
try:
    from .inference_engine import BrainstemInferenceEngine, InferenceConfig, auto_segment_brainstem
    __all__.extend(["BrainstemInferenceEngine", "InferenceConfig", "auto_segment_brainstem"])
except ImportError:
    pass

try:
    from .segmentation_hook import BrainstemSegmentationHook, install_segmentation_hook
    __all__.extend(["BrainstemSegmentationHook", "install_segmentation_hook"])
except ImportError:
    pass
