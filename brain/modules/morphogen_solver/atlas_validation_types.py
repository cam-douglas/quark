#!/usr/bin/env python3
"""Allen Atlas Validation Type Definitions.

Type definitions and data structures for Allen Brain Atlas validation
including coordinate systems, reference data structures, and validation metrics.

Integration: Type definitions for atlas validation system
Rationale: Centralized type definitions for validation components
"""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

class CoordinateSystem(Enum):
    """Coordinate system types for atlas alignment."""
    ALLEN_CCF = "allen_ccf"         # Allen Common Coordinate Framework
    MORPHOGEN_GRID = "morphogen_grid"  # Morphogen solver grid coordinates
    EMBRYONIC_STAGE = "embryonic_stage"  # Embryonic stage coordinates

class ValidationMetric(Enum):
    """Types of validation metrics."""
    DICE_COEFFICIENT = "dice_coefficient"
    HAUSDORFF_DISTANCE = "hausdorff_distance"
    JACCARD_INDEX = "jaccard_index"
    SURFACE_DISTANCE = "surface_distance"

@dataclass
class AtlasReference:
    """Allen Brain Atlas reference data structure."""
    atlas_id: str                    # Atlas identifier
    developmental_stage: str         # E8.5, E9.5, etc.
    coordinate_system: CoordinateSystem
    resolution_um: float            # Spatial resolution (µm)
    dimensions: Tuple[int, int, int] # Grid dimensions
    region_labels: np.ndarray       # Regional segmentation labels
    region_names: Dict[int, str]    # Mapping from label to region name
    reference_url: str              # Download URL for atlas data

@dataclass
class CoordinateTransform:
    """Coordinate transformation between systems."""
    source_system: CoordinateSystem
    target_system: CoordinateSystem
    transformation_matrix: np.ndarray  # 4x4 transformation matrix
    scaling_factors: Tuple[float, float, float]  # X, Y, Z scaling
    translation_offset: Tuple[float, float, float]  # X, Y, Z offset
    rotation_angles: Tuple[float, float, float]  # X, Y, Z rotation (radians)

@dataclass
class ValidationResult:
    """Result of atlas validation."""
    metric_type: ValidationMetric
    metric_value: float             # Computed metric value
    target_threshold: float         # Target threshold for success
    validation_passed: bool         # Whether validation passed
    region_specific_scores: Dict[str, float]  # Per-region scores
    overall_score: float           # Overall validation score
