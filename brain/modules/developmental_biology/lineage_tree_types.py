#!/usr/bin/env python3
"""Lineage Tree Type Definitions.

Type definitions for lineage tree construction including tree nodes,
relationships, visualization parameters, and analysis metrics.

Integration: Type definitions for lineage tree construction system
Rationale: Centralized tree structure definitions with visualization support
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

class TreeNodeType(Enum):
    """Types of nodes in lineage tree."""
    FOUNDING_CELL = "founding_cell"         # Initial progenitor
    PROGENITOR = "progenitor"              # Proliferating progenitor
    COMMITTED_PROGENITOR = "committed_progenitor"  # Fate-committed progenitor
    DIFFERENTIATING = "differentiating"     # Undergoing differentiation
    DIFFERENTIATED = "differentiated"       # Fully differentiated

class DivisionEvent(Enum):
    """Types of division events in lineage tree."""
    SYMMETRIC_PROLIFERATIVE = "symmetric_proliferative"
    SYMMETRIC_DIFFERENTIATIVE = "symmetric_differentiative"
    ASYMMETRIC = "asymmetric"
    TERMINAL = "terminal"                   # Final division before differentiation

@dataclass
class LineageTreeNode:
    """Node in lineage tree representing a cell."""
    cell_id: str                           # Unique cell identifier
    node_type: TreeNodeType                # Type of node
    parent_id: Optional[str]               # Parent cell ID
    children_ids: List[str]                # Children cell IDs
    generation: int                        # Generation number
    birth_time: float                      # Developmental time of birth
    division_time: Optional[float]         # Time of division (if divided)
    division_type: Optional[DivisionEvent] # Type of division
    cell_fate: Optional[str]              # Committed cell fate
    molecular_markers: Dict[str, float]    # Marker expression at key timepoints
    spatial_position: Tuple[float, float, float]  # 3D position in neural tube
    lineage_barcode_ids: List[str]        # Associated barcode IDs

@dataclass
class LineageTreeMetrics:
    """Metrics for lineage tree analysis."""
    total_nodes: int                       # Total cells in tree
    total_divisions: int                   # Total division events
    max_generation: int                    # Maximum generation reached
    tree_depth: int                        # Depth of tree
    branching_factor: float               # Average branching factor
    symmetric_division_ratio: float       # Ratio of symmetric divisions
    asymmetric_division_ratio: float      # Ratio of asymmetric divisions
    fate_diversity: int                   # Number of different fates
    temporal_span: float                  # Time span of tree (weeks)

@dataclass
class TreeVisualizationConfig:
    """Configuration for lineage tree visualization."""
    node_size_base: float = 10.0          # Base node size
    node_size_scale_factor: float = 1.5   # Size scaling factor
    edge_thickness_base: float = 2.0      # Base edge thickness
    color_scheme: str = "developmental"    # Color scheme (developmental, fate, generation)
    show_molecular_markers: bool = True    # Whether to show marker info
    show_division_types: bool = True       # Whether to show division type info
    show_temporal_info: bool = True        # Whether to show timing info
    layout_algorithm: str = "hierarchical" # Layout algorithm (hierarchical, force, circular)
