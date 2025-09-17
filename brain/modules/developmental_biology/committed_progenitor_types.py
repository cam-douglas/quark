"""
Committed Progenitor Types

This module defines types and data structures for committed neural progenitors.

Author: Quark AI
Date: 2025-01-27
"""

from typing import Dict, Tuple
from dataclasses import dataclass
from .neuroepithelial_cells import NeuroepithelialCell
from .lineage_barcode_types import LineageTag
from .ventricular_zone_organizer import ZoneType
from .progenitor_type_classifier import LineagePotency


@dataclass
class CommittedProgenitor:
    """Represents a committed neural progenitor cell"""
    cell_id: str
    parent_cell: NeuroepithelialCell
    lineage_potency: LineagePotency
    lineage_barcode: LineageTag
    commitment_time: float
    zone_type: ZoneType
    position: Tuple[float, float, float]
    molecular_markers: Dict[str, float]
