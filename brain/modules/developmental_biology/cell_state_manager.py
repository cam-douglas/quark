#!/usr/bin/env python3
"""Cell State Management System.

Manages cell state updates, competency evaluation, and fate commitment
for neuroepithelial cells including molecular marker updates and
morphogen responsiveness during embryonic development.

Integration: State management component for neuroepithelial cell system
Rationale: Focused state management logic separated from main cell class
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

from .neuroepithelial_cell_types import (
    NeuroepithelialCellType, CellCyclePhase, DivisionType, CompetencyWindow
)
from .molecular_markers import MolecularMarkerDatabase

logger = logging.getLogger(__name__)

class CellStateManager:
    """Manager for neuroepithelial cell state updates.
    
    Handles cell state transitions, competency evaluation, molecular
    marker updates, and fate commitment decisions for neuroepithelial
    cells during embryonic neural development.
    """
    
    def __init__(self):
        """Initialize cell state manager."""
        self.marker_database = MolecularMarkerDatabase()
        logger.info("Initialized CellStateManager")
    
    def create_competency_windows(self, cell_type: NeuroepithelialCellType) -> List[CompetencyWindow]:
        """Create competency windows based on cell type."""
        windows = []
        
        if cell_type == NeuroepithelialCellType.EARLY_MULTIPOTENT:
            # Early multipotent cells can become any neural cell type
            windows.extend([
                CompetencyWindow('motor_neuron', 8.5, 11.0, 0.9, ['SHH']),
                CompetencyWindow('interneuron', 8.0, 12.0, 0.8, ['SHH', 'BMP']),
                CompetencyWindow('neural_crest', 8.0, 10.0, 0.7, ['BMP', 'WNT']),
                CompetencyWindow('oligodendrocyte', 10.0, 14.0, 0.6, ['SHH']),
                CompetencyWindow('astrocyte', 11.0, 16.0, 0.5, ['BMP'])
            ])
            
        elif cell_type == NeuroepithelialCellType.LATE_MULTIPOTENT:
            # Late multipotent cells have restricted competency
            windows.extend([
                CompetencyWindow('interneuron', 9.0, 13.0, 0.8, ['SHH', 'BMP']),
                CompetencyWindow('oligodendrocyte', 10.5, 15.0, 0.7, ['SHH']),
                CompetencyWindow('astrocyte', 12.0, 18.0, 0.6, ['BMP'])
            ])
        
        return windows
    
    def update_molecular_markers(self, cell_type: NeuroepithelialCellType,
                                developmental_time: float,
                                morphogen_levels: Dict[str, float]) -> Dict[str, float]:
        """Update molecular marker expression levels."""
        return self.marker_database.calculate_marker_expression(
            cell_type, developmental_time, morphogen_levels)
    
    def evaluate_fate_signals(self, competency_windows: List[CompetencyWindow],
                             developmental_time: float,
                             morphogen_levels: Dict[str, float]) -> Optional[Tuple[str, float]]:
        """Evaluate fate commitment signals from morphogens."""
        fate_signals = {}
        
        for window in competency_windows:
            if window.competency_start <= developmental_time <= window.competency_end:
                # Check if required signals are present
                signal_strength = window.competency_strength
                
                for required_signal in window.required_signals:
                    if required_signal in morphogen_levels:
                        signal_level = morphogen_levels[required_signal]
                        signal_strength *= signal_level
                
                fate_signals[window.fate_type] = signal_strength
        
        # Return strongest fate signal
        if fate_signals:
            strongest_fate = max(fate_signals.items(), key=lambda x: x[1])
            return strongest_fate
        
        return None
    
    def get_division_type_probabilities(self, cell_type: NeuroepithelialCellType) -> Dict[DivisionType, float]:
        """Get probabilities for different division types."""
        if cell_type == NeuroepithelialCellType.EARLY_MULTIPOTENT:
            # Early cells favor symmetric proliferative divisions
            return {
                DivisionType.SYMMETRIC_PROLIFERATIVE: 0.7,
                DivisionType.ASYMMETRIC: 0.25,
                DivisionType.SYMMETRIC_DIFFERENTIATIVE: 0.05
            }
        elif cell_type == NeuroepithelialCellType.LATE_MULTIPOTENT:
            # Late cells increase asymmetric divisions
            return {
                DivisionType.SYMMETRIC_PROLIFERATIVE: 0.4,
                DivisionType.ASYMMETRIC: 0.5,
                DivisionType.SYMMETRIC_DIFFERENTIATIVE: 0.1
            }
        elif cell_type == NeuroepithelialCellType.COMMITTED_PROGENITOR:
            # Committed cells favor differentiative divisions
            return {
                DivisionType.SYMMETRIC_PROLIFERATIVE: 0.2,
                DivisionType.ASYMMETRIC: 0.3,
                DivisionType.SYMMETRIC_DIFFERENTIATIVE: 0.5
            }
        else:
            # Default equal probabilities
            return {
                DivisionType.SYMMETRIC_PROLIFERATIVE: 0.33,
                DivisionType.ASYMMETRIC: 0.33,
                DivisionType.SYMMETRIC_DIFFERENTIATIVE: 0.34
            }
    
    def get_cell_cycle_length(self, cell_type: NeuroepithelialCellType) -> float:
        """Get appropriate cell cycle length for cell type."""
        if cell_type == NeuroepithelialCellType.EARLY_MULTIPOTENT:
            return 8.0   # Fast cycling early (hours)
        elif cell_type == NeuroepithelialCellType.LATE_MULTIPOTENT:
            return 12.0  # Moderate cycling
        else:
            return 16.0  # Slower cycling when committed
    
    def validate_cell_state(self, cell_type: NeuroepithelialCellType,
                           molecular_markers: Dict[str, float]) -> Dict[str, bool]:
        """Validate cell state consistency."""
        return self.marker_database.validate_marker_expression(molecular_markers, cell_type)
