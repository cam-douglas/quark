#!/usr/bin/env python3
"""Neuroepithelial Cell Implementation.

Main implementation of neuroepithelial cells including cell properties,
state transitions, and integration with morphogen gradient systems for
embryonic neural development.

Integration: Core cell implementation for developmental biology system
Rationale: Main neuroepithelial cell coordinator with focused responsibilities
"""

from typing import Dict, List, Optional, Tuple
import uuid
import logging

from .neuroepithelial_cell_types import (
    NeuroepithelialCellType, CellCyclePhase, DivisionType, CompetencyWindow
)
from .cell_state_manager import CellStateManager

logger = logging.getLogger(__name__)

class NeuroepithelialCell:
    """Neuroepithelial cell implementation for embryonic neural development.
    
    Represents individual neuroepithelial cells with molecular marker
    expression, cell cycle state, lineage tracking, and morphogen
    responsiveness during embryonic neural tube development.
    """
    
    def __init__(self, cell_type: NeuroepithelialCellType,
                 position: Tuple[float, float, float],
                 developmental_time: float,
                 parent_cell_id: Optional[str] = None):
        """Initialize neuroepithelial cell.
        
        Args:
            cell_type: Type of neuroepithelial cell
            position: 3D position in neural tube (µm coordinates)
            developmental_time: Current developmental time (weeks)
            parent_cell_id: ID of parent cell (None for initial cells)
        """
        # Cell identity
        self.cell_id = str(uuid.uuid4())
        self.cell_type = cell_type
        self.parent_cell_id = parent_cell_id
        self.generation = 0 if parent_cell_id is None else 1
        
        # Spatial properties
        self.position = position
        self.apical_basal_polarity = 1.0
        
        # Temporal properties
        self.birth_time = developmental_time
        self.current_age_hours = 0.0
        
        # Cell cycle state
        self.cell_cycle_phase = CellCyclePhase.G1
        self.cell_cycle_progress = 0.0
        
        # Initialize state manager
        self.state_manager = CellStateManager()
        
        # Cell state
        self.molecular_markers: Dict[str, float] = {}
        self.morphogen_levels: Dict[str, float] = {}
        self.competency_windows = self.state_manager.create_competency_windows(cell_type)
        self.cell_cycle_length_hours = self.state_manager.get_cell_cycle_length(cell_type)
        
        # Fate and division
        self.fate_commitment: Optional[str] = None
        self.commitment_strength = 0.0
        self.division_count = 0
        self.last_division_time = 0.0
        self.division_type_history: List[DivisionType] = []
        
        # Initialize state
        self._initialize_cell_state(developmental_time)
        
        logger.debug(f"Initialized {cell_type.value} cell: {self.cell_id}")
    
    def _initialize_cell_state(self, developmental_time: float) -> None:
        """Initialize cell state based on type and developmental time."""
        # Initialize molecular markers
        self.molecular_markers = self.state_manager.update_molecular_markers(
            self.cell_type, developmental_time, {})
    
    def update_cell_state(self, dt_hours: float, developmental_time: float,
                         morphogen_levels: Dict[str, float]) -> None:
        """Update cell state for one timestep.
        
        Args:
            dt_hours: Time step in hours
            developmental_time: Current developmental time (weeks)
            morphogen_levels: Current morphogen concentrations
        """
        # Update basic properties
        self.current_age_hours += dt_hours
        self.morphogen_levels = morphogen_levels.copy()
        
        # Update molecular markers
        self.molecular_markers = self.state_manager.update_molecular_markers(
            self.cell_type, developmental_time, morphogen_levels)
        
        # Update cell cycle
        self._update_cell_cycle(dt_hours)
        
        # Update competency and fate commitment
        self._update_competency_and_commitment(developmental_time, morphogen_levels)
        
        # Update spatial properties
        self._update_spatial_properties(dt_hours)
    
    def _update_cell_cycle(self, dt_hours: float) -> None:
        """Update cell cycle progression."""
        # Progress through current phase
        phase_duration = self._get_phase_duration(self.cell_cycle_phase)
        self.cell_cycle_progress += dt_hours / phase_duration
        
        # Check for phase transition
        if self.cell_cycle_progress >= 1.0:
            self.cell_cycle_progress = 0.0
            self._advance_cell_cycle_phase()
    
    def _get_phase_duration(self, phase: CellCyclePhase) -> float:
        """Get duration of specific cell cycle phase in hours."""
        total_cycle = self.cell_cycle_length_hours
        
        phase_fractions = {
            CellCyclePhase.G1: 0.5,   # 50% of cycle
            CellCyclePhase.S: 0.25,   # 25% of cycle
            CellCyclePhase.G2: 0.15,  # 15% of cycle
            CellCyclePhase.M: 0.1,    # 10% of cycle
            CellCyclePhase.G0: float('inf')  # Quiescent
        }
        
        return total_cycle * phase_fractions[phase]
    
    def _advance_cell_cycle_phase(self) -> None:
        """Advance to next cell cycle phase."""
        phase_sequence = [CellCyclePhase.G1, CellCyclePhase.S, CellCyclePhase.G2, CellCyclePhase.M]
        
        if self.cell_cycle_phase == CellCyclePhase.M:
            # Complete division, return to G1
            self.cell_cycle_phase = CellCyclePhase.G1
            self._trigger_cell_division()
        elif self.cell_cycle_phase in phase_sequence:
            current_index = phase_sequence.index(self.cell_cycle_phase)
            self.cell_cycle_phase = phase_sequence[(current_index + 1) % len(phase_sequence)]
    
    def _trigger_cell_division(self) -> None:
        """Trigger cell division event."""
        self.division_count += 1
        self.last_division_time = self.current_age_hours
        logger.debug(f"Cell {self.cell_id} completed division #{self.division_count}")
    
    def _update_competency_and_commitment(self, developmental_time: float,
                                        morphogen_levels: Dict[str, float]) -> None:
        """Update cell competency and fate commitment."""
        if self.fate_commitment is None:
            strongest_signal = self.state_manager.evaluate_fate_signals(
                self.competency_windows, developmental_time, morphogen_levels)
            
            if strongest_signal and strongest_signal[1] > 0.8:
                self.fate_commitment = strongest_signal[0]
                self.commitment_strength = strongest_signal[1]
                
                if self.cell_type != NeuroepithelialCellType.COMMITTED_PROGENITOR:
                    self.cell_type = NeuroepithelialCellType.COMMITTED_PROGENITOR
                    logger.debug(f"Cell {self.cell_id} committed to {self.fate_commitment}")
    
    def _update_spatial_properties(self, dt_hours: float) -> None:
        """Update spatial properties including polarity."""
        if self.cell_cycle_phase == CellCyclePhase.M:
            self.apical_basal_polarity *= 0.9  # Disrupted during mitosis
        else:
            self.apical_basal_polarity = min(1.0, self.apical_basal_polarity + 0.1 * dt_hours)
    
    def can_divide(self) -> bool:
        """Check if cell is ready to divide."""
        return (self.cell_cycle_phase == CellCyclePhase.M and 
                self.cell_cycle_progress >= 0.9)
    
    def get_division_type_probabilities(self) -> Dict[DivisionType, float]:
        """Get probabilities for different division types."""
        return self.state_manager.get_division_type_probabilities(self.cell_type)
    
    def export_cell_state(self) -> Dict[str, any]:
        """Export complete cell state information."""
        return {
            'cell_id': self.cell_id,
            'cell_type': self.cell_type.value,
            'parent_cell_id': self.parent_cell_id,
            'generation': self.generation,
            'position': self.position,
            'birth_time': self.birth_time,
            'current_age_hours': self.current_age_hours,
            'cell_cycle_phase': self.cell_cycle_phase.value,
            'cell_cycle_progress': self.cell_cycle_progress,
            'molecular_markers': self.molecular_markers,
            'morphogen_levels': self.morphogen_levels,
            'fate_commitment': self.fate_commitment,
            'commitment_strength': self.commitment_strength,
            'division_count': self.division_count,
            'apical_basal_polarity': self.apical_basal_polarity,
            'can_divide': self.can_divide(),
            'division_probabilities': self.get_division_type_probabilities()
        }