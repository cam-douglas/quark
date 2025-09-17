"""
Cell Cycle Timing Engine for Developmental Biology

This module implements a comprehensive cell cycle timing system that models
G1, S, G2, M phases with checkpoints and developmental stage-dependent timing.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time


class CellCyclePhase(Enum):
    """Cell cycle phases"""
    G1 = "G1"
    S = "S" 
    G2 = "G2"
    M = "M"
    G0 = "G0"  # Quiescent state


class CheckpointType(Enum):
    """Cell cycle checkpoints"""
    G1_S = "G1_S"
    G2_M = "G2_M"
    SPINDLE_ASSEMBLY = "SPINDLE_ASSEMBLY"
    DNA_DAMAGE = "DNA_DAMAGE"


@dataclass
class CellCycleState:
    """Current state of a cell in the cycle"""
    cell_id: str
    current_phase: CellCyclePhase
    phase_start_time: float
    phase_duration: float
    total_cycle_time: float
    checkpoint_status: Dict[CheckpointType, bool]
    developmental_stage: str
    is_arrested: bool = False
    arrest_reason: Optional[str] = None


@dataclass
class PhaseTiming:
    """Timing parameters for each cell cycle phase"""
    g1_duration: float
    s_duration: float
    g2_duration: float
    m_duration: float
    total_cycle_time: float


class CellCycleTimingEngine:
    """
    Core cell cycle timing engine that manages phase transitions,
    checkpoints, and developmental stage-dependent timing.
    """
    
    def __init__(self):
        """Initialize the cell cycle timing engine"""
        self.active_cells: Dict[str, CellCycleState] = {}
        self.phase_timings: Dict[str, PhaseTiming] = {}
        self.checkpoint_requirements: Dict[CheckpointType, callable] = {}
        self._setup_default_timings()
        self._setup_checkpoint_requirements()
    
    def _setup_default_timings(self) -> None:
        """Setup default phase timings for different developmental stages"""
        # Early embryonic (rapid division)
        self.phase_timings["early_embryonic"] = PhaseTiming(
            g1_duration=0.5,  # hours
            s_duration=2.0,
            g2_duration=0.5,
            m_duration=1.0,
            total_cycle_time=4.0
        )
        
        # Mid embryonic (moderate division)
        self.phase_timings["mid_embryonic"] = PhaseTiming(
            g1_duration=2.0,
            s_duration=4.0,
            g2_duration=1.5,
            m_duration=1.5,
            total_cycle_time=9.0
        )
        
        # Late embryonic (slower division)
        self.phase_timings["late_embryonic"] = PhaseTiming(
            g1_duration=6.0,
            s_duration=8.0,
            g2_duration=3.0,
            m_duration=2.0,
            total_cycle_time=19.0
        )
        
        # Fetal (very slow division)
        self.phase_timings["fetal"] = PhaseTiming(
            g1_duration=12.0,
            s_duration=12.0,
            g2_duration=6.0,
            m_duration=3.0,
            total_cycle_time=33.0
        )
    
    def _setup_checkpoint_requirements(self) -> None:
        """Setup checkpoint requirement functions"""
        self.checkpoint_requirements = {
            CheckpointType.G1_S: self._check_g1_s_checkpoint,
            CheckpointType.G2_M: self._check_g2_m_checkpoint,
            CheckpointType.SPINDLE_ASSEMBLY: self._check_spindle_assembly,
            CheckpointType.DNA_DAMAGE: self._check_dna_damage
        }
    
    def add_cell(self, cell_id: str, developmental_stage: str) -> None:
        """Add a new cell to the cycle tracking"""
        if cell_id in self.active_cells:
            raise ValueError(f"Cell {cell_id} already exists in cycle tracking")
        
        timing = self.phase_timings.get(developmental_stage, self.phase_timings["mid_embryonic"])
        
        self.active_cells[cell_id] = CellCycleState(
            cell_id=cell_id,
            current_phase=CellCyclePhase.G1,
            phase_start_time=time.time(),
            phase_duration=timing.g1_duration,
            total_cycle_time=timing.total_cycle_time,
            checkpoint_status={checkpoint: False for checkpoint in CheckpointType},
            developmental_stage=developmental_stage
        )
    
    def update_cell_cycle(self, cell_id: str) -> Optional[CellCyclePhase]:
        """
        Update cell cycle for a specific cell and return new phase if transitioned
        
        Returns:
            New phase if transitioned, None if still in same phase
        """
        if cell_id not in self.active_cells:
            return None
        
        cell = self.active_cells[cell_id]
        current_time = time.time()
        time_in_phase = current_time - cell.phase_start_time
        
        # Check if phase duration has been reached
        if time_in_phase >= cell.phase_duration:
            return self._transition_to_next_phase(cell_id)
        
        return None
    
    def _transition_to_next_phase(self, cell_id: str) -> CellCyclePhase:
        """Transition cell to next phase in cycle"""
        cell = self.active_cells[cell_id]
        timing = self.phase_timings[cell.developmental_stage]
        
        # Determine next phase
        if cell.current_phase == CellCyclePhase.G1:
            next_phase = CellCyclePhase.S
            next_duration = timing.s_duration
        elif cell.current_phase == CellCyclePhase.S:
            next_phase = CellCyclePhase.G2
            next_duration = timing.g2_duration
        elif cell.current_phase == CellCyclePhase.G2:
            next_phase = CellCyclePhase.M
            next_duration = timing.m_duration
        elif cell.current_phase == CellCyclePhase.M:
            next_phase = CellCyclePhase.G1
            next_duration = timing.g1_duration
        else:
            # G0 state - no automatic transition
            return cell.current_phase
        
        # Check if checkpoints are satisfied before transition
        if not self._check_phase_transition_checkpoints(cell_id, next_phase):
            # Arrest cell if checkpoints not met
            cell.is_arrested = True
            cell.arrest_reason = f"Checkpoint failure before {next_phase.value}"
            return cell.current_phase
        
        # Update cell state
        cell.current_phase = next_phase
        cell.phase_start_time = time.time()
        cell.phase_duration = next_duration
        cell.is_arrested = False
        cell.arrest_reason = None
        
        return next_phase
    
    def _check_phase_transition_checkpoints(self, cell_id: str, next_phase: CellCyclePhase) -> bool:
        """Check if all required checkpoints are satisfied for phase transition"""
        cell = self.active_cells[cell_id]
        
        # Determine required checkpoints based on transition
        required_checkpoints = []
        if cell.current_phase == CellCyclePhase.G1 and next_phase == CellCyclePhase.S:
            required_checkpoints = [CheckpointType.G1_S, CheckpointType.DNA_DAMAGE]
        elif cell.current_phase == CellCyclePhase.G2 and next_phase == CellCyclePhase.M:
            required_checkpoints = [CheckpointType.G2_M, CheckpointType.SPINDLE_ASSEMBLY]
        
        # Check each required checkpoint
        for checkpoint in required_checkpoints:
            if not self.checkpoint_requirements[checkpoint](cell_id):
                return False
        
        return True
    
    def _check_g1_s_checkpoint(self, cell_id: str) -> bool:
        """Check G1/S checkpoint - cell size and growth factors"""
        cell = self.active_cells[cell_id]
        # Simulate checkpoint based on developmental stage
        # Early embryonic cells have relaxed checkpoints
        if cell.developmental_stage == "early_embryonic":
            return np.random.random() > 0.1  # 90% pass rate
        else:
            return np.random.random() > 0.05  # 95% pass rate
    
    def _check_g2_m_checkpoint(self, cell_id: str) -> bool:
        """Check G2/M checkpoint - DNA replication completion"""
        cell = self.active_cells[cell_id]
        # Simulate checkpoint based on developmental stage
        if cell.developmental_stage == "early_embryonic":
            return np.random.random() > 0.05  # 95% pass rate
        else:
            return np.random.random() > 0.02  # 98% pass rate
    
    def _check_spindle_assembly(self, cell_id: str) -> bool:
        """Check spindle assembly checkpoint"""
        cell = self.active_cells[cell_id]
        # Simulate checkpoint - more likely to fail in early stages
        if cell.developmental_stage == "early_embryonic":
            return np.random.random() > 0.15  # 85% pass rate
        else:
            return np.random.random() > 0.05  # 95% pass rate
    
    def _check_dna_damage(self, cell_id: str) -> bool:
        """Check DNA damage checkpoint"""
        cell = self.active_cells[cell_id]
        # Simulate checkpoint - early embryonic cells more tolerant
        if cell.developmental_stage == "early_embryonic":
            return np.random.random() > 0.1  # 90% pass rate
        else:
            return np.random.random() > 0.02  # 98% pass rate
    
    def get_cell_cycle_status(self, cell_id: str) -> Optional[CellCycleState]:
        """Get current cell cycle status"""
        return self.active_cells.get(cell_id)
    
    def get_phase_progress(self, cell_id: str) -> float:
        """Get progress through current phase (0.0 to 1.0)"""
        if cell_id not in self.active_cells:
            return 0.0
        
        cell = self.active_cells[cell_id]
        current_time = time.time()
        time_in_phase = current_time - cell.phase_start_time
        return min(time_in_phase / cell.phase_duration, 1.0)
    
    def arrest_cell(self, cell_id: str, reason: str) -> None:
        """Arrest cell in current phase"""
        if cell_id in self.active_cells:
            cell = self.active_cells[cell_id]
            cell.is_arrested = True
            cell.arrest_reason = reason
    
    def release_cell_arrest(self, cell_id: str) -> None:
        """Release cell from arrest"""
        if cell_id in self.active_cells:
            cell = self.active_cells[cell_id]
            cell.is_arrested = False
            cell.arrest_reason = None
    
    def remove_cell(self, cell_id: str) -> None:
        """Remove cell from cycle tracking"""
        if cell_id in self.active_cells:
            del self.active_cells[cell_id]
    
    def get_cycle_statistics(self) -> Dict[str, int]:
        """Get statistics about cells in different phases"""
        stats = {phase.value: 0 for phase in CellCyclePhase}
        stats["arrested"] = 0
        
        for cell in self.active_cells.values():
            if cell.is_arrested:
                stats["arrested"] += 1
            else:
                stats[cell.current_phase.value] += 1
        
        return stats
