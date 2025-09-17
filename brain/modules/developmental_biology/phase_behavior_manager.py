"""
Phase Behavior Manager

This module manages phase-specific behaviors and processes for each
cell cycle phase, including DNA replication, chromosome condensation,
and mitotic spindle formation.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .cell_cycle_timing_engine import CellCyclePhase, CellCycleState


@dataclass
class PhaseBehavior:
    """Behavior configuration for a specific cell cycle phase"""
    phase: CellCyclePhase
    key_processes: List[str]
    regulatory_proteins: List[str]
    energy_requirements: float
    sensitivity_factors: Dict[str, float]


@dataclass
class DNAReplicationState:
    """State of DNA replication during S phase"""
    replication_progress: float  # 0.0 to 1.0
    replication_forks: int
    replication_speed: float
    error_rate: float
    checkpoint_activation: bool


@dataclass
class MitoticState:
    """State of mitosis during M phase"""
    chromosome_condensation: float  # 0.0 to 1.0
    spindle_formation: float  # 0.0 to 1.0
    chromosome_alignment: float  # 0.0 to 1.0
    cytokinesis_progress: float  # 0.0 to 1.0


class PhaseBehaviorManager:
    """
    Manages phase-specific behaviors and processes for each cell cycle phase.
    Handles DNA replication, chromosome condensation, and mitotic spindle formation.
    """
    
    def __init__(self):
        """Initialize phase behavior manager"""
        self.phase_behaviors: Dict[CellCyclePhase, PhaseBehavior] = {}
        self.dna_replication_states: Dict[str, DNAReplicationState] = {}
        self.mitotic_states: Dict[str, MitoticState] = {}
        self._setup_phase_behaviors()
    
    def _setup_phase_behaviors(self) -> None:
        """Setup behavior configurations for each phase"""
        # G1 phase behavior
        self.phase_behaviors[CellCyclePhase.G1] = PhaseBehavior(
            phase=CellCyclePhase.G1,
            key_processes=[
                "cell_growth",
                "protein_synthesis",
                "organelle_duplication",
                "growth_factor_response"
            ],
            regulatory_proteins=["Cyclin D", "CDK4/6", "p21", "p27"],
            energy_requirements=1.0,
            sensitivity_factors={
                "growth_factor_signaling": 0.8,
                "nutrient_availability": 0.9,
                "cell_size": 0.7
            }
        )
        
        # S phase behavior
        self.phase_behaviors[CellCyclePhase.S] = PhaseBehavior(
            phase=CellCyclePhase.S,
            key_processes=[
                "dna_replication",
                "histone_synthesis",
                "centrosome_duplication",
                "checkpoint_monitoring"
            ],
            regulatory_proteins=["Cyclin A", "CDK2", "PCNA", "DNA polymerase"],
            energy_requirements=1.5,
            sensitivity_factors={
                "dna_damage": 0.9,
                "replication_stress": 0.8,
                "nucleotide_availability": 0.7
            }
        )
        
        # G2 phase behavior
        self.phase_behaviors[CellCyclePhase.G2] = PhaseBehavior(
            phase=CellCyclePhase.G2,
            key_processes=[
                "dna_repair",
                "protein_synthesis",
                "mitotic_preparation",
                "checkpoint_activation"
            ],
            regulatory_proteins=["Cyclin A", "Cyclin B", "CDK1", "Wee1"],
            energy_requirements=1.2,
            sensitivity_factors={
                "dna_damage": 0.95,
                "chromosome_integrity": 0.9,
                "mitotic_readiness": 0.8
            }
        )
        
        # M phase behavior
        self.phase_behaviors[CellCyclePhase.M] = PhaseBehavior(
            phase=CellCyclePhase.M,
            key_processes=[
                "chromosome_condensation",
                "spindle_formation",
                "chromosome_segregation",
                "cytokinesis"
            ],
            regulatory_proteins=["Cyclin B", "CDK1", "Condensin", "Cohesin"],
            energy_requirements=2.0,
            sensitivity_factors={
                "spindle_assembly": 0.9,
                "chromosome_attachment": 0.95,
                "tension_sensing": 0.8
            }
        )
        
        # G0 phase behavior (quiescent)
        self.phase_behaviors[CellCyclePhase.G0] = PhaseBehavior(
            phase=CellCyclePhase.G0,
            key_processes=[
                "metabolic_maintenance",
                "dna_repair",
                "differentiation_signaling"
            ],
            regulatory_proteins=["p21", "p27", "p53", "RB"],
            energy_requirements=0.3,
            sensitivity_factors={
                "growth_factor_withdrawal": 0.9,
                "differentiation_signals": 0.7,
                "stress_response": 0.8
            }
        )
    
    def initialize_phase_behavior(self, cell_id: str, phase: CellCyclePhase) -> None:
        """Initialize phase-specific behavior for a cell"""
        if phase == CellCyclePhase.S:
            # Initialize DNA replication state
            self.dna_replication_states[cell_id] = DNAReplicationState(
                replication_progress=0.0,
                replication_forks=0,
                replication_speed=1.0,
                error_rate=1e-9,
                checkpoint_activation=False
            )
        elif phase == CellCyclePhase.M:
            # Initialize mitotic state
            self.mitotic_states[cell_id] = MitoticState(
                chromosome_condensation=0.0,
                spindle_formation=0.0,
                chromosome_alignment=0.0,
                cytokinesis_progress=0.0
            )
    
    def update_phase_behavior(self, cell_id: str, phase: CellCyclePhase, 
                            time_delta: float, cell_state: CellCycleState) -> Dict[str, float]:
        """
        Update phase-specific behavior for a cell
        
        Args:
            cell_id: Identifier for the cell
            phase: Current cell cycle phase
            time_delta: Time elapsed since last update
            cell_state: Current cell state
            
        Returns:
            Dictionary of phase-specific progress metrics
        """
        if phase not in self.phase_behaviors:
            return {}
        
        behavior = self.phase_behaviors[phase]
        progress_metrics = {}
        
        # Update phase-specific processes
        if phase == CellCyclePhase.G1:
            progress_metrics = self._update_g1_behavior(cell_id, time_delta, cell_state)
        elif phase == CellCyclePhase.S:
            progress_metrics = self._update_s_phase_behavior(cell_id, time_delta, cell_state)
        elif phase == CellCyclePhase.G2:
            progress_metrics = self._update_g2_behavior(cell_id, time_delta, cell_state)
        elif phase == CellCyclePhase.M:
            progress_metrics = self._update_m_phase_behavior(cell_id, time_delta, cell_state)
        elif phase == CellCyclePhase.G0:
            progress_metrics = self._update_g0_behavior(cell_id, time_delta, cell_state)
        
        return progress_metrics
    
    def _update_g1_behavior(self, cell_id: str, time_delta: float, 
                           cell_state: CellCycleState) -> Dict[str, float]:
        """Update G1 phase behavior"""
        # Simulate cell growth and protein synthesis
        growth_rate = 0.1 * time_delta  # 10% per hour
        protein_synthesis = 0.15 * time_delta
        
        return {
            "cell_growth": min(growth_rate, 1.0),
            "protein_synthesis": min(protein_synthesis, 1.0),
            "organelle_duplication": min(0.05 * time_delta, 1.0),
            "growth_factor_response": min(0.2 * time_delta, 1.0)
        }
    
    def _update_s_phase_behavior(self, cell_id: str, time_delta: float, 
                                cell_state: CellCycleState) -> Dict[str, float]:
        """Update S phase behavior (DNA replication)"""
        if cell_id not in self.dna_replication_states:
            self.initialize_phase_behavior(cell_id, CellCyclePhase.S)
        
        replication_state = self.dna_replication_states[cell_id]
        
        # Update DNA replication progress
        replication_rate = replication_state.replication_speed * time_delta
        replication_state.replication_progress = min(
            replication_state.replication_progress + replication_rate, 1.0
        )
        
        # Simulate replication fork dynamics
        if replication_state.replication_progress < 0.5:
            replication_state.replication_forks = min(
                replication_state.replication_forks + 1, 50
            )
        else:
            replication_state.replication_forks = max(
                replication_state.replication_forks - 1, 0
            )
        
        # Check for replication errors
        if np.random.random() < replication_state.error_rate * time_delta:
            replication_state.checkpoint_activation = True
        
        return {
            "dna_replication": replication_state.replication_progress,
            "replication_forks": replication_state.replication_forks / 50.0,
            "histone_synthesis": min(0.3 * time_delta, 1.0),
            "centrosome_duplication": min(0.2 * time_delta, 1.0),
            "checkpoint_activation": 1.0 if replication_state.checkpoint_activation else 0.0
        }
    
    def _update_g2_behavior(self, cell_id: str, time_delta: float, 
                           cell_state: CellCycleState) -> Dict[str, float]:
        """Update G2 phase behavior"""
        # Simulate DNA repair and mitotic preparation
        repair_rate = 0.2 * time_delta
        mitotic_prep = 0.25 * time_delta
        
        return {
            "dna_repair": min(repair_rate, 1.0),
            "protein_synthesis": min(0.1 * time_delta, 1.0),
            "mitotic_preparation": min(mitotic_prep, 1.0),
            "checkpoint_activation": min(0.3 * time_delta, 1.0)
        }
    
    def _update_m_phase_behavior(self, cell_id: str, time_delta: float, 
                                cell_state: CellCycleState) -> Dict[str, float]:
        """Update M phase behavior (mitosis)"""
        if cell_id not in self.mitotic_states:
            self.initialize_phase_behavior(cell_id, CellCyclePhase.M)
        
        mitotic_state = self.mitotic_states[cell_id]
        
        # Update mitotic progression
        condensation_rate = 0.4 * time_delta
        spindle_rate = 0.3 * time_delta
        alignment_rate = 0.2 * time_delta
        cytokinesis_rate = 0.25 * time_delta
        
        mitotic_state.chromosome_condensation = min(
            mitotic_state.chromosome_condensation + condensation_rate, 1.0
        )
        mitotic_state.spindle_formation = min(
            mitotic_state.spindle_formation + spindle_rate, 1.0
        )
        mitotic_state.chromosome_alignment = min(
            mitotic_state.chromosome_alignment + alignment_rate, 1.0
        )
        mitotic_state.cytokinesis_progress = min(
            mitotic_state.cytokinesis_progress + cytokinesis_rate, 1.0
        )
        
        return {
            "chromosome_condensation": mitotic_state.chromosome_condensation,
            "spindle_formation": mitotic_state.spindle_formation,
            "chromosome_alignment": mitotic_state.chromosome_alignment,
            "cytokinesis": mitotic_state.cytokinesis_progress
        }
    
    def _update_g0_behavior(self, cell_id: str, time_delta: float, 
                           cell_state: CellCycleState) -> Dict[str, float]:
        """Update G0 phase behavior (quiescent)"""
        # Simulate maintenance and repair processes
        maintenance_rate = 0.05 * time_delta
        repair_rate = 0.1 * time_delta
        
        return {
            "metabolic_maintenance": min(maintenance_rate, 1.0),
            "dna_repair": min(repair_rate, 1.0),
            "differentiation_signaling": min(0.02 * time_delta, 1.0)
        }
    
    def get_phase_energy_requirement(self, phase: CellCyclePhase) -> float:
        """Get energy requirement for a specific phase"""
        if phase in self.phase_behaviors:
            return self.phase_behaviors[phase].energy_requirements
        return 1.0
    
    def get_phase_sensitivity_factors(self, phase: CellCyclePhase) -> Dict[str, float]:
        """Get sensitivity factors for a specific phase"""
        if phase in self.phase_behaviors:
            return self.phase_behaviors[phase].sensitivity_factors
        return {}
    
    def cleanup_cell_data(self, cell_id: str) -> None:
        """Clean up phase-specific data for a cell"""
        if cell_id in self.dna_replication_states:
            del self.dna_replication_states[cell_id]
        if cell_id in self.mitotic_states:
            del self.mitotic_states[cell_id]
    
    def get_phase_statistics(self) -> Dict[str, int]:
        """Get statistics about cells in different phases"""
        stats = {
            "dna_replicating": len(self.dna_replication_states),
            "mitotic": len(self.mitotic_states)
        }
        return stats
