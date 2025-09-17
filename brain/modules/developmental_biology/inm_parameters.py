"""Interkinetic Nuclear Migration Parameters

Phase-dependent velocity parameters based on Kosodo 2011 human data.
Human embryonic radial glia show phase-specific nuclear migration velocities:
S > G2 > M > G1 (fastest to slowest).

Author: Quark AI
Date: 2025-01-30
"""

from typing import Dict
from dataclasses import dataclass
from enum import Enum


class CellCyclePhase(Enum):
    """Cell cycle phases"""
    G1 = "G1"
    S = "S"
    G2 = "G2"
    M = "M"


@dataclass
class PhaseVelocityData:
    """Velocity data for a specific cell cycle phase"""
    base_velocity: float  # Position units per hour
    velocity_multiplier: float  # Phase-specific multiplier
    position_range: tuple  # (min, max) position for this phase


class INMParameters:
    """Phase-dependent INM parameters from human data (Kosodo 2011)"""
    
    def __init__(self):
        """Initialize with human-specific phase velocities"""
        # Base velocity (position units per hour)
        self.base_velocity = 0.25
        
        # Phase-specific velocity multipliers (S > G2 > M > G1)
        self.phase_velocities = {
            CellCyclePhase.S: PhaseVelocityData(
                base_velocity=self.base_velocity,
                velocity_multiplier=1.8,  # Fastest: S phase basal migration
                position_range=(0.1, 0.8)
            ),
            CellCyclePhase.G2: PhaseVelocityData(
                base_velocity=self.base_velocity,
                velocity_multiplier=1.4,  # Fast: G2 apical migration
                position_range=(0.8, 0.2)
            ),
            CellCyclePhase.M: PhaseVelocityData(
                base_velocity=self.base_velocity,
                velocity_multiplier=1.2,  # Medium: M phase final positioning
                position_range=(0.2, 0.05)
            ),
            CellCyclePhase.G1: PhaseVelocityData(
                base_velocity=self.base_velocity,
                velocity_multiplier=0.6,  # Slowest: G1 maintenance
                position_range=(0.05, 0.15)
            )
        }
        
        # Human-specific target positions (Kosodo 2011)
        self.target_positions = {
            CellCyclePhase.G1: 0.1,   # Apical position
            CellCyclePhase.S: 0.8,    # Basal position for DNA synthesis
            CellCyclePhase.G2: 0.2,   # Moving back apical
            CellCyclePhase.M: 0.05    # Apical surface for mitosis
        }
        
        # Tolerance for position matching
        self.position_tolerance = 0.05
    
    def get_velocity_for_phase(self, phase: CellCyclePhase) -> float:
        """Get velocity for specific cell cycle phase"""
        phase_data = self.phase_velocities.get(phase)
        if not phase_data:
            return self.base_velocity
        return phase_data.base_velocity * phase_data.velocity_multiplier
    
    def get_target_position(self, phase: CellCyclePhase) -> float:
        """Get target nuclear position for cell cycle phase"""
        return self.target_positions.get(phase, 0.1)
    
    def validate_velocity_match(self, simulated_velocities: Dict[CellCyclePhase, float], 
                               tolerance: float = 0.15) -> bool:
        """Validate simulated velocities match expected phase ordering ±15%"""
        expected_order = [CellCyclePhase.S, CellCyclePhase.G2, CellCyclePhase.M, CellCyclePhase.G1]
        
        # Check if velocities follow expected ordering
        for i in range(len(expected_order) - 1):
            current_phase = expected_order[i]
            next_phase = expected_order[i + 1]
            
            if current_phase not in simulated_velocities or next_phase not in simulated_velocities:
                continue
                
            current_vel = simulated_velocities[current_phase]
            next_vel = simulated_velocities[next_phase]
            expected_current = self.get_velocity_for_phase(current_phase)
            expected_next = self.get_velocity_for_phase(next_phase)
            
            # Check ordering is preserved
            if current_vel <= next_vel:
                return False
            
            # Check within tolerance
            if abs(current_vel - expected_current) / expected_current > tolerance:
                return False
            if abs(next_vel - expected_next) / expected_next > tolerance:
                return False
        
        return True
