"""INM Velocity Validator

Validates interkinetic nuclear migration velocities against human data
with ±15% tolerance. Ensures phase ordering S > G2 > M > G1.

Author: Quark AI
Date: 2025-01-30
"""

from typing import Dict, List
import numpy as np
from dataclasses import dataclass

from .inm_parameters import CellCyclePhase, INMParameters
from .inm_engine import INMEngine


@dataclass
class INMValidationResult:
    """Result of INM velocity validation"""
    phase: CellCyclePhase
    expected_velocity: float
    simulated_velocity: float
    relative_error: float
    within_tolerance: bool


class INMVelocityValidator:
    """Validates INM velocities against human phase-dependent data"""
    
    def __init__(self, tolerance: float = 0.15):
        """Initialize validator with tolerance (±15% default)"""
        self.tolerance = tolerance
        self.parameters = INMParameters()
    
    def validate_engine_velocities(self, engine: INMEngine) -> List[INMValidationResult]:
        """Validate velocities from an INM engine"""
        results = []
        simulated_velocities = engine.get_phase_velocities()
        
        for phase in CellCyclePhase:
            expected = self.parameters.get_velocity_for_phase(phase)
            simulated = simulated_velocities.get(phase, 0.0)
            
            relative_error = abs(simulated - expected) / expected if expected > 0 else 1.0
            within_tolerance = relative_error <= self.tolerance
            
            results.append(INMValidationResult(
                phase=phase,
                expected_velocity=expected,
                simulated_velocity=simulated,
                relative_error=relative_error,
                within_tolerance=within_tolerance
            ))
        
        return results
    
    def validate_phase_ordering(self, velocities: Dict[CellCyclePhase, float]) -> bool:
        """Validate that velocities follow expected ordering S > G2 > M > G1"""
        return self.parameters.validate_velocity_match(velocities, self.tolerance)
    
    def get_validation_summary(self, results: List[INMValidationResult]) -> Dict[str, any]:
        """Get summary of validation results"""
        total = len(results)
        passed = sum(1 for r in results if r.within_tolerance)
        
        return {
            "total_phases": total,
            "phases_passed": passed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "max_error": max(r.relative_error for r in results) if results else 0.0,
            "mean_error": np.mean([r.relative_error for r in results]) if results else 0.0,
            "all_within_tolerance": passed == total
        }
