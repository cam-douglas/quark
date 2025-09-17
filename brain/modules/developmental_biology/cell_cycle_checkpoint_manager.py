"""
Cell Cycle Checkpoint Manager

This module manages cell cycle checkpoints and regulatory mechanisms
that control phase transitions and cell cycle arrest.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from .cell_cycle_timing_engine import CheckpointType, CellCycleState
from .checkpoint_condition_evaluator import CheckpointConditionEvaluator
from .regulatory_factor_manager import RegulatoryFactorManager


@dataclass
class CheckpointResult:
    """Result of a checkpoint evaluation"""
    passed: bool
    confidence: float
    failure_reason: Optional[str] = None
    required_conditions: List[str] = None


@dataclass
class CheckpointConfiguration:
    """Configuration for a specific checkpoint"""
    checkpoint_type: CheckpointType
    required_conditions: List[str]
    pass_threshold: float
    developmental_sensitivity: Dict[str, float]  # stage -> sensitivity factor


class CellCycleCheckpointManager:
    """
    Manages cell cycle checkpoints with developmental stage-specific
    sensitivity and regulatory mechanisms.
    """
    
    def __init__(self):
        """Initialize checkpoint manager"""
        self.checkpoint_configs: Dict[CheckpointType, CheckpointConfiguration] = {}
        self.condition_evaluator = CheckpointConditionEvaluator()
        self.regulatory_manager = RegulatoryFactorManager()
        self._setup_default_checkpoints()
    
    def _setup_default_checkpoints(self) -> None:
        """Setup default checkpoint configurations"""
        # G1/S checkpoint - cell size and growth factors
        self.checkpoint_configs[CheckpointType.G1_S] = CheckpointConfiguration(
            checkpoint_type=CheckpointType.G1_S,
            required_conditions=[
                "sufficient_cell_size",
                "growth_factor_signaling",
                "nutrient_availability",
                "dna_integrity"
            ],
            pass_threshold=0.8,
            developmental_sensitivity={
                "early_embryonic": 0.6,  # More relaxed
                "mid_embryonic": 0.7,
                "late_embryonic": 0.8,
                "fetal": 0.9  # Most strict
            }
        )
        
        # G2/M checkpoint - DNA replication completion
        self.checkpoint_configs[CheckpointType.G2_M] = CheckpointConfiguration(
            checkpoint_type=CheckpointType.G2_M,
            required_conditions=[
                "dna_replication_complete",
                "chromosome_condensation",
                "centrosome_duplication",
                "dna_damage_repair"
            ],
            pass_threshold=0.85,
            developmental_sensitivity={
                "early_embryonic": 0.7,
                "mid_embryonic": 0.8,
                "late_embryonic": 0.85,
                "fetal": 0.9
            }
        )
        
        # Spindle assembly checkpoint
        self.checkpoint_configs[CheckpointType.SPINDLE_ASSEMBLY] = CheckpointConfiguration(
            checkpoint_type=CheckpointType.SPINDLE_ASSEMBLY,
            required_conditions=[
                "spindle_formation",
                "chromosome_attachment",
                "tension_check",
                "microtubule_stability"
            ],
            pass_threshold=0.9,
            developmental_sensitivity={
                "early_embryonic": 0.8,
                "mid_embryonic": 0.85,
                "late_embryonic": 0.9,
                "fetal": 0.95
            }
        )
        
        # DNA damage checkpoint
        self.checkpoint_configs[CheckpointType.DNA_DAMAGE] = CheckpointConfiguration(
            checkpoint_type=CheckpointType.DNA_DAMAGE,
            required_conditions=[
                "dna_damage_absence",
                "repair_completion",
                "checkpoint_activation",
                "apoptosis_avoidance"
            ],
            pass_threshold=0.95,
            developmental_sensitivity={
                "early_embryonic": 0.8,  # More tolerant
                "mid_embryonic": 0.9,
                "late_embryonic": 0.95,
                "fetal": 0.98
            }
        )
    
    
    def evaluate_checkpoint(self, cell_state: CellCycleState, checkpoint_type: CheckpointType) -> CheckpointResult:
        """
        Evaluate a specific checkpoint for a cell
        
        Args:
            cell_state: Current state of the cell
            checkpoint_type: Type of checkpoint to evaluate
            
        Returns:
            CheckpointResult with pass/fail status and details
        """
        if checkpoint_type not in self.checkpoint_configs:
            return CheckpointResult(
                passed=False,
                confidence=0.0,
                failure_reason=f"Unknown checkpoint type: {checkpoint_type}"
            )
        
        config = self.checkpoint_configs[checkpoint_type]
        
        # Get developmental sensitivity
        sensitivity = config.developmental_sensitivity.get(
            cell_state.developmental_stage, 
            config.pass_threshold
        )
        
        # Evaluate each required condition using condition evaluator
        condition_results = []
        failed_conditions = []
        
        for condition in config.required_conditions:
            condition_result = self.condition_evaluator.evaluate_condition(condition, cell_state)
            condition_results.append(condition_result.passed)
            
            if not condition_result.passed:
                failed_conditions.append(condition)
        
        # Calculate overall checkpoint score
        checkpoint_score = np.mean(condition_results)
        
        # Apply regulatory factors using regulatory manager
        regulatory_modifier = self.regulatory_manager.calculate_regulatory_modifier(
            checkpoint_type, cell_state.developmental_stage
        )
        final_score = checkpoint_score * regulatory_modifier
        
        # Determine if checkpoint passes
        passed = final_score >= sensitivity
        
        # Calculate confidence
        confidence = min(final_score / sensitivity, 1.0) if passed else final_score / sensitivity
        
        return CheckpointResult(
            passed=passed,
            confidence=confidence,
            failure_reason=None if passed else f"Failed conditions: {failed_conditions}",
            required_conditions=config.required_conditions
        )
    
    def update_regulatory_factor(self, factor_name: str, value: float) -> None:
        """Update a regulatory factor value"""
        self.regulatory_manager.update_factor(factor_name, value)
    
    def get_checkpoint_status(self, checkpoint_type: CheckpointType) -> Dict[str, Any]:
        """Get current status of a checkpoint type"""
        if checkpoint_type not in self.checkpoint_configs:
            return {"error": f"Unknown checkpoint type: {checkpoint_type}"}
        
        config = self.checkpoint_configs[checkpoint_type]
        return {
            "checkpoint_type": checkpoint_type.value,
            "required_conditions": config.required_conditions,
            "pass_threshold": config.pass_threshold,
            "developmental_sensitivity": config.developmental_sensitivity,
            "regulatory_modifier": self.regulatory_manager.calculate_regulatory_modifier(
                checkpoint_type, "mid_embryonic"
            )
        }
    
    def get_all_checkpoint_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all checkpoints"""
        return {
            checkpoint.value: self.get_checkpoint_status(checkpoint)
            for checkpoint in CheckpointType
        }
    
    def simulate_stress_response(self, stress_type: str, intensity: float) -> None:
        """Simulate cellular stress response"""
        self.regulatory_manager.simulate_stress_response(stress_type, intensity)
    
    def simulate_growth_factor_stimulation(self, intensity: float) -> None:
        """Simulate growth factor stimulation"""
        self.regulatory_manager.simulate_growth_factor_stimulation(intensity)
