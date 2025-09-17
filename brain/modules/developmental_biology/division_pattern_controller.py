"""
Division Pattern Controller

This module coordinates symmetric vs asymmetric division decisions,
spindle orientation, and division plane control for cell division patterns.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .division_pattern_types import (
    DivisionType, DivisionDecision, SpindleState, CellPolarityState, 
    DivisionPlaneState, DivisionPatternPredictor
)
from .spindle_orientation_controller import SpindleOrientationController, SpindleOrientationResult
from .division_plane_controller import DivisionPlaneController, DivisionPlaneResult
from .notch_oscillator import NotchOscillator  # NEW


class DivisionPatternController:
    """
    Main controller for cell division patterns, coordinating all aspects
    of division decision-making, spindle orientation, and plane positioning.
    """
    
    def __init__(self):
        """Initialize division pattern controller"""
        self.pattern_predictor = DivisionPatternPredictor()
        self.spindle_controller = SpindleOrientationController()
        self.plane_controller = DivisionPlaneController()
        self.notch_oscillator = NotchOscillator()  # NEW
        
        # Active division states
        self.active_divisions: Dict[str, DivisionDecision] = {}
        self.spindle_states: Dict[str, SpindleState] = {}
        self.polarity_states: Dict[str, CellPolarityState] = {}
        self.plane_states: Dict[str, DivisionPlaneState] = {}
    
    def make_division_decision(self, cell_id: str, developmental_stage: str,
                             regulatory_factors: Dict[str, float],
                             cell_polarity: Optional[CellPolarityState] = None) -> DivisionDecision:
        """
        Make a complete division decision for a cell
        
        Args:
            cell_id: Identifier for the cell
            developmental_stage: Current developmental stage
            regulatory_factors: Current regulatory factor levels
            cell_polarity: Current cell polarity state (optional)
            
        Returns:
            Complete division decision with all parameters
        """
        # Initialize Notch oscillator for this cell if needed
        if cell_id not in self.notch_oscillator.cell_states:
            self.notch_oscillator.initialize_cell(cell_id)
        
        # Get Notch-modulated division bias
        notch_bias = self.notch_oscillator.get_division_bias(cell_id)
        
        # Predict division type with Notch modulation
        division_type = self.pattern_predictor.predict_division_type(
            developmental_stage, regulatory_factors
        )
        
        # Modulate division type based on Notch bias
        if notch_bias < 0.4:
            # Force symmetric if Notch is low
            from .division_pattern_types import DivisionType
            division_type = DivisionType.SYMMETRIC_PROLIFERATIVE
        elif notch_bias > 0.6:
            # Force asymmetric if Notch is high
            division_type = DivisionType.ASYMMETRIC_NEUROGENIC
        
        # Calculate confidence
        confidence = self.pattern_predictor.calculate_division_confidence(
            division_type, developmental_stage, regulatory_factors
        )
        
        # Get or create cell polarity state
        if cell_polarity is None:
            cell_polarity = self._create_default_polarity_state(developmental_stage)
        
        # Calculate spindle orientation
        spindle_result = self.spindle_controller.calculate_spindle_orientation(
            cell_polarity, developmental_stage, regulatory_factors
        )
        
        # Calculate division plane
        cell_geometry = self._get_cell_geometry(cell_id)
        plane_result = self.plane_controller.calculate_division_plane(
            spindle_result, cell_polarity, cell_geometry, regulatory_factors
        )
        
        # Create division decision
        division_decision = DivisionDecision(
            division_type=division_type,
            spindle_orientation=spindle_result.orientation,
            cell_polarity=cell_polarity.polarity_type,
            division_plane=plane_result.normal_vector,
            confidence=confidence,
            regulatory_factors=regulatory_factors.copy()
        )
        
        # Store states
        self.active_divisions[cell_id] = division_decision
        self.spindle_states[cell_id] = SpindleState(
            orientation=spindle_result.orientation,
            angle_to_axis=spindle_result.angle_to_axis,
            stability=spindle_result.stability,
            centrosome_separation=0.0,
            microtubule_density=0.0
        )
        self.polarity_states[cell_id] = cell_polarity
        self.plane_states[cell_id] = DivisionPlaneState(
            normal_vector=plane_result.normal_vector,
            position=plane_result.position,
            stability=plane_result.stability,
            regulatory_signals=plane_result.regulatory_signals
        )
        
        return division_decision
    
    def _create_default_polarity_state(self, developmental_stage: str) -> CellPolarityState:
        """Create default polarity state based on developmental stage"""
        # Early embryonic: mostly unpolarized
        if developmental_stage == "early_embryonic":
            polarity_type = "unpolarized"
            polarity_strength = 0.2
        # Mid embryonic: developing polarity
        elif developmental_stage == "mid_embryonic":
            polarity_type = "bipolar"
            polarity_strength = 0.6
        # Late embryonic: strong polarity
        elif developmental_stage == "late_embryonic":
            polarity_type = "bipolar"
            polarity_strength = 0.8
        # Fetal: very strong polarity
        else:  # fetal
            polarity_type = "bipolar"
            polarity_strength = 0.9
        
        from .division_pattern_types import CellPolarity
        
        return CellPolarityState(
            polarity_type=CellPolarity(polarity_type),
            apical_markers={"Par3": 0.5, "aPKC": 0.5, "Crumbs": 0.5},
            basal_markers={"Lgl": 0.5, "Scrib": 0.5, "Dlg": 0.5},
            polarity_strength=polarity_strength,
            orientation_vector=(0.0, 1.0, 0.0)  # Apical-basal axis
        )
    
    def _get_cell_geometry(self, cell_id: str) -> Dict[str, float]:
        """Get cell geometry parameters"""
        # Default geometry - in real implementation, this would come from cell state
        return {
            "volume": 1.0,
            "aspect_ratio": 1.0,
            "surface_area": 6.0,
            "height": 1.0,
            "width": 1.0,
            "depth": 1.0
        }
    
    def update_division_progress(self, cell_id: str, time_delta: float,
                               regulatory_factors: Dict[str, float]) -> Dict[str, float]:
        """
        Update division progress for a cell
        
        Args:
            cell_id: Identifier for the cell
            time_delta: Time elapsed since last update
            regulatory_factors: Current regulatory factor levels
            
        Returns:
            Dictionary of division progress metrics
        """
        if cell_id not in self.active_divisions:
            return {}
        
        # Update spindle state
        if cell_id in self.spindle_states:
            self.spindle_states[cell_id] = self.spindle_controller.update_spindle_state(
                self.spindle_states[cell_id], time_delta, regulatory_factors
            )
        
        # Update plane state
        if cell_id in self.plane_states:
            self.plane_states[cell_id] = self.plane_controller.update_division_plane_state(
                self.plane_states[cell_id], time_delta, regulatory_factors
            )
        
        # Calculate progress metrics
        progress_metrics = self._calculate_division_progress(cell_id)
        
        return progress_metrics
    
    def _calculate_division_progress(self, cell_id: str) -> Dict[str, float]:
        """Calculate division progress metrics"""
        metrics = {}
        
        if cell_id in self.spindle_states:
            spindle_state = self.spindle_states[cell_id]
            metrics.update({
                "spindle_stability": spindle_state.stability,
                "centrosome_separation": spindle_state.centrosome_separation,
                "microtubule_density": spindle_state.microtubule_density,
                "spindle_angle": spindle_state.angle_to_axis
            })
        
        if cell_id in self.plane_states:
            plane_state = self.plane_states[cell_id]
            metrics.update({
                "plane_stability": plane_state.stability,
                "plane_normal_x": plane_state.normal_vector[0],
                "plane_normal_y": plane_state.normal_vector[1],
                "plane_normal_z": plane_state.normal_vector[2]
            })
        
        if cell_id in self.polarity_states:
            polarity_state = self.polarity_states[cell_id]
            metrics.update({
                "polarity_strength": polarity_state.polarity_strength,
                "apical_marker_avg": np.mean(list(polarity_state.apical_markers.values())),
                "basal_marker_avg": np.mean(list(polarity_state.basal_markers.values()))
            })
        
        return metrics
    
    def complete_division(self, cell_id: str) -> Optional[Dict[str, any]]:
        """
        Complete division and return daughter cell information
        
        Args:
            cell_id: Identifier for the dividing cell
            
        Returns:
            Dictionary with daughter cell information or None if division not ready
        """
        if cell_id not in self.active_divisions:
            return None
        
        division_decision = self.active_divisions[cell_id]
        
        # Check if division is ready (spindle stable, plane stable)
        if cell_id in self.spindle_states and cell_id in self.plane_states:
            spindle_state = self.spindle_states[cell_id]
            plane_state = self.plane_states[cell_id]
            
            if spindle_state.stability < 0.8 or plane_state.stability < 0.8:
                return None  # Not ready yet
        
        # Create daughter cell information
        daughter_info = {
            "parent_cell_id": cell_id,
            "division_type": division_decision.division_type.value,
            "spindle_orientation": division_decision.spindle_orientation.value,
            "cell_polarity": division_decision.cell_polarity.value,
            "division_plane": division_decision.division_plane,
            "confidence": division_decision.confidence,
            "daughter_1": {
                "cell_id": f"{cell_id}_daughter_1",
                "inherited_polarity": self._calculate_daughter_polarity(division_decision, 1),
                "volume_ratio": 0.5  # Default, would be calculated from plane
            },
            "daughter_2": {
                "cell_id": f"{cell_id}_daughter_2",
                "inherited_polarity": self._calculate_daughter_polarity(division_decision, 2),
                "volume_ratio": 0.5  # Default, would be calculated from plane
            }
        }
        
        # Clean up division state
        self._cleanup_division_state(cell_id)
        
        return daughter_info
    
    def _calculate_daughter_polarity(self, division_decision: DivisionDecision, 
                                   daughter_number: int) -> str:
        """Calculate inherited polarity for daughter cell"""
        if division_decision.division_type == DivisionType.SYMMETRIC_PROLIFERATIVE:
            # Both daughters inherit progenitor polarity
            return "bipolar"
        elif division_decision.division_type == DivisionType.SYMMETRIC_DIFFERENTIATIVE:
            # Both daughters become differentiated
            return "unpolarized"
        else:  # ASYMMETRIC
            if daughter_number == 1:
                # First daughter remains progenitor
                return "bipolar"
            else:
                # Second daughter differentiates
                return "unpolarized"
    
    def _cleanup_division_state(self, cell_id: str) -> None:
        """Clean up division state for completed division"""
        if cell_id in self.active_divisions:
            del self.active_divisions[cell_id]
        if cell_id in self.spindle_states:
            del self.spindle_states[cell_id]
        if cell_id in self.polarity_states:
            del self.polarity_states[cell_id]
        if cell_id in self.plane_states:
            del self.plane_states[cell_id]
    
    def get_division_statistics(self) -> Dict[str, int]:
        """Get statistics about active divisions"""
        division_types = {}
        spindle_orientations = {}
        
        for division in self.active_divisions.values():
            div_type = division.division_type.value
            division_types[div_type] = division_types.get(div_type, 0) + 1
            
            spindle_orient = division.spindle_orientation.value
            spindle_orientations[spindle_orient] = spindle_orientations.get(spindle_orient, 0) + 1
        
        return {
            "active_divisions": len(self.active_divisions),
            "division_types": division_types,
            "spindle_orientations": spindle_orientations
        }
    
    def get_cell_division_status(self, cell_id: str) -> Optional[Dict[str, any]]:
        """Get division status for a specific cell"""
        if cell_id not in self.active_divisions:
            return None
        
        division_decision = self.active_divisions[cell_id]
        progress_metrics = self._calculate_division_progress(cell_id)
        
        return {
            "division_type": division_decision.division_type.value,
            "spindle_orientation": division_decision.spindle_orientation.value,
            "cell_polarity": division_decision.cell_polarity.value,
            "confidence": division_decision.confidence,
            "progress_metrics": progress_metrics
        }
