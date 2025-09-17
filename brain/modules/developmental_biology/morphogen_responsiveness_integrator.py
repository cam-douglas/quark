"""
Morphogen Responsiveness Integrator

This module integrates ventricular zone organization with the completed
morphogen gradient system from the foundation layer.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .ventricular_zone_organizer import CellArchitecture
from .gene_expression_calculator import GeneExpressionCalculator, GeneExpressionProfile, MorphogenType
from .morphogen_exposure_calculator import MorphogenExposureCalculator


@dataclass
class CellFateDecision:
    """Cell fate decision based on morphogen response"""
    cell_id: str
    fate_commitment: float  # 0.0 to 1.0
    fate_type: str  # "progenitor", "neuron", "glial", "undecided"
    morphogen_contribution: Dict[MorphogenType, float]
    decision_confidence: float


class MorphogenResponsivenessIntegrator:
    """
    Integrates ventricular zone organization with morphogen gradients
    from the foundation layer system.
    """
    
    def __init__(self):
        """Initialize morphogen responsiveness integrator"""
        self.exposure_calculator = MorphogenExposureCalculator()
        self.gene_expression_calculator = GeneExpressionCalculator()
    
    def calculate_morphogen_exposure(self, cell_architecture: CellArchitecture) -> Dict[MorphogenType, float]:
        """Calculate morphogen exposure for cell using exposure calculator"""
        return self.exposure_calculator.calculate_morphogen_exposure(cell_architecture)
    
    def calculate_gene_expression(self, cell_architecture: CellArchitecture,
                                morphogen_exposure: Dict[MorphogenType, float]) -> GeneExpressionProfile:
        """Calculate gene expression profile using gene expression calculator"""
        return self.gene_expression_calculator.calculate_gene_expression(
            cell_architecture.cell_id, morphogen_exposure
        )
    
    def make_cell_fate_decision(self, cell_architecture: CellArchitecture,
                              gene_expression: GeneExpressionProfile) -> CellFateDecision:
        """Make cell fate decision using gene expression calculator"""
        # Calculate fate commitment
        fate_commitment = self.gene_expression_calculator.calculate_fate_commitment(gene_expression)
        
        # Determine fate type
        fate_type = self.gene_expression_calculator.determine_fate_type(gene_expression, fate_commitment)
        
        # Calculate morphogen contribution
        morphogen_contribution = self.gene_expression_calculator.calculate_morphogen_contribution(gene_expression)
        
        # Calculate decision confidence
        decision_confidence = self.gene_expression_calculator.calculate_decision_confidence(gene_expression, fate_commitment)
        
        return CellFateDecision(
            cell_id=cell_architecture.cell_id,
            fate_commitment=fate_commitment,
            fate_type=fate_type,
            morphogen_contribution=morphogen_contribution,
            decision_confidence=decision_confidence
        )
    
    def integrate_morphogen_responsiveness(self, cell_architectures: Dict[str, CellArchitecture]) -> Dict[str, CellFateDecision]:
        """
        Integrate morphogen responsiveness for all cells
        
        Args:
            cell_architectures: Dictionary of cell architectures
            
        Returns:
            Dictionary mapping cell IDs to fate decisions
        """
        cell_fate_decisions = {}
        
        for cell_id, architecture in cell_architectures.items():
            # Calculate morphogen exposure
            morphogen_exposure = self.calculate_morphogen_exposure(architecture)
            
            # Calculate gene expression
            gene_expression = self.calculate_gene_expression(architecture, morphogen_exposure)
            
            # Make fate decision
            fate_decision = self.make_cell_fate_decision(architecture, gene_expression)
            
            cell_fate_decisions[cell_id] = fate_decision
        
        return cell_fate_decisions
    
    def get_morphogen_statistics(self, cell_fate_decisions: Dict[str, CellFateDecision]) -> Dict[str, float]:
        """Get statistics about morphogen responsiveness"""
        if not cell_fate_decisions:
            return {}
        
        fate_counts = {}
        morphogen_contributions = {morphogen: [] for morphogen in MorphogenType}
        total_confidence = 0.0
        
        for decision in cell_fate_decisions.values():
            # Count fate types
            fate_type = decision.fate_type
            fate_counts[fate_type] = fate_counts.get(fate_type, 0) + 1
            
            # Collect morphogen contributions
            for morphogen, contribution in decision.morphogen_contribution.items():
                morphogen_contributions[morphogen].append(contribution)
            
            # Sum confidence
            total_confidence += decision.decision_confidence
        
        total_cells = len(cell_fate_decisions)
        
        # Calculate average morphogen contributions
        avg_contributions = {}
        for morphogen, contributions in morphogen_contributions.items():
            if contributions:
                avg_contributions[morphogen.value] = np.mean(contributions)
            else:
                avg_contributions[morphogen.value] = 0.0
        
        return {
            "total_cells": total_cells,
            "fate_distribution": {fate: count/total_cells for fate, count in fate_counts.items()},
            "average_confidence": total_confidence / total_cells,
            "morphogen_contributions": avg_contributions
        }
