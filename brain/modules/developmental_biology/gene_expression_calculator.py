"""
Gene Expression Calculator

This module calculates gene expression profiles based on morphogen exposure
and handles gene regulatory networks for morphogen responsiveness.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from .signaling_types import GrowthFactorType


class MorphogenType(Enum):
    """Types of morphogens from foundation layer"""
    SHH = "SHH"  # Sonic Hedgehog
    BMP = "BMP"  # Bone Morphogenetic Protein
    WNT = "WNT"  # Wnt signaling
    FGF = "FGF"  # Fibroblast Growth Factor


@dataclass
class GeneExpressionProfile:
    """Gene expression profile for cell"""
    cell_id: str
    morphogen_responsive_genes: Dict[str, float]  # Gene name -> expression level
    transcription_factors: Dict[str, float]  # TF name -> activity level
    downstream_targets: Dict[str, float]  # Target gene -> expression level


class GeneExpressionCalculator:
    """
    Calculates gene expression profiles based on morphogen exposure
    and handles gene regulatory networks.
    """
    
    def __init__(self):
        """Initialize gene expression calculator"""
        self.gene_regulatory_networks: Dict[str, Dict[str, float]] = {}
        self._setup_gene_regulatory_networks()
    
    def _setup_gene_regulatory_networks(self) -> None:
        """Setup gene regulatory networks for morphogen response"""
        self.gene_regulatory_networks = {
            "SHH_pathway": {
                "GLI1": 0.8,  # Transcription factor
                "GLI2": 0.6,
                "GLI3": 0.4,
                "PTCH1": 0.7,  # Receptor
                "SMO": 0.5,  # Signal transducer
                "FOXA2": 0.3,  # Downstream target
                "NKX2-2": 0.2
            },
            "BMP_pathway": {
                "SMAD1": 0.7,  # Transcription factor
                "SMAD5": 0.6,
                "SMAD8": 0.5,
                "BMPR1A": 0.8,  # Receptor
                "BMPR2": 0.7,
                "MSX1": 0.4,  # Downstream target
                "MSX2": 0.3
            },
            "WNT_pathway": {
                "CTNNB1": 0.9,  # β-catenin
                "TCF7": 0.6,  # Transcription factor
                "LEF1": 0.5,
                "FZD1": 0.7,  # Receptor
                "LRP6": 0.6,
                "AXIN2": 0.4,  # Downstream target
                "MYC": 0.3
            },
            "FGF_pathway": {
                "FOS": 0.8,  # Transcription factor
                "JUN": 0.7,
                "EGR1": 0.6,
                "FGFR1": 0.9,  # Receptor
                "FGFR2": 0.8,
                "SPRY1": 0.5,  # Downstream target
                "SPRY2": 0.4
            }
        }
    
    def calculate_gene_expression(self, cell_id: str, morphogen_exposure: Dict[MorphogenType, float]) -> GeneExpressionProfile:
        """
        Calculate gene expression profile based on morphogen exposure
        
        Args:
            cell_id: Unique cell identifier
            morphogen_exposure: Morphogen exposure levels
            
        Returns:
            GeneExpressionProfile with expression levels
        """
        morphogen_responsive_genes = {}
        transcription_factors = {}
        downstream_targets = {}
        
        # Calculate expression for each morphogen pathway
        for morphogen_type, exposure in morphogen_exposure.items():
            pathway_genes = self._get_pathway_genes(morphogen_type)
            
            for gene_name, base_expression in pathway_genes.items():
                # Calculate expression based on exposure
                expression = base_expression * exposure
                
                # Categorize genes
                if gene_name in ["GLI1", "GLI2", "SMAD1", "CTNNB1", "FOS"]:
                    transcription_factors[gene_name] = expression
                elif gene_name in ["FOXA2", "MSX1", "AXIN2", "SPRY1"]:
                    downstream_targets[gene_name] = expression
                else:
                    morphogen_responsive_genes[gene_name] = expression
        
        return GeneExpressionProfile(
            cell_id=cell_id,
            morphogen_responsive_genes=morphogen_responsive_genes,
            transcription_factors=transcription_factors,
            downstream_targets=downstream_targets
        )
    
    def _get_pathway_genes(self, morphogen_type: MorphogenType) -> Dict[str, float]:
        """Get genes for specific morphogen pathway"""
        pathway_map = {
            MorphogenType.SHH: "SHH_pathway",
            MorphogenType.BMP: "BMP_pathway",
            MorphogenType.WNT: "WNT_pathway",
            MorphogenType.FGF: "FGF_pathway"
        }
        
        pathway_name = pathway_map.get(morphogen_type, "SHH_pathway")
        return self.gene_regulatory_networks.get(pathway_name, {})
    
    def calculate_fate_commitment(self, gene_expression: GeneExpressionProfile) -> float:
        """Calculate overall fate commitment level"""
        # Weight transcription factors by importance
        tf_weights = {
            "GLI1": 0.3,  # SHH pathway
            "SMAD1": 0.3,  # BMP pathway
            "CTNNB1": 0.2,  # WNT pathway
            "FOS": 0.2   # FGF pathway
        }
        
        total_commitment = 0.0
        total_weight = 0.0
        
        for tf_name, weight in tf_weights.items():
            if tf_name in gene_expression.transcription_factors:
                expression = gene_expression.transcription_factors[tf_name]
                total_commitment += expression * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return min(1.0, total_commitment / total_weight)
    
    def determine_fate_type(self, gene_expression: GeneExpressionProfile, fate_commitment: float) -> str:
        """Determine cell fate type"""
        fate_decision_thresholds = {
            "progenitor_commitment": 0.6,
            "neuron_commitment": 0.7,
            "glial_commitment": 0.5,
            "differentiation_threshold": 0.8
        }
        
        if fate_commitment < fate_decision_thresholds["progenitor_commitment"]:
            return "undecided"
        elif fate_commitment < fate_decision_thresholds["neuron_commitment"]:
            return "progenitor"
        elif fate_commitment < fate_decision_thresholds["differentiation_threshold"]:
            # Check for neuron vs glial markers
            if gene_expression.transcription_factors.get("GLI1", 0) > 0.5:
                return "neuron"
            else:
                return "glial"
        else:
            return "differentiated"
    
    def calculate_morphogen_contribution(self, gene_expression: GeneExpressionProfile) -> Dict[MorphogenType, float]:
        """Calculate contribution of each morphogen to fate decision"""
        contribution = {}
        
        # Map transcription factors to morphogens
        tf_morphogen_map = {
            "GLI1": MorphogenType.SHH,
            "SMAD1": MorphogenType.BMP,
            "CTNNB1": MorphogenType.WNT,
            "FOS": MorphogenType.FGF
        }
        
        for tf_name, morphogen_type in tf_morphogen_map.items():
            if tf_name in gene_expression.transcription_factors:
                expression = gene_expression.transcription_factors[tf_name]
                contribution[morphogen_type] = expression
            else:
                contribution[morphogen_type] = 0.0
        
        return contribution
    
    def calculate_decision_confidence(self, gene_expression: GeneExpressionProfile, fate_commitment: float) -> float:
        """Calculate confidence in fate decision"""
        # Base confidence from fate commitment
        base_confidence = fate_commitment
        
        # Additional confidence from gene expression consistency
        tf_count = len(gene_expression.transcription_factors)
        target_count = len(gene_expression.downstream_targets)
        
        expression_consistency = min(1.0, (tf_count + target_count) / 10.0)
        
        confidence = (base_confidence + expression_consistency) / 2.0
        
        return min(1.0, confidence)
    
    def get_gene_expression_statistics(self, gene_expressions: List[GeneExpressionProfile]) -> Dict[str, float]:
        """Get statistics about gene expression patterns"""
        if not gene_expressions:
            return {}
        
        # Collect all transcription factors
        all_tfs = set()
        all_targets = set()
        
        for expression in gene_expressions:
            all_tfs.update(expression.transcription_factors.keys())
            all_targets.update(expression.downstream_targets.keys())
        
        # Calculate average expression levels
        tf_averages = {}
        for tf in all_tfs:
            expressions = [expr.transcription_factors.get(tf, 0.0) for expr in gene_expressions]
            tf_averages[tf] = np.mean(expressions)
        
        target_averages = {}
        for target in all_targets:
            expressions = [expr.downstream_targets.get(target, 0.0) for expr in gene_expressions]
            target_averages[target] = np.mean(expressions)
        
        return {
            "total_cells": len(gene_expressions),
            "transcription_factor_averages": tf_averages,
            "downstream_target_averages": target_averages,
            "unique_tfs": len(all_tfs),
            "unique_targets": len(all_targets)
        }
