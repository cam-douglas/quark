#!/usr/bin/env python3
"""SHH Gene Expression System Coordinator.

Coordinates SHH gene expression components including thresholds database,
expression mapping, and spatial domain analysis.

Integration: Component of SHH gradient system  
Rationale: Unified interface for SHH gene expression analysis
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import logging

from .spatial_grid import SpatialGrid
from .gene_thresholds_database import GeneThresholdsDatabase
from .shh_expression_mapper import SHHExpressionMapper
from .shh_domain_analyzer import SHHDomainAnalyzer

logger = logging.getLogger(__name__)

class SHHGeneExpressionMapper:
    """SHH gene expression system coordinator.
    
    Provides unified interface for SHH gene expression analysis including
    threshold management, expression mapping, and spatial domain analysis.
    
    Key Components:
    - GeneThresholdsDatabase: Experimental threshold data
    - SHHExpressionMapper: Expression mapping algorithms
    - SHHDomainAnalyzer: Spatial domain analysis and validation
    """
    
    def __init__(self, spatial_grid: SpatialGrid):
        """Initialize SHH gene expression system.
        
        Args:
            spatial_grid: 3D spatial grid containing SHH concentrations
        """
        self.grid = spatial_grid
        
        # Initialize component systems
        self.thresholds_db = GeneThresholdsDatabase()
        self.expression_mapper = SHHExpressionMapper(spatial_grid, self.thresholds_db)
        self.domain_analyzer = SHHDomainAnalyzer(
            spatial_grid, self.thresholds_db, self.expression_mapper
        )
        
        logger.info("Initialized SHH gene expression system coordinator")
        logger.info(f"Components: ThresholdsDB, ExpressionMapper, DomainAnalyzer")
        logger.info(f"Available genes: {len(self.get_available_genes())}")
    
    def get_gene_expression_map(self, gene_name: str, 
                               expression_mode: str = 'binary') -> np.ndarray:
        """Generate gene expression map based on SHH concentration thresholds.
        
        Args:
            gene_name: Target gene name
            expression_mode: 'binary' or 'graded' expression mapping
            
        Returns:
            Expression map array (0-1 values)
        """
        return self.expression_mapper.get_gene_expression_map(gene_name, expression_mode)
    
    def get_all_expression_maps(self, expression_mode: str = 'binary') -> Dict[str, np.ndarray]:
        """Get expression maps for all genes.
        
        Args:
            expression_mode: 'binary' or 'graded' expression mapping
            
        Returns:
            Dictionary of gene name -> expression map
        """
        return self.expression_mapper.get_all_expression_maps(expression_mode)
    
    def get_dorsal_ventral_profiles(self, genes: Optional[List[str]] = None,
                                   x_pos: Optional[int] = None, 
                                   z_pos: Optional[int] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """Get dorsal-ventral gene expression profiles.
        
        Args:
            genes: List of genes to analyze (all if None)
            x_pos: A-P position (uses center if None)
            z_pos: L-R position (uses center if None)
            
        Returns:
            Dictionary of gene -> {y_coords, expression_profile}
        """
        return self.expression_mapper.get_dorsal_ventral_profiles(genes, x_pos, z_pos)
    
    def analyze_spatial_domains(self) -> Dict[str, Dict[str, Any]]:
        """Analyze spatial domains of gene expression.
        
        Returns:
            Dictionary of domain analysis results
        """
        return self.domain_analyzer.analyze_spatial_domains()
    
    def validate_expression_patterns(self) -> Dict[str, Any]:
        """Validate gene expression patterns against expected biology.
        
        Returns:
            Dictionary of validation results
        """
        return self.domain_analyzer.get_comprehensive_validation()
    
    def get_available_genes(self) -> List[str]:
        """Get list of all available genes."""
        return self.thresholds_db.get_available_genes()
    
    def get_gene_info(self, gene_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific gene.
        
        Args:
            gene_name: Target gene name
            
        Returns:
            Dictionary of gene information or None if not found
        """
        threshold = self.thresholds_db.get_threshold(gene_name)
        if threshold is None:
            return None
        
        return {
            "gene_name": threshold.gene_name,
            "threshold_concentration_nM": threshold.threshold_concentration,
            "expression_type": threshold.expression_type,
            "spatial_domain": threshold.spatial_domain,
            "confidence": threshold.confidence
        }
    
    def compare_expression_patterns(self, gene1: str, gene2: str) -> Dict[str, Any]:
        """Compare expression patterns between two genes.
        
        Args:
            gene1: First gene name
            gene2: Second gene name
            
        Returns:
            Dictionary of comparison metrics
        """
        return self.expression_mapper.compare_expression_patterns(gene1, gene2)
    
    def validate_dorsal_ventral_organization(self) -> Dict[str, Any]:
        """Validate dorsal-ventral organization of gene expression.
        
        Returns:
            Dictionary of validation results
        """
        return self.domain_analyzer.validate_dorsal_ventral_organization()
    
    def validate_gene_hierarchy(self) -> Dict[str, Any]:
        """Validate gene expression hierarchy.
        
        Returns:
            Dictionary of hierarchy validation results
        """
        return self.domain_analyzer.validate_gene_hierarchy()
    
    def get_expression_hierarchy(self) -> Dict[str, Any]:
        """Analyze gene expression hierarchy based on domain sizes.
        
        Returns:
            Dictionary of hierarchy analysis
        """
        return self.expression_mapper.get_expression_hierarchy()
    
    def analyze_expression_boundaries(self, gene1: str, gene2: str) -> Dict[str, Any]:
        """Analyze boundaries between gene expression domains.
        
        Args:
            gene1: First gene name
            gene2: Second gene name
            
        Returns:
            Dictionary of boundary analysis
        """
        return self.domain_analyzer.analyze_expression_boundaries(gene1, gene2)
    
    def clear_expression_cache(self) -> None:
        """Clear cached expression maps."""
        self.expression_mapper.clear_cache()
    
    def update_expression_maps(self, genes: Optional[List[str]] = None) -> None:
        """Update cached expression maps after SHH concentration changes.
        
        Args:
            genes: List of genes to update (all if None)
        """
        self.expression_mapper.update_expression_maps(genes)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive gene expression system status.
        
        Returns:
            Dictionary of system status
        """
        return {
            "system_info": {
                "available_genes": len(self.get_available_genes()),
                "genes": self.get_available_genes(),
                "grid_dimensions": {
                    "x": self.grid.dimensions.x_size,
                    "y": self.grid.dimensions.y_size,
                    "z": self.grid.dimensions.z_size
                },
                "resolution_um": self.grid.resolution
            },
            "components": {
                "thresholds_database": self.thresholds_db.get_database_summary(),
                "expression_mapper": self.expression_mapper.get_mapper_statistics(),
                "domain_analyzer": self.domain_analyzer.get_analyzer_status()
            },
            "validation": self.validate_expression_patterns()
        }