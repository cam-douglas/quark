#!/usr/bin/env python3
"""BMP Gene Expression System Coordinator.

Coordinates BMP gene expression components including thresholds database,
expression mapping, and spatial domain analysis for dorsal neural fates.

Integration: Component of BMP gradient system  
Rationale: Unified interface for BMP gene expression analysis
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import logging

from .spatial_grid import SpatialGrid
from .gene_thresholds_database import GeneThresholdsDatabase, GeneExpressionThreshold
from .shh_expression_mapper import SHHExpressionMapper
from .shh_domain_analyzer import SHHDomainAnalyzer

logger = logging.getLogger(__name__)

class BMPGeneExpressionMapper:
    """BMP gene expression system coordinator.
    
    Provides unified interface for BMP gene expression analysis including
    threshold management, expression mapping, and spatial domain analysis
    for dorsal neural tube patterning.
    
    Key Components:
    - Custom BMP gene thresholds for dorsal specification
    - Expression mapping algorithms adapted for BMP
    - Spatial domain analysis for neural crest and dorsal interneurons
    """
    
    def __init__(self, spatial_grid: SpatialGrid):
        """Initialize BMP gene expression system.
        
        Args:
            spatial_grid: 3D spatial grid containing BMP concentrations
        """
        self.grid = spatial_grid
        
        # Initialize BMP-specific thresholds
        self.thresholds_db = self._create_bmp_thresholds_db()
        self.expression_mapper = SHHExpressionMapper(spatial_grid, self.thresholds_db)
        self.domain_analyzer = SHHDomainAnalyzer(
            spatial_grid, self.thresholds_db, self.expression_mapper
        )
        
        logger.info("Initialized BMP gene expression system coordinator")
        logger.info(f"Available BMP-responsive genes: {len(self.get_available_genes())}")
    
    def _create_bmp_thresholds_db(self) -> GeneThresholdsDatabase:
        """Create BMP-specific gene expression thresholds database."""
        bmp_db = GeneThresholdsDatabase()
        
        # Clear default SHH thresholds and add BMP-specific ones
        bmp_db.thresholds = []
        
        # BMP-responsive genes for dorsal neural tube specification
        # Based on: Liem et al. (1995) Cell; Wine-Lee et al. (2004) Development
        bmp_thresholds = [
            # High BMP thresholds (dorsal-most domains)
            GeneExpressionThreshold(
                gene_name='Msx1',
                threshold_concentration=6.0,     # nM
                expression_type='activation',
                spatial_domain='roof_plate',     # Roof plate specification
                confidence=0.94
            ),
            GeneExpressionThreshold(
                gene_name='Msx2',
                threshold_concentration=5.5,     # nM
                expression_type='activation',
                spatial_domain='roof_plate',     # Roof plate specification
                confidence=0.91
            ),
            GeneExpressionThreshold(
                gene_name='Zic1',
                threshold_concentration=4.0,     # nM
                expression_type='activation',
                spatial_domain='dorsal_neural_tube',  # Dorsal progenitors
                confidence=0.88
            ),
            
            # Medium BMP thresholds (neural crest)
            GeneExpressionThreshold(
                gene_name='Snail2',
                threshold_concentration=3.0,     # nM
                expression_type='activation',
                spatial_domain='neural_crest',   # Neural crest specification
                confidence=0.92
            ),
            GeneExpressionThreshold(
                gene_name='FoxD3',
                threshold_concentration=2.8,     # nM
                expression_type='activation',
                spatial_domain='neural_crest',   # Neural crest maintenance
                confidence=0.89
            ),
            GeneExpressionThreshold(
                gene_name='Sox10',
                threshold_concentration=2.5,     # nM
                expression_type='activation',
                spatial_domain='neural_crest',   # Neural crest differentiation
                confidence=0.95
            ),
            
            # Lower BMP thresholds (dorsal interneurons)
            GeneExpressionThreshold(
                gene_name='Pax3',
                threshold_concentration=2.0,     # nM
                expression_type='activation',
                spatial_domain='dorsal_interneurons',  # Dorsal interneuron progenitors
                confidence=0.93
            ),
            GeneExpressionThreshold(
                gene_name='Pax7',
                threshold_concentration=1.8,     # nM
                expression_type='activation',
                spatial_domain='dorsal_interneurons',  # Dorsal interneuron progenitors
                confidence=0.87
            ),
            
            # BMP-mediated SHH repression
            GeneExpressionThreshold(
                gene_name='SHH_inhibition',
                threshold_concentration=1.5,     # nM
                expression_type='activation',    # BMP activates SHH inhibition
                spatial_domain='dorsal_tube',    # BMP inhibits SHH signaling
                confidence=0.96
            ),
            
            # BMP pathway activation markers
            GeneExpressionThreshold(
                gene_name='Smad1',
                threshold_concentration=1.0,     # nM
                expression_type='activation',    # Direct BMP target
                spatial_domain='bmp_responsive', # All BMP-responsive cells
                confidence=0.98
            ),
            GeneExpressionThreshold(
                gene_name='Id1',
                threshold_concentration=0.8,     # nM
                expression_type='activation',    # BMP downstream target
                spatial_domain='bmp_responsive', # BMP pathway activity
                confidence=0.92
            )
        ]
        
        # Add all BMP thresholds to database
        for threshold in bmp_thresholds:
            bmp_db.add_threshold(threshold)
        
        return bmp_db
    
    def get_gene_expression_map(self, gene_name: str, 
                               expression_mode: str = 'binary') -> np.ndarray:
        """Generate gene expression map based on BMP concentration thresholds.
        
        Args:
            gene_name: Target gene name
            expression_mode: 'binary' or 'graded' expression mapping
            
        Returns:
            Expression map array (0-1 values)
        """
        # Use BMP concentrations instead of SHH
        if 'BMP' not in self.grid.concentrations:
            raise ValueError("BMP concentration not found in spatial grid")
        
        # Temporarily replace SHH with BMP for expression mapping
        original_shh = self.grid.concentrations.get('SHH')
        self.grid.concentrations['SHH'] = self.grid.concentrations['BMP']
        
        try:
            expression_map = self.expression_mapper.get_gene_expression_map(gene_name, expression_mode)
        finally:
            # Restore original SHH concentrations
            if original_shh is not None:
                self.grid.concentrations['SHH'] = original_shh
            elif 'SHH' in self.grid.concentrations:
                del self.grid.concentrations['SHH']
        
        return expression_map
    
    def get_all_expression_maps(self, expression_mode: str = 'binary') -> Dict[str, np.ndarray]:
        """Get expression maps for all BMP-responsive genes.
        
        Args:
            expression_mode: 'binary' or 'graded' expression mapping
            
        Returns:
            Dictionary of gene name -> expression map
        """
        expression_maps = {}
        available_genes = self.get_available_genes()
        
        for gene_name in available_genes:
            try:
                expression_map = self.get_gene_expression_map(gene_name, expression_mode)
                expression_maps[gene_name] = expression_map
            except Exception as e:
                logger.warning(f"Failed to generate BMP expression map for {gene_name}: {e}")
        
        return expression_maps
    
    def validate_expression_patterns(self) -> Dict[str, Any]:
        """Validate BMP gene expression patterns against expected biology.
        
        Returns:
            Dictionary of validation results
        """
        validation = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "gene_validations": {}
        }
        
        try:
            # Check for proper dorsal expression bias
            dorsal_genes = ['Msx1', 'Msx2', 'Zic1', 'Pax3', 'Pax7']
            
            for gene in dorsal_genes:
                if gene in self.get_available_genes():
                    expr_map = self.get_gene_expression_map(gene)
                    
                    if np.sum(expr_map > 0.5) > 0:
                        # Check if expression is biased toward dorsal (low Y values)
                        indices = np.where(expr_map > 0.5)
                        mean_y = np.mean(indices[1])
                        dorsal_bias = mean_y < (self.grid.dimensions.y_size * 0.4)
                        
                        validation["gene_validations"][gene] = {
                            "dorsal_bias": dorsal_bias,
                            "mean_y_position": float(mean_y)
                        }
                        
                        if not dorsal_bias:
                            validation["warnings"].append(f"{gene} not biased toward dorsal domain")
        
        except Exception as e:
            validation["is_valid"] = False
            validation["errors"].append(f"BMP expression validation failed: {str(e)}")
        
        return validation
    
    def get_available_genes(self) -> List[str]:
        """Get list of all available BMP-responsive genes."""
        return self.thresholds_db.get_available_genes()
    
    def get_dorsal_ventral_profiles(self, genes: Optional[List[str]] = None,
                                   x_pos: Optional[int] = None, 
                                   z_pos: Optional[int] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """Get dorsal-ventral BMP gene expression profiles.
        
        Args:
            genes: List of genes to analyze (all if None)
            x_pos: A-P position (uses center if None)
            z_pos: L-R position (uses center if None)
            
        Returns:
            Dictionary of gene -> {y_coords, expression_profile}
        """
        if genes is None:
            genes = self.get_available_genes()
        
        if x_pos is None:
            x_pos = self.grid.dimensions.x_size // 2
        if z_pos is None:
            z_pos = self.grid.dimensions.z_size // 2
        
        profiles = {}
        y_coords = np.arange(self.grid.dimensions.y_size) * self.grid.resolution
        
        for gene in genes:
            try:
                expression_map = self.get_gene_expression_map(gene, 'binary')
                expression_profile = expression_map[x_pos, :, z_pos]
                
                profiles[gene] = {
                    "y_coordinates_um": y_coords,
                    "expression_profile": expression_profile,
                    "expression_extent_um": float(np.sum(expression_profile)) * self.grid.resolution
                }
                
            except Exception as e:
                logger.warning(f"Failed to get BMP D-V profile for {gene}: {e}")
        
        return profiles
    
    def clear_expression_cache(self) -> None:
        """Clear cached expression maps."""
        self.expression_mapper.clear_cache()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive BMP gene expression system status.
        
        Returns:
            Dictionary of system status
        """
        return {
            "system_info": {
                "morphogen": "BMP",
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
