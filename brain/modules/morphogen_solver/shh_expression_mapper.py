#!/usr/bin/env python3
"""SHH Expression Mapping System.

Core expression mapping logic for converting SHH concentration gradients
to gene expression patterns during neural tube patterning.

Integration: Component of SHH gene expression system
Rationale: Focused module for expression mapping algorithms
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import logging

from .spatial_grid import SpatialGrid
from .gene_thresholds_database import GeneThresholdsDatabase, GeneExpressionThreshold

logger = logging.getLogger(__name__)

class SHHExpressionMapper:
    """SHH expression mapping system.
    
    Converts SHH concentration gradients to gene expression maps using
    experimentally-validated concentration thresholds.
    
    Key Features:
    - Binary and graded expression mapping
    - Sigmoid activation functions
    - Expression map caching for performance
    - Multiple expression modes
    """
    
    def __init__(self, spatial_grid: SpatialGrid, thresholds_db: GeneThresholdsDatabase):
        """Initialize SHH expression mapper.
        
        Args:
            spatial_grid: 3D spatial grid containing SHH concentrations
            thresholds_db: Gene expression thresholds database
        """
        self.grid = spatial_grid
        self.thresholds_db = thresholds_db
        
        # Expression maps cache
        self.expression_maps: Dict[str, np.ndarray] = {}
        
        logger.info("Initialized SHH expression mapper")
        logger.info(f"Available genes: {len(self.thresholds_db.get_available_genes())}")
    
    def get_gene_expression_map(self, gene_name: str, 
                               expression_mode: str = 'binary') -> np.ndarray:
        """Generate gene expression map based on SHH concentration thresholds.
        
        Args:
            gene_name: Target gene name
            expression_mode: 'binary' or 'graded' expression mapping
            
        Returns:
            Expression map array (0-1 values)
        """
        # Get threshold information
        threshold_info = self.thresholds_db.get_threshold(gene_name)
        if threshold_info is None:
            available_genes = self.thresholds_db.get_available_genes()
            raise ValueError(f"Unknown gene: {gene_name}. Available: {available_genes}")
        
        # Check if SHH concentration exists
        if 'SHH' not in self.grid.concentrations:
            raise ValueError("SHH concentration not found in spatial grid")
        
        # Check cache first
        cache_key = f"{gene_name}_{expression_mode}"
        if cache_key in self.expression_maps:
            return self.expression_maps[cache_key]
        
        shh_conc = self.grid.concentrations['SHH']
        
        # Generate expression map based on mode
        if expression_mode == 'binary':
            expression_map = self._generate_binary_map(shh_conc, threshold_info)
        elif expression_mode == 'graded':
            expression_map = self._generate_graded_map(shh_conc, threshold_info)
        else:
            raise ValueError(f"Unknown expression mode: {expression_mode}")
        
        # Cache the result
        self.expression_maps[cache_key] = expression_map
        
        return expression_map
    
    def _generate_binary_map(self, shh_concentration: np.ndarray, 
                           threshold: GeneExpressionThreshold) -> np.ndarray:
        """Generate binary expression map.
        
        Args:
            shh_concentration: SHH concentration array
            threshold: Gene expression threshold
            
        Returns:
            Binary expression map (0 or 1)
        """
        if threshold.expression_type == 'activation':
            # Gene expressed above threshold
            expression_map = (shh_concentration >= threshold.threshold_concentration).astype(float)
        else:  # repression
            # Gene expressed below threshold (SHH represses it)
            expression_map = (shh_concentration < threshold.threshold_concentration).astype(float)
        
        return expression_map
    
    def _generate_graded_map(self, shh_concentration: np.ndarray,
                           threshold: GeneExpressionThreshold) -> np.ndarray:
        """Generate graded expression map using sigmoid function.
        
        Args:
            shh_concentration: SHH concentration array
            threshold: Gene expression threshold
            
        Returns:
            Graded expression map (0 to 1)
        """
        if threshold.expression_type == 'activation':
            # Sigmoid activation
            expression_map = self._sigmoid_activation(
                shh_concentration, threshold.threshold_concentration
            )
        else:  # repression
            # Sigmoid repression (inverted activation)
            expression_map = 1.0 - self._sigmoid_activation(
                shh_concentration, threshold.threshold_concentration
            )
        
        return expression_map
    
    def _sigmoid_activation(self, concentration: np.ndarray, threshold: float,
                          steepness: float = 2.0) -> np.ndarray:
        """Calculate sigmoid activation function.
        
        Args:
            concentration: SHH concentration array
            threshold: Half-maximal concentration
            steepness: Sigmoid steepness parameter
            
        Returns:
            Sigmoid values (0 to 1)
        """
        # Sigmoid function: 1 / (1 + exp(-steepness * (C - threshold)))
        exponent = -steepness * (concentration - threshold)
        # Clip to prevent overflow
        exponent = np.clip(exponent, -500, 500)
        return 1.0 / (1.0 + np.exp(exponent))
    
    def get_all_expression_maps(self, expression_mode: str = 'binary') -> Dict[str, np.ndarray]:
        """Get expression maps for all genes.
        
        Args:
            expression_mode: 'binary' or 'graded' expression mapping
            
        Returns:
            Dictionary of gene name -> expression map
        """
        expression_maps = {}
        available_genes = self.thresholds_db.get_available_genes()
        
        for gene_name in available_genes:
            try:
                expression_map = self.get_gene_expression_map(gene_name, expression_mode)
                expression_maps[gene_name] = expression_map
            except Exception as e:
                logger.warning(f"Failed to generate expression map for {gene_name}: {e}")
        
        return expression_maps
    
    def get_dorsal_ventral_profiles(self, genes: Optional[List[str]] = None,
                                   x_pos: Optional[int] = None, 
                                   z_pos: Optional[int] = None,
                                   expression_mode: str = 'binary') -> Dict[str, Dict[str, np.ndarray]]:
        """Get dorsal-ventral gene expression profiles.
        
        Args:
            genes: List of genes to analyze (all if None)
            x_pos: A-P position (uses center if None)
            z_pos: L-R position (uses center if None)
            expression_mode: 'binary' or 'graded' expression mapping
            
        Returns:
            Dictionary of gene -> {y_coords, expression_profile}
        """
        if genes is None:
            genes = self.thresholds_db.get_available_genes()
        
        if x_pos is None:
            x_pos = self.grid.dimensions.x_size // 2
        if z_pos is None:
            z_pos = self.grid.dimensions.z_size // 2
        
        profiles = {}
        y_coords = np.arange(self.grid.dimensions.y_size) * self.grid.resolution
        
        for gene in genes:
            try:
                expression_map = self.get_gene_expression_map(gene, expression_mode)
                expression_profile = expression_map[x_pos, :, z_pos]
                
                profiles[gene] = {
                    "y_coordinates_um": y_coords,
                    "expression_profile": expression_profile,
                    "expression_extent_um": float(np.sum(expression_profile)) * self.grid.resolution
                }
                
            except Exception as e:
                logger.warning(f"Failed to get D-V profile for {gene}: {e}")
        
        return profiles
    
    def compare_expression_patterns(self, gene1: str, gene2: str,
                                   expression_mode: str = 'binary') -> Dict[str, Any]:
        """Compare expression patterns between two genes.
        
        Args:
            gene1: First gene name
            gene2: Second gene name
            expression_mode: Expression mapping mode
            
        Returns:
            Dictionary of comparison metrics
        """
        try:
            map1 = self.get_gene_expression_map(gene1, expression_mode)
            map2 = self.get_gene_expression_map(gene2, expression_mode)
            
            # Calculate overlap metrics
            intersection = np.logical_and(map1 > 0.5, map2 > 0.5)
            union = np.logical_or(map1 > 0.5, map2 > 0.5)
            
            overlap_voxels = int(np.sum(intersection))
            union_voxels = int(np.sum(union))
            
            # Jaccard index (intersection over union)
            jaccard_index = overlap_voxels / union_voxels if union_voxels > 0 else 0.0
            
            # Dice coefficient (2 * intersection / (|A| + |B|))
            gene1_voxels = int(np.sum(map1 > 0.5))
            gene2_voxels = int(np.sum(map2 > 0.5))
            dice_coefficient = (2 * overlap_voxels) / (gene1_voxels + gene2_voxels) if (gene1_voxels + gene2_voxels) > 0 else 0.0
            
            # Spatial correlation
            correlation = float(np.corrcoef(map1.flatten(), map2.flatten())[0, 1])
            
            return {
                "gene1": gene1,
                "gene2": gene2,
                "expression_mode": expression_mode,
                "overlap_metrics": {
                    "jaccard_index": jaccard_index,
                    "dice_coefficient": dice_coefficient,
                    "spatial_correlation": correlation
                },
                "domain_sizes": {
                    f"{gene1}_voxels": gene1_voxels,
                    f"{gene2}_voxels": gene2_voxels,
                    "overlap_voxels": overlap_voxels,
                    "union_voxels": union_voxels
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to compare {gene1} and {gene2}: {e}")
            return {"error": str(e)}
    
    def get_expression_hierarchy(self, expression_mode: str = 'binary') -> Dict[str, Any]:
        """Analyze gene expression hierarchy based on domain sizes.
        
        Args:
            expression_mode: Expression mapping mode
            
        Returns:
            Dictionary of hierarchy analysis
        """
        # Get activation genes only (for proper hierarchy analysis)
        activation_thresholds = self.thresholds_db.get_thresholds_by_type('activation')
        
        # Sort by threshold concentration (highest first)
        activation_thresholds.sort(key=lambda x: x.threshold_concentration, reverse=True)
        
        hierarchy = {
            "genes": [],
            "domain_sizes": {},
            "hierarchy_valid": True,
            "violations": []
        }
        
        for threshold in activation_thresholds:
            try:
                expression_map = self.get_gene_expression_map(threshold.gene_name, expression_mode)
                domain_size = int(np.sum(expression_map > 0.5))
                
                hierarchy["genes"].append(threshold.gene_name)
                hierarchy["domain_sizes"][threshold.gene_name] = {
                    "voxels": domain_size,
                    "threshold_nM": threshold.threshold_concentration
                }
                
            except Exception as e:
                logger.warning(f"Failed to analyze hierarchy for {threshold.gene_name}: {e}")
        
        # Check hierarchy validity (higher threshold genes should have smaller domains)
        genes = hierarchy["genes"]
        for i in range(len(genes) - 1):
            current_gene = genes[i]
            next_gene = genes[i + 1]
            
            current_size = hierarchy["domain_sizes"][current_gene]["voxels"]
            next_size = hierarchy["domain_sizes"][next_gene]["voxels"]
            
            if current_size > next_size:
                hierarchy["hierarchy_valid"] = False
                hierarchy["violations"].append({
                    "higher_threshold_gene": current_gene,
                    "lower_threshold_gene": next_gene,
                    "higher_domain_size": current_size,
                    "lower_domain_size": next_size
                })
        
        return hierarchy
    
    def update_expression_maps(self, genes: Optional[List[str]] = None) -> None:
        """Update cached expression maps after SHH concentration changes.
        
        Args:
            genes: List of genes to update (all if None)
        """
        if genes is None:
            # Clear all cached maps
            self.expression_maps.clear()
            logger.info("Cleared all expression map cache")
        else:
            # Clear specific gene maps
            keys_to_remove = [key for key in self.expression_maps.keys() 
                            if any(key.startswith(f"{gene}_") for gene in genes)]
            for key in keys_to_remove:
                del self.expression_maps[key]
            logger.info(f"Updated expression maps for genes: {genes}")
    
    def get_mapper_statistics(self) -> Dict[str, Any]:
        """Get expression mapper statistics.
        
        Returns:
            Dictionary of mapper statistics
        """
        return {
            "available_genes": len(self.thresholds_db.get_available_genes()),
            "cached_maps": len(self.expression_maps),
            "cache_keys": list(self.expression_maps.keys()),
            "grid_dimensions": {
                "x": self.grid.dimensions.x_size,
                "y": self.grid.dimensions.y_size,
                "z": self.grid.dimensions.z_size,
                "resolution_um": self.grid.resolution
            },
            "shh_available": 'SHH' in self.grid.concentrations
        }
    
    def clear_cache(self) -> None:
        """Clear expression map cache."""
        self.expression_maps.clear()
        logger.info("Cleared expression map cache")
