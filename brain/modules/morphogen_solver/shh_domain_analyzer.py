#!/usr/bin/env python3
"""SHH Spatial Domain Analysis System.

Analyzes spatial domains of gene expression patterns and validates
biological accuracy of SHH-induced gene expression.

Integration: Component of SHH gene expression system
Rationale: Focused module for spatial analysis and validation
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import logging

from .spatial_grid import SpatialGrid
from .gene_thresholds_database import GeneThresholdsDatabase
from .shh_expression_mapper import SHHExpressionMapper

logger = logging.getLogger(__name__)

class SHHDomainAnalyzer:
    """SHH spatial domain analysis system.
    
    Analyzes spatial organization of gene expression domains and validates
    against expected biological patterns for neural tube patterning.
    
    Key Features:
    - Spatial domain statistics
    - Dorsal-ventral organization validation
    - Gene expression hierarchy analysis
    - Biological pattern validation
    """
    
    def __init__(self, spatial_grid: SpatialGrid, 
                 thresholds_db: GeneThresholdsDatabase,
                 expression_mapper: SHHExpressionMapper):
        """Initialize SHH domain analyzer.
        
        Args:
            spatial_grid: 3D spatial grid
            thresholds_db: Gene thresholds database
            expression_mapper: Expression mapping system
        """
        self.grid = spatial_grid
        self.thresholds_db = thresholds_db
        self.expression_mapper = expression_mapper
        
        logger.info("Initialized SHH domain analyzer")
    
    def analyze_spatial_domains(self, expression_mode: str = 'binary') -> Dict[str, Dict[str, Any]]:
        """Analyze spatial domains of gene expression.
        
        Args:
            expression_mode: 'binary' or 'graded' expression mapping
            
        Returns:
            Dictionary of domain analysis results
        """
        domain_analysis = {}
        available_genes = self.thresholds_db.get_available_genes()
        
        for gene_name in available_genes:
            threshold = self.thresholds_db.get_threshold(gene_name)
            if threshold is None:
                continue
            
            try:
                # Get expression map
                expression_map = self.expression_mapper.get_gene_expression_map(
                    gene_name, expression_mode
                )
                
                # Calculate domain statistics
                domain_stats = self._calculate_domain_statistics(expression_map)
                
                domain_analysis[gene_name] = {
                    "spatial_domain": threshold.spatial_domain,
                    "expression_type": threshold.expression_type,
                    "threshold_nM": threshold.threshold_concentration,
                    "confidence": threshold.confidence,
                    "statistics": domain_stats
                }
                
            except Exception as e:
                logger.warning(f"Failed to analyze domain for {gene_name}: {e}")
        
        return domain_analysis
    
    def _calculate_domain_statistics(self, expression_map: np.ndarray) -> Dict[str, Any]:
        """Calculate spatial statistics for expression domain.
        
        Args:
            expression_map: Binary or graded expression map
            
        Returns:
            Dictionary of spatial statistics
        """
        # Basic statistics
        total_voxels = expression_map.size
        expressed_voxels = int(np.sum(expression_map > 0.5))
        expression_fraction = expressed_voxels / total_voxels
        
        if expressed_voxels == 0:
            return {
                "total_voxels": total_voxels,
                "expressed_voxels": 0,
                "expression_fraction": 0.0,
                "centroid": {"x": 0.0, "y": 0.0, "z": 0.0},
                "extent_um": {"x": 0.0, "y": 0.0, "z": 0.0},
                "volume_um3": 0.0
            }
        
        # Find expressed voxel indices
        indices = np.where(expression_map > 0.5)
        
        # Calculate centroid
        centroid_x = float(np.mean(indices[0]))
        centroid_y = float(np.mean(indices[1]))
        centroid_z = float(np.mean(indices[2]))
        
        # Calculate spatial extent
        x_extent = float(np.ptp(indices[0])) * self.grid.resolution
        y_extent = float(np.ptp(indices[1])) * self.grid.resolution
        z_extent = float(np.ptp(indices[2])) * self.grid.resolution
        
        # Calculate volume
        voxel_volume = self.grid.resolution ** 3
        volume_um3 = expressed_voxels * voxel_volume
        
        return {
            "total_voxels": total_voxels,
            "expressed_voxels": expressed_voxels,
            "expression_fraction": expression_fraction,
            "centroid": {
                "x": centroid_x,
                "y": centroid_y,
                "z": centroid_z
            },
            "extent_um": {
                "x": x_extent,
                "y": y_extent,
                "z": z_extent
            },
            "volume_um3": volume_um3
        }
    
    def validate_dorsal_ventral_organization(self, expression_mode: str = 'binary') -> Dict[str, Any]:
        """Validate dorsal-ventral organization of gene expression.
        
        Args:
            expression_mode: Expression mapping mode
            
        Returns:
            Dictionary of validation results
        """
        validation = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "gene_validations": {}
        }
        
        # Define expected gene groups
        ventral_genes = ['Nkx2.2', 'Olig2', 'HB9', 'Nkx6.1']  # Should be ventral
        dorsal_genes = ['Pax6', 'Pax7', 'Msx1']               # Should be dorsal
        
        try:
            # Validate ventral genes
            for gene in ventral_genes:
                if gene in self.thresholds_db.get_available_genes():
                    validation_result = self._validate_ventral_bias(gene, expression_mode)
                    validation["gene_validations"][gene] = validation_result
                    
                    if not validation_result["ventral_bias"]:
                        validation["warnings"].append(f"{gene} not biased toward ventral domain")
            
            # Validate dorsal genes
            for gene in dorsal_genes:
                if gene in self.thresholds_db.get_available_genes():
                    validation_result = self._validate_dorsal_bias(gene, expression_mode)
                    validation["gene_validations"][gene] = validation_result
                    
                    if not validation_result["dorsal_bias"]:
                        validation["warnings"].append(f"{gene} not biased toward dorsal domain")
        
        except Exception as e:
            validation["is_valid"] = False
            validation["errors"].append(f"D-V validation failed: {str(e)}")
        
        return validation
    
    def _validate_ventral_bias(self, gene_name: str, expression_mode: str) -> Dict[str, Any]:
        """Validate that gene shows ventral expression bias.
        
        Args:
            gene_name: Gene to validate
            expression_mode: Expression mapping mode
            
        Returns:
            Dictionary of validation results
        """
        try:
            expression_map = self.expression_mapper.get_gene_expression_map(gene_name, expression_mode)
            
            if np.sum(expression_map > 0.5) == 0:
                return {
                    "ventral_bias": False,
                    "mean_y_position": 0.0,
                    "reason": "No expression detected"
                }
            
            # Calculate mean Y position of expression
            indices = np.where(expression_map > 0.5)
            mean_y = np.mean(indices[1])
            
            # Ventral bias: expression should be in bottom 60% of grid (higher Y values)
            ventral_threshold = self.grid.dimensions.y_size * 0.6
            ventral_bias = mean_y > ventral_threshold
            
            return {
                "ventral_bias": ventral_bias,
                "mean_y_position": float(mean_y),
                "ventral_threshold": ventral_threshold,
                "y_fraction": mean_y / self.grid.dimensions.y_size
            }
            
        except Exception as e:
            return {
                "ventral_bias": False,
                "error": str(e)
            }
    
    def _validate_dorsal_bias(self, gene_name: str, expression_mode: str) -> Dict[str, Any]:
        """Validate that gene shows dorsal expression bias.
        
        Args:
            gene_name: Gene to validate
            expression_mode: Expression mapping mode
            
        Returns:
            Dictionary of validation results
        """
        try:
            expression_map = self.expression_mapper.get_gene_expression_map(gene_name, expression_mode)
            
            if np.sum(expression_map > 0.5) == 0:
                return {
                    "dorsal_bias": False,
                    "mean_y_position": 0.0,
                    "reason": "No expression detected"
                }
            
            # Calculate mean Y position of expression
            indices = np.where(expression_map > 0.5)
            mean_y = np.mean(indices[1])
            
            # Dorsal bias: expression should be in top 40% of grid (lower Y values)
            dorsal_threshold = self.grid.dimensions.y_size * 0.4
            dorsal_bias = mean_y < dorsal_threshold
            
            return {
                "dorsal_bias": dorsal_bias,
                "mean_y_position": float(mean_y),
                "dorsal_threshold": dorsal_threshold,
                "y_fraction": mean_y / self.grid.dimensions.y_size
            }
            
        except Exception as e:
            return {
                "dorsal_bias": False,
                "error": str(e)
            }
    
    def validate_gene_hierarchy(self, expression_mode: str = 'binary') -> Dict[str, Any]:
        """Validate gene expression hierarchy.
        
        Higher threshold genes should have smaller expression domains.
        
        Args:
            expression_mode: Expression mapping mode
            
        Returns:
            Dictionary of hierarchy validation results
        """
        validation = {
            "is_valid": True,
            "violations": [],
            "gene_hierarchy": []
        }
        
        try:
            # Get activation genes sorted by threshold
            activation_thresholds = self.thresholds_db.get_thresholds_by_type('activation')
            activation_thresholds.sort(key=lambda x: x.threshold_concentration, reverse=True)
            
            # Calculate domain sizes
            gene_domains = []
            for threshold in activation_thresholds:
                try:
                    expression_map = self.expression_mapper.get_gene_expression_map(
                        threshold.gene_name, expression_mode
                    )
                    domain_size = int(np.sum(expression_map > 0.5))
                    
                    gene_domains.append({
                        "gene": threshold.gene_name,
                        "threshold_nM": threshold.threshold_concentration,
                        "domain_size": domain_size
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to get domain size for {threshold.gene_name}: {e}")
            
            validation["gene_hierarchy"] = gene_domains
            
            # Check hierarchy violations
            for i in range(len(gene_domains) - 1):
                current = gene_domains[i]
                next_gene = gene_domains[i + 1]
                
                if current["domain_size"] > next_gene["domain_size"]:
                    validation["is_valid"] = False
                    validation["violations"].append({
                        "higher_threshold_gene": current["gene"],
                        "higher_threshold_nM": current["threshold_nM"],
                        "higher_domain_size": current["domain_size"],
                        "lower_threshold_gene": next_gene["gene"],
                        "lower_threshold_nM": next_gene["threshold_nM"],
                        "lower_domain_size": next_gene["domain_size"]
                    })
        
        except Exception as e:
            validation["is_valid"] = False
            validation["error"] = str(e)
        
        return validation
    
    def analyze_expression_boundaries(self, gene1: str, gene2: str,
                                    expression_mode: str = 'binary') -> Dict[str, Any]:
        """Analyze boundaries between gene expression domains.
        
        Args:
            gene1: First gene name
            gene2: Second gene name
            expression_mode: Expression mapping mode
            
        Returns:
            Dictionary of boundary analysis
        """
        try:
            map1 = self.expression_mapper.get_gene_expression_map(gene1, expression_mode)
            map2 = self.expression_mapper.get_gene_expression_map(gene2, expression_mode)
            
            # Find boundary voxels (adjacent to both domains)
            # Dilate each domain by 1 voxel and find intersection
            from scipy.ndimage import binary_dilation
            
            # Create binary masks
            mask1 = map1 > 0.5
            mask2 = map2 > 0.5
            
            # Dilate masks
            dilated1 = binary_dilation(mask1)
            dilated2 = binary_dilation(mask2)
            
            # Find boundary region
            boundary = np.logical_and(dilated1, dilated2) & ~(mask1 | mask2)
            
            # Calculate boundary statistics
            boundary_voxels = int(np.sum(boundary))
            total_boundary_length = boundary_voxels * self.grid.resolution
            
            # Find boundary centroid
            if boundary_voxels > 0:
                boundary_indices = np.where(boundary)
                boundary_centroid = {
                    "x": float(np.mean(boundary_indices[0])),
                    "y": float(np.mean(boundary_indices[1])),
                    "z": float(np.mean(boundary_indices[2]))
                }
            else:
                boundary_centroid = {"x": 0.0, "y": 0.0, "z": 0.0}
            
            return {
                "gene1": gene1,
                "gene2": gene2,
                "boundary_analysis": {
                    "boundary_voxels": boundary_voxels,
                    "boundary_length_um": total_boundary_length,
                    "boundary_centroid": boundary_centroid,
                    "has_clear_boundary": boundary_voxels > 0
                },
                "domain_separation": {
                    "overlap_voxels": int(np.sum(mask1 & mask2)),
                    "gap_exists": boundary_voxels > 0 and np.sum(mask1 & mask2) == 0
                }
            }
            
        except Exception as e:
            return {"error": f"Boundary analysis failed: {str(e)}"}
    
    def get_comprehensive_validation(self, expression_mode: str = 'binary') -> Dict[str, Any]:
        """Get comprehensive validation of SHH gene expression patterns.
        
        Args:
            expression_mode: Expression mapping mode
            
        Returns:
            Dictionary of comprehensive validation results
        """
        validation = {
            "overall_valid": True,
            "validation_components": {},
            "summary": {}
        }
        
        try:
            # Spatial domain analysis
            domain_analysis = self.analyze_spatial_domains(expression_mode)
            validation["validation_components"]["spatial_domains"] = domain_analysis
            
            # Dorsal-ventral organization
            dv_validation = self.validate_dorsal_ventral_organization(expression_mode)
            validation["validation_components"]["dorsal_ventral"] = dv_validation
            
            if not dv_validation["is_valid"]:
                validation["overall_valid"] = False
            
            # Gene hierarchy validation
            hierarchy_validation = self.validate_gene_hierarchy(expression_mode)
            validation["validation_components"]["gene_hierarchy"] = hierarchy_validation
            
            if not hierarchy_validation["is_valid"]:
                validation["overall_valid"] = False
            
            # Summary statistics
            total_genes = len(domain_analysis)
            genes_with_expression = sum(1 for gene_data in domain_analysis.values() 
                                      if gene_data["statistics"]["expressed_voxels"] > 0)
            
            validation["summary"] = {
                "total_genes_analyzed": total_genes,
                "genes_with_expression": genes_with_expression,
                "expression_rate": genes_with_expression / total_genes if total_genes > 0 else 0.0,
                "dv_validation_warnings": len(dv_validation["warnings"]),
                "hierarchy_violations": len(hierarchy_validation.get("violations", [])),
                "expression_mode": expression_mode
            }
            
        except Exception as e:
            validation["overall_valid"] = False
            validation["error"] = str(e)
        
        return validation
    
    def get_analyzer_status(self) -> Dict[str, Any]:
        """Get domain analyzer status and statistics.
        
        Returns:
            Dictionary of analyzer status
        """
        return {
            "analyzer_info": {
                "available_genes": len(self.thresholds_db.get_available_genes()),
                "grid_dimensions": {
                    "x": self.grid.dimensions.x_size,
                    "y": self.grid.dimensions.y_size,
                    "z": self.grid.dimensions.z_size
                },
                "resolution_um": self.grid.resolution
            },
            "validation_capabilities": [
                "spatial_domain_analysis",
                "dorsal_ventral_organization", 
                "gene_hierarchy_validation",
                "expression_boundary_analysis",
                "comprehensive_validation"
            ],
            "expression_modes": ["binary", "graded"]
        }
