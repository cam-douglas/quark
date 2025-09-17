#!/usr/bin/env python3
"""Cell Fate Specification System Coordinator.

Coordinates cell fate specification components including cell fate types,
rule engine, and spatial analysis for neural tube development.

Integration: Main interface for cell fate specification system
Rationale: Unified coordinator maintaining clean API and integrations
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import logging

from .spatial_grid import SpatialGrid
from .shh_gene_expression import SHHGeneExpressionMapper
from .cell_fate_types import CellFateRulesDatabase, NeuralCellType
from .fate_rule_engine import FateRuleEngine
from .ra_gradient_solver import RAGradientSolver
from .fgf8_gradient_solver import FGF8GradientSolver

logger = logging.getLogger(__name__)

class CellFateSpecifier:
    """Cell fate specification system coordinator.
    
    Coordinates cell fate specification including rule management, 
    rule application, and spatial analysis for neural tube development.
    
    Key Components:
    - CellFateRulesDatabase: Cell type definitions and rules
    - FateRuleEngine: Rule application and conflict resolution
    - Spatial analysis and validation
    """
    
    def __init__(self, spatial_grid: SpatialGrid, gene_expression: SHHGeneExpressionMapper):
        """Initialize cell fate specifier.
        
        Args:
            spatial_grid: 3D spatial grid
            gene_expression: SHH gene expression system
        """
        self.grid = spatial_grid
        self.gene_expression = gene_expression
        
        # Initialize component systems
        self.rules_db = CellFateRulesDatabase()
        self.rule_engine = FateRuleEngine(self.rules_db)
        
        # NEW: RA and FGF8 solvers
        self.ra_solver = RAGradientSolver()
        self.fgf8_solver = FGF8GradientSolver()
        
        # Cell fate maps storage
        self.cell_fate_maps: Dict[str, np.ndarray] = {}
        
        logger.info("Initialized cell fate specifier coordinator")
        logger.info(f"Components: RulesDatabase, RuleEngine")
        logger.info(f"Cell fate rules: {len(self.rules_db.get_all_rules())}")
    
    def specify_cell_fates(self, expression_mode: str = 'binary') -> Dict[str, np.ndarray]:
        """Specify cell fates based on current gene expression patterns.
        
        Args:
            expression_mode: 'binary' or 'graded' gene expression
            
        Returns:
            Dictionary of cell_type -> spatial fate map
        """
        # Get all gene expression maps
        gene_maps = self.gene_expression.get_all_expression_maps(expression_mode)
        
        # NEW: add RA and FGF8 concentration maps for rules that require them
        ra_map = self._compute_ra_map()
        fgf8_map = self._compute_fgf8_map()
        gene_maps["RA"] = ra_map
        gene_maps["FGF8"] = fgf8_map
        
        # Get grid shape
        grid_shape = (self.grid.dimensions.x_size, 
                     self.grid.dimensions.y_size, 
                     self.grid.dimensions.z_size)
        
        # Apply fate rules
        cell_fate_maps = self.rule_engine.apply_fate_rules(gene_maps, grid_shape)
        
        # Store results
        self.cell_fate_maps = cell_fate_maps
        
        return cell_fate_maps
    
    def get_cell_type_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for each cell type.
        
        Returns:
            Dictionary of cell type statistics
        """
        if not self.cell_fate_maps:
            return {}
        
        statistics = {}
        voxel_volume = self.grid.resolution ** 3
        
        for cell_type, fate_map in self.cell_fate_maps.items():
            cell_voxels = int(np.sum(fate_map > 0.5))
            total_volume = cell_voxels * voxel_volume
            
            if cell_voxels > 0:
                # Calculate centroid
                indices = np.where(fate_map > 0.5)
                centroid = {
                    "x": float(np.mean(indices[0])),
                    "y": float(np.mean(indices[1])),
                    "z": float(np.mean(indices[2]))
                }
                
                # Calculate spatial extent
                extent = {
                    "x": float(np.ptp(indices[0])) * self.grid.resolution,
                    "y": float(np.ptp(indices[1])) * self.grid.resolution,
                    "z": float(np.ptp(indices[2])) * self.grid.resolution
                }
            else:
                centroid = {"x": 0.0, "y": 0.0, "z": 0.0}
                extent = {"x": 0.0, "y": 0.0, "z": 0.0}
            
            statistics[cell_type] = {
                "voxel_count": cell_voxels,
                "volume_um3": total_volume,
                "fraction_of_total": cell_voxels / fate_map.size,
                "centroid": centroid,
                "spatial_extent_um": extent
            }
        
        return statistics
    
    def get_dorsal_ventral_cell_distribution(self, x_pos: Optional[int] = None,
                                           z_pos: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """Get dorsal-ventral distribution of cell types.
        
        Args:
            x_pos: A-P position (center if None)
            z_pos: L-R position (center if None)
            
        Returns:
            Dictionary of cell type distributions
        """
        if not self.cell_fate_maps:
            return {}
        
        if x_pos is None:
            x_pos = self.grid.dimensions.x_size // 2
        if z_pos is None:
            z_pos = self.grid.dimensions.z_size // 2
        
        y_coords = np.arange(self.grid.dimensions.y_size) * self.grid.resolution
        distributions = {}
        
        for cell_type, fate_map in self.cell_fate_maps.items():
            profile = fate_map[x_pos, :, z_pos]
            
            # Find expression boundaries
            expressed_indices = np.where(profile > 0.5)[0]
            
            if len(expressed_indices) > 0:
                dorsal_boundary = float(y_coords[expressed_indices[0]])
                ventral_boundary = float(y_coords[expressed_indices[-1]])
                domain_height = ventral_boundary - dorsal_boundary
            else:
                dorsal_boundary = ventral_boundary = domain_height = 0.0
            
            distributions[cell_type] = {
                "y_coordinates_um": y_coords.tolist(),
                "cell_fate_profile": profile.tolist(),
                "dorsal_boundary_um": dorsal_boundary,
                "ventral_boundary_um": ventral_boundary,
                "domain_height_um": domain_height,
                "domain_center_um": (dorsal_boundary + ventral_boundary) / 2
            }
        
        return distributions
    
    def validate_cell_fate_patterns(self) -> Dict[str, Any]:
        """Validate cell fate patterns against expected biology.
        
        Returns:
            Dictionary of validation results
        """
        validation = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "cell_type_validations": {}
        }
        
        if not self.cell_fate_maps:
            validation["is_valid"] = False
            validation["errors"].append("No cell fate maps available")
            return validation
        
        try:
            # Check dorsal-ventral organization
            dv_distributions = self.get_dorsal_ventral_cell_distribution()
            
            # Expected ventral-to-dorsal order
            expected_order = [
                'floor_plate', 'v3_interneuron', 'motor_neuron', 
                'v2_interneuron', 'v1_interneuron', 'v0_interneuron',
                'dorsal_interneuron', 'neural_crest', 'roof_plate'
            ]
            
            # Check if cell types appear in correct dorsal-ventral order
            cell_centers = {}
            for cell_type in expected_order:
                if cell_type in dv_distributions:
                    cell_centers[cell_type] = dv_distributions[cell_type]['domain_center_um']
            
            # Validate ordering (ventral cells should have higher Y coordinates)
            for i in range(len(expected_order) - 1):
                current_type = expected_order[i]
                next_type = expected_order[i + 1]
                
                if current_type in cell_centers and next_type in cell_centers:
                    if cell_centers[current_type] < cell_centers[next_type]:
                        validation["warnings"].append(
                            f"{current_type} appears dorsal to {next_type} (unexpected)"
                        )
            
            # Check for missing expected cell types
            for cell_type in ['motor_neuron', 'v2_interneuron', 'v0_interneuron']:
                if cell_type not in self.cell_fate_maps or np.sum(self.cell_fate_maps[cell_type]) == 0:
                    validation["warnings"].append(f"Missing or empty {cell_type} domain")
            
            validation["cell_type_validations"] = {
                "expected_order": expected_order,
                "detected_cell_types": list(self.cell_fate_maps.keys()),
                "cell_centers_um": cell_centers
            }
        
        except Exception as e:
            validation["is_valid"] = False
            validation["errors"].append(f"Validation failed: {str(e)}")
        
        return validation
    
    def analyze_rule_coverage(self) -> Dict[str, Any]:
        """Analyze rule coverage against available gene expression data.
        
        Returns:
            Dictionary of coverage analysis
        """
        gene_maps = self.gene_expression.get_all_expression_maps('binary')
        return self.rule_engine.analyze_rule_coverage(gene_maps)
    
    def get_conflict_analysis(self) -> Dict[str, Any]:
        """Get analysis of cell fate assignment conflicts.
        
        Returns:
            Dictionary of conflict analysis
        """
        return self.rule_engine.get_conflict_analysis(self.cell_fate_maps)
    
    def get_specifier_status(self) -> Dict[str, Any]:
        """Get comprehensive cell fate specifier status.
        
        Returns:
            Dictionary of specifier status
        """
        return {
            "specifier_info": {
                "cell_types_available": len(self.cell_fate_maps),
                "grid_dimensions": {
                    "x": self.grid.dimensions.x_size,
                    "y": self.grid.dimensions.y_size,
                    "z": self.grid.dimensions.z_size
                },
                "resolution_um": self.grid.resolution
            },
            "components": {
                "rules_database": self.rules_db.get_database_summary(),
                "rule_engine": self.rule_engine.get_engine_statistics()
            },
            "statistics": self.get_cell_type_statistics(),
            "validation": self.validate_cell_fate_patterns(),
            "coverage_analysis": self.analyze_rule_coverage(),
            "conflict_analysis": self.get_conflict_analysis()
        }
    
    def reset_cell_fates(self) -> None:
        """Reset cell fate maps."""
        self.cell_fate_maps.clear()
        self.rule_engine.clear_cache()
        logger.info("Reset cell fate specification system")

    # -------------------- NEW HELPERS --------------------
    def _compute_ra_map(self) -> np.ndarray:
        gx, gy, gz = (self.grid.dimensions.x_size,
                      self.grid.dimensions.y_size,
                      self.grid.dimensions.z_size)
        res = self.grid.resolution
        ra = np.zeros((gx, gy, gz), dtype=float)
        for x in range(gx):
            for y in range(gy):
                for z in range(gz):
                    # Convert voxel indices to µm coords: dorsal at y=0
                    ra[x, y, z] = self.ra_solver.concentration((x*res, y*res, z*res))
        # normalize to [0,1]
        vmax = float(np.max(ra)) or 1.0
        return ra / vmax

    def _compute_fgf8_map(self) -> np.ndarray:
        gx, gy, gz = (self.grid.dimensions.x_size,
                      self.grid.dimensions.y_size,
                      self.grid.dimensions.z_size)
        res = self.grid.resolution
        fgf = np.zeros((gx, gy, gz), dtype=float)
        for x in range(gx):
            for y in range(gy):
                for z in range(gz):
                    # Place organiser at (0,0,0) in this simple model
                    fgf[x, y, z] = self.fgf8_solver.concentration((x*res, y*res, (z - gz//2)*res))
        vmax = float(np.max(fgf)) or 1.0
        return fgf / vmax