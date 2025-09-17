#!/usr/bin/env python3
"""Fate Rule Engine for Cell Fate Specification.

Applies cell fate rules to gene expression patterns and resolves conflicts
during neural tube cell fate specification.

Integration: Core processing engine for cell fate specification
Rationale: Focused module for rule application and conflict resolution
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import logging

from .cell_fate_types import CellFateRule, NeuralCellType, CellFateRulesDatabase

logger = logging.getLogger(__name__)

class FateRuleEngine:
    """Engine for applying cell fate rules to gene expression patterns.
    
    Processes gene expression maps through cell fate rules to determine
    spatial cell fate patterns with conflict resolution.
    
    Key Features:
    - Rule-based cell fate determination
    - Priority-based conflict resolution
    - Spatial fate mapping
    - Rule application validation
    """
    
    def __init__(self, rules_database: CellFateRulesDatabase):
        """Initialize fate rule engine.
        
        Args:
            rules_database: Cell fate rules database
        """
        self.rules_db = rules_database
        
        # Rule application cache
        self.rule_cache: Dict[str, np.ndarray] = {}
        
        logger.info("Initialized fate rule engine")
        logger.info(f"Rules available: {len(self.rules_db.get_all_rules())}")
    
    def apply_fate_rules(self, gene_expression_maps: Dict[str, np.ndarray],
                        grid_shape: Tuple[int, int, int]) -> Dict[str, np.ndarray]:
        """Apply all cell fate rules to gene expression maps.
        
        Args:
            gene_expression_maps: Dictionary of gene -> expression map
            grid_shape: 3D grid shape (x, y, z)
            
        Returns:
            Dictionary of cell_type -> fate map
        """
        cell_fate_maps = {}
        
        # Apply each fate rule
        for rule in self.rules_db.get_all_rules():
            fate_map = self._apply_single_rule(rule, gene_expression_maps, grid_shape)
            cell_fate_maps[rule.cell_type.value] = fate_map
        
        # Resolve conflicts using priority
        resolved_maps = self._resolve_fate_conflicts(cell_fate_maps)
        
        return resolved_maps
    
    def _apply_single_rule(self, rule: CellFateRule, 
                          gene_maps: Dict[str, np.ndarray],
                          grid_shape: Tuple[int, int, int]) -> np.ndarray:
        """Apply single cell fate rule.
        
        Args:
            rule: Cell fate rule to apply
            gene_maps: Dictionary of gene expression maps
            grid_shape: 3D grid shape
            
        Returns:
            Binary fate map for this cell type
        """
        fate_map = np.ones(grid_shape, dtype=bool)
        
        # Check required genes (all must be expressed)
        for gene in rule.required_genes:
            if gene in gene_maps:
                fate_map &= (gene_maps[gene] > 0.5)
            else:
                # If required gene not available, no cells of this type
                logger.warning(f"Required gene {gene} not available for {rule.cell_type.value}")
                return np.zeros(grid_shape, dtype=bool)
        
        # Check excluded genes (none must be expressed)
        for gene in rule.excluded_genes:
            if gene in gene_maps:
                fate_map &= (gene_maps[gene] <= 0.5)
        
        return fate_map
    
    def _resolve_fate_conflicts(self, cell_fate_maps: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Resolve conflicts when multiple cell types are specified for same voxel.
        
        Uses priority-based resolution: higher priority rules override lower priority.
        
        Args:
            cell_fate_maps: Dictionary of cell type -> fate map
            
        Returns:
            Dictionary of resolved cell fate maps
        """
        resolved_maps = {}
        
        if not cell_fate_maps:
            return resolved_maps
        
        grid_shape = next(iter(cell_fate_maps.values())).shape
        
        # Create priority map to track highest priority at each voxel
        priority_map = np.zeros(grid_shape, dtype=int)
        
        # Sort rules by priority (highest first)
        sorted_rules = sorted(self.rules_db.get_all_rules(), 
                            key=lambda x: x.priority, reverse=True)
        
        for rule in sorted_rules:
            cell_type = rule.cell_type.value
            if cell_type in cell_fate_maps:
                fate_map = cell_fate_maps[cell_type]
                
                # Apply this fate where it has priority and no higher priority fate exists
                mask = fate_map & (priority_map < rule.priority)
                
                # Update priority map
                priority_map[mask] = rule.priority
                
                # Store resolved fate map
                resolved_maps[cell_type] = mask.astype(float)
        
        return resolved_maps
    
    def analyze_rule_coverage(self, gene_expression_maps: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze how well rules cover available gene expression data.
        
        Args:
            gene_expression_maps: Available gene expression maps
            
        Returns:
            Dictionary of coverage analysis
        """
        available_genes = set(gene_expression_maps.keys())
        
        coverage_analysis = {
            "available_genes": sorted(list(available_genes)),
            "required_genes": self.rules_db.get_required_genes(),
            "excluded_genes": self.rules_db.get_excluded_genes(),
            "rule_coverage": {},
            "missing_genes": [],
            "unused_genes": []
        }
        
        # Check coverage for each rule
        for rule in self.rules_db.get_all_rules():
            required_available = [gene for gene in rule.required_genes if gene in available_genes]
            required_missing = [gene for gene in rule.required_genes if gene not in available_genes]
            
            excluded_available = [gene for gene in rule.excluded_genes if gene in available_genes]
            excluded_missing = [gene for gene in rule.excluded_genes if gene not in available_genes]
            
            coverage_analysis["rule_coverage"][rule.cell_type.value] = {
                "required_genes": rule.required_genes,
                "required_available": required_available,
                "required_missing": required_missing,
                "excluded_genes": rule.excluded_genes,
                "excluded_available": excluded_available,
                "excluded_missing": excluded_missing,
                "can_apply": len(required_missing) == 0,
                "priority": rule.priority,
                "confidence": rule.confidence
            }
        
        # Find missing and unused genes
        all_rule_genes = set(self.rules_db.get_required_genes() + self.rules_db.get_excluded_genes())
        coverage_analysis["missing_genes"] = sorted(list(all_rule_genes - available_genes))
        coverage_analysis["unused_genes"] = sorted(list(available_genes - all_rule_genes))
        
        return coverage_analysis
    
    def validate_rule_application(self, gene_expression_maps: Dict[str, np.ndarray],
                                 cell_fate_maps: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Validate rule application results.
        
        Args:
            gene_expression_maps: Input gene expression maps
            cell_fate_maps: Output cell fate maps
            
        Returns:
            Dictionary of validation results
        """
        validation = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "rule_validations": {}
        }
        
        # Validate each rule's application
        for rule in self.rules_db.get_all_rules():
            cell_type = rule.cell_type.value
            
            if cell_type not in cell_fate_maps:
                validation["errors"].append(f"Missing fate map for {cell_type}")
                validation["is_valid"] = False
                continue
            
            fate_map = cell_fate_maps[cell_type]
            cell_count = int(np.sum(fate_map > 0.5))
            
            rule_validation = {
                "cell_type": cell_type,
                "cell_count": cell_count,
                "has_cells": cell_count > 0,
                "priority": rule.priority,
                "confidence": rule.confidence
            }
            
            # Check if rule requirements are met where cells are specified
            if cell_count > 0:
                cell_indices = np.where(fate_map > 0.5)
                
                # Validate required genes
                for gene in rule.required_genes:
                    if gene in gene_expression_maps:
                        gene_expr = gene_expression_maps[gene]
                        gene_values_at_cells = gene_expr[cell_indices]
                        
                        # Check if required gene is expressed in cell fate domain
                        expr_fraction = np.mean(gene_values_at_cells > 0.5)
                        
                        if expr_fraction < 0.8:  # Allow some tolerance
                            validation["warnings"].append(
                                f"{cell_type}: Required gene {gene} not well expressed "
                                f"({expr_fraction:.1%} of cells)"
                            )
                
                # Validate excluded genes
                for gene in rule.excluded_genes:
                    if gene in gene_expression_maps:
                        gene_expr = gene_expression_maps[gene]
                        gene_values_at_cells = gene_expr[cell_indices]
                        
                        # Check if excluded gene is properly excluded
                        expr_fraction = np.mean(gene_values_at_cells > 0.5)
                        
                        if expr_fraction > 0.2:  # Allow some tolerance
                            validation["warnings"].append(
                                f"{cell_type}: Excluded gene {gene} expressed "
                                f"({expr_fraction:.1%} of cells)"
                            )
            
            validation["rule_validations"][cell_type] = rule_validation
        
        return validation
    
    def get_conflict_analysis(self, cell_fate_maps: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze conflicts between cell fate assignments.
        
        Args:
            cell_fate_maps: Cell fate maps before conflict resolution
            
        Returns:
            Dictionary of conflict analysis
        """
        if not cell_fate_maps:
            return {"no_data": True}
        
        grid_shape = next(iter(cell_fate_maps.values())).shape
        total_voxels = np.prod(grid_shape)
        
        # Count overlaps between cell types
        overlap_matrix = {}
        conflict_voxels = np.zeros(grid_shape, dtype=int)
        
        cell_types = list(cell_fate_maps.keys())
        
        for i, cell_type1 in enumerate(cell_types):
            overlap_matrix[cell_type1] = {}
            
            for j, cell_type2 in enumerate(cell_types):
                if i != j:
                    map1 = cell_fate_maps[cell_type1] > 0.5
                    map2 = cell_fate_maps[cell_type2] > 0.5
                    overlap = np.sum(map1 & map2)
                    overlap_matrix[cell_type1][cell_type2] = int(overlap)
                else:
                    overlap_matrix[cell_type1][cell_type2] = 0
        
        # Count total conflicts per voxel
        for fate_map in cell_fate_maps.values():
            conflict_voxels += (fate_map > 0.5).astype(int)
        
        conflict_counts = {
            "no_assignment": int(np.sum(conflict_voxels == 0)),
            "single_assignment": int(np.sum(conflict_voxels == 1)),
            "multiple_assignment": int(np.sum(conflict_voxels > 1)),
            "max_conflicts": int(np.max(conflict_voxels))
        }
        
        return {
            "overlap_matrix": overlap_matrix,
            "conflict_counts": conflict_counts,
            "conflict_fraction": conflict_counts["multiple_assignment"] / total_voxels,
            "assignment_coverage": (conflict_counts["single_assignment"] + 
                                  conflict_counts["multiple_assignment"]) / total_voxels
        }
    
    def get_engine_statistics(self) -> Dict[str, Any]:
        """Get fate rule engine statistics.
        
        Returns:
            Dictionary of engine statistics
        """
        rules = self.rules_db.get_all_rules()
        
        return {
            "engine_info": {
                "total_rules": len(rules),
                "cell_types": len(set(rule.cell_type for rule in rules)),
                "cache_size": len(self.rule_cache)
            },
            "rule_statistics": {
                "priority_range": (
                    min(rule.priority for rule in rules) if rules else 0,
                    max(rule.priority for rule in rules) if rules else 0
                ),
                "confidence_stats": {
                    "mean": sum(rule.confidence for rule in rules) / len(rules) if rules else 0,
                    "min": min(rule.confidence for rule in rules) if rules else 0,
                    "max": max(rule.confidence for rule in rules) if rules else 0
                },
                "gene_requirements": {
                    "unique_required": len(self.rules_db.get_required_genes()),
                    "unique_excluded": len(self.rules_db.get_excluded_genes())
                }
            },
            "database_validation": self.rules_db.validate_rules()
        }
    
    def clear_cache(self) -> None:
        """Clear rule application cache."""
        self.rule_cache.clear()
        logger.info("Cleared fate rule engine cache")
