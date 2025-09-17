#!/usr/bin/env python3
"""Cell Fate Types and Rules Definitions.

Defines neural cell types and fate specification rules for neural tube
development based on morphogen-induced gene expression patterns.

Integration: Core definitions for cell fate specification system
Rationale: Centralized cell type and rule definitions with biological validation
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class NeuralCellType(Enum):
    """Neural cell types specified by morphogen gradients."""
    # Ventral cell types (high SHH)
    FLOOR_PLATE = "floor_plate"
    V3_INTERNEURON = "v3_interneuron"
    MOTOR_NEURON = "motor_neuron"
    V2_INTERNEURON = "v2_interneuron"
    V1_INTERNEURON = "v1_interneuron"
    V0_INTERNEURON = "v0_interneuron"
    
    # Dorsal cell types (low SHH)
    DORSAL_INTERNEURON = "dorsal_interneuron"
    NEURAL_CREST = "neural_crest"
    ROOF_PLATE = "roof_plate"

@dataclass
class CellFateRule:
    """Rule for cell fate specification based on gene expression."""
    cell_type: NeuralCellType
    required_genes: List[str]      # Genes that must be expressed
    excluded_genes: List[str]      # Genes that must NOT be expressed
    priority: int                  # Priority for conflicting rules (higher = priority)
    confidence: float = 1.0        # Confidence in this rule (0-1)
    
    def __post_init__(self):
        """Validate rule parameters."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0 and 1")
        if self.priority < 0:
            raise ValueError("Priority must be non-negative")

class CellFateRulesDatabase:
    """Database of cell fate specification rules.
    
    Contains experimentally-validated rules for neural cell fate specification
    based on morphogen-induced gene expression patterns during neural tube
    dorsal-ventral patterning.
    
    Key Features:
    - Literature-backed fate rules
    - Priority-based conflict resolution
    - Confidence scoring for experimental validation
    - Easy rule lookup and filtering
    """
    
    def __init__(self):
        """Initialize cell fate rules database."""
        self.fate_rules: List[CellFateRule] = []
        self._load_default_rules()
        
        logger.info(f"Loaded {len(self.fate_rules)} cell fate rules")
    
    def _load_default_rules(self) -> None:
        """Load default cell fate specification rules.
        
        Based on experimental data from neural tube patterning studies:
        - Jessell (2000) Nature Reviews Neuroscience
        - Briscoe & Ericson (2001) Current Opinion in Neurobiology
        - Dessaud et al. (2008) Development
        """
        self.fate_rules = [
            # Floor plate (highest SHH, ventral-most)
            CellFateRule(
                cell_type=NeuralCellType.FLOOR_PLATE,
                required_genes=['Nkx2.2', 'Olig2'],
                excluded_genes=['Pax6', 'Pax7'],
                priority=10,
                confidence=0.95
            ),
            
            # V3 interneurons (high SHH)
            CellFateRule(
                cell_type=NeuralCellType.V3_INTERNEURON,
                required_genes=['Nkx2.2'],
                excluded_genes=['Olig2', 'Pax6'],
                priority=9,
                confidence=0.92
            ),
            
            # Motor neurons (high SHH)
            CellFateRule(
                cell_type=NeuralCellType.MOTOR_NEURON,
                required_genes=['Olig2', 'HB9'],
                excluded_genes=['Nkx2.2', 'Pax6'],
                priority=8,
                confidence=0.98
            ),
            
            # V2 interneurons (medium SHH)
            CellFateRule(
                cell_type=NeuralCellType.V2_INTERNEURON,
                required_genes=['Nkx6.1'],
                excluded_genes=['Nkx2.2', 'Olig2', 'Pax6'],
                priority=7,
                confidence=0.90
            ),
            
            # V1 interneurons (medium SHH)
            CellFateRule(
                cell_type=NeuralCellType.V1_INTERNEURON,
                required_genes=['Dbx1'],
                excluded_genes=['Nkx6.1', 'Pax6'],
                priority=6,
                confidence=0.88
            ),
            
            # V0 interneurons (low SHH)
            CellFateRule(
                cell_type=NeuralCellType.V0_INTERNEURON,
                required_genes=['Pax6'],
                excluded_genes=['Dbx1', 'Pax7'],
                priority=5,
                confidence=0.85
            ),
            
            # Dorsal interneurons (very low SHH)
            CellFateRule(
                cell_type=NeuralCellType.DORSAL_INTERNEURON,
                required_genes=['Pax7'],
                excluded_genes=['Pax6', 'Msx1'],
                priority=4,
                confidence=0.82
            ),
            
            # Neural crest (no SHH)
            CellFateRule(
                cell_type=NeuralCellType.NEURAL_CREST,
                required_genes=['Msx1'],
                excluded_genes=['Pax6', 'Pax7'],
                priority=3,
                confidence=0.80
            ),
            
            # Roof plate (no SHH, dorsal-most)
            CellFateRule(
                cell_type=NeuralCellType.ROOF_PLATE,
                required_genes=['Msx1'],
                excluded_genes=['Pax6', 'Pax7', 'BMP_inhibition'],
                priority=2,
                confidence=0.75
            )
        ]
    
    def get_all_rules(self) -> List[CellFateRule]:
        """Get all cell fate rules."""
        return self.fate_rules.copy()
    
    def get_rules_by_cell_type(self, cell_type: NeuralCellType) -> List[CellFateRule]:
        """Get rules for specific cell type.
        
        Args:
            cell_type: Neural cell type
            
        Returns:
            List of matching rules
        """
        return [rule for rule in self.fate_rules if rule.cell_type == cell_type]
    
    def get_rules_by_priority(self, min_priority: int = 0) -> List[CellFateRule]:
        """Get rules with minimum priority.
        
        Args:
            min_priority: Minimum priority threshold
            
        Returns:
            List of rules with priority >= min_priority
        """
        return [rule for rule in self.fate_rules if rule.priority >= min_priority]
    
    def get_high_confidence_rules(self, min_confidence: float = 0.9) -> List[CellFateRule]:
        """Get high-confidence rules.
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of high-confidence rules
        """
        return [rule for rule in self.fate_rules if rule.confidence >= min_confidence]
    
    def get_rules_requiring_gene(self, gene_name: str) -> List[CellFateRule]:
        """Get rules that require specific gene.
        
        Args:
            gene_name: Gene name
            
        Returns:
            List of rules requiring this gene
        """
        return [rule for rule in self.fate_rules if gene_name in rule.required_genes]
    
    def get_rules_excluding_gene(self, gene_name: str) -> List[CellFateRule]:
        """Get rules that exclude specific gene.
        
        Args:
            gene_name: Gene name
            
        Returns:
            List of rules excluding this gene
        """
        return [rule for rule in self.fate_rules if gene_name in rule.excluded_genes]
    
    def add_rule(self, rule: CellFateRule) -> None:
        """Add new cell fate rule.
        
        Args:
            rule: Cell fate rule to add
        """
        # Check for duplicate cell type with same priority
        existing = [r for r in self.fate_rules 
                   if r.cell_type == rule.cell_type and r.priority == rule.priority]
        
        if existing:
            logger.warning(f"Rule for {rule.cell_type.value} with priority {rule.priority} already exists")
        
        self.fate_rules.append(rule)
        logger.info(f"Added rule for {rule.cell_type.value} (priority {rule.priority})")
    
    def remove_rule(self, cell_type: NeuralCellType, priority: int) -> bool:
        """Remove cell fate rule.
        
        Args:
            cell_type: Cell type
            priority: Rule priority
            
        Returns:
            True if removed, False if not found
        """
        for i, rule in enumerate(self.fate_rules):
            if rule.cell_type == cell_type and rule.priority == priority:
                removed_rule = self.fate_rules.pop(i)
                logger.info(f"Removed rule for {removed_rule.cell_type.value}")
                return True
        return False
    
    def get_available_cell_types(self) -> List[str]:
        """Get list of all available cell types."""
        return list(set(rule.cell_type.value for rule in self.fate_rules))
    
    def get_required_genes(self) -> List[str]:
        """Get list of all genes required by any rule."""
        genes = set()
        for rule in self.fate_rules:
            genes.update(rule.required_genes)
        return sorted(list(genes))
    
    def get_excluded_genes(self) -> List[str]:
        """Get list of all genes excluded by any rule."""
        genes = set()
        for rule in self.fate_rules:
            genes.update(rule.excluded_genes)
        return sorted(list(genes))
    
    def validate_rules(self) -> Dict[str, Any]:
        """Validate rule database consistency.
        
        Returns:
            Dictionary of validation results
        """
        validation = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        # Check for missing cell types
        expected_types = set(NeuralCellType)
        defined_types = set(rule.cell_type for rule in self.fate_rules)
        missing_types = expected_types - defined_types
        
        if missing_types:
            validation["warnings"].append(f"Missing rules for cell types: {[t.value for t in missing_types]}")
        
        # Check for conflicting priorities
        priority_conflicts = {}
        for rule in self.fate_rules:
            key = (rule.cell_type, rule.priority)
            if key in priority_conflicts:
                validation["errors"].append(f"Duplicate priority {rule.priority} for {rule.cell_type.value}")
                validation["is_valid"] = False
            else:
                priority_conflicts[key] = rule
        
        # Statistics
        validation["statistics"] = {
            "total_rules": len(self.fate_rules),
            "cell_types_covered": len(defined_types),
            "unique_required_genes": len(self.get_required_genes()),
            "unique_excluded_genes": len(self.get_excluded_genes()),
            "high_confidence_rules": len(self.get_high_confidence_rules()),
            "priority_range": (
                min(rule.priority for rule in self.fate_rules) if self.fate_rules else 0,
                max(rule.priority for rule in self.fate_rules) if self.fate_rules else 0
            )
        }
        
        return validation
    
    def get_database_summary(self) -> Dict[str, Any]:
        """Get comprehensive database summary."""
        return {
            "rule_count": len(self.fate_rules),
            "cell_types": self.get_available_cell_types(),
            "required_genes": self.get_required_genes(),
            "excluded_genes": self.get_excluded_genes(),
            "confidence_stats": {
                "mean": sum(rule.confidence for rule in self.fate_rules) / len(self.fate_rules),
                "min": min(rule.confidence for rule in self.fate_rules),
                "max": max(rule.confidence for rule in self.fate_rules)
            } if self.fate_rules else {},
            "priority_distribution": {
                rule.priority: len([r for r in self.fate_rules if r.priority == rule.priority])
                for rule in self.fate_rules
            },
            "validation": self.validate_rules()
        }
