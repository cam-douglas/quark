#!/usr/bin/env python3
"""Tree Structure Validator.

Validation system for lineage tree structural integrity including
consistency checks, orphan detection, and temporal validation.

Integration: Validation component for lineage tree system
Rationale: Focused validation logic separated from tree builder
"""

from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class TreeStructureValidator:
    """Validator for lineage tree structural integrity.
    
    Provides comprehensive validation of tree structure including
    parent-child consistency, generation validation, and temporal
    consistency checks.
    """
    
    def __init__(self):
        """Initialize tree structure validator."""
        logger.info("Initialized TreeStructureValidator")
    
    def validate_tree_integrity(self, tree_nodes: Dict[str, any], 
                               root_id: Optional[str] = None) -> Dict[str, any]:
        """Validate lineage tree structural integrity."""
        validation_results = {
            'structural_integrity': True,
            'orphaned_nodes': [],
            'circular_references': [],
            'missing_parents': [],
            'inconsistent_generations': [],
            'temporal_inconsistencies': []
        }
        
        nodes_to_check = self._get_nodes_to_check(tree_nodes, root_id)
        
        for node_id in nodes_to_check:
            node = tree_nodes[node_id]
            
            # Check parent-child consistency
            if node.parent_id is not None:
                if node.parent_id not in tree_nodes:
                    validation_results['missing_parents'].append(node_id)
                    validation_results['structural_integrity'] = False
                else:
                    parent_node = tree_nodes[node.parent_id]
                    if node_id not in parent_node.children_ids:
                        validation_results['orphaned_nodes'].append(node_id)
                        validation_results['structural_integrity'] = False
                    
                    # Check generation consistency
                    if node.generation != parent_node.generation + 1:
                        validation_results['inconsistent_generations'].append(node_id)
                        validation_results['structural_integrity'] = False
                    
                    # Check temporal consistency
                    if node.birth_time < parent_node.birth_time:
                        validation_results['temporal_inconsistencies'].append(node_id)
                        validation_results['structural_integrity'] = False
        
        return validation_results
    
    def _get_nodes_to_check(self, tree_nodes: Dict[str, any], 
                           root_id: Optional[str]) -> List[str]:
        """Get list of nodes to check for validation."""
        if root_id:
            # Get descendants of specific root
            return self._get_all_descendants(tree_nodes, root_id)
        else:
            # Check all nodes
            return list(tree_nodes.keys())
    
    def _get_all_descendants(self, tree_nodes: Dict[str, any], root_id: str) -> List[str]:
        """Get all descendant nodes from root."""
        if root_id not in tree_nodes:
            return []
        
        descendants = [root_id]
        
        def collect_descendants(node_id: str):
            if node_id in tree_nodes:
                node = tree_nodes[node_id]
                for child_id in node.children_ids:
                    descendants.append(child_id)
                    collect_descendants(child_id)
        
        collect_descendants(root_id)
        
        return descendants
