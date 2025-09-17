#!/usr/bin/env python3
"""Lineage Tree Builder.

Main coordinator for dynamic lineage tree construction including node
creation, relationship tracking, and integration with tree validation.

Integration: Tree building coordinator for lineage tree system
Rationale: Main tree building coordinator with focused responsibilities
"""

from typing import Dict, List, Optional, Set
import logging

from .lineage_tree_types import (
    LineageTreeNode, TreeNodeType, DivisionEvent, LineageTreeMetrics
)
from .tree_structure_validator import TreeStructureValidator

logger = logging.getLogger(__name__)

class LineageTreeBuilder:
    """Builder for dynamic lineage tree construction.
    
    Main coordinator for constructing and maintaining lineage trees
    during simulation with real-time updates and validation.
    """
    
    def __init__(self):
        """Initialize lineage tree builder."""
        self.tree_nodes: Dict[str, LineageTreeNode] = {}
        self.root_nodes: Set[str] = set()
        self.leaf_nodes: Set[str] = set()
        self.validator = TreeStructureValidator()
        
        logger.info("Initialized LineageTreeBuilder")
    
    def create_founding_node(self, cell_id: str, birth_time: float,
                           position: tuple, molecular_markers: Dict[str, float],
                           barcode_ids: List[str]) -> LineageTreeNode:
        """Create founding node for lineage tree."""
        founding_node = LineageTreeNode(
            cell_id=cell_id,
            node_type=TreeNodeType.FOUNDING_CELL,
            parent_id=None,
            children_ids=[],
            generation=0,
            birth_time=birth_time,
            division_time=None,
            division_type=None,
            cell_fate=None,
            molecular_markers=molecular_markers.copy(),
            spatial_position=position,
            lineage_barcode_ids=barcode_ids.copy()
        )
        
        self.tree_nodes[cell_id] = founding_node
        self.root_nodes.add(cell_id)
        self.leaf_nodes.add(cell_id)
        
        logger.info(f"Created founding node: {cell_id}")
        
        return founding_node
    
    def add_division_event(self, parent_id: str, daughter_ids: List[str],
                          division_time: float, division_type: DivisionEvent,
                          daughter_positions: List[tuple],
                          daughter_markers: List[Dict[str, float]],
                          daughter_barcodes: List[List[str]]) -> List[LineageTreeNode]:
        """Add division event to lineage tree."""
        if parent_id not in self.tree_nodes:
            logger.error(f"Parent node {parent_id} not found in tree")
            return []
        
        parent_node = self.tree_nodes[parent_id]
        
        # Update parent node
        parent_node.children_ids = daughter_ids.copy()
        parent_node.division_time = division_time
        parent_node.division_type = division_type
        
        # Remove parent from leaf nodes
        self.leaf_nodes.discard(parent_id)
        
        # Create daughter nodes
        daughter_nodes = []
        
        for i, daughter_id in enumerate(daughter_ids):
            # Determine daughter node type
            if division_type == DivisionEvent.SYMMETRIC_DIFFERENTIATIVE:
                node_type = TreeNodeType.DIFFERENTIATING
            elif division_type == DivisionEvent.ASYMMETRIC and i == 1:
                node_type = TreeNodeType.DIFFERENTIATING
            else:
                node_type = TreeNodeType.PROGENITOR
            
            daughter_node = LineageTreeNode(
                cell_id=daughter_id,
                node_type=node_type,
                parent_id=parent_id,
                children_ids=[],
                generation=parent_node.generation + 1,
                birth_time=division_time,
                division_time=None,
                division_type=None,
                cell_fate=None,
                molecular_markers=daughter_markers[i].copy() if i < len(daughter_markers) else {},
                spatial_position=daughter_positions[i] if i < len(daughter_positions) else (0, 0, 0),
                lineage_barcode_ids=daughter_barcodes[i].copy() if i < len(daughter_barcodes) else []
            )
            
            self.tree_nodes[daughter_id] = daughter_node
            self.leaf_nodes.add(daughter_id)
            daughter_nodes.append(daughter_node)
        
        logger.info(f"Added division: {parent_id} → {len(daughter_ids)} daughters ({division_type.value})")
        
        return daughter_nodes
    
    def update_node_fate_commitment(self, cell_id: str, committed_fate: str,
                                   commitment_time: float) -> bool:
        """Update node with fate commitment information."""
        if cell_id not in self.tree_nodes:
            logger.error(f"Cell {cell_id} not found in tree")
            return False
        
        node = self.tree_nodes[cell_id]
        node.cell_fate = committed_fate
        node.node_type = TreeNodeType.COMMITTED_PROGENITOR
        
        logger.info(f"Cell {cell_id} committed to {committed_fate} at E{commitment_time}")
        
        return True
    
    def update_node_markers(self, cell_id: str, updated_markers: Dict[str, float]) -> bool:
        """Update node molecular markers."""
        if cell_id not in self.tree_nodes:
            return False
        
        node = self.tree_nodes[cell_id]
        node.molecular_markers.update(updated_markers)
        
        return True
    
    def get_lineage_path(self, cell_id: str) -> List[str]:
        """Get complete lineage path from root to specific cell."""
        if cell_id not in self.tree_nodes:
            return []
        
        path = []
        current_id = cell_id
        
        # Traverse up to root
        while current_id is not None:
            path.append(current_id)
            current_node = self.tree_nodes[current_id]
            current_id = current_node.parent_id
        
        # Reverse to get root → target path
        path.reverse()
        
        return path
    
    def get_all_descendants(self, cell_id: str) -> List[str]:
        """Get all descendant cells from specific cell."""
        if cell_id not in self.tree_nodes:
            return []
        
        descendants = []
        
        def collect_descendants(node_id: str):
            node = self.tree_nodes[node_id]
            for child_id in node.children_ids:
                descendants.append(child_id)
                collect_descendants(child_id)
        
        collect_descendants(cell_id)
        
        return descendants
    
    def calculate_tree_metrics(self, root_id: Optional[str] = None) -> LineageTreeMetrics:
        """Calculate comprehensive tree metrics."""
        if root_id:
            nodes_to_analyze = [root_id] + self.get_all_descendants(root_id)
        else:
            nodes_to_analyze = list(self.tree_nodes.keys())
        
        if not nodes_to_analyze:
            return LineageTreeMetrics(0, 0, 0, 0, 0.0, 0.0, 0.0, 0, 0.0)
        
        # Calculate basic metrics
        total_nodes = len(nodes_to_analyze)
        total_divisions = sum(1 for node_id in nodes_to_analyze 
                            if self.tree_nodes[node_id].division_time is not None)
        
        generations = [self.tree_nodes[node_id].generation for node_id in nodes_to_analyze]
        max_generation = max(generations) if generations else 0
        tree_depth = max_generation + 1
        
        # Calculate branching factor
        total_children = sum(len(self.tree_nodes[node_id].children_ids) 
                           for node_id in nodes_to_analyze)
        branching_factor = total_children / max(1, total_divisions)
        
        # Calculate division ratios
        division_types = [self.tree_nodes[node_id].division_type 
                         for node_id in nodes_to_analyze 
                         if self.tree_nodes[node_id].division_type is not None]
        
        symmetric_count = sum(1 for dt in division_types 
                            if dt in [DivisionEvent.SYMMETRIC_PROLIFERATIVE, 
                                     DivisionEvent.SYMMETRIC_DIFFERENTIATIVE])
        asymmetric_count = sum(1 for dt in division_types if dt == DivisionEvent.ASYMMETRIC)
        
        total_typed_divisions = len(division_types)
        symmetric_ratio = symmetric_count / max(1, total_typed_divisions)
        asymmetric_ratio = asymmetric_count / max(1, total_typed_divisions)
        
        # Calculate fate diversity and temporal span
        fates = set(self.tree_nodes[node_id].cell_fate 
                   for node_id in nodes_to_analyze 
                   if self.tree_nodes[node_id].cell_fate is not None)
        fate_diversity = len(fates)
        
        birth_times = [self.tree_nodes[node_id].birth_time for node_id in nodes_to_analyze]
        temporal_span = max(birth_times) - min(birth_times) if birth_times else 0.0
        
        return LineageTreeMetrics(
            total_nodes=total_nodes,
            total_divisions=total_divisions,
            max_generation=max_generation,
            tree_depth=tree_depth,
            branching_factor=branching_factor,
            symmetric_division_ratio=symmetric_ratio,
            asymmetric_division_ratio=asymmetric_ratio,
            fate_diversity=fate_diversity,
            temporal_span=temporal_span
        )
    
    def validate_tree_integrity(self, root_id: Optional[str] = None) -> Dict[str, any]:
        """Validate tree structural integrity."""
        return self.validator.validate_tree_integrity(self.tree_nodes, root_id)