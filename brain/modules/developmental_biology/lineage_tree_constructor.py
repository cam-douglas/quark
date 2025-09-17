#!/usr/bin/env python3
"""Lineage Tree Construction System.

Main coordinator for lineage tree construction including dynamic tree
building during simulation and comprehensive lineage analysis.

Integration: Main lineage tree coordinator for developmental biology
Rationale: Main tree construction coordinator with focused responsibilities
"""

from typing import Dict, List, Optional
import logging

from .lineage_tree_types import DivisionEvent, TreeVisualizationConfig
from .lineage_tree_builder import LineageTreeBuilder
from .lineage_tree_analyzer import LineageTreeAnalyzer
from .neuroepithelial_cells import NeuroepithelialCell

logger = logging.getLogger(__name__)

class LineageTreeConstructor:
    """Main coordinator for lineage tree construction.
    
    Coordinates dynamic lineage tree building during simulation with
    real-time updates and comprehensive analysis capabilities.
    """
    
    def __init__(self):
        """Initialize lineage tree construction system."""
        self.tree_builder = LineageTreeBuilder()
        self.tree_analyzer = LineageTreeAnalyzer(self.tree_builder)
        
        # Tracking state
        self.active_simulations: Dict[str, str] = {}
        self.simulation_trees: Dict[str, List[str]] = {}
        
        logger.info("Initialized LineageTreeConstructor")
    
    def start_lineage_simulation(self, simulation_id: str, 
                                founding_cells: List[NeuroepithelialCell]) -> bool:
        """Start lineage tree construction for simulation."""
        logger.info(f"Starting lineage simulation: {simulation_id}")
        
        root_cell_ids = []
        
        # Create founding nodes for each cell
        for cell in founding_cells:
            barcode_ids = [f"barcode_{i}" for i in range(3)]
            
            founding_node = self.tree_builder.create_founding_node(
                cell_id=cell.cell_id,
                birth_time=cell.birth_time,
                position=cell.position,
                molecular_markers=cell.molecular_markers,
                barcode_ids=barcode_ids
            )
            
            root_cell_ids.append(cell.cell_id)
        
        # Register simulation
        self.simulation_trees[simulation_id] = root_cell_ids
        if root_cell_ids:
            self.active_simulations[simulation_id] = root_cell_ids[0]
        
        logger.info(f"Simulation {simulation_id} started with {len(founding_cells)} founding cells")
        
        return True
    
    def record_cell_division(self, simulation_id: str, parent_cell: NeuroepithelialCell,
                           daughter_cells: List[NeuroepithelialCell],
                           division_type: DivisionEvent) -> bool:
        """Record cell division event in lineage tree."""
        if simulation_id not in self.active_simulations:
            logger.error(f"Simulation {simulation_id} not found")
            return False
        
        # Extract daughter information
        daughter_ids = [cell.cell_id for cell in daughter_cells]
        daughter_positions = [cell.position for cell in daughter_cells]
        daughter_markers = [cell.molecular_markers for cell in daughter_cells]
        daughter_barcodes = [["barcode_1", "barcode_2"] for _ in daughter_cells]
        
        # Record division in tree
        daughter_nodes = self.tree_builder.add_division_event(
            parent_id=parent_cell.cell_id,
            daughter_ids=daughter_ids,
            division_time=parent_cell.current_age_hours / 168.0,
            division_type=division_type,
            daughter_positions=daughter_positions,
            daughter_markers=daughter_markers,
            daughter_barcodes=daughter_barcodes
        )
        
        logger.info(f"Recorded division: {parent_cell.cell_id} → {len(daughter_cells)} daughters")
        
        return len(daughter_nodes) == len(daughter_cells)
    
    def update_cell_fate_commitment(self, simulation_id: str, cell_id: str,
                                   committed_fate: str, commitment_time: float) -> bool:
        """Update cell fate commitment in lineage tree."""
        if simulation_id not in self.active_simulations:
            return False
        
        success = self.tree_builder.update_node_fate_commitment(
            cell_id, committed_fate, commitment_time)
        
        if success:
            logger.info(f"Updated fate commitment: {cell_id} → {committed_fate}")
        
        return success
    
    def get_lineage_tree_for_simulation(self, simulation_id: str) -> Dict[str, any]:
        """Get complete lineage tree for simulation."""
        if simulation_id not in self.simulation_trees:
            return {}
        
        root_ids = self.simulation_trees[simulation_id]
        complete_tree = {}
        
        for root_id in root_ids:
            tree_structure = self._build_tree_structure(root_id)
            complete_tree[root_id] = tree_structure
        
        return complete_tree
    
    def _build_tree_structure(self, root_id: str) -> Dict[str, any]:
        """Build hierarchical tree structure from root."""
        if root_id not in self.tree_builder.tree_nodes:
            return {}
        
        root_node = self.tree_builder.tree_nodes[root_id]
        
        tree_structure = {
            'node_info': {
                'cell_id': root_node.cell_id,
                'node_type': root_node.node_type.value,
                'generation': root_node.generation,
                'birth_time': root_node.birth_time,
                'cell_fate': root_node.cell_fate,
                'division_type': root_node.division_type.value if root_node.division_type else None,
                'children_count': len(root_node.children_ids)
            },
            'children': {}
        }
        
        # Recursively build children
        for child_id in root_node.children_ids:
            tree_structure['children'][child_id] = self._build_tree_structure(child_id)
        
        return tree_structure
    
    def generate_lineage_visualization_data(self, simulation_id: str,
                                          config: TreeVisualizationConfig) -> Dict[str, any]:
        """Generate data for lineage tree visualization."""
        if simulation_id not in self.simulation_trees:
            return {}
        
        root_ids = self.simulation_trees[simulation_id]
        visualization_data = {
            'nodes': [],
            'edges': [],
            'metadata': {
                'simulation_id': simulation_id,
                'root_count': len(root_ids),
                'config': config
            }
        }
        
        # Process each tree
        for root_id in root_ids:
            nodes_to_process = [root_id] + self.tree_builder.get_all_descendants(root_id)
            
            for node_id in nodes_to_process:
                node = self.tree_builder.tree_nodes[node_id]
                
                # Node visualization data
                node_data = {
                    'id': node_id,
                    'label': f"{node_id[:8]}...",
                    'type': node.node_type.value,
                    'generation': node.generation,
                    'birth_time': node.birth_time,
                    'fate': node.cell_fate,
                    'position': node.spatial_position
                }
                
                visualization_data['nodes'].append(node_data)
                
                # Edge visualization data
                if node.parent_id:
                    edge_data = {
                        'source': node.parent_id,
                        'target': node_id,
                        'division_type': node.division_type.value if node.division_type else 'unknown'
                    }
                    
                    visualization_data['edges'].append(edge_data)
        
        return visualization_data
    
    def export_simulation_summary(self, simulation_id: str) -> Dict[str, any]:
        """Export comprehensive simulation summary."""
        if simulation_id not in self.active_simulations:
            return {}
        
        tree_analysis = self.tree_analyzer.export_tree_analysis()
        root_ids = self.simulation_trees[simulation_id]
        
        summary = {
            'simulation_id': simulation_id,
            'root_cells': root_ids,
            'tree_analysis': tree_analysis,
            'simulation_status': {
                'active': simulation_id in self.active_simulations,
                'tree_construction_operational': True,
                'lineage_tracking_functional': tree_analysis['tree_metrics']['total_divisions'] > 0
            }
        }
        
        return summary