#!/usr/bin/env python3
"""Lineage Tree Analyzer.

Analysis system for lineage trees including tree metrics calculation,
lineage pattern analysis, and developmental progression assessment
with comprehensive tree structure evaluation.

Integration: Analysis component for lineage tree system
Rationale: Focused analysis logic separated from tree construction
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

from .lineage_tree_types import LineageTreeNode, LineageTreeMetrics, DivisionEvent
from .lineage_tree_builder import LineageTreeBuilder

logger = logging.getLogger(__name__)

class LineageTreeAnalyzer:
    """Analyzer for lineage tree patterns and metrics.
    
    Provides comprehensive analysis of lineage trees including
    developmental patterns, division statistics, fate progression,
    and temporal dynamics during embryonic development.
    """
    
    def __init__(self, tree_builder: LineageTreeBuilder):
        """Initialize lineage tree analyzer.
        
        Args:
            tree_builder: Lineage tree builder instance
        """
        self.tree_builder = tree_builder
        
        logger.info("Initialized LineageTreeAnalyzer")
    
    def analyze_division_patterns(self, root_id: Optional[str] = None) -> Dict[str, any]:
        """Analyze division patterns in lineage tree.
        
        Args:
            root_id: Root node for analysis (None for all trees)
            
        Returns:
            Division pattern analysis
        """
        nodes_to_analyze = self._get_analysis_nodes(root_id)
        
        # Collect division events
        division_events = []
        for node_id in nodes_to_analyze:
            node = self.tree_builder.tree_nodes[node_id]
            if node.division_type is not None:
                division_events.append({
                    'node_id': node_id,
                    'division_type': node.division_type,
                    'division_time': node.division_time,
                    'generation': node.generation,
                    'children_count': len(node.children_ids)
                })
        
        # Analyze patterns
        if not division_events:
            return {'no_divisions': True}
        
        # Division type distribution
        type_counts = {}
        for event in division_events:
            div_type = event['division_type']
            type_counts[div_type.value] = type_counts.get(div_type.value, 0) + 1
        
        total_divisions = len(division_events)
        type_frequencies = {dt: count/total_divisions for dt, count in type_counts.items()}
        
        # Temporal division analysis
        division_times = [event['division_time'] for event in division_events]
        temporal_analysis = {
            'earliest_division': min(division_times),
            'latest_division': max(division_times),
            'division_span': max(division_times) - min(division_times),
            'average_division_time': np.mean(division_times),
            'division_frequency': total_divisions / (max(division_times) - min(division_times)) if len(set(division_times)) > 1 else 0
        }
        
        # Generation-based analysis
        generation_divisions = {}
        for event in division_events:
            gen = event['generation']
            if gen not in generation_divisions:
                generation_divisions[gen] = []
            generation_divisions[gen].append(event)
        
        generation_analysis = {}
        for gen, events in generation_divisions.items():
            generation_analysis[f'generation_{gen}'] = {
                'division_count': len(events),
                'avg_children': np.mean([event['children_count'] for event in events]),
                'division_types': [event['division_type'].value for event in events]
            }
        
        return {
            'total_divisions': total_divisions,
            'division_type_frequencies': type_frequencies,
            'temporal_analysis': temporal_analysis,
            'generation_analysis': generation_analysis,
            'division_events': division_events
        }
    
    def analyze_fate_progression(self, root_id: Optional[str] = None) -> Dict[str, any]:
        """Analyze cell fate progression in lineage tree.
        
        Args:
            root_id: Root node for analysis
            
        Returns:
            Fate progression analysis
        """
        nodes_to_analyze = self._get_analysis_nodes(root_id)
        
        # Collect fate information
        fate_commitments = []
        uncommitted_cells = []
        
        for node_id in nodes_to_analyze:
            node = self.tree_builder.tree_nodes[node_id]
            
            if node.cell_fate is not None:
                fate_commitments.append({
                    'cell_id': node_id,
                    'fate': node.cell_fate,
                    'generation': node.generation,
                    'commitment_time': node.birth_time  # Simplified
                })
            else:
                uncommitted_cells.append({
                    'cell_id': node_id,
                    'generation': node.generation,
                    'node_type': node.node_type.value
                })
        
        # Analyze fate distribution
        fate_counts = {}
        for commitment in fate_commitments:
            fate = commitment['fate']
            fate_counts[fate] = fate_counts.get(fate, 0) + 1
        
        total_committed = len(fate_commitments)
        fate_frequencies = {fate: count/total_committed for fate, count in fate_counts.items()} if total_committed > 0 else {}
        
        # Analyze commitment timing
        if fate_commitments:
            commitment_times = [fc['commitment_time'] for fc in fate_commitments]
            timing_analysis = {
                'earliest_commitment': min(commitment_times),
                'latest_commitment': max(commitment_times),
                'commitment_span': max(commitment_times) - min(commitment_times),
                'average_commitment_time': np.mean(commitment_times)
            }
        else:
            timing_analysis = {}
        
        # Generation-based fate analysis
        generation_fates = {}
        for commitment in fate_commitments:
            gen = commitment['generation']
            if gen not in generation_fates:
                generation_fates[gen] = []
            generation_fates[gen].append(commitment['fate'])
        
        generation_fate_analysis = {}
        for gen, fates in generation_fates.items():
            unique_fates = set(fates)
            generation_fate_analysis[f'generation_{gen}'] = {
                'committed_cells': len(fates),
                'unique_fates': len(unique_fates),
                'fate_diversity': list(unique_fates)
            }
        
        return {
            'total_committed_cells': total_committed,
            'total_uncommitted_cells': len(uncommitted_cells),
            'commitment_ratio': total_committed / len(nodes_to_analyze) if nodes_to_analyze else 0,
            'fate_frequencies': fate_frequencies,
            'timing_analysis': timing_analysis,
            'generation_fate_analysis': generation_fate_analysis,
            'uncommitted_by_generation': self._group_by_generation(uncommitted_cells)
        }
    
    def _get_analysis_nodes(self, root_id: Optional[str]) -> List[str]:
        """Get nodes for analysis based on root specification."""
        if root_id:
            return [root_id] + self.tree_builder.get_all_descendants(root_id)
        else:
            return list(self.tree_builder.tree_nodes.keys())
    
    def _group_by_generation(self, cells: List[Dict]) -> Dict[int, int]:
        """Group cells by generation and count."""
        generation_counts = {}
        for cell in cells:
            gen = cell['generation']
            generation_counts[gen] = generation_counts.get(gen, 0) + 1
        
        return generation_counts
    
    def export_tree_analysis(self, root_id: Optional[str] = None) -> Dict[str, any]:
        """Export comprehensive tree analysis.
        
        Args:
            root_id: Root node for analysis
            
        Returns:
            Complete tree analysis
        """
        # Calculate tree metrics
        tree_metrics = self.tree_builder.calculate_tree_metrics(root_id)
        
        # Analyze division patterns
        division_analysis = self.analyze_division_patterns(root_id)
        
        # Analyze fate progression
        fate_analysis = self.analyze_fate_progression(root_id)
        
        # Validate tree integrity
        integrity_validation = self.tree_builder.validate_tree_integrity(root_id)
        
        analysis = {
            'tree_metrics': {
                'total_nodes': tree_metrics.total_nodes,
                'total_divisions': tree_metrics.total_divisions,
                'max_generation': tree_metrics.max_generation,
                'tree_depth': tree_metrics.tree_depth,
                'branching_factor': tree_metrics.branching_factor,
                'temporal_span_weeks': tree_metrics.temporal_span
            },
            'division_patterns': division_analysis,
            'fate_progression': fate_analysis,
            'tree_integrity': integrity_validation,
            'summary_statistics': {
                'nodes_per_generation': tree_metrics.total_nodes / max(1, tree_metrics.tree_depth),
                'divisions_per_week': tree_metrics.total_divisions / max(0.1, tree_metrics.temporal_span),
                'fate_commitment_efficiency': fate_analysis.get('commitment_ratio', 0.0),
                'tree_completeness': 1.0 if integrity_validation['structural_integrity'] else 0.0
            }
        }
        
        return analysis
