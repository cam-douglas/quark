#!/usr/bin/env python3
"""
🧠 Organic Connectome Enhancer
==============================

Enhances and maintains organic brain-like connectome configurations between all agents
following biological principles and developmental constraints.

Features:
- Biological connectivity patterns (small-world, scale-free)
- Developmental stage-appropriate connections
- Organic growth and pruning algorithms
- Cross-module integration patterns
- Real-time connectivity monitoring
- Adaptive plasticity rules

Author: Quark Brain Simulation Team
Created: 2025-01-21
"""

import os, sys
import json
import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclasses import data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclass, asdict
import scipy.sparse as sp
from scipy.stats import powerlaw, norm
import pandas as pd

# Add quark root to path
QUARK_ROOT = Path(__file__).parent.parent
sys.path.append(str(QUARK_ROOT))

# Import brain modules
try:
    from brain_architecture.neural_core.connectome.connectome_manager import ConnectomeManager
    from brain_architecture.neural_core.connectome.schema import ConnectomeConfig
    from brain_architecture.neural_core.connectome.validators import ensure_small_world, enforce_ei_balance
except ImportError as e:
    print(f"Warning: Could not import connectome modules: {e}")

@dataclass
class BiologicalConstraints:
    """Biological constraints for organic connectome development."""
    stage: str  # fetal, neonate, early_postnatal
    max_degree: int  # Maximum connections per node
    clustering_target: float  # Target clustering coefficient
    path_length_target: float  # Target average path length
    small_world_sigma: float  # Small-world metric target
    modularity_target: float  # Target modularity
    connection_density: float  # Overall connection density
    growth_rate: float  # Rate of new connection formation
    pruning_rate: float  # Rate of connection removal
    plasticity_window: float  # Time window for plasticity
    
@dataclass
class ConnectomeMetrics:
    """Metrics for evaluating connectome quality."""
    clustering_coefficient: float
    average_path_length: float
    small_world_sigma: float
    modularity: float
    density: float
    degree_distribution_fit: float
    biological_plausibility: float
    developmental_appropriateness: float
    cross_module_integration: float

class BiologicalConnectomeGenerator:
    """Generates biologically plausible connectome patterns."""
    
    def __init__(self):
        self.brain_areas = {
            'prefrontal_cortex': {'type': 'cortical', 'layer': 'association', 'excitatory_ratio': 0.8},
            'motor_cortex': {'type': 'cortical', 'layer': 'primary', 'excitatory_ratio': 0.8},
            'somatosensory_cortex': {'type': 'cortical', 'layer': 'primary', 'excitatory_ratio': 0.8},
            'visual_cortex': {'type': 'cortical', 'layer': 'primary', 'excitatory_ratio': 0.8},
            'auditory_cortex': {'type': 'cortical', 'layer': 'primary', 'excitatory_ratio': 0.8},
            'thalamus': {'type': 'subcortical', 'layer': 'relay', 'excitatory_ratio': 0.9},
            'basal_ganglia': {'type': 'subcortical', 'layer': 'action_selection', 'excitatory_ratio': 0.3},
            'hippocampus': {'type': 'archicortex', 'layer': 'memory', 'excitatory_ratio': 0.85},
            'amygdala': {'type': 'subcortical', 'layer': 'emotion', 'excitatory_ratio': 0.7},
            'cerebellum': {'type': 'subcortical', 'layer': 'motor_learning', 'excitatory_ratio': 0.8},
            'brainstem': {'type': 'subcortical', 'layer': 'autonomic', 'excitatory_ratio': 0.6},
            'working_memory': {'type': 'cortical', 'layer': 'association', 'excitatory_ratio': 0.8},
            'default_mode_network': {'type': 'network', 'layer': 'default', 'excitatory_ratio': 0.8},
            'salience_network': {'type': 'network', 'layer': 'attention', 'excitatory_ratio': 0.8},
            'conscious_agent': {'type': 'network', 'layer': 'integration', 'excitatory_ratio': 0.8}
        }
        
        # Developmental stage constraints
        self.stage_constraints = {
            'fetal': BiologicalConstraints(
                stage='fetal',
                max_degree=50,
                clustering_target=0.3,
                path_length_target=3.5,
                small_world_sigma=2.0,
                modularity_target=0.6,
                connection_density=0.1,
                growth_rate=0.8,
                pruning_rate=0.1,
                plasticity_window=0.9
            ),
            'neonate': BiologicalConstraints(
                stage='neonate',
                max_degree=150,
                clustering_target=0.45,
                path_length_target=3.0,
                small_world_sigma=3.0,
                modularity_target=0.5,
                connection_density=0.3,
                growth_rate=0.6,
                pruning_rate=0.3,
                plasticity_window=0.7
            ),
            'early_postnatal': BiologicalConstraints(
                stage='early_postnatal',
                max_degree=300,
                clustering_target=0.6,
                path_length_target=2.5,
                small_world_sigma=5.0,
                modularity_target=0.4,
                connection_density=0.5,
                growth_rate=0.4,
                pruning_rate=0.5,
                plasticity_window=0.5
            )
        }
        
    def create_brain_like_topology(self, nodes: List[str], stage: str = 'fetal') -> nx.Graph:
        """Create brain-like network topology with biological constraints."""
        constraints = self.stage_constraints[stage]
        
        # Initialize graph
        G = nx.Graph()
        G.add_nodes_from(nodes)
        
        # Group nodes by brain areas
        area_nodes = self._group_nodes_by_area(nodes)
        
        # 1. Create intra-area connections (local clustering)
        for area, area_node_list in area_nodes.items():
            if len(area_node_list) > 1:
                self._add_intra_area_connections(G, area_node_list, constraints)
        
        # 2. Create inter-area connections (long-range)
        self._add_inter_area_connections(G, area_nodes, constraints)
        
        # 3. Add small-world properties
        self._enhance_small_world_properties(G, constraints)
        
        # 4. Enforce biological constraints
        self._enforce_biological_constraints(G, constraints)
        
        return G
        
    def _group_nodes_by_area(self, nodes: List[str]) -> Dict[str, List[str]]:
        """Group nodes by brain area based on naming convention."""
        area_nodes = {area: [] for area in self.brain_areas.keys()}
        area_nodes['unknown'] = []
        
        for node in nodes:
            assigned = False
            for area in self.brain_areas.keys():
                if area.lower() in node.lower() or any(keyword in node.lower() 
                    for keyword in area.replace('_', ' ').split()):
                    area_nodes[area].append(node)
                    assigned = True
                    break
            
            if not assigned:
                area_nodes['unknown'].append(node)
        
        # Remove empty groups
        return {area: nodes for area, nodes in area_nodes.items() if nodes}
        
    def _add_intra_area_connections(self, G: nx.Graph, nodes: List[str], constraints: BiologicalConstraints):
        """Add connections within a brain area."""
        n_nodes = len(nodes)
        if n_nodes < 2:
            return
            
        # Create small-world network within area
        # Start with ring lattice
        k = min(6, n_nodes - 1)  # Initial degree
        for i in range(n_nodes):
            for j in range(1, k//2 + 1):
                if i != (i + j) % n_nodes:
                    G.add_edge(nodes[i], nodes[(i + j) % n_nodes])
                if i != (i - j) % n_nodes:
                    G.add_edge(nodes[i], nodes[(i - j) % n_nodes])
        
        # Rewire some edges for small-world properties
        rewiring_prob = 0.1 * constraints.plasticity_window
        edges_to_rewire = list(G.edges())
        
        for u, v in edges_to_rewire:
            if np.random.random() < rewiring_prob:
                # Remove edge and add new random edge
                G.remove_edge(u, v)
                possible_targets = [n for n in nodes if n != u and not G.has_edge(u, n)]
                if possible_targets:
                    new_target = np.random.choice(possible_targets)
                    G.add_edge(u, new_target)
                    
    def _add_inter_area_connections(self, G: nx.Graph, area_nodes: Dict[str, List[str]], 
                                  constraints: BiologicalConstraints):
        """Add connections between brain areas based on known anatomy."""
        
        # Define biologically plausible inter-area connections
        anatomical_connections = {
            'prefrontal_cortex': ['thalamus', 'basal_ganglia', 'working_memory', 'conscious_agent'],
            'thalamus': ['prefrontal_cortex', 'motor_cortex', 'somatosensory_cortex', 'visual_cortex', 'auditory_cortex'],
            'basal_ganglia': ['prefrontal_cortex', 'motor_cortex', 'thalamus'],
            'hippocampus': ['prefrontal_cortex', 'default_mode_network', 'conscious_agent'],
            'visual_cortex': ['thalamus', 'prefrontal_cortex', 'conscious_agent'],
            'auditory_cortex': ['thalamus', 'prefrontal_cortex', 'conscious_agent'],
            'motor_cortex': ['thalamus', 'basal_ganglia', 'cerebellum', 'somatosensory_cortex'],
            'cerebellum': ['motor_cortex', 'prefrontal_cortex', 'brainstem'],
            'amygdala': ['prefrontal_cortex', 'hippocampus', 'brainstem'],
            'working_memory': ['prefrontal_cortex', 'thalamus', 'conscious_agent'],
            'default_mode_network': ['prefrontal_cortex', 'hippocampus', 'conscious_agent'],
            'salience_network': ['prefrontal_cortex', 'thalamus', 'conscious_agent'],
            'conscious_agent': ['prefrontal_cortex', 'thalamus', 'working_memory', 'default_mode_network', 
                              'salience_network', 'visual_cortex', 'auditory_cortex', 'hippocampus']
        }
        
        # Add inter-area connections
        for source_area, target_areas in anatomical_connections.items():
            if source_area not in area_nodes:
                continue
                
            source_nodes = area_nodes[source_area]
            
            for target_area in target_areas:
                if target_area not in area_nodes:
                    continue
                    
                target_nodes = area_nodes[target_area]
                
                # Calculate connection probability based on anatomical strength
                base_prob = constraints.connection_density * 0.1  # Lower for inter-area
                
                # Adjust probability based on area types
                source_info = self.brain_areas.get(source_area, {})
                target_info = self.brain_areas.get(target_area, {})
                
                # Higher probability for cortical-subcortical connections
                if (source_info.get('type') == 'cortical' and target_info.get('type') == 'subcortical') or \
                   (source_info.get('type') == 'subcortical' and target_info.get('type') == 'cortical'):
                    base_prob *= 2.0
                
                # Add connections
                for source_node in source_nodes:
                    for target_node in target_nodes:
                        if np.random.random() < base_prob:
                            G.add_edge(source_node, target_node)
                            
    def _enhance_small_world_properties(self, G: nx.Graph, constraints: BiologicalConstraints):
        """Enhance small-world properties of the network."""
        current_clustering = nx.average_clustering(G)
        target_clustering = constraints.clustering_target
        
        if current_clustering < target_clustering:
            # Add more local connections to increase clustering
            nodes = list(G.nodes())
            for _ in range(int(len(nodes) * 0.1)):  # Add 10% more local connections
                node = np.random.choice(nodes)
                neighbors = list(G.neighbors(node))
                
                if len(neighbors) >= 2:
                    # Try to connect two neighbors (triangle formation)
                    neighbor1, neighbor2 = np.random.choice(neighbors, 2, replace=False)
                    if not G.has_edge(neighbor1, neighbor2):
                        G.add_edge(neighbor1, neighbor2)
                        
    def _enforce_biological_constraints(self, G: nx.Graph, constraints: BiologicalConstraints):
        """Enforce biological constraints on the network."""
        
        # 1. Enforce maximum degree constraint
        nodes_to_prune = []
        for node in G.nodes():
            degree = G.degree(node)
            if degree > constraints.max_degree:
                # Remove random edges to meet constraint
                neighbors = list(G.neighbors(node))
                edges_to_remove = degree - constraints.max_degree
                targets_to_remove = np.random.choice(neighbors, edges_to_remove, replace=False)
                for target in targets_to_remove:
                    if G.has_edge(node, target):
                        G.remove_edge(node, target)
        
        # 2. Ensure connectivity
        if not nx.is_connected(G):
            # Connect disconnected components
            components = list(nx.connected_components(G))
            if len(components) > 1:
                main_component = max(components, key=len)
                for component in components:
                    if component != main_component:
                        # Connect to main component
                        source = np.random.choice(list(component))
                        target = np.random.choice(list(main_component))
                        G.add_edge(source, target)
                        
    def apply_developmental_changes(self, G: nx.Graph, from_stage: str, to_stage: str) -> nx.Graph:
        """Apply developmental changes when transitioning between stages."""
        from_constraints = self.stage_constraints[from_stage]
        to_constraints = self.stage_constraints[to_stage]
        
        # Create new graph as copy
        new_G = G.copy()
        
        # Growth phase - add new connections
        growth_factor = to_constraints.growth_rate
        current_edges = len(new_G.edges())
        target_new_edges = int(current_edges * growth_factor * 0.1)  # 10% growth per transition
        
        nodes = list(new_G.nodes())
        for _ in range(target_new_edges):
            node1, node2 = np.random.choice(nodes, 2, replace=False)
            if not new_G.has_edge(node1, node2):
                # Add edge with probability based on distance and area compatibility
                if self._should_connect_nodes(node1, node2, to_constraints):
                    new_G.add_edge(node1, node2)
        
        # Pruning phase - remove weak connections
        pruning_factor = to_constraints.pruning_rate
        current_edges = list(new_G.edges())
        edges_to_remove = int(len(current_edges) * pruning_factor * 0.05)  # 5% pruning per transition
        
        # Preferentially remove edges that don't contribute to small-world properties
        edge_scores = []
        for edge in current_edges:
            score = self._calculate_edge_importance(new_G, edge)
            edge_scores.append((edge, score))
        
        # Sort by importance (lower score = less important)
        edge_scores.sort(key=lambda x: x[1])
        
        for i in range(min(edges_to_remove, len(edge_scores))):
            edge_to_remove = edge_scores[i][0]
            new_G.remove_edge(*edge_to_remove)
            
            # Ensure connectivity is maintained
            if not nx.is_connected(new_G):
                # Re-add the edge if it breaks connectivity
                new_G.add_edge(*edge_to_remove)
        
        # Enforce new stage constraints
        self._enforce_biological_constraints(new_G, to_constraints)
        
        return new_G
        
    def _should_connect_nodes(self, node1: str, node2: str, constraints: BiologicalConstraints) -> bool:
        """Determine if two nodes should be connected based on biological principles."""
        
        # Extract area information
        area1 = self._get_node_area(node1)
        area2 = self._get_node_area(node2)
        
        # Same area - higher probability
        if area1 == area2:
            return np.random.random() < 0.3
        
        # Check anatomical plausibility
        area1_info = self.brain_areas.get(area1, {})
        area2_info = self.brain_areas.get(area2, {})
        
        # Cortical-subcortical connections are common
        if (area1_info.get('type') == 'cortical' and area2_info.get('type') == 'subcortical') or \
           (area1_info.get('type') == 'subcortical' and area2_info.get('type') == 'cortical'):
            return np.random.random() < 0.1
        
        # Network-level connections (conscious agent connects to many areas)
        if 'conscious' in node1.lower() or 'conscious' in node2.lower():
            return np.random.random() < 0.2
        
        # Default low probability for other connections
        return np.random.random() < 0.05
        
    def _get_node_area(self, node: str) -> str:
        """Extract brain area from node name."""
        for area in self.brain_areas.keys():
            if area.lower() in node.lower():
                return area
        return 'unknown'
        
    def _calculate_edge_importance(self, G: nx.Graph, edge: Tuple[str, str]) -> float:
        """Calculate importance of an edge for network properties."""
        u, v = edge
        
        # Calculate local clustering contribution
        u_neighbors = set(G.neighbors(u))
        v_neighbors = set(G.neighbors(v))
        common_neighbors = len(u_neighbors & v_neighbors)
        
        # Calculate betweenness centrality contribution
        try:
            # Remove edge temporarily
            G.remove_edge(u, v)
            path_length_without = nx.average_shortest_path_length(G) if nx.is_connected(G) else float('inf')
            G.add_edge(u, v)  # Add back
            
            path_length_with = nx.average_shortest_path_length(G) if nx.is_connected(G) else float('inf')
            
            path_importance = abs(path_length_without - path_length_with)
        except:
            path_importance = 0.0
        
        # Combine metrics (higher score = more important)
        importance = common_neighbors * 0.5 + path_importance * 0.5
        
        return importance

class OrganicConnectomeEnhancer:
    """Main class for enhancing and maintaining organic connectome configurations."""
    
    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or QUARK_ROOT
        self.connectome_dir = self.base_dir / 'brain_modules' / 'connectome'
        self.enhancement_dir = self.base_dir / 'training' / 'connectome_enhancements'
        self.enhancement_dir.mkdir(parents=True, exist_ok=True)
        
        self.setup_logging()
        
        # Initialize components
        self.bio_generator = BiologicalConnectomeGenerator()
        self.current_connectome = None
        self.enhancement_history = []
        
        # Try to load existing connectome
        try:
            self.connectome_manager = ConnectomeManager()
            self.load_current_connectome()
        except Exception as e:
            self.logger.warning(f"Could not load connectome manager: {e}")
            self.connectome_manager = None
            
    def setup_logging(self):
        """Setup logging for connectome enhancer."""
        self.logger = logging.getLogger("organic_connectome_enhancer")
        
    def load_current_connectome(self):
        """Load current connectome configuration."""
        try:
            connectome_file = self.connectome_dir / 'exports' / 'connectome.json'
            if connectome_file.exists():
                with open(connectome_file, 'r') as f:
                    data = json.load(f)
                    
                # Convert to NetworkX graph
                self.current_connectome = nx.Graph()
                
                # Add nodes
                for node_data in data.get('nodes', []):
                    node_id = node_data.get('id', '')
                    self.current_connectome.add_node(node_id, **node_data)
                
                # Add edges
                for edge_data in data.get('links', []):
                    source = edge_data.get('source', '')
                    target = edge_data.get('target', '')
                    weight = edge_data.get('weight', 1.0)
                    self.current_connectome.add_edge(source, target, weight=weight)
                    
                self.logger.info(f"Loaded connectome with {len(self.current_connectome.nodes())} nodes and {len(self.current_connectome.edges())} edges")
                
        except Exception as e:
            self.logger.error(f"Error loading current connectome: {e}")
            self.current_connectome = None
            
    def analyze_current_connectome(self) -> ConnectomeMetrics:
        """Analyze current connectome and calculate biological metrics."""
        if not self.current_connectome or len(self.current_connectome.nodes()) == 0:
            return ConnectomeMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        G = self.current_connectome
        
        # Basic network metrics
        try:
            clustering = nx.average_clustering(G)
        except:
            clustering = 0.0
            
        try:
            if nx.is_connected(G):
                avg_path_length = nx.average_shortest_path_length(G)
            else:
                # Calculate for largest component
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                avg_path_length = nx.average_shortest_path_length(subgraph)
        except:
            avg_path_length = float('inf')
            
        # Small-world sigma
        try:
            # Compare to random graph
            n = len(G.nodes())
            m = len(G.edges())
            p = 2 * m / (n * (n - 1)) if n > 1 else 0
            
            random_graph = nx.erdos_renyi_graph(n, p)
            random_clustering = nx.average_clustering(random_graph)
            random_path_length = nx.average_shortest_path_length(random_graph) if nx.is_connected(random_graph) else avg_path_length
            
            if random_clustering > 0 and random_path_length > 0:
                sigma = (clustering / random_clustering) / (avg_path_length / random_path_length)
            else:
                sigma = 0.0
        except:
            sigma = 0.0
            
        # Modularity
        try:
            # Use community detection
            communities = nx.community.greedy_modularity_communities(G)
            modularity = nx.community.modularity(G, communities)
        except:
            modularity = 0.0
            
        # Density
        density = nx.density(G)
        
        # Degree distribution fit (test for scale-free properties)
        degrees = [G.degree(node) for node in G.nodes()]
        if degrees:
            try:
                # Fit power law
                from scipy import stats
                degree_hist, bin_edges = np.histogram(degrees, bins=20)
                degree_hist = degree_hist[degree_hist > 0]  # Remove zeros
                if len(degree_hist) > 3:
                    # Simple correlation test
                    x = np.arange(1, len(degree_hist) + 1)
                    log_x = np.log(x)
                    log_y = np.log(degree_hist)
                    correlation, _ = stats.pearsonr(log_x, log_y)
                    degree_fit = abs(correlation)
                else:
                    degree_fit = 0.0
            except:
                degree_fit = 0.0
        else:
            degree_fit = 0.0
            
        # Biological plausibility (composite score)
        biological_plausibility = self._calculate_biological_plausibility(G)
        
        # Developmental appropriateness
        developmental_appropriateness = self._calculate_developmental_appropriateness(G)
        
        # Cross-module integration
        cross_module_integration = self._calculate_cross_module_integration(G)
        
        return ConnectomeMetrics(
            clustering_coefficient=clustering,
            average_path_length=avg_path_length,
            small_world_sigma=sigma,
            modularity=modularity,
            density=density,
            degree_distribution_fit=degree_fit,
            biological_plausibility=biological_plausibility,
            developmental_appropriateness=developmental_appropriateness,
            cross_module_integration=cross_module_integration
        )
        
    def _calculate_biological_plausibility(self, G: nx.Graph) -> float:
        """Calculate biological plausibility score."""
        score = 0.0
        factors = 0
        
        # 1. Small-world properties (biological brains are small-world)
        try:
            clustering = nx.average_clustering(G)
            if 0.3 <= clustering <= 0.7:  # Typical range for brain networks
                score += 1.0
            else:
                score += max(0, 1.0 - abs(clustering - 0.5) / 0.5)
            factors += 1
        except:
            pass
            
        # 2. Degree distribution (should not be too uniform)
        degrees = [G.degree(node) for node in G.nodes()]
        if degrees:
            degree_variance = np.var(degrees)
            mean_degree = np.mean(degrees)
            cv = np.sqrt(degree_variance) / mean_degree if mean_degree > 0 else 0
            if 0.5 <= cv <= 2.0:  # Reasonable coefficient of variation
                score += 1.0
            else:
                score += max(0, 1.0 - abs(cv - 1.0) / 1.0)
            factors += 1
            
        # 3. Modularity (brain networks are modular)
        try:
            communities = nx.community.greedy_modularity_communities(G)
            modularity = nx.community.modularity(G, communities)
            if 0.3 <= modularity <= 0.7:  # Typical range
                score += 1.0
            else:
                score += max(0, 1.0 - abs(modularity - 0.5) / 0.5)
            factors += 1
        except:
            pass
            
        return score / factors if factors > 0 else 0.0
        
    def _calculate_developmental_appropriateness(self, G: nx.Graph) -> float:
        """Calculate developmental appropriateness score."""
        # This would ideally compare against known developmental patterns
        # For now, use network complexity as proxy
        
        n_nodes = len(G.nodes())
        n_edges = len(G.edges())
        
        if n_nodes == 0:
            return 0.0
            
        # Calculate network efficiency
        density = nx.density(G)
        
        # Developmental appropriateness based on complexity
        # Fetal: low complexity, Neonate: medium, Early postnatal: high
        if density < 0.2:
            return 1.0  # Appropriate for fetal stage
        elif density < 0.4:
            return 0.8  # Appropriate for neonate stage
        elif density < 0.6:
            return 0.9  # Appropriate for early postnatal
        else:
            return 0.6  # May be overly complex
            
    def _calculate_cross_module_integration(self, G: nx.Graph) -> float:
        """Calculate cross-module integration score."""
        # Group nodes by brain areas
        area_nodes = {}
        for node in G.nodes():
            area = self.bio_generator._get_node_area(node)
            if area not in area_nodes:
                area_nodes[area] = []
            area_nodes[area].append(node)
        
        if len(area_nodes) <= 1:
            return 0.0
            
        # Calculate inter-area connections
        total_inter_connections = 0
        total_possible_inter = 0
        
        areas = list(area_nodes.keys())
        for i, area1 in enumerate(areas):
            for j, area2 in enumerate(areas[i+1:], i+1):
                nodes1 = area_nodes[area1]
                nodes2 = area_nodes[area2]
                
                # Count actual connections
                actual_connections = 0
                for n1 in nodes1:
                    for n2 in nodes2:
                        if G.has_edge(n1, n2):
                            actual_connections += 1
                            
                total_inter_connections += actual_connections
                total_possible_inter += len(nodes1) * len(nodes2)
        
        if total_possible_inter == 0:
            return 0.0
            
        integration_density = total_inter_connections / total_possible_inter
        
        # Normalize to 0-1 range (typical brain integration is around 5-15%)
        normalized_score = min(1.0, integration_density / 0.1)
        
        return normalized_score
        
    def enhance_connectome(self, target_stage: str = 'neonate') -> nx.Graph:
        """Enhance current connectome to be more biologically plausible."""
        self.logger.info(f"Enhancing connectome for {target_stage} stage")
        
        if not self.current_connectome:
            self.logger.warning("No current connectome loaded, creating new one")
            # Create basic connectome from available brain modules
            nodes = [f"{area}:n{i}" for area in self.bio_generator.brain_areas.keys() for i in range(10)]
            self.current_connectome = self.bio_generator.create_brain_like_topology(nodes, target_stage)
        
        # Analyze current state
        current_metrics = self.analyze_current_connectome()
        self.logger.info(f"Current metrics: {asdict(current_metrics)}")
        
        # Create enhanced connectome
        nodes = list(self.current_connectome.nodes())
        enhanced_connectome = self.bio_generator.create_brain_like_topology(nodes, target_stage)
        
        # Preserve some existing connections that are already good
        preservation_rate = 0.7  # Preserve 70% of existing connections that meet criteria
        
        for edge in self.current_connectome.edges():
            if np.random.random() < preservation_rate:
                # Check if edge is biologically plausible
                if self._is_edge_biologically_plausible(edge[0], edge[1]):
                    if not enhanced_connectome.has_edge(edge[0], edge[1]):
                        weight = self.current_connectome.edges[edge].get('weight', 1.0)
                        enhanced_connectome.add_edge(edge[0], edge[1], weight=weight)
        
        # Analyze enhanced connectome
        enhanced_metrics = self.analyze_current_connectome()
        
        # Save enhancement record
        enhancement_record = {
            'timestamp': datetime.now().isoformat(),
            'target_stage': target_stage,
            'original_metrics': asdict(current_metrics),
            'enhanced_metrics': asdict(enhanced_metrics),
            'nodes_count': len(enhanced_connectome.nodes()),
            'edges_count': len(enhanced_connectome.edges())
        }
        
        self.enhancement_history.append(enhancement_record)
        
        # Update current connectome
        self.current_connectome = enhanced_connectome
        
        self.logger.info(f"Enhanced connectome: {len(enhanced_connectome.nodes())} nodes, {len(enhanced_connectome.edges())} edges")
        self.logger.info(f"Enhanced metrics: {asdict(enhanced_metrics)}")
        
        return enhanced_connectome
        
    def _is_edge_biologically_plausible(self, node1: str, node2: str) -> bool:
        """Check if an edge is biologically plausible."""
        return self.bio_generator._should_connect_nodes(node1, node2, 
                                                      self.bio_generator.stage_constraints['neonate'])
        
    def apply_developmental_progression(self, stages: List[str]) -> List[nx.Graph]:
        """Apply developmental progression through multiple stages."""
        self.logger.info(f"Applying developmental progression: {' -> '.join(stages)}")
        
        progression_results = []
        current_graph = self.current_connectome
        
        for i, stage in enumerate(stages):
            if i == 0:
                # First stage - enhance to target
                enhanced_graph = self.enhance_connectome(stage)
            else:
                # Subsequent stages - apply developmental changes
                previous_stage = stages[i-1]
                enhanced_graph = self.bio_generator.apply_developmental_changes(
                    current_graph, previous_stage, stage
                )
            
            # Analyze stage result
            stage_metrics = self.analyze_current_connectome()
            
            progression_results.append({
                'stage': stage,
                'graph': enhanced_graph.copy(),
                'metrics': stage_metrics,
                'timestamp': datetime.now().isoformat()
            })
            
            current_graph = enhanced_graph
            self.current_connectome = enhanced_graph
            
            self.logger.info(f"Completed {stage} stage: {asdict(stage_metrics)}")
        
        return [result['graph'] for result in progression_results]
        
    def save_enhanced_connectome(self, filename: str = None) -> str:
        """Save enhanced connectome to file."""
        if not self.current_connectome:
            raise ValueError("No connectome to save")
            
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'enhanced_connectome_{timestamp}.json'
            
        output_file = self.enhancement_dir / filename
        
        # Convert NetworkX graph to JSON format
        connectome_data = {
            'directed': False,
            'multigraph': False,
            'graph': {
                'enhancement_timestamp': datetime.now().isoformat(),
                'nodes_count': len(self.current_connectome.nodes()),
                'edges_count': len(self.current_connectome.edges())
            },
            'nodes': [],
            'links': []
        }
        
        # Add nodes
        for node in self.current_connectome.nodes(data=True):
            node_data = {
                'id': node[0],
                **node[1]  # Include node attributes
            }
            connectome_data['nodes'].append(node_data)
        
        # Add edges
        for edge in self.current_connectome.edges(data=True):
            edge_data = {
                'source': edge[0],
                'target': edge[1],
                'weight': edge[2].get('weight', 1.0),
                **{k: v for k, v in edge[2].items() if k != 'weight'}
            }
            connectome_data['links'].append(edge_data)
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(connectome_data, f, indent=2)
            
        self.logger.info(f"Saved enhanced connectome to {output_file}")
        return str(output_file)
        
    def create_enhancement_visualization(self) -> str:
        """Create visualization of connectome enhancement."""
        if not self.current_connectome:
            return None
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🧠 Organic Connectome Enhancement Analysis', fontsize=16, fontweight='bold')
        
        # 1. Network topology
        pos = nx.spring_layout(self.current_connectome, k=3, iterations=50, seed=42)
        
        # Color nodes by brain area
        node_colors = []
        area_color_map = {
            'prefrontal': '#FF6B35', 'motor': '#4ECDC4', 'sensory': '#45B7D1',
            'visual': '#96CEB4', 'auditory': '#FFEAA7', 'thalamus': '#DDA0DD',
            'basal': '#F39C12', 'hippocampus': '#E74C3C', 'amygdala': '#9B59B6',
            'cerebellum': '#1ABC9C', 'brainstem': '#34495E', 'working': '#F1C40F',
            'default': '#E67E22', 'salience': '#3498DB', 'conscious': '#E91E63',
            'unknown': '#95A5A6'
        }
        
        for node in self.current_connectome.nodes():
            area = self.bio_generator._get_node_area(node)
            color = area_color_map.get(area.split('_')[0], area_color_map['unknown'])
            node_colors.append(color)
        
        nx.draw_networkx_nodes(self.current_connectome, pos, node_color=node_colors, 
                              node_size=30, alpha=0.8, ax=axes[0, 0])
        nx.draw_networkx_edges(self.current_connectome, pos, alpha=0.3, width=0.5, ax=axes[0, 0])
        axes[0, 0].set_title('Enhanced Connectome Topology')
        axes[0, 0].axis('off')
        
        # 2. Degree distribution
        degrees = [self.current_connectome.degree(node) for node in self.current_connectome.nodes()]
        axes[0, 1].hist(degrees, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_xlabel('Node Degree')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Degree Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Enhancement metrics over time
        if self.enhancement_history:
            timestamps = [datetime.fromisoformat(h['timestamp']) for h in self.enhancement_history]
            clustering_scores = [h['enhanced_metrics']['clustering_coefficient'] for h in self.enhancement_history]
            biological_scores = [h['enhanced_metrics']['biological_plausibility'] for h in self.enhancement_history]
            
            axes[1, 0].plot(timestamps, clustering_scores, 'o-', label='Clustering', color='blue')
            axes[1, 0].plot(timestamps, biological_scores, 's-', label='Biological Plausibility', color='green')
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_title('Enhancement Progress')
            axes[1, 0].legend()
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Brain area connectivity matrix
        area_nodes = {}
        for node in self.current_connectome.nodes():
            area = self.bio_generator._get_node_area(node)
            if area not in area_nodes:
                area_nodes[area] = []
            area_nodes[area].append(node)
        
        areas = list(area_nodes.keys())
        connectivity_matrix = np.zeros((len(areas), len(areas)))
        
        for i, area1 in enumerate(areas):
            for j, area2 in enumerate(areas):
                if i != j:
                    nodes1 = area_nodes[area1]
                    nodes2 = area_nodes[area2]
                    connections = 0
                    for n1 in nodes1:
                        for n2 in nodes2:
                            if self.current_connectome.has_edge(n1, n2):
                                connections += 1
                    if len(nodes1) > 0 and len(nodes2) > 0:
                        connectivity_matrix[i, j] = connections / (len(nodes1) * len(nodes2))
        
        im = axes[1, 1].imshow(connectivity_matrix, cmap='viridis', aspect='auto')
        axes[1, 1].set_xticks(range(len(areas)))
        axes[1, 1].set_yticks(range(len(areas)))
        axes[1, 1].set_xticklabels([area[:8] for area in areas], rotation=45)
        axes[1, 1].set_yticklabels([area[:8] for area in areas])
        axes[1, 1].set_title('Inter-Area Connectivity')
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        viz_file = self.enhancement_dir / f'connectome_enhancement_{timestamp}.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Created enhancement visualization: {viz_file}")
        return str(viz_file)
        
    def get_enhancement_summary(self) -> Dict[str, Any]:
        """Get summary of connectome enhancements."""
        if not self.current_connectome:
            return {}
            
        current_metrics = self.analyze_current_connectome()
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'current_connectome': {
                'nodes': len(self.current_connectome.nodes()),
                'edges': len(self.current_connectome.edges()),
                'metrics': asdict(current_metrics)
            },
            'enhancement_history': self.enhancement_history,
            'biological_plausibility_score': current_metrics.biological_plausibility,
            'developmental_appropriateness': current_metrics.developmental_appropriateness,
            'cross_module_integration': current_metrics.cross_module_integration,
            'recommendations': self._generate_enhancement_recommendations(current_metrics)
        }
        
        return summary
        
    def _generate_enhancement_recommendations(self, metrics: ConnectomeMetrics) -> List[str]:
        """Generate recommendations for further enhancement."""
        recommendations = []
        
        if metrics.clustering_coefficient < 0.3:
            recommendations.append("Increase local clustering by adding more triangular connections")
            
        if metrics.small_world_sigma < 2.0:
            recommendations.append("Enhance small-world properties by balancing local and long-range connections")
            
        if metrics.modularity < 0.3:
            recommendations.append("Improve modularity by strengthening intra-area connections")
            
        if metrics.cross_module_integration < 0.1:
            recommendations.append("Increase cross-module integration for better consciousness emergence")
            
        if metrics.biological_plausibility < 0.7:
            recommendations.append("Apply more stringent biological constraints")
            
        if not recommendations:
            recommendations.append("Connectome appears well-optimized for current stage")
            
        return recommendations

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Organic Connectome Enhancer')
    parser.add_argument('--enhance', action='store_true', help='Enhance current connectome')
    parser.add_argument('--stage', type=str, choices=['fetal', 'neonate', 'early_postnatal'], 
                       default='neonate', help='Target developmental stage')
    parser.add_argument('--progression', action='store_true', help='Apply full developmental progression')
    parser.add_argument('--analyze', action='store_true', help='Analyze current connectome')
    parser.add_argument('--visualize', action='store_true', help='Create visualization')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize enhancer
    enhancer = OrganicConnectomeEnhancer()
    
    if args.analyze:
        # Analyze current connectome
        metrics = enhancer.analyze_current_connectome()
        print("🧠 Current Connectome Analysis:")
        print(f"  Clustering Coefficient: {metrics.clustering_coefficient:.3f}")
        print(f"  Average Path Length: {metrics.average_path_length:.3f}")
        print(f"  Small-World Sigma: {metrics.small_world_sigma:.3f}")
        print(f"  Modularity: {metrics.modularity:.3f}")
        print(f"  Density: {metrics.density:.3f}")
        print(f"  Biological Plausibility: {metrics.biological_plausibility:.3f}")
        print(f"  Developmental Appropriateness: {metrics.developmental_appropriateness:.3f}")
        print(f"  Cross-Module Integration: {metrics.cross_module_integration:.3f}")
    
    if args.enhance:
        # Enhance connectome
        enhanced_graph = enhancer.enhance_connectome(args.stage)
        output_file = enhancer.save_enhanced_connectome()
        print(f"✅ Enhanced connectome saved to: {output_file}")
        
        # Show improvement
        summary = enhancer.get_enhancement_summary()
        print(f"📊 Enhancement complete:")
        print(f"  Nodes: {summary['current_connectome']['nodes']}")
        print(f"  Edges: {summary['current_connectome']['edges']}")
        print(f"  Biological Plausibility: {summary['biological_plausibility_score']:.3f}")
    
    if args.progression:
        # Apply full developmental progression
        stages = ['fetal', 'neonate', 'early_postnatal']
        graphs = enhancer.apply_developmental_progression(stages)
        
        print(f"✅ Applied developmental progression through {len(stages)} stages")
        for i, stage in enumerate(stages):
            print(f"  {stage}: {len(graphs[i].nodes())} nodes, {len(graphs[i].edges())} edges")
    
    if args.visualize:
        # Create visualization
        viz_file = enhancer.create_enhancement_visualization()
        if viz_file:
            print(f"📈 Created visualization: {viz_file}")
    
    # Print summary
    summary = enhancer.get_enhancement_summary()
    if summary.get('recommendations'):
        print("\n💡 Recommendations:")
        for rec in summary['recommendations']:
            print(f"  - {rec}")

if __name__ == '__main__':
    main()

