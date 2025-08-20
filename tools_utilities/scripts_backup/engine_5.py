"""
Synthesis Engine - Knowledge integration and pattern recognition
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclasses import data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclass, field
from collections import defaultdict
import logging
import json

@dataclass
class KnowledgeNode:
    """Represents a node in the knowledge graph"""
    id: str
    content: Any
    type: str
    connections: List[str] = field(default_factory=list)
    confidence: float = 0.5
    creation_time: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Pattern:
    """Represents a discovered pattern"""
    id: str
    pattern_type: str
    elements: List[str]
    strength: float
    frequency: int
    contexts: List[str] = field(default_factory=list)
    
@dataclass 
class Insight:
    """Represents a synthesized insight"""
    id: str
    content: str
    confidence: float
    supporting_patterns: List[str]
    implications: List[str] = field(default_factory=list)
    novelty_score: float = 0.0

class SynthesisEngine:
    """
    Core synthesis engine for pattern recognition and knowledge integration.
    
    Combines insights from curiosity and exploration to form coherent
    understanding and generate new knowledge.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Knowledge storage
        self.knowledge_graph: Dict[str, KnowledgeNode] = {}
        self.patterns: Dict[str, Pattern] = {}
        self.insights: Dict[str, Insight] = {}
        
        # Synthesis parameters
        self.pattern_threshold = self.config.get('pattern_threshold', 0.6)
        self.insight_threshold = self.config.get('insight_threshold', 0.7)
        self.max_graph_size = self.config.get('max_graph_size', 10000)
        
        # Pattern recognition settings
        self.min_pattern_frequency = self.config.get('min_pattern_frequency', 3)
        self.connection_strength_threshold = self.config.get('connection_threshold', 0.5)
        
    def add_knowledge(self, 
                      content: Any,
                      knowledge_type: str,
                      metadata: Optional[Dict] = None) -> str:
        """
        Add new knowledge to the synthesis system.
        
        Returns the ID of the created knowledge node.
        """
        node_id = f"{knowledge_type}_{len(self.knowledge_graph)}"
        
        # Create knowledge node
        node = KnowledgeNode(
            id=node_id,
            content=content,
            type=knowledge_type,
            creation_time=len(self.knowledge_graph),
            metadata=metadata or {}
        )
        
        # Add to graph
        self.knowledge_graph[node_id] = node
        
        # Find connections with existing knowledge
        self._find_and_create_connections(node)
        
        # Trigger pattern recognition
        self._update_patterns(node)
        
        # Check for new insights
        self._generate_insights()
        
        self.logger.debug(f"Added knowledge node: {node_id} ({knowledge_type})")
        
        # Maintain graph size
        if len(self.knowledge_graph) > self.max_graph_size:
            self._prune_knowledge_graph()
            
        return node_id
        
    def synthesize_insights(self, 
                           focus_area: Optional[str] = None,
                           max_insights: int = 5) -> List[Insight]:
        """
        Generate insights by synthesizing patterns and knowledge.
        
        Args:
            focus_area: Specific area to focus synthesis on
            max_insights: Maximum number of insights to generate
            
        Returns:
            List of generated insights
        """
        candidate_insights = []
        
        # Get relevant patterns
        relevant_patterns = self._get_relevant_patterns(focus_area)
        
        # Generate insights from pattern combinations
        for pattern_combo in self._generate_pattern_combinations(relevant_patterns):
            insight = self._synthesize_insight_from_patterns(pattern_combo)
            if insight and insight.confidence >= self.insight_threshold:
                candidate_insights.append(insight)
                
        # Generate insights from knowledge connections
        strong_connections = self._find_strong_connections(focus_area)
        for connection in strong_connections:
            insight = self._synthesize_insight_from_connection(connection)
            if insight and insight.confidence >= self.insight_threshold:
                candidate_insights.append(insight)
                
        # Rank and filter insights
        ranked_insights = self._rank_insights(candidate_insights)
        
        # Store new insights
        for insight in ranked_insights[:max_insights]:
            self.insights[insight.id] = insight
            
        self.logger.info(f"Generated {len(ranked_insights[:max_insights])} new insights")
        
        return ranked_insights[:max_insights]
        
    def find_patterns(self, 
                      pattern_types: Optional[List[str]] = None,
                      min_strength: float = 0.5) -> List[Pattern]:
        """
        Find patterns in the knowledge graph.
        
        Args:
            pattern_types: Types of patterns to look for
            min_strength: Minimum pattern strength threshold
            
        Returns:
            List of discovered patterns
        """
        discovered_patterns = []
        
        # Sequence patterns
        if not pattern_types or 'sequence' in pattern_types:
            sequence_patterns = self._find_sequence_patterns()
            discovered_patterns.extend(sequence_patterns)
            
        # Clustering patterns  
        if not pattern_types or 'cluster' in pattern_types:
            cluster_patterns = self._find_cluster_patterns()
            discovered_patterns.extend(cluster_patterns)
            
        # Causal patterns
        if not pattern_types or 'causal' in pattern_types:
            causal_patterns = self._find_causal_patterns()
            discovered_patterns.extend(causal_patterns)
            
        # Hierarchical patterns
        if not pattern_types or 'hierarchical' in pattern_types:
            hierarchical_patterns = self._find_hierarchical_patterns()
            discovered_patterns.extend(hierarchical_patterns)
            
        # Filter by strength
        strong_patterns = [
            p for p in discovered_patterns 
            if p.strength >= min_strength
        ]
        
        # Store new patterns
        for pattern in strong_patterns:
            if pattern.id not in self.patterns:
                self.patterns[pattern.id] = pattern
                
        return strong_patterns
        
    def integrate_knowledge(self, 
                           knowledge_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Integrate knowledge from multiple sources.
        
        Returns integration results and conflicts.
        """
        integration_results = {
            'integrated_nodes': [],
            'conflicts': [],
            'new_connections': [],
            'confidence_updates': []
        }
        
        for source in knowledge_sources:
            node_id = self.add_knowledge(
                content=source.get('content'),
                knowledge_type=source.get('type', 'unknown'),
                metadata=source.get('metadata', {})
            )
            
            integration_results['integrated_nodes'].append(node_id)
            
            # Check for conflicts with existing knowledge
            conflicts = self._detect_conflicts(node_id)
            integration_results['conflicts'].extend(conflicts)
            
        # Resolve conflicts and update confidences
        self._resolve_conflicts(integration_results['conflicts'])
        
        return integration_results
        
    def get_knowledge_summary(self, 
                             focus_type: Optional[str] = None) -> Dict[str, Any]:
        """Get a summary of current knowledge state."""
        
        # Filter nodes by type if specified
        if focus_type:
            nodes = [n for n in self.knowledge_graph.values() if n.type == focus_type]
        else:
            nodes = list(self.knowledge_graph.values())
            
        # Calculate statistics
        node_types = defaultdict(int)
        total_connections = 0
        avg_confidence = 0.0
        
        for node in nodes:
            node_types[node.type] += 1
            total_connections += len(node.connections)
            avg_confidence += node.confidence
            
        if nodes:
            avg_confidence /= len(nodes)
            avg_connections = total_connections / len(nodes)
        else:
            avg_connections = 0
            
        return {
            'total_nodes': len(nodes),
            'node_types': dict(node_types),
            'total_patterns': len(self.patterns),
            'total_insights': len(self.insights),
            'average_confidence': avg_confidence,
            'average_connections': avg_connections,
            'strong_patterns': len([p for p in self.patterns.values() if p.strength > 0.7]),
            'high_confidence_insights': len([i for i in self.insights.values() if i.confidence > 0.8])
        }
        
    def _find_and_create_connections(self, new_node: KnowledgeNode):
        """Find and create connections between new node and existing knowledge."""
        for existing_id, existing_node in self.knowledge_graph.items():
            if existing_id == new_node.id:
                continue
                
            # Calculate connection strength
            strength = self._calculate_connection_strength(new_node, existing_node)
            
            if strength >= self.connection_strength_threshold:
                # Create bidirectional connection
                new_node.connections.append(existing_id)
                existing_node.connections.append(new_node.id)
                
    def _calculate_connection_strength(self, 
                                      node1: KnowledgeNode,
                                      node2: KnowledgeNode) -> float:
        """Calculate strength of connection between two knowledge nodes."""
        
        # Type similarity
        type_similarity = 1.0 if node1.type == node2.type else 0.3
        
        # Content similarity (simplified)
        content_similarity = self._calculate_content_similarity(
            node1.content, node2.content
        )
        
        # Metadata similarity
        metadata_similarity = self._calculate_metadata_similarity(
            node1.metadata, node2.metadata
        )
        
        # Weighted combination
        strength = (
            0.4 * content_similarity +
            0.3 * type_similarity +
            0.3 * metadata_similarity
        )
        
        return min(1.0, strength)
        
    def _calculate_content_similarity(self, content1: Any, content2: Any) -> float:
        """Calculate similarity between content objects."""
        # Simple string-based similarity
        str1 = str(content1).lower()
        str2 = str(content2).lower()
        
        if not str1 or not str2:
            return 0.0
            
        # Word overlap
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
        
    def _calculate_metadata_similarity(self, 
                                      meta1: Dict[str, Any],
                                      meta2: Dict[str, Any]) -> float:
        """Calculate similarity between metadata dictionaries."""
        if not meta1 and not meta2:
            return 1.0
        if not meta1 or not meta2:
            return 0.0
            
        # Key overlap
        keys1 = set(meta1.keys())
        keys2 = set(meta2.keys())
        
        key_overlap = len(keys1.intersection(keys2)) / len(keys1.union(keys2))
        
        # Value similarity for common keys
        value_similarity = 0.0
        common_keys = keys1.intersection(keys2)
        
        if common_keys:
            for key in common_keys:
                if meta1[key] == meta2[key]:
                    value_similarity += 1.0
            value_similarity /= len(common_keys)
            
        return (key_overlap + value_similarity) / 2.0
        
    def _update_patterns(self, new_node: KnowledgeNode):
        """Update pattern recognition with new knowledge."""
        # This is a simplified version - can be greatly enhanced
        
        # Look for recurring content patterns
        content_str = str(new_node.content).lower()
        
        # Update pattern frequencies
        for existing_pattern in self.patterns.values():
            # Check if new node matches existing pattern
            if self._node_matches_pattern(new_node, existing_pattern):
                existing_pattern.frequency += 1
                if new_node.id not in existing_pattern.elements:
                    existing_pattern.elements.append(new_node.id)
                    
    def _node_matches_pattern(self, node: KnowledgeNode, pattern: Pattern) -> bool:
        """Check if a node matches an existing pattern."""
        # Simplified pattern matching
        if pattern.pattern_type == 'type_cluster':
            return node.type in pattern.contexts
        elif pattern.pattern_type == 'content_similarity':
            # Check content similarity with pattern elements
            for element_id in pattern.elements[:3]:  # Check first few elements
                if element_id in self.knowledge_graph:
                    element_node = self.knowledge_graph[element_id]
                    similarity = self._calculate_content_similarity(
                        node.content, element_node.content
                    )
                    if similarity > 0.6:
                        return True
        
        return False
        
    def _generate_insights(self):
        """Generate insights from current patterns and knowledge."""
        # This is triggered automatically when new knowledge is added
        # For now, just check if we have enough patterns for insights
        
        if len(self.patterns) >= 3:
            # Try to generate insights from pattern combinations
            recent_patterns = list(self.patterns.values())[-5:]
            self._try_generate_insight_from_patterns(recent_patterns)
            
    def _try_generate_insight_from_patterns(self, patterns: List[Pattern]):
        """Try to generate insights from a set of patterns."""
        if len(patterns) < 2:
            return
            
        # Look for patterns that share elements
        for i, pattern1 in enumerate(patterns):
            for pattern2 in patterns[i+1:]:
                shared_elements = set(pattern1.elements).intersection(set(pattern2.elements))
                
                if len(shared_elements) >= 2:
                    # Potential insight from pattern intersection
                    insight = self._create_insight_from_intersection(
                        pattern1, pattern2, shared_elements
                    )
                    
                    if insight.confidence >= self.insight_threshold:
                        self.insights[insight.id] = insight
                        
    def _create_insight_from_intersection(self, 
                                         pattern1: Pattern,
                                         pattern2: Pattern,
                                         shared_elements: Set[str]) -> Insight:
        """Create insight from pattern intersection."""
        
        insight_content = (
            f"Pattern intersection detected: {pattern1.pattern_type} and "
            f"{pattern2.pattern_type} share {len(shared_elements)} elements. "
            f"This suggests a deeper connection between these domains."
        )
        
        # Calculate confidence based on pattern strengths and overlap
        confidence = (pattern1.strength + pattern2.strength) / 2.0
        confidence *= len(shared_elements) / max(len(pattern1.elements), len(pattern2.elements))
        
        insight = Insight(
            id=f"insight_{len(self.insights)}",
            content=insight_content,
            confidence=confidence,
            supporting_patterns=[pattern1.id, pattern2.id],
            novelty_score=0.7  # Intersection insights are moderately novel
        )
        
        return insight
        
    def _get_relevant_patterns(self, focus_area: Optional[str]) -> List[Pattern]:
        """Get patterns relevant to a focus area."""
        if not focus_area:
            return list(self.patterns.values())
            
        relevant = []
        for pattern in self.patterns.values():
            if (focus_area in pattern.contexts or 
                focus_area in pattern.pattern_type or
                any(focus_area in element for element in pattern.elements)):
                relevant.append(pattern)
                
        return relevant
        
    def _generate_pattern_combinations(self, 
                                      patterns: List[Pattern]) -> List[List[Pattern]]:
        """Generate meaningful combinations of patterns."""
        combinations = []
        
        # Pairs
        for i, pattern1 in enumerate(patterns):
            for pattern2 in patterns[i+1:]:
                combinations.append([pattern1, pattern2])
                
        # Triplets for strong patterns
        strong_patterns = [p for p in patterns if p.strength > 0.7]
        for i, pattern1 in enumerate(strong_patterns):
            for j, pattern2 in enumerate(strong_patterns[i+1:], i+1):
                for pattern3 in strong_patterns[j+1:]:
                    combinations.append([pattern1, pattern2, pattern3])
                    
        return combinations
        
    def _synthesize_insight_from_patterns(self, 
                                         pattern_combo: List[Pattern]) -> Optional[Insight]:
        """Synthesize insight from a combination of patterns."""
        if len(pattern_combo) < 2:
            return None
            
        # Create insight content
        pattern_types = [p.pattern_type for p in pattern_combo]
        avg_strength = sum(p.strength for p in pattern_combo) / len(pattern_combo)
        
        content = (
            f"Synthesis of patterns {', '.join(pattern_types)} reveals "
            f"interconnected structure with strength {avg_strength:.2f}. "
            f"This suggests systemic relationships worth investigating."
        )
        
        confidence = min(avg_strength, 0.9)  # Cap confidence
        
        insight = Insight(
            id=f"synthesis_{len(self.insights)}",
            content=content,
            confidence=confidence,
            supporting_patterns=[p.id for p in pattern_combo],
            novelty_score=min(len(pattern_combo) * 0.2, 1.0)
        )
        
        return insight
        
    def _find_strong_connections(self, focus_area: Optional[str]) -> List[Tuple[str, str]]:
        """Find strong connections in the knowledge graph."""
        strong_connections = []
        
        for node_id, node in self.knowledge_graph.items():
            if focus_area and focus_area not in str(node.content) and focus_area != node.type:
                continue
                
            for connected_id in node.connections:
                if connected_id in self.knowledge_graph:
                    connected_node = self.knowledge_graph[connected_id]
                    strength = self._calculate_connection_strength(node, connected_node)
                    
                    if strength > 0.7:
                        # Avoid duplicates (both directions)
                        pair = tuple(sorted([node_id, connected_id]))
                        if pair not in strong_connections:
                            strong_connections.append(pair)
                            
        return strong_connections
        
    def _synthesize_insight_from_connection(self, 
                                           connection: Tuple[str, str]) -> Optional[Insight]:
        """Synthesize insight from a strong connection."""
        node1_id, node2_id = connection
        
        if node1_id not in self.knowledge_graph or node2_id not in self.knowledge_graph:
            return None
            
        node1 = self.knowledge_graph[node1_id]
        node2 = self.knowledge_graph[node2_id]
        
        strength = self._calculate_connection_strength(node1, node2)
        
        content = (
            f"Strong connection detected between {node1.type} and {node2.type} "
            f"(strength: {strength:.2f}). This relationship may reveal "
            f"important structural insights."
        )
        
        insight = Insight(
            id=f"connection_{len(self.insights)}",
            content=content,
            confidence=strength * 0.9,  # Slightly discounted
            supporting_patterns=[],  # Connection-based, not pattern-based
            novelty_score=strength
        )
        
        return insight
        
    def _rank_insights(self, insights: List[Insight]) -> List[Insight]:
        """Rank insights by importance/quality."""
        
        def insight_score(insight: Insight) -> float:
            score = (
                insight.confidence * 0.4 +
                insight.novelty_score * 0.3 +
                len(insight.supporting_patterns) * 0.1 +
                len(insight.implications) * 0.2
            )
            return score
            
        return sorted(insights, key=insight_score, reverse=True)
        
    def _find_sequence_patterns(self) -> List[Pattern]:
        """Find temporal or logical sequence patterns."""
        # Simplified sequence pattern detection
        sequences = []
        
        # Group nodes by creation time
        time_sorted_nodes = sorted(
            self.knowledge_graph.values(),
            key=lambda n: n.creation_time
        )
        
        # Look for sequences of similar types
        current_sequence = []
        current_type = None
        
        for node in time_sorted_nodes:
            if node.type == current_type:
                current_sequence.append(node.id)
            else:
                if len(current_sequence) >= self.min_pattern_frequency:
                    pattern = Pattern(
                        id=f"sequence_{len(self.patterns)}",
                        pattern_type="temporal_sequence",
                        elements=current_sequence.copy(),
                        strength=min(len(current_sequence) / 10.0, 1.0),
                        frequency=len(current_sequence),
                        contexts=[current_type] if current_type else []
                    )
                    sequences.append(pattern)
                    
                current_sequence = [node.id]
                current_type = node.type
                
        return sequences
        
    def _find_cluster_patterns(self) -> List[Pattern]:
        """Find clustering patterns in the knowledge graph."""
        clusters = []
        
        # Simple clustering by node type
        type_clusters = defaultdict(list)
        
        for node in self.knowledge_graph.values():
            type_clusters[node.type].append(node.id)
            
        for node_type, node_ids in type_clusters.items():
            if len(node_ids) >= self.min_pattern_frequency:
                pattern = Pattern(
                    id=f"cluster_{node_type}_{len(self.patterns)}",
                    pattern_type="type_cluster",
                    elements=node_ids,
                    strength=min(len(node_ids) / 20.0, 1.0),
                    frequency=len(node_ids),
                    contexts=[node_type]
                )
                clusters.append(pattern)
                
        return clusters
        
    def _find_causal_patterns(self) -> List[Pattern]:
        """Find causal relationship patterns."""
        # Simplified causal pattern detection
        # In practice, this would use more sophisticated causal inference
        
        causal_patterns = []
        
        # Look for nodes that frequently appear together in connections
        co_occurrence = defaultdict(int)
        
        for node in self.knowledge_graph.values():
            for connection in node.connections:
                pair = tuple(sorted([node.id, connection]))
                co_occurrence[pair] += 1
                
        for (node1_id, node2_id), frequency in co_occurrence.items():
            if frequency >= self.min_pattern_frequency:
                pattern = Pattern(
                    id=f"causal_{len(self.patterns)}",
                    pattern_type="causal_relationship", 
                    elements=[node1_id, node2_id],
                    strength=min(frequency / 10.0, 1.0),
                    frequency=frequency
                )
                causal_patterns.append(pattern)
                
        return causal_patterns
        
    def _find_hierarchical_patterns(self) -> List[Pattern]:
        """Find hierarchical patterns in knowledge structure."""
        # Simplified hierarchical pattern detection
        hierarchies = []
        
        # Look for nodes with many connections (potential hub nodes)
        hub_nodes = []
        
        for node in self.knowledge_graph.values():
            if len(node.connections) >= 5:  # Threshold for hub status
                hub_nodes.append(node.id)
                
        if len(hub_nodes) >= 2:
            pattern = Pattern(
                id=f"hierarchy_{len(self.patterns)}",
                pattern_type="hierarchical_structure",
                elements=hub_nodes,
                strength=len(hub_nodes) / 10.0,
                frequency=len(hub_nodes)
            )
            hierarchies.append(pattern)
            
        return hierarchies
        
    def _detect_conflicts(self, node_id: str) -> List[Dict[str, Any]]:
        """Detect conflicts between new node and existing knowledge."""
        conflicts = []
        
        if node_id not in self.knowledge_graph:
            return conflicts
            
        new_node = self.knowledge_graph[node_id]
        
        # Check for contradictory information
        for existing_id, existing_node in self.knowledge_graph.items():
            if existing_id == node_id:
                continue
                
            # Simple conflict detection based on content similarity with opposite metadata
            content_sim = self._calculate_content_similarity(
                new_node.content, existing_node.content
            )
            
            if content_sim > 0.7:  # Similar content
                # Check for contradictory metadata or low confidence
                if (existing_node.confidence < 0.3 or new_node.confidence < 0.3):
                    conflicts.append({
                        'type': 'low_confidence_conflict',
                        'nodes': [node_id, existing_id],
                        'similarity': content_sim
                    })
                    
        return conflicts
        
    def _resolve_conflicts(self, conflicts: List[Dict[str, Any]]):
        """Resolve conflicts between knowledge nodes."""
        for conflict in conflicts:
            if conflict['type'] == 'low_confidence_conflict':
                # Update confidence based on conflict resolution
                nodes = conflict['nodes']
                
                for node_id in nodes:
                    if node_id in self.knowledge_graph:
                        node = self.knowledge_graph[node_id]
                        # Reduce confidence for conflicting information
                        node.confidence *= 0.8
                        
    def _prune_knowledge_graph(self):
        """Prune knowledge graph to maintain manageable size."""
        # Remove nodes with lowest confidence first
        sorted_nodes = sorted(
            self.knowledge_graph.values(),
            key=lambda n: n.confidence
        )
        
        # Remove bottom 10%
        nodes_to_remove = sorted_nodes[:len(sorted_nodes) // 10]
        
        for node in nodes_to_remove:
            # Remove connections to this node
            for other_node in self.knowledge_graph.values():
                if node.id in other_node.connections:
                    other_node.connections.remove(node.id)
                    
            # Remove the node
            del self.knowledge_graph[node.id]
            
        self.logger.info(f"Pruned {len(nodes_to_remove)} nodes from knowledge graph")
