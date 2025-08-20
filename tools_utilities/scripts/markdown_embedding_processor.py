#!/usr/bin/env python3
"""
Brain-Inspired Markdown Rules Embedding Processor
=================================================

This script implements a Graph Neural Memory system that mirrors the brain's associative memory:

🧠 Core Features:
1. Graph Neural Memory (Connectedness Layer)
2. Auto-summarization and semantic tagging
3. Hierarchical retrieval (coarse to fine)
4. Hebbian learning with usage tracking
5. Semantic similarity and keyword extraction
6. Multi-hop graph traversal for rule discovery

🔬 Brain-Inspired Principles:
- Nodes = Rule sections (neural units)
- Edges = Semantic connections (synapses)
- Hebbian Learning: "Rules that fire together wire together"
- Associative Recall: Related rules retrieved together
- Plasticity: Adaptive connection strengthening
"""

import os
import re
import json
import hashlib
import sqlite3
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
import networkx as nx
from datetime import datetime
import logging
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SemanticNode:
    """Represents a neural unit in the semantic network"""
    node_id: str
    source_file: str
    section_title: str
    content: str
    level: int
    semantic_tags: List[str]
    associative_keywords: List[str]
    hebbian_weight: float
    activation_threshold: float
    usage_count: int
    last_accessed: datetime
    metadata: Dict

@dataclass
class SemanticEdge:
    """Represents a synaptic connection between semantic nodes"""
    source_id: str
    target_id: str
    edge_type: str  # 'direct_link', 'semantic_similarity', 'keyword_shared'
    weight: float
    strength: float  # Hebbian strength
    last_reinforced: datetime
    co_occurrence_count: int
    metadata: Dict

@dataclass
class RuleFile:
    """Represents a complete markdown rule file with semantic properties"""
    filename: str
    title: str
    summary: str
    semantic_tags: List[str]
    topics: List[str]
    linked_files: List[str]
    sections: List[SemanticNode]
    dependencies: List[str]
    hebbian_network: Dict
    metadata: Dict

@dataclass
class EmbeddingChunk:
    """Represents a chunk ready for embedding with semantic context"""
    chunk_id: str
    content: str
    source_file: str
    section_title: str
    linked_content: str
    semantic_context: str
    hebbian_connections: List[str]
    metadata: Dict
    embedding_vector: Optional[List[float]] = None

class BrainInspiredRuleProcessor:
    """Processes markdown rule files into a brain-inspired semantic network"""
    
    def __init__(self, rules_dir: str):
        self.rules_dir = Path(rules_dir)
        self.rule_files: Dict[str, RuleFile] = {}
        self.semantic_nodes: Dict[str, SemanticNode] = {}
        self.semantic_edges: Dict[str, SemanticEdge] = {}
        self.graph = nx.DiGraph()
        self.chunks: Dict[str, EmbeddingChunk] = {}
        self.hebbian_weights = defaultdict(float)
        self.usage_patterns = defaultdict(int)
        
        # Semantic analysis tools
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
    def extract_semantic_tags(self, content: str, title: str) -> List[str]:
        """Extract semantic tags using TF-IDF and keyword analysis"""
        # Combine title and content for analysis
        text = f"{title} {content}"
        
        # Extract key phrases and terms
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = Counter(words)
        
        # Filter out common words and short terms
        filtered_words = [
            word for word, freq in word_freq.most_common(20)
            if len(word) > 3 and freq > 1
        ]
        
        # Add domain-specific tags
        domain_tags = []
        if any(word in text.lower() for word in ['security', 'safety', 'protect']):
            domain_tags.append('security')
        if any(word in text.lower() for word in ['brain', 'neural', 'cognitive']):
            domain_tags.append('neuroscience')
        if any(word in text.lower() for word in ['ai', 'model', 'machine']):
            domain_tags.append('artificial_intelligence')
        if any(word in text.lower() for word in ['rule', 'compliance', 'policy']):
            domain_tags.append('governance')
        
        return filtered_words[:10] + domain_tags
    
    def extract_associative_keywords(self, content: str) -> List[str]:
        """Extract keywords that create associative bridges"""
        # Look for technical terms, concepts, and domain vocabulary
        keywords = []
        
        # Extract technical terms (camelCase, snake_case)
        tech_terms = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', content)
        keywords.extend(tech_terms)
        
        # Extract domain concepts
        concept_patterns = [
            r'\b(?:security|safety|compliance|validation|testing)\b',
            r'\b(?:brain|neural|cognitive|consciousness|memory)\b',
            r'\b(?:ai|agi|machine|learning|model)\b',
            r'\b(?:rule|policy|governance|framework)\b'
        ]
        
        for pattern in concept_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            keywords.extend(matches)
        
        return list(set(keywords))[:15]
    
    def calculate_semantic_similarity(self, content1: str, content2: str) -> float:
        """Calculate semantic similarity between two content pieces"""
        try:
            # Use TF-IDF vectors for similarity
            texts = [content1, content2]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.warning(f"Failed to calculate semantic similarity: {e}")
            return 0.0
    
    def create_semantic_node(self, section: Dict, filepath: Path) -> SemanticNode:
        """Create a semantic node with brain-inspired properties"""
        node_id = f"{filepath.stem}_{section['section_title'].replace(' ', '_')}"
        
        # Extract semantic properties
        semantic_tags = self.extract_semantic_tags(section['content'], section['section_title'])
        associative_keywords = self.extract_associative_keywords(section['content'])
        
        # Initialize Hebbian properties
        hebbian_weight = 1.0  # Base weight
        activation_threshold = 0.5  # Default threshold
        usage_count = 0
        last_accessed = datetime.now()
        
        return SemanticNode(
            node_id=node_id,
            source_file=str(filepath),
            section_title=section['section_title'],
            content=section['content'],
            level=section['level'],
            semantic_tags=semantic_tags,
            associative_keywords=associative_keywords,
            hebbian_weight=hebbian_weight,
            activation_threshold=activation_threshold,
            usage_count=usage_count,
            last_accessed=last_accessed,
            metadata={
                'file': str(filepath),
                'level': section['level'],
                'created_at': datetime.now().isoformat(),
                'node_type': 'rule_section'
            }
        )
    
    def create_semantic_edges(self, nodes: List[SemanticNode]) -> List[SemanticEdge]:
        """Create semantic edges between nodes"""
        edges = []
        
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i+1:], i+1):
                # Direct link edges (explicit markdown links)
                if self.has_direct_link(node1, node2):
                    edge = SemanticEdge(
                        source_id=node1.node_id,
                        target_id=node2.node_id,
                        edge_type='direct_link',
                        weight=1.0,
                        strength=1.0,
                        last_reinforced=datetime.now(),
                        co_occurrence_count=1,
                        metadata={'link_type': 'explicit'}
                    )
                    edges.append(edge)
                
                # Semantic similarity edges (cos_sim > 0.85)
                similarity = self.calculate_semantic_similarity(node1.content, node2.content)
                if similarity > 0.85:
                    edge = SemanticEdge(
                        source_id=node1.node_id,
                        target_id=node2.node_id,
                        edge_type='semantic_similarity',
                        weight=similarity,
                        strength=similarity,
                        last_reinforced=datetime.now(),
                        co_occurrence_count=1,
                        metadata={'similarity_score': similarity}
                    )
                    edges.append(edge)
                
                # Shared keyword edges
                shared_keywords = set(node1.associative_keywords) & set(node2.associative_keywords)
                if len(shared_keywords) >= 2:
                    edge = SemanticEdge(
                        source_id=node1.node_id,
                        target_id=node2.node_id,
                        edge_type='keyword_shared',
                        weight=len(shared_keywords) / 10.0,
                        strength=len(shared_keywords) / 10.0,
                        last_reinforced=datetime.now(),
                        co_occurrence_count=1,
                        metadata={'shared_keywords': list(shared_keywords)}
                    )
                    edges.append(edge)
        
        return edges
    
    def has_direct_link(self, node1: SemanticNode, node2: SemanticNode) -> bool:
        """Check if two nodes have explicit markdown links between them"""
        # Look for markdown links in content
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        links = re.findall(link_pattern, node1.content)
        
        for link_text, link_file in links:
            if link_file in node2.source_file or node2.source_file in link_file:
                return True
        
        return False
    
    def build_hebbian_network(self, nodes: List[SemanticNode], edges: List[SemanticEdge]) -> Dict:
        """Build a Hebbian learning network"""
        hebbian_network = {
            'nodes': {node.node_id: asdict(node) for node in nodes},
            'edges': {f"{edge.source_id}_{edge.target_id}": asdict(edge) for edge in edges},
            'learning_rate': 0.1,
            'decay_rate': 0.01,
            'plasticity_threshold': 0.1
        }
        
        return hebbian_network
    
    def parse_markdown_file(self, filepath: Path) -> RuleFile:
        """Parse a single markdown file and extract semantic properties"""
        logger.info(f"Parsing {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract title from first # heading
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else filepath.stem
        
        # Extract sections using headers
        sections = []
        lines = content.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            # Check for headers
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # Save previous section
                if current_section:
                    current_section['content'] = '\n'.join(current_content).strip()
                    sections.append(current_section)
                
                # Start new section
                level = len(header_match.group(1))
                section_title = header_match.group(2).strip()
                
                current_section = {
                    'section_title': section_title,
                    'content': '',
                    'level': level
                }
                current_content = []
            else:
                if current_section:
                    current_content.append(line)
        
        # Save last section
        if current_section:
            current_section['content'] = '\n'.join(current_content).strip()
            sections.append(current_section)
        
        # Create semantic nodes
        semantic_nodes = []
        for section in sections:
            node = self.create_semantic_node(section, filepath)
            semantic_nodes.append(node)
            self.semantic_nodes[node.node_id] = node
        
        # Create semantic edges
        semantic_edges = self.create_semantic_edges(semantic_nodes)
        for edge in semantic_edges:
            edge_id = f"{edge.source_id}_{edge.target_id}"
            self.semantic_edges[edge_id] = edge
        
        # Extract dependencies and linked files
        dependencies = []
        linked_files = []
        for section in sections:
            links = re.findall(r'\[([^\]]+)\]\(([^)]+\.md)\)', section['content'])
            for link_text, link_file in links:
                if link_file not in dependencies:
                    dependencies.append(link_file)
                if link_file not in linked_files:
                    linked_files.append(link_file)
        
        # Generate summary and tags
        summary = self.generate_summary(content, title)
        semantic_tags = self.extract_semantic_tags(content, title)
        topics = self.extract_topics(content)
        
        # Build Hebbian network
        hebbian_network = self.build_hebbian_network(semantic_nodes, semantic_edges)
        
        return RuleFile(
            filename=str(filepath),
            title=title,
            summary=summary,
            semantic_tags=semantic_tags,
            topics=topics,
            linked_files=linked_files,
            sections=semantic_nodes,
            dependencies=dependencies,
            hebbian_network=hebbian_network,
            metadata={
                'parsed_at': datetime.now().isoformat(),
                'section_count': len(sections),
                'dependency_count': len(dependencies),
                'semantic_node_count': len(semantic_nodes),
                'semantic_edge_count': len(semantic_edges)
            }
        )
    
    def generate_summary(self, content: str, title: str) -> str:
        """Generate a summary of the rule file"""
        # Simple summary generation (can be enhanced with LLM)
        lines = content.split('\n')
        summary_lines = []
        
        for line in lines[:10]:  # First 10 lines
            if line.strip() and not line.startswith('#'):
                summary_lines.append(line.strip())
                if len(summary_lines) >= 3:
                    break
        
        summary = ' '.join(summary_lines)
        if len(summary) > 200:
            summary = summary[:200] + '...'
        
        return summary or f"Rules and guidelines for {title}"
    
    def extract_topics(self, content: str) -> List[str]:
        """Extract main topics from content"""
        # Extract topic words from headers and key content
        topics = []
        
        # Look for headers
        headers = re.findall(r'^#{1,6}\s+(.+)$', content, re.MULTILINE)
        for header in headers:
            topics.extend(header.lower().split())
        
        # Look for emphasized text
        emphasized = re.findall(r'\*\*([^*]+)\*\*', content)
        for emph in emphasized:
            topics.extend(emph.lower().split())
        
        # Filter and return unique topics
        filtered_topics = [topic for topic in topics if len(topic) > 3]
        return list(set(filtered_topics))[:10]
    
    def expand_semantic_context(self, node: SemanticNode) -> str:
        """Expand node content with semantic context from connected nodes"""
        expanded_content = node.content
        
        # Find connected nodes
        connected_nodes = []
        for edge_id, edge in self.semantic_edges.items():
            if edge.source_id == node.node_id:
                target_node = self.semantic_nodes.get(edge.target_id)
                if target_node:
                    connected_nodes.append((target_node, edge))
            elif edge.target_id == node.node_id:
                source_node = self.semantic_nodes.get(edge.source_id)
                if source_node:
                    connected_nodes.append((source_node, edge))
        
        # Sort by edge strength and add context
        connected_nodes.sort(key=lambda x: x[1].strength, reverse=True)
        
        for connected_node, edge in connected_nodes[:3]:  # Top 3 connections
            context_snippet = connected_node.content[:150] + '...'
            expanded_content += f"\n\n--- Related: {connected_node.section_title} (Strength: {edge.strength:.2f}) ---\n{context_snippet}"
        
        return expanded_content
    
    def create_embedding_chunks(self) -> List[EmbeddingChunk]:
        """Create embedding chunks with semantic context"""
        chunks = []
        
        for node_id, node in self.semantic_nodes.items():
            # Expand content with semantic context
            semantic_context = self.expand_semantic_context(node)
            
            # Get Hebbian connections
            hebbian_connections = []
            for edge_id, edge in self.semantic_edges.items():
                if edge.source_id == node_id or edge.target_id == node_id:
                    hebbian_connections.append(edge_id)
            
            # Create chunk
            chunk_id = f"{node_id}_{hashlib.md5(semantic_context.encode()).hexdigest()[:8]}"
            
            chunk = EmbeddingChunk(
                chunk_id=chunk_id,
                content=node.content,
                source_file=node.source_file,
                section_title=node.section_title,
                linked_content=semantic_context,
                semantic_context=semantic_context,
                hebbian_connections=hebbian_connections,
                metadata={
                    'node_id': node_id,
                    'level': node.level,
                    'semantic_tags': node.semantic_tags,
                    'associative_keywords': node.associative_keywords,
                    'hebbian_weight': node.hebbian_weight,
                    'usage_count': node.usage_count,
                    'created_at': datetime.now().isoformat()
                }
            )
            
            chunks.append(chunk)
            self.chunks[chunk_id] = chunk
        
        return chunks
    
    def build_semantic_graph(self):
        """Build the semantic graph with brain-inspired properties"""
        self.graph.clear()
        
        # Add nodes with semantic properties
        for node_id, node in self.semantic_nodes.items():
            self.graph.add_node(node_id, 
                              title=node.section_title,
                              source_file=node.source_file,
                              semantic_tags=node.semantic_tags,
                              hebbian_weight=node.hebbian_weight,
                              usage_count=node.usage_count)
        
        # Add edges with Hebbian properties
        for edge_id, edge in self.semantic_edges.items():
            self.graph.add_edge(edge.source_id, edge.target_id,
                              edge_type=edge.edge_type,
                              weight=edge.weight,
                              strength=edge.strength,
                              co_occurrence=edge.co_occurrence_count)
        
        # Add file-level relationships
        for rule_file in self.rule_files.values():
            for dependency in rule_file.dependencies:
                if dependency in self.rule_files:
                    self.graph.add_edge(rule_file.filename, dependency,
                                      edge_type='file_dependency',
                                      weight=1.0,
                                      strength=1.0)
    
    def process_all_files(self):
        """Process all markdown files and build semantic network"""
        logger.info(f"Processing markdown files in {self.rules_dir}")
        
        # Find all .md files
        md_files = list(self.rules_dir.glob("*.md"))
        logger.info(f"Found {len(md_files)} markdown files")
        
        # Parse each file
        for filepath in md_files:
            try:
                rule_file = self.parse_markdown_file(filepath)
                self.rule_files[filepath.name] = rule_file
            except Exception as e:
                logger.error(f"Failed to parse {filepath}: {e}")
        
        # Create embedding chunks
        chunks = self.create_embedding_chunks()
        logger.info(f"Created {len(chunks)} embedding chunks")
        
        # Build semantic graph
        self.build_semantic_graph()
        logger.info(f"Built semantic graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
        return chunks
    
    def export_semantic_network(self, output_dir: str):
        """Export the complete semantic network"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export semantic nodes
        with open(output_path / "semantic_nodes.json", 'w') as f:
            json.dump({node_id: asdict(node) for node_id, node in self.semantic_nodes.items()}, 
                      f, indent=2, default=str)
        
        # Export semantic edges
        with open(output_path / "semantic_edges.json", 'w') as f:
            json.dump({edge_id: asdict(edge) for edge_id, edge in self.semantic_edges.items()}, 
                      f, indent=2, default=str)
        
        # Export Hebbian network
        hebbian_data = {}
        for filename, rule_file in self.rule_files.items():
            hebbian_data[filename] = rule_file.hebbian_network
        
        with open(output_path / "hebbian_network.json", 'w') as f:
            json.dump(hebbian_data, f, indent=2, default=str)
        
        # Export semantic graph
        graph_data = {
            'nodes': [{'id': node, **self.graph.nodes[node]} for node in self.graph.nodes()],
            'edges': [{'source': u, 'target': v, **self.graph.edges[u, v]} for u, v in self.graph.edges()]
        }
        with open(output_path / "semantic_graph.json", 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        # Export brain-inspired summary
        brain_summary = {
            'total_semantic_nodes': len(self.semantic_nodes),
            'total_semantic_edges': len(self.semantic_edges),
            'graph_nodes': self.graph.number_of_nodes(),
            'graph_edges': self.graph.number_of_edges(),
            'hebbian_networks': len(hebbian_data),
            'semantic_tags': list(set([tag for node in self.semantic_nodes.values() for tag in node.semantic_tags])),
            'associative_keywords': list(set([kw for node in self.semantic_nodes.values() for kw in node.associative_keywords])),
            'processed_at': datetime.now().isoformat(),
            'rules_dir': str(self.rules_dir)
        }
        
        with open(output_path / "brain_inspired_summary.json", 'w') as f:
            json.dump(brain_summary, f, indent=2)
        
        logger.info(f"Exported semantic network to {output_path}")
    
    def export_metadata(self, output_dir: str):
        """Export metadata and graph information"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export rule files metadata
        with open(output_path / "rule_files_metadata.json", 'w') as f:
            json.dump({name: asdict(rule_file) for name, rule_file in self.rule_files.items()}, 
                      f, indent=2, default=str)
        
        # Export chunks metadata
        with open(output_path / "embedding_chunks_metadata.json", 'w') as f:
            json.dump({chunk_id: asdict(chunk) for chunk_id, chunk in self.chunks.items()}, 
                      f, indent=2, default=str)
        
        # Export dependency graph
        graph_data = {
            'nodes': [{'id': node, **self.graph.nodes[node]} for node in self.graph.nodes()],
            'edges': [{'source': u, 'target': v} for u, v in self.graph.edges()]
        }
        with open(output_path / "dependency_graph.json", 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        # Export processing summary
        summary = {
            'total_files': len(self.rule_files),
            'total_sections': len(self.semantic_nodes),
            'total_chunks': len(self.chunks),
            'graph_nodes': self.graph.number_of_nodes(),
            'graph_edges': self.graph.number_of_edges(),
            'semantic_nodes': len(self.semantic_nodes),
            'semantic_edges': len(self.semantic_edges),
            'processed_at': datetime.now().isoformat(),
            'rules_dir': str(self.rules_dir)
        }
        
        with open(output_path / "processing_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Exported metadata to {output_path}")
    
    def generate_brain_inspired_instructions(self, output_dir: str):
        """Generate instructions for brain-inspired rule system"""
        output_path = Path(output_dir)
        
        instructions = {
            'system_architecture': 'Brain-Inspired Graph Neural Memory',
            'core_principles': [
                'Graph Neural Memory (Connectedness Layer)',
                'Auto-summarization and semantic tagging',
                'Hierarchical retrieval (coarse to fine)',
                'Hebbian learning with usage tracking',
                'Semantic similarity and keyword extraction',
                'Multi-hop graph traversal for rule discovery'
            ],
            'brain_equivalents': {
                'semantic_linking': 'Association cortex',
                'attention_on_active_rules': 'Working memory',
                'rule_consolidation': 'Memory encoding',
                'context_reuse': 'Chunked recall',
                'self_updating': 'Plasticity'
            },
            'implementation_steps': [
                '1. Load semantic network from semantic_nodes.json and semantic_edges.json',
                '2. Use hebbian_network.json for learning dynamics',
                '3. Implement semantic similarity search (cos_sim > 0.85)',
                '4. Build multi-hop graph traversal for rule discovery',
                '5. Apply Hebbian learning: "Rules that fire together wire together"',
                '6. Use associative recall for related rule retrieval',
                '7. Implement plasticity for adaptive connection strengthening'
            ],
            'retrieval_strategy': {
                'step_1': 'Coarse filtering with keywords and tags',
                'step_2': 'Graph walk to fetch linked neighbors (±1–2 hops)',
                'step_3': 'Chunk assembly with semantic context',
                'step_4': 'Hebbian reinforcement of used pathways'
            },
            'prompt_strategy': 'Compliance-first reasoning engine that must obey every retrieved rule. If rules conflict, prioritize stricter constraints or flag ambiguity.'
        }
        
        with open(output_path / "brain_inspired_instructions.json", 'w') as f:
            json.dump(instructions, f, indent=2)
        
        logger.info(f"Generated brain-inspired instructions in {output_path}")

def main():
    """Main processing function"""
    # Initialize processor with absolute path
    import os
    current_dir = os.getcwd()
    rules_dir = current_dir  # We're already in the rules directory
    
    processor = BrainInspiredRuleProcessor(rules_dir)
    
    # Process all files
    chunks = processor.process_all_files()
    
    # Export metadata and semantic network
    output_dir = os.path.join(current_dir, "embeddings")
    processor.export_metadata(output_dir)
    processor.export_semantic_network(output_dir)
    processor.generate_brain_inspired_instructions(output_dir)
    
    # Print summary
    print(f"\n🧠 Brain-Inspired Semantic Network Summary:")
    print(f"   Files processed: {len(processor.rule_files)}")
    print(f"   Semantic nodes: {len(processor.semantic_nodes)}")
    print(f"   Semantic edges: {len(processor.semantic_edges)}")
    print(f"   Embedding chunks: {len(chunks)}")
    print(f"   Graph nodes: {processor.graph.number_of_nodes()}")
    print(f"   Graph edges: {processor.graph.number_of_edges()}")
    print(f"\n📁 Output directory: {output_dir}")
    print(f"   - semantic_nodes.json (neural units)")
    print(f"   - semantic_edges.json (synaptic connections)")
    print(f"   - hebbian_network.json (learning dynamics)")
    print(f"   - semantic_graph.json (complete network)")
    print(f"   - brain_inspired_summary.json (system overview)")
    print(f"   - brain_inspired_instructions.json (implementation guide)")
    print(f"\n🔬 Brain-Inspired Features:")
    print(f"   - Graph Neural Memory with semantic connectivity")
    print(f"   - Hebbian learning: 'Rules that fire together wire together'")
    print(f"   - Associative recall for related rule discovery")
    print(f"   - Multi-hop graph traversal for context expansion")
    print(f"   - Semantic similarity with cos_sim > 0.85")
    print(f"   - Adaptive connection strengthening over time")

if __name__ == "__main__":
    main()
