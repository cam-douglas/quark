#!/usr/bin/env python3
"""
Embedding Usage Example
=======================

This script demonstrates how to use the generated markdown rule embeddings
for RAG (Retrieval-Augmented Generation) and rule discovery.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import networkx as nx

class RuleEmbeddingManager:
    """Manages and queries rule embeddings"""
    
    def __init__(self, embeddings_dir: str):
        self.embeddings_dir = Path(embeddings_dir)
        self.chunks = {}
        self.graph = nx.DiGraph()
        self.rule_files = {}
        
        # Load generated data
        self.load_embeddings()
        self.load_dependency_graph()
        self.load_rule_files()
    
    def load_embeddings(self):
        """Load embedding chunks metadata"""
        chunks_file = self.embeddings_dir / "embedding_chunks_metadata.json"
        if chunks_file.exists():
            with open(chunks_file, 'r') as f:
                self.chunks = json.load(f)
            print(f"Loaded {len(self.chunks)} embedding chunks")
    
    def load_dependency_graph(self):
        """Load dependency graph"""
        graph_file = self.embeddings_dir / "dependency_graph.json"
        if graph_file.exists():
            with open(graph_file, 'r') as f:
                graph_data = json.load(f)
            
            # Reconstruct graph
            for node in graph_data['nodes']:
                self.graph.add_node(node['id'], **{k: v for k, v in node.items() if k != 'id'})
            
            for edge in graph_data['edges']:
                self.graph.add_edge(edge['source'], edge['target'])
            
            print(f"Loaded dependency graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def load_rule_files(self):
        """Load rule files metadata"""
        rules_file = self.embeddings_dir / "rule_files_metadata.json"
        if rules_file.exists():
            with open(rules_file, 'r') as f:
                self.rule_files = json.load(f)
            print(f"Loaded {len(self.rule_files)} rule files")
    
    def search_rules(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search for rules containing the query"""
        results = []
        
        for chunk_id, chunk in self.chunks.items():
            # Simple text search (can be enhanced with semantic similarity)
            if query.lower() in chunk['content'].lower() or query.lower() in chunk['section_title'].lower():
                results.append({
                    'chunk_id': chunk_id,
                    'source_file': chunk['source_file'],
                    'section_title': chunk['section_title'],
                    'content_preview': chunk['content'][:200] + '...',
                    'relevance_score': self.calculate_relevance(query, chunk)
                })
        
        # Sort by relevance and return top results
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:max_results]
    
    def calculate_relevance(self, query: str, chunk: Dict) -> float:
        """Calculate simple relevance score"""
        query_lower = query.lower()
        content_lower = chunk['content'].lower()
        title_lower = chunk['section_title'].lower()
        
        # Title matches are more relevant
        title_score = sum(1 for word in query_lower.split() if word in title_lower) * 2
        content_score = sum(1 for word in query_lower.split() if word in content_lower)
        
        return title_score + content_score
    
    def find_related_rules(self, rule_file: str, max_depth: int = 2) -> List[str]:
        """Find rules related to a specific rule file"""
        related = []
        
        if rule_file in self.graph:
            # Find nodes reachable from this rule
            reachable = nx.descendants(self.graph, rule_file)
            for node in reachable:
                if len(related) < max_depth * 5:  # Limit results
                    related.append(node)
        
        return related
    
    def get_rule_hierarchy(self) -> Dict:
        """Get the hierarchy of rule categories"""
        hierarchy = {
            'priority_0': [],  # Security Rules
            'priority_1': [],  # Core Framework
            'priority_2': [],  # Domain-Specific
            'priority_3': []   # Implementation
        }
        
        # Categorize based on file names and content
        for filename, rule_data in self.rule_files.items():
            if 'security' in filename.lower():
                hierarchy['priority_0'].append(filename)
            elif 'general' in filename.lower() or 'model-behavior' in filename.lower():
                hierarchy['priority_1'].append(filename)
            elif any(keyword in filename.lower() for keyword in ['brain', 'agi', 'ml', 'cognitive']):
                hierarchy['priority_2'].append(filename)
            else:
                hierarchy['priority_3'].append(filename)
        
        return hierarchy
    
    def get_rule_summary(self) -> Dict:
        """Get a summary of all rules"""
        summary = {
            'total_files': len(self.rule_files),
            'total_chunks': len(self.chunks),
            'graph_nodes': self.graph.number_of_nodes(),
            'graph_edges': self.graph.number_of_edges(),
            'categories': self.get_rule_hierarchy(),
            'most_referenced': self.get_most_referenced_rules()
        }
        
        return summary
    
    def get_most_referenced_rules(self) -> List[Dict]:
        """Get rules that are referenced most often"""
        in_degrees = dict(self.graph.in_degree())
        most_referenced = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return [{'rule': rule, 'references': count} for rule, count in most_referenced]

def main():
    """Demonstrate the embedding manager"""
    print("🧠 Rule Embedding Manager Demo")
    print("=" * 50)
    
    # Initialize manager
    embeddings_dir = "embeddings"
    manager = RuleEmbeddingManager(embeddings_dir)
    
    # Show summary
    summary = manager.get_rule_summary()
    print(f"\n📊 Rule System Summary:")
    print(f"   Total Files: {summary['total_files']}")
    print(f"   Total Chunks: {summary['total_chunks']}")
    print(f"   Graph Nodes: {summary['graph_nodes']}")
    print(f"   Graph Edges: {summary['graph_edges']}")
    
    # Show hierarchy
    print(f"\n🏗️  Rule Hierarchy:")
    for priority, rules in summary['categories'].items():
        print(f"   {priority.replace('_', ' ').title()}: {len(rules)} rules")
        for rule in rules[:3]:  # Show first 3
            print(f"     - {rule}")
        if len(rules) > 3:
            print(f"     ... and {len(rules) - 3} more")
    
    # Show most referenced rules
    print(f"\n🔗 Most Referenced Rules:")
    for ref in summary['most_referenced'][:5]:
        print(f"   {ref['rule']}: {ref['references']} references")
    
    # Demo search
    print(f"\n🔍 Search Demo:")
    search_queries = ["security", "brain simulation", "AGI", "testing"]
    
    for query in search_queries:
        print(f"\n   Searching for: '{query}'")
        results = manager.search_rules(query, max_results=3)
        
        for i, result in enumerate(results, 1):
            print(f"     {i}. {result['section_title']} (from {result['source_file']})")
            print(f"        Score: {result['relevance_score']}")
            print(f"        Preview: {result['content_preview']}")
    
    # Demo related rules
    print(f"\n🔗 Related Rules Demo:")
    demo_file = "rules-security.md"
    related = manager.find_related_rules(demo_file)
    print(f"   Rules related to '{demo_file}':")
    for rule in related[:5]:
        print(f"     - {rule}")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"   Use this manager to implement RAG with your preferred embedding model")

if __name__ == "__main__":
    main()
