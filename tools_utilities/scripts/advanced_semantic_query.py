#!/usr/bin/env python3
"""
🧬 Advanced Semantic Query System
Complex rule retrieval and analysis using integrated brain modules
"""

import json
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Set
import re

class AdvancedSemanticQuery:
    """Advanced semantic query system for complex rule analysis"""
    
    def __init__(self):
        self.memory_dir = Path("memory")
        self.load_semantic_network()
        
    def load_semantic_network(self):
        """Load the semantic network and metadata"""
        try:
            # Load metadata
            with open(self.memory_dir / "metadata.json") as f:
                self.metadata = json.load(f)
            
            # Load graph
            with open(self.memory_dir / "rule_graph.json") as f:
                graph_data = json.load(f)
                self.graph = nx.node_link_graph(graph_data)
            
            # Load summaries
            with open(self.memory_dir / "summaries.json") as f:
                self.summaries = json.load(f)
                
            print(f"✅ Loaded semantic network: {len(self.metadata)} chunks, {self.graph.number_of_edges()} edges")
            
        except Exception as e:
            print(f"❌ Failed to load semantic network: {e}")
            raise
    
    def complex_query(self, query: str, max_results: int = 10, 
                     biological_markers: List[str] = None,
                     priority_levels: List[int] = None) -> Dict[str, Any]:
        """Perform complex semantic query with multiple filters"""
        
        print(f"🔍 Complex Query: {query}")
        print(f"   Filters: Markers={biological_markers}, Priorities={priority_levels}")
        
        # Step 1: Text-based relevance search
        relevant_chunks = self._text_relevance_search(query)
        
        # Step 2: Biological marker filtering
        if biological_markers:
            relevant_chunks = self._filter_by_markers(relevant_chunks, biological_markers)
        
        # Step 3: Priority level filtering
        if priority_levels:
            relevant_chunks = self._filter_by_priority(relevant_chunks, priority_levels)
        
        # Step 4: Graph-based context expansion
        expanded_results = self._graph_context_expansion(relevant_chunks, max_results)
        
        # Step 5: Biological compliance scoring
        scored_results = self._biological_compliance_scoring(expanded_results)
        
        return {
            "query": query,
            "filters": {
                "biological_markers": biological_markers,
                "priority_levels": priority_levels
            },
            "results": scored_results,
            "total_found": len(scored_results),
            "biological_compliance": self._calculate_compliance_score(scored_results)
        }
    
    def _text_relevance_search(self, query: str) -> List[Dict]:
        """Search for textually relevant chunks"""
        query_words = set(query.lower().split())
        relevant_chunks = []
        
        for chunk in self.metadata:
            text = chunk.get("text", "").lower()
            chunk_words = set(text.split())
            
            # Calculate word overlap
            overlap = len(query_words & chunk_words)
            if overlap > 0:
                relevance = overlap / len(query_words)
                if relevance > 0.1:  # Threshold for relevance
                    relevant_chunks.append({
                        **chunk,
                        "relevance_score": relevance
                    })
        
        # Sort by relevance
        relevant_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
        return relevant_chunks
    
    def _filter_by_markers(self, chunks: List[Dict], markers: List[str]) -> List[Dict]:
        """Filter chunks by biological markers"""
        filtered = []
        for chunk in chunks:
            chunk_markers = chunk.get("markers", [])
            if any(marker in chunk_markers for marker in markers):
                filtered.append(chunk)
        return filtered
    
    def _filter_by_priority(self, chunks: List[Dict], priorities: List[int]) -> List[Dict]:
        """Filter chunks by priority levels"""
        filtered = []
        for chunk in chunks:
            # Extract priority from filename (e.g., "01-index.md" -> priority 1)
            filename = chunk.get("file", "")
            match = re.match(r"(\d+)-", filename)
            if match:
                priority = int(match.group(1))
                if priority in priorities:
                    filtered.append(chunk)
        return filtered
    
    def _graph_context_expansion(self, chunks: List[Dict], max_results: int) -> List[Dict]:
        """Expand results using graph context"""
        expanded = []
        processed_nodes = set()
        
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id is not None and chunk_id not in processed_nodes:
                expanded.append(chunk)
                processed_nodes.add(chunk_id)
                
                # Add connected nodes (neighbors)
                if chunk_id in self.graph:
                    neighbors = list(self.graph.neighbors(chunk_id))
                    for neighbor_id in neighbors[:3]:  # Add up to 3 neighbors
                        if neighbor_id not in processed_nodes and len(expanded) < max_results:
                            neighbor_data = self._get_chunk_by_id(neighbor_id)
                            if neighbor_data:
                                expanded.append(neighbor_data)
                                processed_nodes.add(neighbor_id)
        
        return expanded[:max_results]
    
    def _get_chunk_by_id(self, chunk_id: int) -> Dict:
        """Get chunk data by ID"""
        for chunk in self.metadata:
            if chunk.get("chunk_id") == chunk_id:
                return chunk
        return None
    
    def _biological_compliance_scoring(self, chunks: List[Dict]) -> List[Dict]:
        """Score chunks for biological compliance"""
        scored_chunks = []
        
        for chunk in chunks:
            markers = chunk.get("markers", [])
            
            # Calculate compliance score
            compliance_score = 0.0
            
            # Critical markers bonus
            critical_markers = {"GFAP", "NeuN"}
            if any(marker in critical_markers for marker in markers):
                compliance_score += 0.3
            
            # Marker diversity bonus
            marker_diversity = len(markers) / 5.0  # Normalize
            compliance_score += marker_diversity * 0.2
            
            # Priority bonus (lower numbers = higher priority)
            filename = chunk.get("file", "")
            match = re.match(r"(\d+)-", filename)
            if match:
                priority = int(match.group(1))
                priority_score = max(0, (20 - priority) / 20.0)  # Higher score for lower priority numbers
                compliance_score += priority_score * 0.3
            
            # Relevance bonus
            relevance = chunk.get("relevance_score", 0.0)
            compliance_score += relevance * 0.2
            
            scored_chunks.append({
                **chunk,
                "biological_compliance_score": min(compliance_score, 1.0)
            })
        
        # Sort by compliance score
        scored_chunks.sort(key=lambda x: x["biological_compliance_score"], reverse=True)
        return scored_chunks
    
    def _calculate_compliance_score(self, chunks: List[Dict]) -> float:
        """Calculate overall biological compliance score"""
        if not chunks:
            return 0.0
        
        total_score = sum(chunk.get("biological_compliance_score", 0.0) for chunk in chunks)
        return total_score / len(chunks)
    
    def analyze_rule_interactions(self, rule_file: str) -> Dict[str, Any]:
        """Analyze how a specific rule file interacts with others"""
        print(f"🔍 Analyzing rule interactions for: {rule_file}")
        
        # Find chunks from this file
        file_chunks = [chunk for chunk in self.metadata if chunk.get("file") == rule_file]
        
        if not file_chunks:
            return {"error": f"File {rule_file} not found"}
        
        # Analyze connections
        connections = []
        for chunk in file_chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id and chunk_id in self.graph:
                neighbors = list(self.graph.neighbors(chunk_id))
                for neighbor_id in neighbors[:5]:  # Top 5 connections
                    neighbor = self._get_chunk_by_id(neighbor_id)
                    if neighbor:
                        connections.append({
                            "source": chunk.get("text", "")[:100],
                            "target": neighbor.get("text", "")[:100],
                            "target_file": neighbor.get("file", ""),
                            "target_markers": neighbor.get("markers", [])
                        })
        
        return {
            "file": rule_file,
            "total_chunks": len(file_chunks),
            "total_connections": len(connections),
            "connections": connections[:10],  # Top 10 connections
            "biological_markers": list(set(marker for chunk in file_chunks for marker in chunk.get("markers", [])))
        }
    
    def find_biological_pathways(self, start_marker: str, end_marker: str, max_path_length: int = 5) -> List[List[str]]:
        """Find biological pathways between markers"""
        print(f"🧬 Finding biological pathways: {start_marker} → {end_marker}")
        
        # Find chunks with start marker
        start_chunks = [chunk for chunk in self.metadata if start_marker in chunk.get("markers", [])]
        end_chunks = [chunk for chunk in self.metadata if end_marker in chunk.get("markers", [])]
        
        if not start_chunks or not end_chunks:
            return []
        
        pathways = []
        
        # Find paths between start and end chunks
        for start_chunk in start_chunks[:3]:  # Check first 3 start chunks
            for end_chunk in end_chunks[:3]:  # Check first 3 end chunks
                start_id = start_chunk.get("chunk_id")
                end_id = end_chunk.get("chunk_id")
                
                if start_id is not None and end_id is not None and start_id in self.graph and end_id in self.graph:
                    try:
                        # Find shortest path
                        path = nx.shortest_path(self.graph, start_id, end_id)
                        if len(path) <= max_path_length:
                            pathway = []
                            for node_id in path:
                                chunk = self._get_chunk_by_id(node_id)
                                if chunk:
                                    pathway.append({
                                        "file": chunk.get("file", ""),
                                        "markers": chunk.get("markers", []),
                                        "text_preview": chunk.get("text", "")[:100]
                                    })
                            pathways.append(pathway)
                    except nx.NetworkXNoPath:
                        continue
        
        return pathways

def main():
    """Main execution function"""
    print("🧬 Advanced Semantic Query System")
    print("=" * 50)
    
    try:
        query_system = AdvancedSemanticQuery()
        
        # Example 1: Complex query with biological markers
        print("\n🔍 Example 1: Security rules with GFAP marker")
        result1 = query_system.complex_query(
            "security rules and compliance",
            biological_markers=["GFAP"],
            max_results=5
        )
        print(f"   Found {result1['total_found']} results")
        print(f"   Compliance score: {result1['biological_compliance']:.3f}")
        
        # Example 2: Rule interaction analysis
        print("\n🔍 Example 2: Analyzing rule interactions")
        interaction = query_system.analyze_rule_interactions("02-rules_security.md")
        print(f"   {interaction['file']}: {interaction['total_connections']} connections")
        
        # Example 3: Biological pathway finding
        print("\n🔍 Example 3: Finding biological pathways")
        pathways = query_system.find_biological_pathways("GFAP", "NeuN", max_path_length=4)
        print(f"   Found {len(pathways)} pathways between GFAP and NeuN")
        
        print("\n✅ Advanced semantic query system operational!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
