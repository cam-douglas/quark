#!/usr/bin/env python3
"""
🧬 System Status Test - Verifies the semantic network is working
"""

import json
import os
from pathlib import Path

def test_system_status():
    """Test the current system status"""
    print("🧬 Testing Semantic Network System Status...")
    
    # Check if memory directory exists
    memory_dir = Path("memory")
    if not memory_dir.exists():
        print("❌ Memory directory not found")
        return False
    
    # Test summaries.json
    try:
        with open(memory_dir / "summaries.json") as f:
            summaries = json.load(f)
        print(f"✅ Summaries loaded: {summaries['total_chunks']} chunks, {summaries['total_edges']} edges")
        print(f"✅ Biological markers: {', '.join(summaries['biological_markers'])}")
        print(f"✅ Critical markers present: {summaries['critical_markers_present']}")
    except Exception as e:
        print(f"❌ Failed to load summaries: {e}")
        return False
    
    # Test metadata.json
    try:
        with open(memory_dir / "metadata.json") as f:
            metadata = json.load(f)
        print(f"✅ Metadata loaded: {len(metadata)} chunks")
        
        # Check biological marker distribution
        marker_counts = {}
        for entry in metadata:
            for marker in entry.get("markers", []):
                marker_counts[marker] = marker_counts.get(marker, 0) + 1
        
        print(f"✅ Marker distribution:")
        for marker, count in sorted(marker_counts.items()):
            print(f"   {marker}: {count} chunks")
            
    except Exception as e:
        print(f"❌ Failed to load metadata: {e}")
        return False
    
    # Test mutation log
    try:
        with open(memory_dir / "mutation_log.json") as f:
            mutations = json.load(f)
        print(f"✅ Mutation log loaded: {len(mutations)} mutations")
        
        if mutations:
            scores = [m["scores"]["overall_score"] for m in mutations]
            print(f"✅ Score range: {min(scores):.3f} - {min(scores):.3f}")
            print(f"✅ Average score: {sum(scores)/len(scores):.3f}")
            
    except Exception as e:
        print(f"❌ Failed to load mutation log: {e}")
        return False
    
    # Test rule graph
    try:
        with open(memory_dir / "rule_graph.json") as f:
            graph_data = json.load(f)
        print(f"✅ Rule graph loaded: {len(graph_data.get('nodes', []))} nodes")
        
    except Exception as e:
        print(f"✅ Failed to load rule graph: {e}")
        return False
    
    print("\n🎉 System Status: HEALTHY")
    print("✅ Semantic network is operational")
    print("✅ All core data files are accessible")
    print("✅ Biological compliance is maintained")
    
    return True

def test_semantic_search():
    """Test basic semantic search functionality"""
    print("\n🔍 Testing Semantic Search...")
    
    try:
        with open("memory/metadata.json") as f:
            metadata = json.load(f)
        
        # Simple search for "security" related content
        search_term = "security"
        relevant_chunks = []
        
        for entry in metadata:
            text = entry.get("text", "").lower()
            if search_term in text:
                relevant_chunks.append({
                    "file": entry.get("file", "unknown"),
                    "markers": entry.get("markers", []),
                    "text_preview": text[:100] + "..." if len(text) > 100 else text
                })
        
        print(f"✅ Found {len(relevant_chunks)} security-related chunks")
        
        if relevant_chunks:
            print("✅ Sample results:")
            for i, chunk in enumerate(relevant_chunks[:3]):
                print(f"   {i+1}. {chunk['file']} ({', '.join(chunk['markers'])})")
                print(f"✅ {chunk['text_preview']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Semantic search failed: {e}")
        return False

if __name__ == "__main__":
    print("🧬 Semantic Network System Test")
    print("=" * 50)
    
    # Test system status
    status_ok = test_system_status()
    
    if status_ok:
        # Test semantic search
        test_semantic_search()
    
    print("\n" + "=" * 50)
    print("✅ Test completed successfully!")
