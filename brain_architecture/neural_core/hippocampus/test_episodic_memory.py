#!/usr/bin/env python3
"""
🧠 Test Episodic Memory Module
Tests the hippocampus episodic memory functionality
"""

import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from episodic_memory import EpisodicMemory

def test_episodic_memory():
    """Test basic episodic memory functionality"""
    print("🧠 Testing QUARK Episodic Memory...")
    
    # Initialize episodic memory
    memory = EpisodicMemory(max_episodes=100, pattern_dim=32)
    print("✅ Episodic memory initialized")
    
    # Store some episodes
    print("\n📝 Storing Episodes...")
    
    episode1_id = memory.store_episode(
        content={"action": "created neural network", "framework": "PyTorch"},
        context={"location": "home", "time": "morning", "mood": "focused"},
        emotional_valence=0.8,
        importance=0.9
    )
    print(f"   Stored episode 1: {episode1_id}")
    
    episode2_id = memory.store_episode(
        content={"action": "debugged code", "issue": "memory leak"},
        context={"location": "office", "time": "afternoon", "mood": "frustrated"},
        emotional_valence=-0.3,
        importance=0.7
    )
    print(f"   Stored episode 2: {episode2_id}")
    
    episode3_id = memory.store_episode(
        content={"action": "solved problem", "solution": "garbage collection"},
        context={"location": "office", "time": "evening", "mood": "satisfied"},
        emotional_valence=0.9,
        importance=0.8
    )
    print(f"   Stored episode 3: {episode3_id}")
    
    # Test retrieval
    print("\n🔍 Testing Memory Retrieval...")
    
    # Query by content
    results = memory.retrieve_episode({"action": "created"}, max_results=3)
    print(f"   Retrieved {len(results)} episodes for 'created' action")
    for episode in results:
        print(f"     - {episode.content['action']} (importance: {episode.importance:.2f})")
    
    # Query by context
    results = memory.retrieve_episode({"location": "office"}, max_results=3)
    print(f"   Retrieved {len(results)} episodes for 'office' location")
    for episode in results:
        print(f"     - {episode.content['action']} (mood: {episode.context['mood']})")
    
    # Test pattern completion
    print("\n🧩 Testing Pattern Completion...")
    
    # Create partial features (simplified)
    partial_features = np.zeros(32)
    partial_features[0] = 0.8  # High emotional valence
    partial_features[1] = 0.9  # High importance
    
    completed_episodes = memory.pattern_completion(partial_features, threshold=0.5)
    print(f"   Pattern completion found {len(completed_episodes)} episodes")
    for episode in completed_episodes:
        print(f"     - {episode.content['action']} (valence: {episode.emotional_valence:.2f})")
    
    # Get memory statistics
    print("\n📊 Memory Statistics:")
    stats = memory.get_memory_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    print("\n✅ Episodic memory test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_episodic_memory()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
