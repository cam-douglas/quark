# tests/test_pillar3_memory_control.py

"""
Purpose: Comprehensive validation test for Pillar 3 (Working Memory & Control).
Inputs: Test scenarios for memory buffer and thalamic relay
Outputs: Validation results and performance metrics
Dependencies: brain_modules.working_memory.memory_buffer, brain_modules.thalamus.thalamic_relay, numpy, pytest
"""

import sys
import os
import numpy as np
import pytest
import time

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from brain_modules.working_memory.memory_buffer import MemoryBuffer, MemoryItem
from brain_modules.thalamus.thalamic_relay import ThalamicRelay

class TestPillar3WorkingMemoryAndControl:
    """Test suite for Pillar 3: Working Memory & Control"""
    
    def test_memory_item_functionality(self):
        """Test basic MemoryItem functionality"""
        item = MemoryItem("test_content")
        
        # Test initialization
        assert item.content == "test_content"
        assert item.activation == 1.0
        
        # Test activation decay
        initial_activation = item.activation
        item.update_activation()
        assert item.activation < initial_activation
        
        # Test refresh
        item.refresh()
        assert item.activation > item.decay_rate
        
        # Test active status
        assert item.is_active(threshold=0.1)
        item.activation = 0.05
        assert not item.is_active(threshold=0.1)
    
    def test_memory_buffer_capacity(self):
        """Test that the memory buffer respects its capacity"""
        buffer = MemoryBuffer(capacity=3)
        
        # Add items up to capacity
        for i in range(3):
            buffer.add_item(f"item_{i}")
        
        assert buffer.get_size() == 3, "Buffer should be at capacity"
        
        # Add one more item
        buffer.add_item("item_3")
        
        assert buffer.get_size() == 3, "Buffer should not exceed capacity"
        
        # Check that the least active item was removed (in this case, the first one)
        contents = [item.content for item in buffer.buffer]
        assert "item_0" not in contents, "The least active item should be removed"
    
    def test_memory_buffer_decay_and_retrieval(self):
        """Test item decay and retrieval of active items"""
        buffer = MemoryBuffer(capacity=5)
        
        # Add items
        for i in range(5):
            buffer.add_item(f"item_{i}")
        
        # Simulate time passing
        for _ in range(10):
            buffer.update_buffer()
        
        # Some items should have decayed
        assert buffer.get_size() < 5, "Some items should have decayed and been removed"
        
        # Retrieve active items
        active_items = buffer.retrieve_items()
        assert len(active_items) == buffer.get_size(), "Should retrieve all active items"
    
    def test_thalamic_relay_initialization(self):
        """Test ThalamicRelay initialization"""
        relay = ThalamicRelay(num_sources=4)
        
        # Test initial attention weights
        expected_weights = np.array([0.25, 0.25, 0.25, 0.25])
        np.testing.assert_allclose(relay.attention_weights, expected_weights, atol=1e-6)
        
        # Test information sources
        assert not relay.information_sources, "Information sources should be empty initially"
    
    def test_thalamic_relay_attention_update(self):
        """Test updating of attentional weights"""
        relay = ThalamicRelay(num_sources=3)
        
        # Update attention
        new_weights = np.array([0.6, 0.2, 0.2])
        relay.update_attention(new_weights)
        
        np.testing.assert_allclose(relay.attention_weights, new_weights, atol=1e-6)
        
        # Test with non-normalized weights
        relay.update_attention(np.array([1, 1, 2]))
        expected_weights = np.array([0.25, 0.25, 0.5])
        np.testing.assert_allclose(relay.attention_weights, expected_weights, atol=1e-6)
    
    def test_thalamic_relay_information_routing(self):
        """Test information routing based on attention"""
        relay = ThalamicRelay(num_sources=3)
        
        # Add information sources
        relay.add_information_source(0, "visual")
        relay.add_information_source(1, "auditory")
        relay.add_information_source(2, "tactile")
        
        # Update attention to focus on visual
        relay.update_attention(np.array([0.8, 0.1, 0.1]))
        
        # Route information
        routed_info = relay.route_information()
        
        # Check that visual information has the highest weight
        assert routed_info["source_0"]["attention_weight"] == 0.8
        assert routed_info["source_1"]["attention_weight"] == 0.1
        assert routed_info["source_2"]["attention_weight"] == 0.1
        
        # Check that data is routed correctly
        assert routed_info["source_0"]["data"] == "visual"

def run_pillar3_validation():
    """Run comprehensive validation of Pillar 3 implementation"""
    print("🧠 Pillar 3 Validation: Working Memory & Control")
    print("=" * 60)
    
    # Create test instance
    test_suite = TestPillar3WorkingMemoryAndControl()
    
    # Run all tests
    test_methods = [
        test_suite.test_memory_item_functionality,
        test_suite.test_memory_buffer_capacity,
        test_suite.test_memory_buffer_decay_and_retrieval,
        test_suite.test_thalamic_relay_initialization,
        test_suite.test_thalamic_relay_attention_update,
        test_suite.test_thalamic_relay_information_routing
    ]
    
    passed_tests = 0
    total_tests = len(test_methods)
    
    for test_method in test_methods:
        try:
            test_method()
            print(f"✅ {test_method.__name__}: PASSED")
            passed_tests += 1
        except Exception as e:
            print(f"❌ {test_method.__name__}: FAILED - {str(e)}")
    
    print("=" * 60)
    print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 Pillar 3 Validation: ALL TESTS PASSED")
        print("✅ Working Memory & Control systems are working correctly")
        return True
    else:
        print("⚠️  Pillar 3 Validation: SOME TESTS FAILED")
        print("🔧 Review implementation and fix issues")
        return False

if __name__ == "__main__":
    success = run_pillar3_validation()
    exit(0 if success else 1)
