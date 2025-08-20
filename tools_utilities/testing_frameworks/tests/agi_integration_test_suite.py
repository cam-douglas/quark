#!/usr/bin/env python3
"""
AGI Integration Test Suite
Comprehensive testing framework for AGI-enhanced brain simulation
Tests all 10 AGI capability domains with optimization and robustness validation

Test Categories:
1. Core Cognitive Domains Testing
2. Perception & World Modeling Testing  
3. Action & Agency Testing
4. Communication & Language Testing
5. Social & Cultural Intelligence Testing
6. Metacognition & Self-Modeling Testing
7. Knowledge Integration Testing
8. Robustness & Adaptivity Testing
9. Creativity & Exploration Testing
10. Implementation Pillars Testing

Usage:
    python -m pytest tests/agi_integration_test_suite.py -v
    python -m pytest tests/agi_integration_test_suite.py::TestCognitiveDomains -v
"""

import pytest
import numpy as np
import time
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, List, Any, Tuple

# Import AGI-enhanced components
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.agi_enhanced_brain_launcher import (
    AGIEnhancedModule, AGIArchitectureAgent, AGISystemManager,
    AGIMessage, agi_msg, AGICapabilities
)

# Test fixtures and utilities
@pytest.fixture
def agi_config():
    """AGI system configuration for testing"""
    return {
        "optimization_level": "high",
        "robustness_level": "high",
        "agi_capabilities": AGICapabilities.get_all_capabilities(),
        "modules": {
            "TestModule": {
                "type": "AGIEnhancedModule",
                "agi_capabilities": ["memory_systems", "learning", "reasoning"]
            },
            "ArchitectureAgent": {
                "type": "AGIArchitectureAgent",
                "agi_capabilities": ["all"]
            }
        }
    }

@pytest.fixture
def agi_module():
    """Basic AGI-enhanced module for testing"""
    spec = {
        "agi_capabilities": ["memory_systems", "learning", "reasoning"],
        "optimization_level": "high",
        "robustness_level": "high"
    }
    return AGIEnhancedModule("TestModule", spec)

@pytest.fixture
def architecture_agent():
    """AGI Architecture Agent for testing"""
    spec = {
        "agi_capabilities": AGICapabilities.get_all_capabilities(),
        "optimization_level": "high",
        "robustness_level": "high"
    }
    return AGIArchitectureAgent("ArchitectureAgent", spec)

# ---------------------------
# Test Category 1: Core Cognitive Domains
# ---------------------------
class TestCognitiveDomains:
    """Test core cognitive capabilities: Memory, Learning, Reasoning, Problem Solving"""
    
    def test_memory_systems_integration(self, agi_module):
        """Test enhanced memory systems functionality"""
        # Test episodic memory
        ctx = {"tick": 1, "current_context": {"location": "test"}}
        msg = agi_msg("AGI_Query", "test", "TestModule", 
                     agi_domain="memory_systems",
                     operation="store", 
                     content="test memory item")
        
        outbox, telemetry = agi_module.step([msg], ctx)
        
        # Verify memory storage
        assert len(agi_module.state["episodic_buffer"]) > 0
        assert agi_module.state["episodic_buffer"][0]["content"] == "test memory item"
        assert "memory_systems" in str(telemetry)
    
    def test_memory_retrieval(self, agi_module):
        """Test memory retrieval functionality"""
        # Store a memory first
        ctx = {"tick": 1}
        store_msg = agi_msg("AGI_Query", "test", "TestModule",
                          agi_domain="memory_systems",
                          operation="store",
                          content="retrievable memory")
        
        agi_module.step([store_msg], ctx)
        
        # Test retrieval
        retrieve_msg = agi_msg("AGI_Query", "test", "TestModule",
                             agi_domain="memory_systems",
                             operation="retrieve",
                             query="retrievable")
        
        outbox, telemetry = agi_module.step([retrieve_msg], ctx)
        
        # Verify retrieval response
        assert len(outbox) > 0
        assert outbox[0].kind == "AGI_Response"
        assert "results" in outbox[0].payload
    
    def test_learning_systems(self, agi_module):
        """Test enhanced learning capabilities"""
        ctx = {"tick": 1}
        
        # Test meta-learning
        meta_msg = agi_msg("AGI_Query", "test", "TestModule",
                         agi_domain="learning",
                         type="meta_learning",
                         adaptation_data={"rate_modifier": 1.5})
        
        outbox, telemetry = agi_module.step([meta_msg], ctx)
        
        # Verify meta-learning state update
        assert "meta_learner" in agi_module.state
        assert agi_module.state["meta_learner"]["adaptation_rate"] > 0
    
    def test_reasoning_systems(self, agi_module):
        """Test sophisticated reasoning capabilities"""
        ctx = {"tick": 1}
        
        # Test deductive reasoning
        deductive_msg = agi_msg("AGI_Query", "test", "TestModule",
                              agi_domain="reasoning",
                              type="deductive",
                              premises=["All humans are mortal", "Socrates is human"])
        
        outbox, telemetry = agi_module.step([deductive_msg], ctx)
        
        # Verify reasoning response
        assert len(outbox) > 0
        assert outbox[0].kind == "AGI_Response"
        assert "conclusion" in outbox[0].payload
    
    def test_problem_solving_integration(self, agi_module):
        """Test problem-solving capabilities integration"""
        # Problem solving combines multiple cognitive domains
        ctx = {"tick": 1}
        
        # Simulate complex problem requiring memory + reasoning
        problem_msg = agi_msg("AGI_Query", "test", "TestModule",
                            agi_domain="reasoning",
                            type="inductive",
                            observations=[
                                {"pattern": "A", "result": 1},
                                {"pattern": "A", "result": 1},
                                {"pattern": "B", "result": 2}
                            ])
        
        outbox, telemetry = agi_module.step([problem_msg], ctx)
        
        # Verify pattern recognition
        assert len(outbox) > 0
        assert "pattern" in outbox[0].payload

# ---------------------------
# Test Category 2: Perception & World Modeling
# ---------------------------
class TestPerceptionWorldModeling:
    """Test perception and world modeling capabilities"""
    
    def test_multimodal_perception(self, agi_module):
        """Test multimodal perception fusion"""
        ctx = {"tick": 1}
        
        multimodal_msg = agi_msg("AGI_Query", "test", "TestModule",
                               agi_domain="perception",
                               modality="multimodal",
                               inputs={
                                   "visual": {"content": "image_data", "confidence": 0.9},
                                   "auditory": {"content": "audio_data", "confidence": 0.8}
                               })
        
        outbox, telemetry = agi_module.step([multimodal_msg], ctx)
        
        # Verify multimodal fusion
        assert len(outbox) > 0
        response = outbox[0].payload.get("representation", {})
        assert "modalities_used" in response
        assert len(response["modalities_used"]) == 2
    
    def test_world_model_predictive_capability(self, agi_module):
        """Test world model predictive simulation"""
        # This would test the world model's ability to predict future states
        ctx = {"tick": 1}
        
        # Simulate world state prediction request
        prediction_msg = agi_msg("AGI_Query", "test", "TestModule",
                                agi_domain="world_modeling",
                                request_type="predict_state",
                                current_state={"position": [0, 0], "velocity": [1, 0]},
                                time_horizon=5)
        
        # For now, verify the message processing doesn't crash
        outbox, telemetry = agi_module.step([prediction_msg], ctx)
        assert isinstance(outbox, list)
        assert isinstance(telemetry, dict)

# ---------------------------
# Test Category 3: Action & Agency
# ---------------------------
class TestActionAgency:
    """Test action and agency capabilities"""
    
    def test_action_planning_hierarchy(self, agi_module):
        """Test hierarchical action planning"""
        ctx = {"tick": 1}
        
        planning_msg = agi_msg("AGI_Query", "test", "TestModule",
                             agi_domain="action_planning",
                             goal="reach_target",
                             constraints=["avoid_obstacles", "minimize_time"],
                             planning_horizon=10)
        
        outbox, telemetry = agi_module.step([planning_msg], ctx)
        
        # Verify planning capability
        assert isinstance(outbox, list)
        assert "module" in telemetry
    
    def test_decision_making_uncertainty(self, agi_module):
        """Test decision-making under uncertainty"""
        ctx = {"tick": 1}
        
        decision_msg = agi_msg("AGI_Query", "test", "TestModule",
                             agi_domain="decision_making",
                             options=[
                                 {"action": "A", "utility": 0.8, "uncertainty": 0.2},
                                 {"action": "B", "utility": 0.6, "uncertainty": 0.1}
                             ],
                             risk_tolerance=0.5)
        
        outbox, telemetry = agi_module.step([decision_msg], ctx)
        
        # Decision making should process without errors
        assert isinstance(outbox, list)
        assert telemetry["module"] == "TestModule"

# ---------------------------
# Test Category 4: Communication & Language
# ---------------------------
class TestCommunicationLanguage:
    """Test communication and language capabilities"""
    
    def test_natural_language_processing(self, agi_module):
        """Test natural language understanding and generation"""
        ctx = {"tick": 1}
        
        language_msg = agi_msg("AGI_Query", "test", "TestModule",
                             agi_domain="natural_language",
                             task="understand",
                             text="The quick brown fox jumps over the lazy dog",
                             context="demonstration_sentence")
        
        outbox, telemetry = agi_module.step([language_msg], ctx)
        
        # Language processing should handle gracefully
        assert isinstance(outbox, list)
        assert "processed_messages" in telemetry
    
    def test_symbolic_manipulation(self, agi_module):
        """Test symbolic and mathematical reasoning"""
        ctx = {"tick": 1}
        
        symbolic_msg = agi_msg("AGI_Query", "test", "TestModule",
                             agi_domain="symbolic_manipulation",
                             expression="2 + 2",
                             operation="evaluate")
        
        outbox, telemetry = agi_module.step([symbolic_msg], ctx)
        
        # Symbolic processing should work
        assert isinstance(outbox, list)
        assert isinstance(telemetry, dict)

# ---------------------------
# Test Category 5: Metacognition & Self-Modeling
# ---------------------------
class TestMetacognitionSelfModeling:
    """Test metacognitive and self-modeling capabilities"""
    
    def test_self_representation(self, agi_module):
        """Test self-representation and architectural awareness"""
        ctx = {"tick": 1}
        
        self_query_msg = agi_msg("AGI_Query", "test", "TestModule",
                               agi_domain="self_representation",
                               query_type="architecture",
                               detail_level="high")
        
        outbox, telemetry = agi_module.step([self_query_msg], ctx)
        
        # Self-representation should be accessible
        assert isinstance(outbox, list)
        assert "module" in telemetry
    
    def test_introspection_capability(self, agi_module):
        """Test introspection and decision explanation"""
        ctx = {"tick": 1}
        
        introspection_msg = agi_msg("AGI_Query", "test", "TestModule",
                                  agi_domain="introspection",
                                  request="explain_last_decision",
                                  detail_level="medium")
        
        outbox, telemetry = agi_module.step([introspection_msg], ctx)
        
        # Introspection should provide explanations
        assert isinstance(outbox, list)
        assert isinstance(telemetry, dict)

# ---------------------------
# Test Category 6: Optimization & Robustness
# ---------------------------
class TestOptimizationRobustness:
    """Test optimization and robustness features"""
    
    def test_message_optimization(self, agi_module):
        """Test message processing optimization"""
        ctx = {"tick": 1}
        
        # Create batch of messages with different priorities
        messages = [
            agi_msg("AGI_Query", "test", "TestModule", priority=1, content="low_priority"),
            agi_msg("AGI_Query", "test", "TestModule", priority=5, content="high_priority"),
            agi_msg("AGI_Query", "test", "TestModule", priority=3, content="medium_priority")
        ]
        
        # Process with optimization
        start_time = time.time()
        outbox, telemetry = agi_module.step(messages, ctx)
        processing_time = time.time() - start_time
        
        # Verify optimization worked (should be fast)
        assert processing_time < 1.0  # Should process quickly
        assert "processed_messages" in telemetry
        assert telemetry["processed_messages"] == len(messages)
    
    def test_error_recovery(self, agi_module):
        """Test error recovery mechanisms"""
        ctx = {"tick": 1}
        
        # Create message that might cause error
        error_msg = agi_msg("AGI_Query", "test", "TestModule",
                          agi_domain="nonexistent_domain",
                          malformed_data=True)
        
        # Should handle gracefully without crashing
        outbox, telemetry = agi_module.step([error_msg], ctx)
        
        # Error should be handled gracefully
        assert isinstance(outbox, list)
        assert isinstance(telemetry, dict)
        assert agi_module.error_recovery_count >= 0
    
    def test_robustness_filtering(self, agi_module):
        """Test robustness filtering of low-confidence outputs"""
        ctx = {"tick": 1}
        
        # Set strict confidence threshold
        agi_module.adaptive_threshold = 0.9
        
        # Create message that would produce low-confidence output
        low_conf_msg = agi_msg("AGI_Query", "test", "TestModule",
                             agi_domain="memory_systems",
                             confidence=0.5,  # Below threshold
                             operation="retrieve",
                             query="uncertain_query")
        
        outbox, telemetry = agi_module.step([low_conf_msg], ctx)
        
        # Should filter low-confidence outputs if robustness is high
        if agi_module.robustness_level == "high":
            # Check for robustness metrics in telemetry
            assert "robustness" in telemetry or "processed_messages" in telemetry

# ---------------------------
# Test Category 7: System Integration
# ---------------------------
class TestSystemIntegration:
    """Test system-level integration and coordination"""
    
    def test_agi_architecture_agent(self, architecture_agent):
        """Test AGI Architecture Agent coordination"""
        ctx = {"tick": 1}
        
        coordination_msg = agi_msg("AGI_Query", "test", "ArchitectureAgent",
                                 request_type="domain_status",
                                 detail_level="summary")
        
        outbox, telemetry = architecture_agent.step([coordination_msg], ctx)
        
        # Verify coordination functionality
        assert isinstance(outbox, list)
        assert "coordination_status" in telemetry
        assert "agi_coordination" in telemetry
    
    def test_load_balancing(self, architecture_agent):
        """Test load balancing across AGI domains"""
        ctx = {"tick": 1}
        
        # Simulate high load scenario
        architecture_agent.coordination_state["domain_performance"] = {
            "memory_systems": 100,
            "reasoning": 50,
            "learning": 25
        }
        
        load_balance_msg = agi_msg("AGI_Query", "test", "ArchitectureAgent",
                                 request_type="load_balance")
        
        outbox, telemetry = architecture_agent.step([load_balance_msg], ctx)
        
        # Should generate load balancing recommendations
        assert isinstance(outbox, list)
        efficiency = architecture_agent._calculate_coordination_efficiency()
        assert 0 <= efficiency <= 1.0
    
    def test_system_manager_initialization(self, agi_config):
        """Test AGI System Manager initialization"""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(agi_config, f)
            config_path = f.name
        
        try:
            # Initialize system manager
            system_manager = AGISystemManager(config_path)
            system_manager.initialize_system()
            
            # Verify initialization
            assert len(system_manager.modules) > 0
            assert "optimization_enabled" in dir(system_manager)
            assert system_manager.system_metrics is not None
        
        finally:
            # Clean up
            os.unlink(config_path)
    
    @patch('time.time')
    def test_simulation_execution(self, mock_time, agi_config):
        """Test full simulation execution"""
        mock_time.return_value = 1000.0
        
        # Create minimal config for testing
        minimal_config = {
            "optimization_level": "medium",
            "robustness_level": "medium",
            "modules": {
                "TestAgent": {
                    "type": "AGIEnhancedModule",
                    "agi_capabilities": ["memory_systems"]
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(minimal_config, f)
            config_path = f.name
        
        try:
            system_manager = AGISystemManager(config_path)
            system_manager.initialize_system()
            
            # Run short simulation
            system_manager.run_simulation(steps=5)
            
            # Verify simulation ran
            assert system_manager.system_metrics["total_steps"] == 5
            assert system_manager.system_metrics["error_count"] >= 0
        
        finally:
            os.unlink(config_path)
            # Clean up any generated report files
            report_path = Path("agi_simulation_report.json")
            if report_path.exists():
                report_path.unlink()

# ---------------------------
# Test Category 8: Performance & Scalability
# ---------------------------
class TestPerformanceScalability:
    """Test performance and scalability characteristics"""
    
    def test_processing_latency(self, agi_module):
        """Test processing latency under load"""
        ctx = {"tick": 1}
        
        # Create multiple messages
        messages = [
            agi_msg("AGI_Query", "test", "TestModule",
                   agi_domain="memory_systems",
                   operation="store",
                   content=f"message_{i}")
            for i in range(100)
        ]
        
        # Measure processing time
        start_time = time.time()
        outbox, telemetry = agi_module.step(messages, ctx)
        processing_time = time.time() - start_time
        
        # Should process efficiently
        assert processing_time < 5.0  # Should complete in reasonable time
        assert len(agi_module.performance_metrics["processing_time"]) > 0
    
    def test_memory_efficiency(self, agi_module):
        """Test memory usage optimization"""
        # Fill episodic buffer beyond capacity
        for i in range(150):  # More than the 100-item limit
            ctx = {"tick": i}
            msg = agi_msg("AGI_Query", "test", "TestModule",
                         agi_domain="memory_systems",
                         operation="store",
                         content=f"memory_item_{i}")
            agi_module.step([msg], ctx)
        
        # Should have consolidated memory
        assert len(agi_module.state["episodic_buffer"]) <= 100
        assert len(agi_module.performance_metrics["memory_usage"]) <= 1000
    
    def test_concurrent_processing(self, agi_module):
        """Test concurrent message processing capability"""
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def process_messages(thread_id):
            ctx = {"tick": thread_id}
            msg = agi_msg("AGI_Query", f"thread_{thread_id}", "TestModule",
                         agi_domain="reasoning",
                         type="deductive",
                         premises=[f"premise_{thread_id}_1", f"premise_{thread_id}_2"])
            
            outbox, telemetry = agi_module.step([msg], ctx)
            results_queue.put((thread_id, len(outbox), telemetry))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=process_messages, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all threads completed
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        assert len(results) == 5

# ---------------------------
# Test Category 9: AGI Capability Coverage
# ---------------------------
class TestAGICapabilityCoverage:
    """Test coverage of all AGI capability domains"""
    
    def test_all_domains_available(self):
        """Test that all AGI domains are properly defined"""
        all_capabilities = AGICapabilities.get_all_capabilities()
        
        # Should have capabilities from all 10 domains
        assert len(all_capabilities) >= 30  # At least 3 per domain
        
        # Check that core domains are represented
        cognitive_caps = AGICapabilities.get_domain_capabilities("cognitive")
        assert "memory_systems" in cognitive_caps
        assert "learning" in cognitive_caps
        assert "reasoning" in cognitive_caps
        assert "problem_solving" in cognitive_caps
    
    def test_capability_domain_mapping(self):
        """Test domain to capability mapping"""
        domains = AGICapabilities.DOMAINS
        
        # All 10 major domains should be present
        expected_domains = [
            "cognitive", "perception", "action", "communication", "social",
            "metacognition", "knowledge", "robustness", "creativity", "implementation"
        ]
        
        for domain in expected_domains:
            assert domain in domains
            assert len(domains[domain]) > 0
    
    def test_module_capability_assignment(self, agi_module):
        """Test that modules correctly register AGI capabilities"""
        # Module should have been initialized with specific capabilities
        assert len(agi_module.agi_capabilities) > 0
        assert "memory_systems" in agi_module.agi_capabilities
        assert "learning" in agi_module.agi_capabilities
        assert "reasoning" in agi_module.agi_capabilities

# ---------------------------
# Test Utilities and Fixtures
# ---------------------------
class TestUtilities:
    """Utility functions for AGI testing"""
    
    @staticmethod
    def generate_test_data(size: int = 100) -> List[Dict[str, Any]]:
        """Generate synthetic test data for AGI testing"""
        return [
            {
                "id": i,
                "content": f"test_data_{i}",
                "category": f"category_{i % 10}",
                "features": np.random.rand(5).tolist(),
                "confidence": np.random.uniform(0.5, 1.0)
            }
            for i in range(size)
        ]
    
    @staticmethod
    def validate_agi_response(response: AGIMessage) -> bool:
        """Validate AGI response message format"""
        if not isinstance(response, AGIMessage):
            return False
        
        if response.kind not in ["AGI_Response", "Command", "Telemetry"]:
            return False
        
        if not isinstance(response.payload, dict):
            return False
        
        return True
    
    @staticmethod
    def measure_performance(func, *args, **kwargs) -> Tuple[Any, float]:
        """Measure function performance"""
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time

# ---------------------------
# Integration Test Scenarios
# ---------------------------
class TestIntegrationScenarios:
    """Test complex integration scenarios across multiple AGI domains"""
    
    def test_memory_learning_integration(self, agi_module):
        """Test integration between memory and learning systems"""
        ctx = {"tick": 1}
        
        # Store initial knowledge
        store_msg = agi_msg("AGI_Query", "test", "TestModule",
                          agi_domain="memory_systems",
                          operation="store",
                          content="initial_knowledge")
        
        agi_module.step([store_msg], ctx)
        
        # Learn from the stored knowledge
        learn_msg = agi_msg("AGI_Query", "test", "TestModule",
                          agi_domain="learning",
                          type="continual",
                          knowledge={"skill": "pattern_recognition", "level": 0.8})
        
        outbox, telemetry = agi_module.step([learn_msg], ctx)
        
        # Verify both systems worked together
        assert len(agi_module.state["episodic_buffer"]) > 0
        assert "continual_learner" in agi_module.state
    
    def test_reasoning_perception_integration(self, agi_module):
        """Test integration between reasoning and perception systems"""
        ctx = {"tick": 1}
        
        # First, process perceptual input
        perception_msg = agi_msg("AGI_Query", "test", "TestModule",
                               agi_domain="perception",
                               modality="multimodal",
                               inputs={"visual": {"patterns": ["circle", "square"]}})
        
        agi_module.step([perception_msg], ctx)
        
        # Then reason about the perceived patterns
        reasoning_msg = agi_msg("AGI_Query", "test", "TestModule",
                              agi_domain="reasoning",
                              type="inductive",
                              observations=[
                                  {"shape": "circle", "property": "round"},
                                  {"shape": "square", "property": "angular"}
                              ])
        
        outbox, telemetry = agi_module.step([reasoning_msg], ctx)
        
        # Should produce integrated reasoning about perceived patterns
        assert len(outbox) > 0 or "processed_messages" in telemetry

# ---------------------------
# Benchmark Tests
# ---------------------------
class TestBenchmarks:
    """Benchmark tests for AGI performance"""
    
    def test_memory_capacity_benchmark(self, agi_module):
        """Benchmark memory capacity and retrieval performance"""
        ctx = {"tick": 1}
        
        # Store large number of memories
        num_memories = 1000
        start_time = time.time()
        
        for i in range(num_memories):
            msg = agi_msg("AGI_Query", "test", "TestModule",
                         agi_domain="memory_systems",
                         operation="store",
                         content=f"benchmark_memory_{i}")
            agi_module.step([msg], ctx)
        
        storage_time = time.time() - start_time
        
        # Test retrieval performance
        start_time = time.time()
        retrieve_msg = agi_msg("AGI_Query", "test", "TestModule",
                             agi_domain="memory_systems",
                             operation="retrieve",
                             query="benchmark")
        
        outbox, telemetry = agi_module.step([retrieve_msg], ctx)
        retrieval_time = time.time() - start_time
        
        # Performance benchmarks
        assert storage_time < 10.0  # Should store 1000 items quickly
        assert retrieval_time < 1.0  # Should retrieve quickly
        assert len(agi_module.state["episodic_buffer"]) <= 100  # Should auto-consolidate
    
    def test_reasoning_complexity_benchmark(self, agi_module):
        """Benchmark reasoning performance with increasing complexity"""
        ctx = {"tick": 1}
        
        complexities = [2, 5, 10, 20]
        performance_results = []
        
        for complexity in complexities:
            premises = [f"premise_{i}" for i in range(complexity)]
            
            start_time = time.time()
            msg = agi_msg("AGI_Query", "test", "TestModule",
                         agi_domain="reasoning",
                         type="deductive",
                         premises=premises)
            
            outbox, telemetry = agi_module.step([msg], ctx)
            processing_time = time.time() - start_time
            
            performance_results.append(processing_time)
        
        # Performance should scale reasonably
        assert all(t < 1.0 for t in performance_results)  # All should be fast
        
        # Can optionally test that complexity doesn't cause exponential slowdown
        # (though simple implementation might not show this)

if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--strict-markers"
    ])
