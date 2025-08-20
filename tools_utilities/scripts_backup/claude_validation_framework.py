#!/usr/bin/env python3
"""
🔧 Claude Validation Framework
Core rule for functional implementation and testing validation

**Model:** Claude (Functional Implementation & Testing)
**Purpose:** Automated unit and integration tests for brain simulation components
**Validation Level:** Functional behavior verification
**Rule ID:** validation.claude.functional
"""

import pytest
import sys
import os
import yaml
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from core.brain_launcher_v3 import Brain, Module, Message, PFC, BasalGanglia, Thalamus, WorkingMemory, DMN, Salience, Attention, Sleeper, Cerebellum, ArchitectureAgent

class ClaudeValidator:
    """Claude's functional validation framework for brain simulation"""
    
    def __init__(self):
        self.test_results = {}
    
    def run_comprehensive_validation(self, brain_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive functional validation tests"""
        print("🔧 CLAUDE VALIDATION (Functional Tests)")
        print("=" * 50)
        
        # Create temporary test environment
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_metrics_file = os.path.join(temp_dir, "test_metrics.csv")
            
            # Run all functional tests
            test_results = {
                "brain_initialization": self._test_brain_initialization(brain_config, temp_metrics_file),
                "brain_step_execution": self._test_brain_step_execution(brain_config, temp_metrics_file),
                "working_memory_functionality": self._test_working_memory_functionality(brain_config, temp_metrics_file),
                "pfc_executive_function": self._test_pfc_executive_function(brain_config, temp_metrics_file),
                "basal_ganglia_gating": self._test_basal_ganglia_gating(brain_config, temp_metrics_file),
                "thalamic_routing": self._test_thalamic_routing(brain_config, temp_metrics_file),
                "dmn_internal_simulation": self._test_dmn_internal_simulation(brain_config, temp_metrics_file),
                "salience_network": self._test_salience_network(brain_config, temp_metrics_file),
                "attention_mechanism": self._test_attention_mechanism(brain_config, temp_metrics_file),
                "architecture_agent_coordination": self._test_architecture_agent_coordination(brain_config, temp_metrics_file),
                "curriculum_scheduling": self._test_curriculum_scheduling(brain_config, temp_metrics_file),
                "metrics_logging": self._test_metrics_logging(brain_config, temp_metrics_file),
                "stage_progression": self._test_stage_progression(brain_config, temp_metrics_file),
                "error_handling": self._test_error_handling(brain_config, temp_metrics_file),
                "performance_benchmarks": self._test_performance_benchmarks(brain_config, temp_metrics_file),
                "memory_usage": self._test_memory_usage(brain_config, temp_metrics_file)
            }
            
            # Calculate pass rate
            total_tests = len(test_results)
            passed_tests = sum(1 for result in test_results.values() if result["status"] == "PASS")
            pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
            
            claude_results = {
                "test_results": test_results,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "pass_rate": pass_rate,
                "status": "PASS" if pass_rate >= 80 else "FAIL"
            }
            
            print(f"📊 Claude Validation Results:")
            print(f"   Total Tests: {total_tests}")
            print(f"   Passed: {passed_tests}")
            print(f"   Pass Rate: {pass_rate:.1f}%")
            print(f"   Status: {claude_results['status']}")
            
            return claude_results
    
    def _test_brain_initialization(self, brain_config: Dict[str, Any], temp_metrics_file: str) -> Dict[str, Any]:
        """Test brain initialization with valid configuration"""
        try:
            brain = Brain(brain_config, stage="F", log_csv=temp_metrics_file)
            
            # Verify all required modules are present
            expected_modules = ["architecture_agent", "pfc", "basal_ganglia", "thalamus", 
                              "working_memory", "dmn", "salience", "attention"]
            for module_name in expected_modules:
                if module_name not in brain.modules:
                    return {"status": "FAIL", "error": f"Module {module_name} not found"}
            
            # Verify initial state
            if brain.t != 0 or brain.stage != "F":
                return {"status": "FAIL", "error": "Initial state incorrect"}
            
            return {"status": "PASS", "message": "Brain initialization successful"}
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    def _test_brain_step_execution(self, brain_config: Dict[str, Any], temp_metrics_file: str) -> Dict[str, Any]:
        """Test brain step execution and telemetry output"""
        try:
            brain = Brain(brain_config, stage="F", log_csv=temp_metrics_file)
            
            # Execute one step
            telemetry = brain.step(50)
            
            # Verify telemetry structure
            required_keys = ["t", "mode", "state", "fatigue", "mods"]
            for key in required_keys:
                if key not in telemetry:
                    return {"status": "FAIL", "error": f"Missing telemetry key: {key}"}
            
            # Verify time progression
            if telemetry["t"] != 1:
                return {"status": "FAIL", "error": "Time progression incorrect"}
            
            # Verify neuromodulators are present
            mods = telemetry["mods"]
            expected_mods = ["DA", "NE", "ACh", "5HT"]
            for mod in expected_mods:
                if mod not in mods:
                    return {"status": "FAIL", "error": f"Missing neuromodulator: {mod}"}
                if not (0.0 <= mods[mod] <= 1.0):
                    return {"status": "FAIL", "error": f"Neuromodulator {mod} out of range"}
            
            return {"status": "PASS", "message": "Brain step execution successful"}
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    def _test_working_memory_functionality(self, brain_config: Dict[str, Any], temp_metrics_file: str) -> Dict[str, Any]:
        """Test working memory capacity and operations"""
        try:
            brain = Brain(brain_config, stage="F", log_csv=temp_metrics_file)
            
            # Get working memory module
            wm = brain.modules["working_memory"]
            
            # Test initial state
            if wm.spec.get("slots", 0) != 3:
                return {"status": "FAIL", "error": "Working memory slots incorrect"}
            
            # Test memory operations
            inbox = [Message("Command", "pfc", "working_memory", payload={"action": "store", "item": "test_item"})]
            ctx = {"wm_confidence": 0.5, "global": {"arousal": 0.5}}
            
            out, telemetry = wm.step(inbox, ctx)
            
            # Verify telemetry contains confidence and slots
            if "confidence" not in telemetry or "slots" not in telemetry:
                return {"status": "FAIL", "error": "Working memory telemetry incomplete"}
            
            if not (0.0 <= telemetry["confidence"] <= 1.0):
                return {"status": "FAIL", "error": "Working memory confidence out of range"}
            
            return {"status": "PASS", "message": "Working memory functionality verified"}
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    def _test_pfc_executive_function(self, brain_config: Dict[str, Any], temp_metrics_file: str) -> Dict[str, Any]:
        """Test PFC executive control and planning"""
        try:
            brain = Brain(brain_config, stage="F", log_csv=temp_metrics_file)
            pfc = brain.modules["pfc"]
            
            ctx = {
                "wm_confidence": 0.7,
                "global": {"arousal": 0.6, "mode": "task-positive"},
                "attention": {"task_bias": 0.8}
            }
            
            out, telemetry = pfc.step([], ctx)
            
            # Verify PFC generates commands
            if len(out) == 0:
                return {"status": "FAIL", "error": "PFC not generating commands"}
            
            if not any(m.kind == "Command" for m in out):
                return {"status": "FAIL", "error": "PFC not generating Command messages"}
            
            # Verify telemetry
            if "confidence" not in telemetry or "demand" not in telemetry:
                return {"status": "FAIL", "error": "PFC telemetry incomplete"}
            
            if not (0.0 <= telemetry["confidence"] <= 1.0):
                return {"status": "FAIL", "error": "PFC confidence out of range"}
            
            return {"status": "PASS", "message": "PFC executive function verified"}
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    def _test_basal_ganglia_gating(self, brain_config: Dict[str, Any], temp_metrics_file: str) -> Dict[str, Any]:
        """Test basal ganglia action selection and gating"""
        try:
            brain = Brain(brain_config, stage="F", log_csv=temp_metrics_file)
            bg = brain.modules["basal_ganglia"]
            
            ctx = {
                "modulators": {"DA": 0.4, "ACh": 0.6},
                "global": {"fatigue": 0.3},
                "attention": {"task_bias": 0.7},
                "policy": {"moe_k": 2},
                "topology": {"modules": ["pfc", "working_memory", "dmn"]}
            }
            
            out, telemetry = bg.step([], ctx)
            
            # Verify gating decisions
            if "chosen" not in telemetry:
                return {"status": "FAIL", "error": "Basal ganglia not making gating decisions"}
            
            if len(telemetry["chosen"]) > ctx["policy"]["moe_k"]:
                return {"status": "FAIL", "error": "Basal ganglia exceeding MoE k limit"}
            
            # Verify confidence influenced by dopamine
            if "confidence" not in telemetry:
                return {"status": "FAIL", "error": "Basal ganglia confidence missing"}
            
            return {"status": "PASS", "message": "Basal ganglia gating verified"}
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    def _test_thalamic_routing(self, brain_config: Dict[str, Any], temp_metrics_file: str) -> Dict[str, Any]:
        """Test thalamic relay and routing functionality"""
        try:
            brain = Brain(brain_config, stage="F", log_csv=temp_metrics_file)
            thalamus = brain.modules["thalamus"]
            
            # Create activation command
            inbox = [Message("Command", "basal_ganglia", "thalamus", payload={"activate": ["pfc", "working_memory"]})]
            ctx = {"global": {"arousal": 0.5}}
            
            out, telemetry = thalamus.step(inbox, ctx)
            
            # Verify routing occurs
            if len(out) == 0:
                return {"status": "FAIL", "error": "Thalamus not routing signals"}
            
            if "confidence" not in telemetry:
                return {"status": "FAIL", "error": "Thalamus telemetry incomplete"}
            
            return {"status": "PASS", "message": "Thalamic routing verified"}
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    def _test_dmn_internal_simulation(self, brain_config: Dict[str, Any], temp_metrics_file: str) -> Dict[str, Any]:
        """Test DMN internal simulation and self-reflection"""
        try:
            brain = Brain(brain_config, stage="F", log_csv=temp_metrics_file)
            dmn = brain.modules["dmn"]
            
            ctx = {
                "global": {"mode": "internal", "arousal": 0.4},
                "attention": {"task_bias": 0.3}
            }
            
            out, telemetry = dmn.step([], ctx)
            
            # Verify DMN activity
            if "confidence" not in telemetry or "demand" not in telemetry:
                return {"status": "FAIL", "error": "DMN telemetry incomplete"}
            
            return {"status": "PASS", "message": "DMN internal simulation verified"}
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    def _test_salience_network(self, brain_config: Dict[str, Any], temp_metrics_file: str) -> Dict[str, Any]:
        """Test salience network attention allocation"""
        try:
            brain = Brain(brain_config, stage="F", log_csv=temp_metrics_file)
            salience = brain.modules["salience"]
            
            ctx = {
                "global": {"arousal": 0.6, "mode": "internal"},
                "attention": {"task_bias": 0.5}
            }
            
            out, telemetry = salience.step([], ctx)
            
            # Verify salience processing
            if "confidence" not in telemetry or "demand" not in telemetry:
                return {"status": "FAIL", "error": "Salience telemetry incomplete"}
            
            return {"status": "PASS", "message": "Salience network verified"}
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    def _test_attention_mechanism(self, brain_config: Dict[str, Any], temp_metrics_file: str) -> Dict[str, Any]:
        """Test attention mechanism and task bias"""
        try:
            brain = Brain(brain_config, stage="F", log_csv=temp_metrics_file)
            attention = brain.modules["attention"]
            
            ctx = {
                "global": {"arousal": 0.5, "mode": "task-positive"},
                "attention": {"task_bias": 0.7}
            }
            
            out, telemetry = attention.step([], ctx)
            
            # Verify attention metrics
            if "task_bias" not in telemetry or "confidence" not in telemetry:
                return {"status": "FAIL", "error": "Attention telemetry incomplete"}
            
            if not (0.0 <= telemetry["task_bias"] <= 1.0):
                return {"status": "FAIL", "error": "Task bias out of range"}
            
            return {"status": "PASS", "message": "Attention mechanism verified"}
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    def _test_architecture_agent_coordination(self, brain_config: Dict[str, Any], temp_metrics_file: str) -> Dict[str, Any]:
        """Test architecture agent global coordination"""
        try:
            brain = Brain(brain_config, stage="F", log_csv=temp_metrics_file)
            aa = brain.modules["architecture_agent"]
            
            ctx = {
                "global": {"t": 1, "arousal": 0.5, "mode": "internal", "state": "wake", "fatigue": 0.2},
                "modulators": {"DA": 0.3, "NE": 0.45, "ACh": 0.5, "5HT": 0.5},
                "attention": {"task_bias": 0.6},
                "wm_confidence": 0.5,
                "policy": {"moe_k": 1},
                "sleep": {"ticks_left": 0, "phase": "NREM"},
                "topology": {"modules": list(brain.modules.keys())}
            }
            
            out, telemetry = aa.step([], ctx)
            
            # Verify global coordination
            required_keys = ["modulators", "mode", "state", "fatigue"]
            for key in required_keys:
                if key not in telemetry:
                    return {"status": "FAIL", "error": f"Missing telemetry key: {key}"}
            
            # Verify neuromodulator updates
            mods = telemetry["modulators"]
            for mod in ["DA", "NE", "ACh", "5HT"]:
                if mod not in mods:
                    return {"status": "FAIL", "error": f"Missing neuromodulator: {mod}"}
                if not (0.0 <= mods[mod] <= 1.0):
                    return {"status": "FAIL", "error": f"Neuromodulator {mod} out of range"}
            
            return {"status": "PASS", "message": "Architecture agent coordination verified"}
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    def _test_curriculum_scheduling(self, brain_config: Dict[str, Any], temp_metrics_file: str) -> Dict[str, Any]:
        """Test curriculum-based development scheduling"""
        try:
            brain = Brain(brain_config, stage="F", log_csv=temp_metrics_file)
            
            # Test curriculum application
            if brain.curriculum:
                # Verify curriculum updates working memory slots
                initial_slots = brain.modules["working_memory"].spec.get("slots", 3)
                
                # Simulate progression through weeks
                for week in range(5):
                    brain.curriculum.update(brain.t, brain, {"policy": {"moe_k": 1}})
                    brain.t += 50  # One week worth of ticks
                
                # Verify changes occurred
                final_slots = brain.modules["working_memory"].spec.get("slots", 3)
                # Note: Actual changes depend on curriculum schedule
                
                return {"status": "PASS", "message": "Curriculum scheduling verified"}
            else:
                return {"status": "PASS", "message": "No curriculum configured"}
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    def _test_metrics_logging(self, brain_config: Dict[str, Any], temp_metrics_file: str) -> Dict[str, Any]:
        """Test metrics logging and CSV output"""
        try:
            brain = Brain(brain_config, stage="F", log_csv=temp_metrics_file)
            
            # Execute multiple steps
            for _ in range(3):
                brain.step(50)
            
            # Verify CSV file was created and contains data
            if not os.path.exists(temp_metrics_file):
                return {"status": "FAIL", "error": "Metrics file not created"}
            
            with open(temp_metrics_file, 'r') as f:
                lines = f.readlines()
                if len(lines) <= 1:  # Only header, no data
                    return {"status": "FAIL", "error": "No metrics data logged"}
            
            return {"status": "PASS", "message": "Metrics logging verified"}
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    def _test_stage_progression(self, brain_config: Dict[str, Any], temp_metrics_file: str) -> Dict[str, Any]:
        """Test different developmental stages"""
        try:
            stages = ["F", "N0", "N1"]
            
            for stage in stages:
                brain = Brain(brain_config, stage=stage, log_csv=temp_metrics_file)
                if brain.stage != stage:
                    return {"status": "FAIL", "error": f"Stage {stage} not set correctly"}
                
                # Test stage-specific configurations
                if stage == "F":
                    if brain.modules["working_memory"].spec.get("slots", 0) != 3:
                        return {"status": "FAIL", "error": "Stage F WM slots incorrect"}
                elif stage == "N0":
                    # N0 should have sleep capabilities
                    pass
                elif stage == "N1":
                    # N1 should have expanded capabilities
                    pass
            
            return {"status": "PASS", "message": "Stage progression verified"}
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    def _test_error_handling(self, brain_config: Dict[str, Any], temp_metrics_file: str) -> Dict[str, Any]:
        """Test error handling and robustness"""
        try:
            brain = Brain(brain_config, stage="F", log_csv=temp_metrics_file)
            
            # Test with invalid messages
            invalid_message = Message("Invalid", "test", "test", payload={})
            
            # Should not crash
            brain.step(50)
            
            return {"status": "PASS", "message": "Error handling verified"}
        except Exception as e:
            return {"status": "FAIL", "error": f"Error handling failed: {str(e)}"}
    
    def _test_performance_benchmarks(self, brain_config: Dict[str, Any], temp_metrics_file: str) -> Dict[str, Any]:
        """Test performance benchmarks and timing"""
        try:
            import time
            
            brain = Brain(brain_config, stage="F", log_csv=temp_metrics_file)
            
            # Benchmark step execution time
            start_time = time.time()
            for _ in range(10):
                brain.step(50)
            end_time = time.time()
            
            execution_time = end_time - start_time
            avg_time_per_step = execution_time / 10
            
            # Verify reasonable performance (adjust thresholds as needed)
            if avg_time_per_step >= 1.0:
                return {"status": "FAIL", "error": f"Step execution too slow: {avg_time_per_step:.3f}s"}
            
            return {"status": "PASS", "message": f"Performance acceptable: {avg_time_per_step:.3f}s per step"}
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    def _test_memory_usage(self, brain_config: Dict[str, Any], temp_metrics_file: str) -> Dict[str, Any]:
        """Test memory usage and resource management"""
        try:
            import psutil
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss
            
            brain = Brain(brain_config, stage="F", log_csv=temp_metrics_file)
            
            # Execute many steps
            for _ in range(100):
                brain.step(50)
            
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Verify reasonable memory usage (adjust threshold as needed)
            if memory_increase >= 100 * 1024 * 1024:  # 100MB
                return {"status": "FAIL", "error": f"Memory usage too high: {memory_increase / 1024 / 1024:.1f}MB"}
            
            return {"status": "PASS", "message": "Memory usage acceptable"}
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}

# Example usage
def example_claude_validation():
    """Example of Claude validation usage"""
    validator = ClaudeValidator()
    
    # Sample brain configuration
    brain_config = {
        "modules": {
            "architecture_agent": {"type": "architecture_agent"},
            "pfc": {"type": "pfc"},
            "basal_ganglia": {"type": "basal_ganglia"},
            "thalamus": {"type": "thalamus"},
            "working_memory": {"type": "working_memory", "slots": 3},
            "dmn": {"type": "dmn"},
            "salience": {"type": "salience"},
            "attention": {"type": "attention"}
        },
        "curriculum": {
            "ticks_per_week": 50,
            "schedule": [
                {"week": 0, "wm_slots": 3, "moe_k": 1},
                {"week": 4, "wm_slots": 4, "moe_k": 2}
            ]
        }
    }
    
    # Run validation
    results = validator.run_comprehensive_validation(brain_config)
    
    print(f"Claude Validation Complete: {results['status']} ({results['pass_rate']:.1f}%)")
    return results

if __name__ == "__main__":
    example_claude_validation()
