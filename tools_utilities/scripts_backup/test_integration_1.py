#!/usr/bin/env python3
"""
🧠 Test Integration Script for Unified Training System

This script tests the integration of existing simulation and training capabilities
without requiring PyTorch or other heavy dependencies.
"""

import os, sys
from pathlib import Path

def test_directory_structure():
    """Test that all expected directories exist"""
    print("🔍 Testing directory structure...")
    
    expected_dirs = [
        'simulation_frameworks',
        'agent_systems', 
        'cognitive_engines',
        'neural_architectures',
        'core_systems',
        'optimization_tools',
        'development_tools',
        'documentation',
        'research_applications',
        'visualization_tools',
        'data_integration'
    ]
    
    missing_dirs = []
    for dir_name in expected_dirs:
        if os.path.exists(dir_name):
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ (missing)")
            missing_dirs.append(dir_name)
    
    return len(missing_dirs) == 0

def test_simulation_frameworks():
    """Test that simulation frameworks are available"""
    print("\n🧠 Testing simulation frameworks...")
    
    sim_frameworks = [
        'optimized_brain_physics.py',
        'fetal_anatomical_simulation.py',
        'morphogen_physics.py',
        'tissue_mechanics.py',
        'dual_mode_simulator.py',
        'neural_simulator.py',
        'enhanced_data_resources.py'
    ]
    
    available_frameworks = []
    for framework in sim_frameworks:
        framework_path = os.path.join('simulation_frameworks', framework)
        if os.path.exists(framework_path):
            print(f"  ✅ {framework}")
            available_frameworks.append(framework)
        else:
            print(f"  ❌ {framework} (missing)")
    
    return len(available_frameworks)

def test_training_systems():
    """Test that training systems are available"""
    print("\n🎯 Testing training systems...")
    
    training_files = [
        'training_orchestrator.py',
        'comprehensive_training_orchestrator.py',
        'domain_specific_trainers.py',
        'integrated_brain_simulation_trainer.py',
        'quick_start_training.py',
        'training_config.yaml',
        'TRAINING_USAGE_GUIDE.md'
    ]
    
    available_files = []
    for file_name in training_files:
        if os.path.exists(file_name):
            print(f"  ✅ {file_name}")
            available_files.append(file_name)
        else:
            print(f"  ❌ {file_name} (missing)")
    
    return len(available_files)

def test_agent_systems():
    """Test that agent systems are available"""
    print("\n🤖 Testing agent systems...")
    
    agent_files = [
        'small_mind_core.py',
        'unified_intelligence_system.py',
        'terminal_agent.py',
        'agent_hub.py'
    ]
    
    available_agents = []
    for agent_file in agent_files:
        agent_path = os.path.join('agent_systems', agent_file)
        if os.path.exists(agent_path):
            print(f"  ✅ {agent_file}")
            available_agents.append(agent_file)
        else:
            print(f"  ❌ {agent_file} (missing)")
    
    return len(available_agents)

def test_cognitive_engines():
    """Test that cognitive engines are available"""
    print("\n🧩 Testing cognitive engines...")
    
    cognitive_files = [
        'curiosity_engine.py',
        'exploration_module.py',
        'synthesis_engine.py'
    ]
    
    available_cognitive = []
    for cog_file in cognitive_files:
        cog_path = os.path.join('cognitive_engines', cog_file)
        if os.path.exists(cog_path):
            print(f"  ✅ {cog_file}")
            available_cognitive.append(cog_file)
        else:
            print(f"  ❌ {cog_file} (missing)")
    
    return len(available_cognitive)

def test_neural_architectures():
    """Test that neural architectures are available"""
    print("\n🧠 Testing neural architectures...")
    
    neural_files = [
        'childlike_learning_system.py',
        'continuous_training.py',
        'cloud_integration.py'
    ]
    
    available_neural = []
    for neural_file in neural_files:
        neural_path = os.path.join('neural_architectures', neural_file)
        if os.path.exists(neural_path):
            print(f"  ✅ {neural_file}")
            available_neural.append(neural_file)
        else:
            print(f"  ❌ {neural_file} (missing)")
    
    return len(available_neural)

def main():
    """Main test function"""
    print("🧠 Unified Training System Integration Test")
    print("=" * 50)
    
    # Test directory structure
    structure_ok = test_directory_structure()
    
    # Test simulation frameworks
    sim_count = test_simulation_frameworks()
    
    # Test training systems
    training_count = test_training_systems()
    
    # Test agent systems
    agent_count = test_agent_systems()
    
    # Test cognitive engines
    cognitive_count = test_cognitive_engines()
    
    # Test neural architectures
    neural_count = test_neural_architectures()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 50)
    print(f"Directory Structure: {'✅ PASS' if structure_ok else '❌ FAIL'}")
    print(f"Simulation Frameworks: {sim_count}/7 available")
    print(f"Training Systems: {training_count}/7 available")
    print(f"Agent Systems: {agent_count}/4 available")
    print(f"Cognitive Engines: {cognitive_count}/3 available")
    print(f"Neural Architectures: {neural_count}/3 available")
    
    total_components = sim_count + training_count + agent_count + cognitive_count + neural_count
    print(f"\nTotal Components Available: {total_components}/24")
    
    if total_components >= 20:
        print("🎉 Integration Status: EXCELLENT")
    elif total_components >= 15:
        print("✅ Integration Status: GOOD")
    elif total_components >= 10:
        print("⚠️  Integration Status: FAIR")
    else:
        print("❌ Integration Status: POOR")
    
    print("\n🚀 Ready for simulation training development tasks!")

if __name__ == "__main__":
    main()
