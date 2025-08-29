#!/usr/bin/env python3
"""
🧠 Test Brain-Task Integration

This script demonstrates the integration between the brain's automatic goal system
and the central task management system.
"""

import sys
import time
from pathlib import Path

# Add the parent directory to the path to import the integration module
sys.path.append(str(Path(__file__).parent))

from brain_task_bridge import BrainTaskIntegrationManager

def test_brain_integration():
    """Test the brain-task integration system"""
    print("🧠 Testing Brain-Task Integration System")
    print("=" * 50)
    
    # Create integration manager
    integration_manager = BrainTaskIntegrationManager()
    
    print("✅ Integration manager created")
    
    # Test goal generation
    print("\n🎯 Testing Brain Goal Generation:")
    test_goals = integration_manager.generate_test_goals()
    
    for i, goal in enumerate(test_goals, 1):
        print(f"  {i}. {goal['title']}")
        print(f"     Priority: {goal['priority']}")
        print(f"     Type: {goal.get('tags', ['general'])[-1]}")
        print(f"     Effort: {goal.get('estimated_effort', 'medium')}")
        print()
    
    # Start integration
    print("🔄 Starting Brain-Task Integration...")
    integration_manager.start_integration()
    
    # Monitor integration for a few cycles
    print("\n📊 Monitoring Integration Status:")
    for i in range(5):
        status = integration_manager.get_integration_status()
        
        print(f"\n--- Cycle {i+1} ---")
        print(f"Integration Status: {status['integration_status']}")
        print(f"Synchronization Active: {status['synchronization_active']}")
        print(f"Brain Tasks Directory: {status['brain_tasks_directory']}")
        
        # Show brain state
        brain_state = status['brain_state']
        consciousness = brain_state['consciousness']
        attention = brain_state['attention']
        
        print(f"Consciousness State:")
        print(f"  Awake: {consciousness['awake']}")
        print(f"  Cognitive Load: {consciousness['cognitive_load']:.2f}")
        print(f"  Learning Mode: {consciousness['learning_mode']}")
        print(f"  Attention Focus: {consciousness['attention_focus']}")
        
        print(f"Attention State:")
        print(f"  Task Bias: {attention['task_bias']:.2f}")
        print(f"  Internal Bias: {attention['internal_bias']:.2f}")
        print(f"  Focus Target: {attention['focus_target']}")
        
        time.sleep(2)
    
    # Stop integration
    print("\n🛑 Stopping Integration...")
    integration_manager.stop_integration()
    
    print("\n✅ Test completed successfully!")
    print("\n📁 Check the following directories for generated content:")
    print(f"  - {integration_manager.synchronizer.task_system_path}/active_tasks/brain_generated/")
    print(f"  - {integration_manager.synchronizer.task_system_path}/active_tasks/brain_generated/TASK_SUMMARY.md")

def test_individual_components():
    """Test individual components of the integration system"""
    print("\n🔧 Testing Individual Components")
    print("=" * 40)
    
    from brain_task_bridge import BrainStateMonitor, BrainGoalGenerator, BrainGoalTranslator
    
    # Test Brain State Monitor
    print("\n1. Testing Brain State Monitor:")
    monitor = BrainStateMonitor()
    state = monitor.get_current_state()
    print(f"   Initial consciousness state: {state['consciousness']}")
    
    # Test Goal Generator
    print("\n2. Testing Goal Generator:")
    generator = BrainGoalGenerator()
    goals = generator.generate_goals(state['consciousness'])
    print(f"   Generated {len(goals)} goals")
    
    for goal in goals:
        print(f"     - {goal['description']} ({goal['priority']} priority)")
    
    # Test Goal Translator
    print("\n3. Testing Goal Translator:")
    translator = BrainGoalTranslator()
    tasks = translator.translate_brain_goals(goals)
    print(f"   Translated to {len(tasks)} tasks")
    
    for task in tasks:
        print(f"     - {task['title']}")
        print(f"       Priority: {task['priority']}")
        print(f"       Effort: {task['estimated_effort']}")
        print(f"       Due: {task['due_date']}")

def main():
    """Main test function"""
    print("🧠 Brain-Task Integration Test Suite")
    print("=" * 60)
    
    try:
        # Test individual components
        test_individual_components()
        
        # Test full integration
        test_brain_integration()
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
