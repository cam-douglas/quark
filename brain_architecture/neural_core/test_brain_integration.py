#!/usr/bin/env python3
"""
🧠 Brain Integration Test
Tests the integration of executive control, working memory, action selection, and information relay
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prefrontal_cortex.executive_control import ExecutiveControl
from working_memory.working_memory import WorkingMemory
from basal_ganglia.action_selection import ActionSelection, Action
from thalamus.information_relay import InformationRelay, SensoryInput

def test_brain_integration():
    """Test basic brain module integration"""
    print("🧠 Testing QUARK Brain Integration...")
    
    # Initialize brain modules
    executive = ExecutiveControl()
    working_memory = WorkingMemory(capacity=8)
    action_selection = ActionSelection()
    thalamus = InformationRelay()
    
    print("✅ Brain modules initialized")
    
    # Test executive control
    print("\n📋 Testing Executive Control...")
    plan = executive.create_plan("Develop new neural network architecture", priority=0.8)
    print(f"   Created plan: {plan.goal}")
    print(f"   Steps: {plan.steps}")
    
    decision = executive.make_decision(["Use PyTorch", "Use TensorFlow", "Use JAX"])
    print(f"   Made decision: {decision.selected}")
    print(f"   Confidence: {decision.confidence:.2f}")
    
    # Test working memory
    print("\n🧠 Testing Working Memory...")
    working_memory.store("Neural network architecture requirements", priority=0.9)
    working_memory.store("PyTorch documentation", priority=0.7)
    working_memory.store("Previous project notes", priority=0.6)
    
    retrieved = working_memory.retrieve("architecture")
    if retrieved:
        print(f"   Retrieved: {retrieved.content}")
    
    # Test action selection
    print("\n🎯 Testing Action Selection...")
    action_selection.add_action(Action(
        action_id="research",
        description="Research neural network architectures",
        expected_reward=0.8,
        confidence=0.7,
        effort=0.6,
        priority=0.8
    ))
    
    action_selection.add_action(Action(
        action_id="implement",
        description="Implement basic architecture",
        expected_reward=0.9,
        confidence=0.5,
        effort=0.8,
        priority=0.9
    ))
    
    selected_action = action_selection.select_action({})
    if selected_action:
        print(f"   Selected action: {selected_action.description}")
    
    # Test thalamus information relay
    print("\n🔄 Testing Information Relay...")
    thalamus.receive_sensory_input(SensoryInput(
        modality="visual",
        content="Code editor with neural network code",
        intensity=0.8,
        priority=0.9,
        timestamp=0.0,
        source_id="code_editor"
    ))
    
    thalamus.set_attention_focus("visual", strength=1.2, duration=5.0, source="executive")
    
    routing_info = thalamus.get_routing_info("visual")
    print(f"   Visual routing targets: {routing_info['targets']}")
    print(f"   Attention strength: {routing_info['attention']:.2f}")
    
    # Test integration
    print("\n🔗 Testing Module Integration...")
    
    # Executive creates plan, working memory stores it
    plan = executive.create_plan("Test brain integration", priority=0.9)
    working_memory.store(f"Plan: {plan.goal}", priority=plan.priority)
    
    # Thalamus receives sensory input about the plan
    thalamus.receive_sensory_input(SensoryInput(
        modality="internal",
        content=f"Plan created: {plan.goal}",
        intensity=0.7,
        priority=0.8,
        timestamp=0.0,
        source_id="executive"
    ))
    
    # Action selection gets new action
    action_selection.add_action(Action(
        action_id="execute_plan",
        description=f"Execute plan: {plan.goal}",
        expected_reward=0.9,
        confidence=0.8,
        effort=0.7,
        priority=0.9
    ))
    
    # Get status from all modules
    print("\n📊 Brain Module Status:")
    print(f"   Executive: {executive.get_status()}")
    print(f"   Working Memory: {working_memory.get_status()}")
    print(f"   Action Selection: {action_selection.get_action_stats()}")
    print(f"   Thalamus: {thalamus.get_status()}")
    
    print("\n✅ Brain integration test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_brain_integration()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
