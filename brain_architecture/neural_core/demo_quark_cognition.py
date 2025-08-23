#!/usr/bin/env python3
"""
🧠 QUARK Cognition Demo
Demonstrates QUARK's brain capabilities for cognitive tasks
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prefrontal_cortex.executive_control import ExecutiveControl
from working_memory.working_memory import WorkingMemory
from basal_ganglia.action_selection import ActionSelection, Action
from thalamus.information_relay import InformationRelay, SensoryInput
from hippocampus.episodic_memory import EpisodicMemory

class QUARKBrain:
    """Integrated QUARK brain system"""
    
    def __init__(self):
        print("🧠 Initializing QUARK Brain...")
        
        # Initialize brain modules
        self.executive = ExecutiveControl()
        self.working_memory = WorkingMemory(capacity=10)
        self.action_selection = ActionSelection()
        self.thalamus = InformationRelay()
        self.episodic_memory = EpisodicMemory(max_episodes=200, pattern_dim=32)
        
        print("✅ QUARK Brain initialized successfully!")
        print(f"   - Executive Control: Ready")
        print(f"   - Working Memory: {self.working_memory.capacity} slots available")
        print(f"   - Action Selection: Ready for decisions")
        print(f"   - Information Relay: Routing established")
        print(f"   - Episodic Memory: {self.episodic_memory.max_episodes} episodes capacity")
    
    def process_task(self, task_description: str):
        """Process a cognitive task using all brain modules"""
        print(f"\n🎯 Processing Task: {task_description}")
        
        # Step 1: Executive Control - Create plan
        print("\n📋 Step 1: Executive Planning...")
        plan = self.executive.create_plan(task_description, priority=0.9)
        print(f"   Created plan with {len(plan.steps)} steps: {plan.steps}")
        
        # Step 2: Working Memory - Store task information
        print("\n🧠 Step 2: Working Memory Storage...")
        self.working_memory.store(f"Task: {task_description}", priority=0.9)
        self.working_memory.store(f"Plan: {plan.steps}", priority=0.8)
        
        # Step 3: Thalamus - Process sensory input about task
        print("\n🔄 Step 3: Information Processing...")
        self.thalamus.receive_sensory_input(SensoryInput(
            modality="internal",
            content=f"New task received: {task_description}",
            intensity=0.9,
            priority=0.9,
            timestamp=0.0,
            source_id="user_input"
        ))
        
        # Step 4: Action Selection - Choose next action
        print("\n🎯 Step 4: Action Selection...")
        self.action_selection.add_action(Action(
            action_id="execute_plan",
            description=f"Execute plan for: {task_description}",
            expected_reward=0.9,
            confidence=0.8,
            effort=0.7,
            priority=0.9
        ))
        
        selected_action = self.action_selection.select_action({})
        if selected_action:
            print(f"   Selected action: {selected_action.description}")
        
        # Step 5: Episodic Memory - Store task episode
        print("\n📝 Step 5: Memory Formation...")
        episode_id = self.episodic_memory.store_episode(
            content={"task": task_description, "plan": plan.steps, "action": selected_action.description if selected_action else "none"},
            context={"source": "user", "priority": "high", "complexity": "medium"},
            emotional_valence=0.7,
            importance=0.9
        )
        print(f"   Stored episode: {episode_id}")
        
        # Step 6: Integration - Show brain status
        print("\n🔗 Step 6: Brain Integration Status...")
        self._show_brain_status()
        
        return {
            "plan": plan,
            "selected_action": selected_action,
            "episode_id": episode_id,
            "brain_status": self._get_brain_status()
        }
    
    def solve_problem(self, problem: str):
        """Use brain capabilities to solve a problem"""
        print(f"\n🧩 Solving Problem: {problem}")
        
        # Check if we have similar problems in memory
        print("\n🔍 Checking Memory for Similar Problems...")
        similar_episodes = self.episodic_memory.retrieve_episode({"task": problem}, max_results=3)
        
        if similar_episodes:
            print(f"   Found {len(similar_episodes)} similar experiences:")
            for episode in similar_episodes:
                print(f"     - {episode.content.get('task', 'Unknown task')}")
        else:
            print("   No similar problems found - this is a new challenge!")
        
        # Create solution plan
        print("\n💡 Creating Solution Plan...")
        solution_plan = self.executive.create_plan(f"Solve: {problem}", priority=0.95)
        print(f"   Solution plan: {solution_plan.steps}")
        
        # Store solution in working memory
        self.working_memory.store(f"Problem: {problem}", priority=0.95)
        self.working_memory.store(f"Solution: {solution_plan.steps}", priority=0.9)
        
        # Store solution episode
        solution_episode = self.episodic_memory.store_episode(
            content={"problem": problem, "solution": solution_plan.steps},
            context={"type": "problem_solving", "success": "pending"},
            emotional_valence=0.8,
            importance=0.95
        )
        
        print(f"   Solution stored as episode: {solution_episode}")
        return solution_plan
    
    def _show_brain_status(self):
        """Display current brain module status"""
        print("   📊 Brain Module Status:")
        print(f"     Executive: {self.executive.get_status()}")
        print(f"     Working Memory: {self.working_memory.get_status()}")
        print(f"     Action Selection: {self.action_selection.get_action_stats()}")
        print(f"     Thalamus: {self.thalamus.get_status()}")
        print(f"     Episodic Memory: {self.episodic_memory.get_memory_stats()}")
    
    def _get_brain_status(self) -> dict:
        """Get comprehensive brain status"""
        return {
            "executive": self.executive.get_status(),
            "working_memory": self.working_memory.get_status(),
            "action_selection": self.action_selection.get_action_stats(),
            "thalamus": self.thalamus.get_status(),
            "episodic_memory": self.episodic_memory.get_memory_stats()
        }

def main():
    """Main demonstration function"""
    print("🧠 QUARK Brain Cognition Demonstration")
    print("=" * 50)
    
    # Initialize QUARK brain
    quark_brain = QUARKBrain()
    
    # Demonstrate task processing
    print("\n" + "=" * 50)
    print("🎯 DEMONSTRATION 1: Task Processing")
    print("=" * 50)
    
    result1 = quark_brain.process_task("Develop a new machine learning algorithm")
    
    # Demonstrate problem solving
    print("\n" + "=" * 50)
    print("🧩 DEMONSTRATION 2: Problem Solving")
    print("=" * 50)
    
    result2 = quark_brain.solve_problem("Optimize neural network training speed")
    
    # Demonstrate memory retrieval
    print("\n" + "=" * 50)
    print("🔍 DEMONSTRATION 3: Memory Retrieval")
    print("=" * 50)
    
    # Retrieve recent memories
    recent_episodes = quark_brain.episodic_memory.retrieve_episode({"type": "problem_solving"}, max_results=5)
    print(f"   Retrieved {len(recent_episodes)} problem-solving episodes")
    
    for episode in recent_episodes:
        print(f"     - {episode.content.get('problem', 'Unknown problem')}")
    
    print("\n" + "=" * 50)
    print("✅ QUARK Brain Demonstration Complete!")
    print("=" * 50)
    
    print("\n🧠 QUARK's Brain Capabilities Demonstrated:")
    print("   ✅ Executive planning and decision-making")
    print("   ✅ Working memory management")
    print("   ✅ Action selection and learning")
    print("   ✅ Information routing and attention")
    print("   ✅ Episodic memory formation and retrieval")
    print("   ✅ Integrated cognitive processing")
    
    return quark_brain

if __name__ == "__main__":
    try:
        brain = main()
    except Exception as e:
        print(f"❌ Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()
