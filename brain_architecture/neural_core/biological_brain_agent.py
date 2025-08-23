#!/usr/bin/env python3
"""
🧠 Biological Brain Agent
Integrates with QUARK's brain architecture and task management system to make intelligent decisions
about task prioritization, execution timing, and resource allocation.
"""

# Ensure repository root is on sys.path for absolute imports like `testing.*`
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_here, '..', '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import time

# Import brain modules
from prefrontal_cortex.executive_control import ExecutiveControl
from working_memory.working_memory import WorkingMemory
from basal_ganglia.action_selection import ActionSelection
from thalamus.information_relay import InformationRelay
from hippocampus.episodic_memory import EpisodicMemory

@dataclass
class TaskDecision:
    """Represents a decision about task execution"""
    task_id: str
    decision: str  # "execute", "defer", "delegate", "reject"
    priority_score: float
    reasoning: str
    estimated_effort: float
    dependencies_met: bool

@dataclass
class ResourceState:
    """Current state of cognitive and computational resources"""
    cognitive_load: float
    working_memory_available: float
    attention_focus: str
    emotional_state: Dict[str, float]
    energy_level: float
    time_available: float

class BiologicalBrainAgent:
    """Biological brain agent for intelligent task decision-making"""
    
    def __init__(self):
        # Initialize brain modules
        self.executive = ExecutiveControl()
        self.working_memory = WorkingMemory(capacity=15)
        self.action_selection = ActionSelection()
        self.thalamus = InformationRelay()
        self.episodic_memory = EpisodicMemory(max_episodes=500, pattern_dim=32)
        
        # Task management state
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.resource_state = ResourceState(
            cognitive_load=0.3,
            working_memory_available=0.7,
            attention_focus="task_management",
            emotional_state={"motivation": 0.8, "stress": 0.2, "focus": 0.7},
            energy_level=0.8,
            time_available=1.0
        )
        
        # Decision parameters
        self.decision_thresholds = {
            "high_priority": 0.8,
            "medium_priority": 0.6,
            "low_priority": 0.4
        }
        
        print("🧠 Biological Brain Agent initialized")
    
    def load_tasks(self) -> Dict[str, Any]:
        """Load current tasks from the task management system"""
        # Simulate task loading for now
        self.active_tasks = {
            "task_1": {"id": "1", "title": "Complete Implementation Checklist", "priority": "HIGH", "status": "IN PROGRESS"},
            "task_2": {"id": "2", "title": "Deploy Core Framework", "priority": "HIGH", "status": "NOT STARTED"},
            "task_3": {"id": "3", "title": "Phase 1: Advanced Cognitive Integration", "priority": "HIGH", "status": "NOT STARTED"},
            "task_4": {"id": "4", "title": "SLM+LLM Integration", "priority": "MEDIUM", "status": "NOT STARTED"},
            "task_5": {"id": "5", "title": "Phase 2: Neural Dynamics", "priority": "MEDIUM", "status": "NOT STARTED"}
        }
        
        self.working_memory.store(f"Loaded {len(self.active_tasks)} tasks", priority=0.9)
        return self.active_tasks
    
    def analyze_task_priorities(self) -> List[Tuple[str, float]]:
        """Analyze and rank tasks by priority using brain modules"""
        if not self.active_tasks:
            self.load_tasks()
        
        task_priorities = []
        
        for task_id, task in self.active_tasks.items():
            priority_score = self._calculate_task_priority(task)
            task_priorities.append((task_id, priority_score))
            
            # Store task analysis in episodic memory
            self.episodic_memory.store_episode(
                content={"task_analysis": task, "priority_score": priority_score},
                context={"analysis_type": "priority_calculation"},
                emotional_valence=0.5,
                importance=priority_score
            )
        
        # Sort by priority score (highest first)
        task_priorities.sort(key=lambda x: x[1], reverse=True)
        return task_priorities
    
    def _calculate_task_priority(self, task: Dict[str, Any]) -> float:
        """Calculate task priority using multiple cognitive factors"""
        base_priority = 0.5
        
        # Priority level adjustment
        if "HIGH" in task.get("priority", ""):
            base_priority += 0.3
        elif "MEDIUM" in task.get("priority", ""):
            base_priority += 0.1
        
        # Status adjustment
        if "IN PROGRESS" in task.get("status", ""):
            base_priority += 0.2
        
        # Emotional state adjustment
        motivation = self.resource_state.emotional_state.get("motivation", 0.5)
        base_priority *= (0.8 + 0.4 * motivation)
        
        return min(1.0, max(0.0, base_priority))
    
    def make_task_decisions(self) -> List[TaskDecision]:
        """Make intelligent decisions about which tasks to execute"""
        task_priorities = self.analyze_task_priorities()
        self._update_resource_state()
        
        decisions = []
        
        for task_id, priority_score in task_priorities:
            decision = self._make_single_task_decision(task_id, priority_score)
            decisions.append(decision)
            
            # Store decision in episodic memory
            self.episodic_memory.store_episode(
                content={"task_decision": decision.__dict__},
                context={"decision_type": "task_execution"},
                emotional_valence=0.6,
                importance=priority_score
            )
        
        return decisions
    
    def _make_single_task_decision(self, task_id: str, priority_score: float) -> TaskDecision:
        """Make decision for a single task"""
        task = self.active_tasks.get(task_id, {})
        resource_available = self._check_resource_availability(task)
        
        # Determine decision based on priority and resources
        if priority_score >= self.decision_thresholds["high_priority"] and resource_available:
            decision_type = "execute"
        elif priority_score >= self.decision_thresholds["medium_priority"] and resource_available:
            decision_type = "execute"
        else:
            decision_type = "defer"
        
        reasoning = self._generate_decision_reasoning(task_id, priority_score, decision_type, resource_available)
        estimated_effort = self._estimate_task_effort(task)
        dependencies_met = True  # Simplified for now
        
        return TaskDecision(
            task_id=task_id,
            decision=decision_type,
            priority_score=priority_score,
            reasoning=reasoning,
            estimated_effort=estimated_effort,
            dependencies_met=dependencies_met
        )
    
    def _check_resource_availability(self, task: Dict[str, Any]) -> bool:
        """Check if sufficient resources are available"""
        if self.resource_state.cognitive_load > 0.8:
            return False
        if self.resource_state.working_memory_available < 0.3:
            return False
        if self.resource_state.energy_level < 0.4:
            return False
        return True
    
    def _generate_decision_reasoning(self, task_id: str, priority_score: float, 
                                   decision: str, resource_available: bool) -> str:
        """Generate reasoning for the decision"""
        if decision == "execute":
            if resource_available:
                reasoning = f"Task {task_id} has high priority ({priority_score:.2f}) and sufficient resources"
            else:
                reasoning = f"Task {task_id} has high priority but insufficient resources - deferring"
        else:
            reasoning = f"Task {task_id} has low priority ({priority_score:.2f}) - deferring"
        
        return reasoning
    
    def _estimate_task_effort(self, task: Dict[str, Any]) -> float:
        """Estimate effort required for task execution"""
        return 0.5  # Default medium effort
    
    def _update_resource_state(self):
        """Update current resource state"""
        wm_status = self.working_memory.get_status()
        self.resource_state.working_memory_available = wm_status["available_slots"] / max(wm_status["used_slots"], 1)
        
        active_task_count = len([t for t in self.active_tasks.values() if "IN PROGRESS" in t.get("status", "")])
        self.resource_state.cognitive_load = min(1.0, active_task_count * 0.2)
    
    def execute_task_plan(self, decisions: List[TaskDecision]) -> Dict[str, Any]:
        """Execute the task plan based on decisions"""
        print("\n🧠 Executing Task Plan...")
        
        execution_results = {
            "executed_tasks": [],
            "deferred_tasks": [],
            "execution_summary": {}
        }
        
        for decision in decisions:
            if decision.decision == "execute" and decision.dependencies_met:
                print(f"   🚀 Executing: {decision.task_id}")
                result = self._execute_single_task(decision)
                execution_results["executed_tasks"].append(result)
            else:
                print(f"   ⏸️  Deferring: {decision.task_id}")
                execution_results["deferred_tasks"].append(decision.task_id)
        
        self._update_resource_state()
        
        # Store execution results
        self.episodic_memory.store_episode(
            content={"execution_results": execution_results},
            context={"execution_type": "task_plan"},
            emotional_valence=0.7,
            importance=0.8
        )
        
        return execution_results
    
    def _execute_single_task(self, decision: TaskDecision) -> Dict[str, Any]:
        """Execute a single task"""
        execution_result = {
            "task_id": decision.task_id,
            "status": "executing",
            "start_time": time.time(),
            "priority_score": decision.priority_score,
            "estimated_effort": decision.estimated_effort
        }
        
        # Update task status
        if decision.task_id in self.active_tasks:
            self.active_tasks[decision.task_id]["status"] = "IN PROGRESS"
        
        # Store execution in working memory
        self.working_memory.store(f"Executing task: {decision.task_id}", priority=0.9)
        
        return execution_result
    
    def get_brain_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the biological brain agent"""
        return {
            "resource_state": {
                "cognitive_load": self.resource_state.cognitive_load,
                "working_memory_available": self.resource_state.working_memory_available,
                "attention_focus": self.resource_state.attention_focus,
                "emotional_state": self.resource_state.emotional_state,
                "energy_level": self.resource_state.energy_level,
                "time_available": self.resource_state.time_available
            },
            "task_management": {
                "active_tasks": len(self.active_tasks),
                "decisions_made": 0
            },
            "brain_modules": {
                "executive": self.executive.get_status(),
                "working_memory": self.working_memory.get_status(),
                "action_selection": self.action_selection.get_action_stats(),
                "thalamus": self.thalamus.get_status(),
                "episodic_memory": self.episodic_memory.get_memory_stats()
            }
        }
    
    def step(self) -> Dict[str, Any]:
        """Step the biological brain agent forward"""
        # Load current tasks
        tasks = self.load_tasks()
        
        # Make task decisions
        decisions = self.make_task_decisions()
        
        # Execute task plan
        execution_results = self.execute_task_plan(decisions)
        
        # Update brain modules
        brain_outputs = {
            "executive": self.executive.step({}),
            "working_memory": self.working_memory.step({}),
            "action_selection": self.action_selection.step({}),
            "thalamus": self.thalamus.step({}),
            "episodic_memory": self.episodic_memory.step({})
        }
        
        return {
            "task_decisions": [d.__dict__ for d in decisions],
            "execution_results": execution_results,
            "brain_outputs": brain_outputs,
            "agent_status": self.get_brain_agent_status()
        }

def main():
    """Main function to demonstrate the biological brain agent"""
    print("🧠 Biological Brain Agent Demonstration")
    print("=" * 50)
    
    # Initialize the agent
    agent = BiologicalBrainAgent()
    
    # Step the agent forward
    print("\n🔄 Stepping Biological Brain Agent...")
    results = agent.step()
    
    # Display results
    print("\n📊 Task Decisions Made:")
    for decision in results["task_decisions"]:
        print(f"   {decision['task_id']}: {decision['decision']} (Priority: {decision['priority_score']:.2f})")
        print(f"     Reasoning: {decision['reasoning']}")
    
    print(f"\n🚀 Executed Tasks: {len(results['execution_results']['executed_tasks'])}")
    print(f"⏸️  Deferred Tasks: {len(results['execution_results']['deferred_tasks'])}")
    
    print("\n🧠 Brain Agent Status:")
    status = results["agent_status"]
    print(f"   Cognitive Load: {status['resource_state']['cognitive_load']:.2f}")
    print(f"   Working Memory Available: {status['resource_state']['working_memory_available']:.2f}")
    print(f"   Motivation: {status['resource_state']['emotional_state']['motivation']:.2f}")
    
    print("\n✅ Biological Brain Agent demonstration complete!")
    return agent

if __name__ == "__main__":
    try:
        agent = main()
    except Exception as e:
        print(f"❌ Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()
