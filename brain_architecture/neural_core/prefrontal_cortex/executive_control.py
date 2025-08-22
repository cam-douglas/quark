#!/usr/bin/env python3
"""
🧠 Executive Control Module - Prefrontal Cortex
Core executive functions for planning, decision-making, and cognitive control
"""

import numpy as np
import time
import urllib.request
import urllib.parse
from typing import Dict, List, Any
from dataclasses import dataclass, field
import json
import os
from testing.testing_frameworks.scientific_validation import ScientificValidator
from brain_architecture.neural_core.basal_ganglia.dopamine_system import DopamineSystem

@dataclass
class Plan:
    """Cognitive plan with steps and priority"""
    goal: str
    steps: List[str]
    priority: float
    status: str = "active"

@dataclass
class Decision:
    """Decision made by executive system"""
    options: List[str]
    selected: str
    confidence: float
    reasoning: str

class ExecutiveControl:
    """Core executive control for planning and decision-making"""
    
    def __init__(self):
        self.plans: List[Plan] = []
        self.decisions: List[Decision] = []
        self.cognitive_resources = {"attention": 1.0, "memory": 1.0}
        self.validator = ScientificValidator()
        self.dopamine_system = DopamineSystem() # Instantiate the dopamine system
        self._last_goal_push_s: float = 0.0
        self._viewer_url: str = os.environ.get("QUARK_VIEWER_URL", "http://127.0.0.1:8011")
        
    def ingest_task_analysis(self, analysis_path: str = "tasks/integrations/brain_analysis/brain_task_analysis.json"):
        """Reads the task analysis file and converts tasks into internal plans."""
        if not os.path.exists(analysis_path):
            print(f"Warning: Task analysis file not found at {analysis_path}")
            return

        with open(analysis_path, 'r') as f:
            analysis = json.load(f)

        # Clear existing plans derived from this system to avoid duplication
        self.plans = [p for p in self.plans if not p.goal.startswith("DEV_TASK:")]

        task_summary = analysis.get("task_summary", {})
        for task_id, task in task_summary.items():
            # Create a goal string that uniquely identifies the development task
            goal_str = f"DEV_TASK: {task.get('title', 'Untitled Task')}"
            
            # Use cognitive load estimate to set priority (higher load = higher priority focus)
            priority = task.get('cognitive_load', 0.5)

            # Create a plan if it's a high-priority task
            if task.get('priority') == 'high' and task.get('status') != 'Completed':
                self.create_plan(goal=goal_str, priority=priority)
        
        # After ingesting development tasks, create a standing goal to self-validate.
        self.create_plan(goal="SELF_VALIDATE: Check internal models against scientific benchmarks", priority=0.9)
        
        print(f"Ingested {len(task_summary)} tasks. Created {len(self.plans)} new high-priority development plans.")

    def create_plan(self, goal: str, priority: float = 0.5) -> Plan:
        """Create new cognitive plan"""
        steps = self._generate_steps(goal)
        plan = Plan(goal=goal, steps=steps, priority=priority)
        self.plans.append(plan)
        return plan
    
    def _generate_steps(self, goal: str) -> List[str]:
        """Generate plan steps based on goal complexity"""
        complexity = len(goal.split())
        if complexity <= 3:
            return ["analyze", "execute", "validate"]
        elif complexity <= 6:
            return ["research", "plan", "implement", "test"]
        else:
            return ["research", "design", "prototype", "test", "deploy"]
    
    def make_decision(self, options: List[str]) -> Decision:
        """Make decision from available options"""
        # Simple selection - in full implementation would use more sophisticated evaluation
        selected = options[0] if options else ""
        confidence = 0.7
        reasoning = f"Selected {selected} based on current context"
        
        decision = Decision(
            options=options,
            selected=selected,
            confidence=confidence,
            reasoning=reasoning
        )
        self.decisions.append(decision)
        return decision
    
    def get_status(self) -> Dict[str, Any]:
        """Get current executive control status"""
        return {
            "active_plans": len([p for p in self.plans if p.status == "active"]),
            "total_decisions": len(self.decisions),
            "cognitive_resources": self.cognitive_resources.copy()
        }
    
    def step(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Step executive control forward"""
        # Process new goals
        if "new_goals" in inputs:
            for goal in inputs["new_goals"]:
                self.create_plan(goal)
        
        # Make decisions if requested
        if "decision_requests" in inputs:
            for request in inputs["decision_requests"]:
                self.make_decision(request.get("options", []))
        
        # --- Goal-Driven Motivation Loop ---
        # Find the highest priority active development task
        active_dev_plans = [p for p in self.plans if p.goal.startswith("DEV_TASK:") and p.status == "active"]
        if active_dev_plans:
            highest_priority_plan = max(active_dev_plans, key=lambda p: p.priority)
            # Set motivational bias based on the task's priority (cognitive load)
            # We scale it to the -0.2 to 0.2 range expected by the dopamine system
            bias = (highest_priority_plan.priority - 0.5) * 0.4 
            self.dopamine_system.set_motivational_bias(bias)
            print(f"PFC: Setting motivational bias to {bias:.2f} for task '{highest_priority_plan.goal}'")
            # Push priority to live viewer (/goal) occasionally
            try:
                now = time.time()
                if now - self._last_goal_push_s > 2.0:
                    pr = max(0.0, min(1.0, float(highest_priority_plan.priority)))
                    url = f"{self._viewer_url}/goal?priority={pr:.3f}"
                    req = urllib.request.Request(url, method='GET')
                    urllib.request.urlopen(req, timeout=0.5).read()
                    self._last_goal_push_s = now
            except Exception:
                pass
        else:
            # If no active dev tasks, reset motivational bias
            self.dopamine_system.set_motivational_bias(0.0)

        # Periodically run self-validation if it's an active goal
        if any(p.goal.startswith("SELF_VALIDATE") and p.status == "active" for p in self.plans):
            # In a real simulation, this data would come from the live brain model
            mock_agi_data = {'activity': np.random.rand(100)} 
            self.validator.run_validation(mock_agi_data, "neural_bench")

        return self.get_status()
