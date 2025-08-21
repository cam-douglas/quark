#!/usr/bin/env python3
"""
🧠 Executive Control Module - Prefrontal Cortex
Core executive functions for planning, decision-making, and cognitive control
"""

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass, field

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
        
        return self.get_status()
