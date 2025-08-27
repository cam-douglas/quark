# brain_architecture/neural_core/prefrontal_cortex/plan.py
from typing import List
from dataclasses import dataclass

@dataclass
class Plan:
    """Cognitive plan with steps and priority"""
    goal: str
    steps: List[str]
    priority: float
    status: str = "active"
