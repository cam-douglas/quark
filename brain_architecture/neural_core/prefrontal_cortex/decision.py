# brain_architecture/neural_core/prefrontal_cortex/decision.py
from typing import List
from dataclasses import dataclass

@dataclass
class Decision:
    """Decision made by executive system"""
    options: List[str]
    selected: str
    confidence: float
    reasoning: str
