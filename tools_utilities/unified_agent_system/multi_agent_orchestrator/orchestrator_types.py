"""
Multi-Agent Orchestrator Types

Type definitions and enums for the multi-agent orchestrator system.

Author: Quark AI
Date: 2025-01-27
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class AgentRole(Enum):
    """Engineering roles from .quark/rules/engineering-roles.mdc"""
    PROGRAM_ORCHESTRATOR = "program_orchestrator"
    DEVELOPMENTAL_NEUROBIOLOGIST = "developmental_neurobiologist"
    MOLECULAR_GENETICIST = "molecular_geneticist"
    COMPUTATIONAL_BIOLOGIST = "computational_biologist"
    TISSUE_MORPHOGENESIS_ENGINEER = "tissue_morphogenesis_engineer"
    CONNECTOMICS_ENGINEER = "connectomics_engineer"
    CIRCUIT_BUILDER = "circuit_builder"
    NEUROPLASTICITY_LEARNING_SCIENTIST = "neuroplasticity_learning_scientist"
    SELF_ORGANIZATION_ENGINEER = "self_organization_engineer"
    MULTIMODAL_SENSORY_ENGINEER = "multimodal_sensory_engineer"
    COGNITIVE_ARCHITECT = "cognitive_architect"
    NEUROMORPHIC_SYSTEMS_ENGINEER = "neuromorphic_systems_engineer"
    DATA_PIPELINES_ENGINEER = "data_pipelines_engineer"
    EVALUATION_BEHAVIOR_SCIENTIST = "evaluation_behavior_scientist"
    SAFETY_ETHICS_OFFICER = "safety_ethics_officer"
    PRODUCT_HCI = "product_hci"
    QA_REPRODUCIBILITY_ENGINEER = "qa_reproducibility_engineer"


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SpecializedTask:
    """Represents a task delegated to a specialized engineering agent"""
    id: str
    title: str
    description: str
    role: AgentRole
    priority: int = 1  # 1-5, higher is more urgent
    deliverables: List[str] = None
    dependencies: List[str] = None
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    compliance_checked: bool = False
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.deliverables is None:
            self.deliverables = []
        if self.dependencies is None:
            self.dependencies = []
