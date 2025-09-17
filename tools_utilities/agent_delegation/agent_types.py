"""
Agent Delegation Types

Type definitions and enums for the agent delegation system.

Author: Quark AI
Date: 2025-01-27
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DelegatedTask:
    """Represents a task delegated to a background agent"""
    id: str
    title: str
    description: str
    agent_type: str  # "cline", "compliance", "testing", "documentation"
    priority: int = 1  # 1-5, higher is more urgent
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
