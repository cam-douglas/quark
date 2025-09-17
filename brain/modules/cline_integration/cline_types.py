"""
Cline Types - Core type definitions for Cline integration

Defines the fundamental types used throughout the Cline integration system
for task representation, execution results, and complexity assessment.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any


class TaskComplexity(Enum):
    """Task complexity levels for autonomous delegation"""
    SIMPLE = 1      # Basic file edits, single function changes
    MODERATE = 2    # Multi-file changes, refactoring
    COMPLEX = 3     # Architecture changes, new module creation
    CRITICAL = 4    # Brain architecture modifications, biological systems


class ClineTaskType(Enum):
    """Types of tasks that can be delegated to Cline"""
    CODE_GENERATION = "code_generation"
    FILE_EDITING = "file_editing"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    REFACTORING = "refactoring"
    BROWSER_TESTING = "browser_testing"
    COMMAND_EXECUTION = "command_execution"


@dataclass
class CodingTask:
    """Represents a coding task for autonomous execution"""
    description: str
    task_type: ClineTaskType
    complexity: TaskComplexity
    files_involved: List[str]
    biological_constraints: bool = True
    context: Optional[Dict[str, Any]] = None
    working_directory: Optional[str] = None


@dataclass
class TaskResult:
    """Result of autonomous task execution"""
    success: bool
    output: str
    files_modified: List[str]
    commands_executed: List[str]
    biological_compliance: bool
    error_message: Optional[str] = None
    execution_time: Optional[float] = None


@dataclass
class BrainContext:
    """Brain architecture context for Cline integration"""
    current_phase: str
    active_modules: List[str]
    neural_architecture: Dict[str, Any]
    biological_constraints: Dict[str, Any]
    morphogen_status: Dict[str, Any]
    foundation_layer_status: Dict[str, Any]
    compliance_rules: List[str]


class ExecutionMode(Enum):
    """Execution modes for Cline adapter"""
    AUTONOMOUS = "autonomous"
    GUIDED = "guided"
    SIMULATION = "simulation"


class BiologicalCompliance(Enum):
    """Biological compliance levels"""
    STRICT = "strict"          # Full AlphaGenome validation
    MODERATE = "moderate"      # Basic biological constraints
    PERMISSIVE = "permissive"  # Minimal validation
    DISABLED = "disabled"      # No biological validation
