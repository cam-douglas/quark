"""
Cline Integration Module - Autonomous coding with Quark State System

This module provides seamless integration between Cline's autonomous coding
capabilities and your existing Quark State System task management.

Key Components:
- task_integration_core: Core integration with existing task loader
- task_converter: Convert Quark tasks to Cline tasks  
- status_reporter: Comprehensive status reporting
- cline_adapter: Main Cline autonomous coding interface
- cline_mcp_server: MCP bridge for Cursor integration

Usage:
    from brain.modules.cline_integration import execute_foundation_layer_tasks_autonomously
    
    # Execute Foundation Layer tasks autonomously
    results = await execute_foundation_layer_tasks_autonomously(max_tasks=3)
"""

# Import main integration functions
from .task_integration_core import (
    QuarkClineIntegration,
    execute_foundation_layer_tasks_autonomously,
    execute_task_by_name
)

# Import status reporting functions  
from .status_reporter import (
    get_quark_cline_status,
    get_foundation_layer_status,
    generate_progress_report,
    get_task_priority_matrix
)

# Import task conversion utilities
from .task_converter import (
    convert_quark_task_to_cline_task,
    get_foundation_layer_file_mapping
)

# Import core Cline adapter and types
from .cline_adapter import ClineAdapter
from .cline_types import CodingTask, TaskComplexity, ClineTaskType
from .brain_context_provider import BrainContextProvider
from .biological_validator import BiologicalValidator
from .mcp_executor import MCPExecutor

__all__ = [
    # Core integration
    "QuarkClineIntegration",
    "execute_foundation_layer_tasks_autonomously", 
    "execute_task_by_name",
    
    # Status reporting
    "get_quark_cline_status",
    "get_foundation_layer_status", 
    "generate_progress_report",
    "get_task_priority_matrix",
    
    # Task conversion
    "convert_quark_task_to_cline_task",
    "get_foundation_layer_file_mapping",
    
    # Cline adapter and types
    "ClineAdapter",
    "CodingTask", 
    "TaskComplexity",
    "ClineTaskType",
    
    # Core components
    "BrainContextProvider",
    "BiologicalValidator",
    "MCPExecutor"
]

# Module metadata
__version__ = "1.0.0"
__author__ = "Quark Brain Architecture Team"
__description__ = "Autonomous coding integration with Quark State System"
