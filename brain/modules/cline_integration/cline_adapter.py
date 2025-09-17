"""
Cline Adapter Core - Main interface for autonomous coding

Simplified core adapter that coordinates between brain context, biological
validation, and MCP execution for autonomous coding tasks.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

from brain.architecture.neural_core.cognitive_systems.resource_management.manager_core import ResourceManager

from .cline_types import CodingTask, TaskResult, TaskComplexity, ClineTaskType
from .brain_context_provider import BrainContextProvider
from .biological_validator import BiologicalValidator
from .mcp_executor import MCPExecutor


class ClineAdapter:
    """
    Core adapter for Cline autonomous coding with Quark Brain Architecture
    
    Coordinates brain context, biological validation, and MCP execution
    for safe autonomous coding within biological constraints.
    """
    
    def __init__(self, resource_manager: Optional[ResourceManager] = None):
        """Initialize Cline adapter with brain architecture integration"""
        self.resource_manager = resource_manager
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.workspace_path = Path("/Users/camdouglas/quark")
        self.autonomous_threshold = TaskComplexity.MODERATE
        
        # Core components
        self.brain_context_provider = BrainContextProvider(self.workspace_path)
        self.biological_validator = BiologicalValidator(self.workspace_path)
        self.mcp_executor = MCPExecutor(self.workspace_path)
        
        self.logger.info("Cline adapter initialized with brain architecture integration")

    async def autonomous_code_generation(self, task: Union[str, CodingTask]) -> TaskResult:
        """
        Execute autonomous code generation with full brain context
        
        Args:
            task: Task description string or CodingTask object
            
        Returns:
            TaskResult with execution details and biological compliance status
        """
        if isinstance(task, str):
            task = CodingTask(
                description=task,
                task_type=ClineTaskType.CODE_GENERATION,
                complexity=self._assess_task_complexity(task),
                files_involved=[]
            )
        
        self.logger.info(f"Starting autonomous code generation: {task.description}")
        
        # Load brain context
        brain_context = await self.get_brain_context()
        
        # Check biological constraints
        if task.biological_constraints:
            compliance_check = await self.biological_validator.validate_biological_compliance(task)
            if not compliance_check:
                return TaskResult(
                    success=False,
                    output="",
                    files_modified=[],
                    commands_executed=[],
                    biological_compliance=False,
                    error_message="Task violates biological constraints"
                )
        
        # Execute task via MCP server
        try:
            result = await self.mcp_executor.execute_via_mcp(task, brain_context)
            
            # Post-execution validation
            if task.biological_constraints and result.success:
                post_compliance = await self.biological_validator.validate_post_execution_compliance(result)
                result.biological_compliance = post_compliance
            
            return result
            
        except Exception as e:
            self.logger.error(f"Autonomous code generation failed: {e}")
            return TaskResult(
                success=False,
                output="",
                files_modified=[],
                commands_executed=[],
                biological_compliance=False,
                error_message=str(e)
            )

    async def handle_complex_coding_task(self, task: CodingTask) -> TaskResult:
        """
        Handle complex coding tasks with intelligent delegation
        
        This method decides whether to delegate to Cline or handle locally
        based on task complexity and brain architecture constraints.
        """
        self.logger.info(f"Handling complex coding task: {task.description}")
        
        # Assess whether to delegate to Cline
        should_delegate = await self._should_delegate_to_cline(task)
        
        if should_delegate:
            return await self.autonomous_code_generation(task)
        else:
            return await self._handle_task_locally(task)

    async def browser_automation_testing(self, test_scenario: str, app_url: str = "http://localhost:3000") -> TaskResult:
        """
        Execute browser automation testing for neural interfaces
        
        Args:
            test_scenario: Description of testing scenario
            app_url: URL of the application to test
            
        Returns:
            TaskResult with testing outcomes
        """
        task = CodingTask(
            description=f"Test neural interface: {test_scenario}",
            task_type=ClineTaskType.BROWSER_TESTING,
            complexity=TaskComplexity.MODERATE,
            files_involved=[],
            context={"app_url": app_url, "test_scenario": test_scenario}
        )
        
        return await self.mcp_executor.execute_browser_automation(task)

    async def get_brain_context(self) -> Dict[str, Any]:
        """
        Get comprehensive brain architecture context for Cline
        
        Returns:
            Dictionary containing current brain state and constraints
        """
        brain_context = await self.brain_context_provider.get_brain_context()
        
        # Convert to dictionary format for backward compatibility
        return {
            "current_phase": brain_context.current_phase,
            "active_modules": brain_context.active_modules,
            "neural_architecture": brain_context.neural_architecture,
            "biological_constraints": brain_context.biological_constraints,
            "morphogen_status": brain_context.morphogen_status,
            "foundation_layer_status": brain_context.foundation_layer_status,
            "compliance_rules": brain_context.compliance_rules
        }

    def _assess_task_complexity(self, description: str) -> TaskComplexity:
        """Assess task complexity based on description"""
        description_lower = description.lower()
        
        if any(term in description_lower for term in ["architecture", "brain", "neural", "morphogen"]):
            return TaskComplexity.CRITICAL
        elif any(term in description_lower for term in ["refactor", "multiple files", "system"]):
            return TaskComplexity.COMPLEX
        elif any(term in description_lower for term in ["edit", "modify", "update"]):
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE

    async def _should_delegate_to_cline(self, task: CodingTask) -> bool:
        """Determine if task should be delegated to Cline"""
        # Delegate if complexity exceeds threshold
        if task.complexity.value >= self.autonomous_threshold.value:
            return True
        
        # Delegate specific task types
        if task.task_type in [ClineTaskType.BROWSER_TESTING, ClineTaskType.COMMAND_EXECUTION]:
            return True
        
        # Don't delegate critical brain architecture changes during foundation phase
        if task.complexity == TaskComplexity.CRITICAL:
            brain_context = await self.get_brain_context()
            if brain_context.get("current_phase") == "Foundation Layer - SHH System Complete":
                return False
        
        return False

    async def _handle_task_locally(self, task: CodingTask) -> TaskResult:
        """Handle task locally without Cline delegation"""
        self.logger.info(f"Handling task locally: {task.description}")
        
        return TaskResult(
            success=True,
            output=f"Task handled locally: {task.description}",
            files_modified=[],
            commands_executed=[],
            biological_compliance=True
        )
