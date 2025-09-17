"""
MCP Executor - Handles MCP server communication for Cline integration

Manages execution of tasks via MCP server, browser automation,
and biological compliance validation.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from .cline_types import CodingTask, TaskResult


class MCPExecutor:
    """
    Handles MCP server communication and task execution
    
    Manages interaction with Cline MCP server for autonomous task execution
    with biological constraint validation and result processing.
    """
    
    def __init__(self, workspace_path: Path):
        """Initialize MCP executor"""
        self.workspace_path = workspace_path
        self.mcp_server_path = workspace_path / "brain/modules/cline_integration/dist/cline_mcp_server.js"
        self.logger = logging.getLogger(__name__)
    
    async def execute_via_mcp(self, task: CodingTask, brain_context: Dict[str, Any]) -> TaskResult:
        """Execute task via MCP server communication"""
        import time
        start_time = time.time()
        
        # Prepare MCP command
        mcp_command = [
            "node", str(self.mcp_server_path),
            "cline_execute_task",
            "--task", task.description,
            "--context", json.dumps(brain_context),
            "--biological_constraints", str(task.biological_constraints).lower(),
            "--working_directory", str(task.working_directory or self.workspace_path)
        ]
        
        try:
            # Execute MCP command
            process = await asyncio.create_subprocess_exec(
                *mcp_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace_path)
            )
            
            stdout, stderr = await process.communicate()
            execution_time = time.time() - start_time
            
            if process.returncode == 0:
                # Parse successful result
                result_data = json.loads(stdout.decode())
                return TaskResult(
                    success=True,
                    output=result_data.get("output", ""),
                    files_modified=result_data.get("files_modified", []),
                    commands_executed=result_data.get("commands_executed", []),
                    biological_compliance=True,  # Will be validated separately
                    execution_time=execution_time
                )
            else:
                return TaskResult(
                    success=False,
                    output=stdout.decode(),
                    files_modified=[],
                    commands_executed=[],
                    biological_compliance=False,
                    error_message=stderr.decode(),
                    execution_time=execution_time
                )
                
        except Exception as e:
            return TaskResult(
                success=False,
                output="",
                files_modified=[],
                commands_executed=[],
                biological_compliance=False,
                error_message=f"MCP execution failed: {e}",
                execution_time=time.time() - start_time
            )

    async def execute_browser_automation(self, task: CodingTask) -> TaskResult:
        """Execute browser automation testing"""
        mcp_command = [
            "node", str(self.mcp_server_path),
            "cline_browser_automation",
            "--action", "test_app",
            "--url", task.context.get("app_url", "http://localhost:3000"),
            "--test_scenario", task.context.get("test_scenario", "")
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *mcp_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                result_data = json.loads(stdout.decode())
                return TaskResult(
                    success=True,
                    output=result_data.get("output", ""),
                    files_modified=[],
                    commands_executed=[],
                    biological_compliance=True
                )
            else:
                return TaskResult(
                    success=False,
                    output="",
                    files_modified=[],
                    commands_executed=[],
                    biological_compliance=False,
                    error_message=stderr.decode()
                )
                
        except Exception as e:
            return TaskResult(
                success=False,
                output="",
                files_modified=[],
                commands_executed=[],
                biological_compliance=False,
                error_message=f"Browser automation failed: {e}"
            )
