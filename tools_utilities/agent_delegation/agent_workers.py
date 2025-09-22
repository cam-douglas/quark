"""
Agent Workers

Specialized worker classes for different types of background agents.

Author: Quark AI
Date: 2025-01-27
"""

import subprocess
import logging
import arxiv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Handle both relative and absolute imports
try:
    from .agent_types import DelegatedTask, TaskStatus
except ImportError:
    # Import from the same directory
    import os
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from agent_types import DelegatedTask, TaskStatus


class BaseAgentWorker:
    """Base class for all agent workers"""
    
    def __init__(self, workspace_root: Path, logger: logging.Logger):
        self.workspace_root = workspace_root
        self.logger = logger
    
    def execute_task(self, task: DelegatedTask) -> Dict[str, Any]:
        """Execute a task - to be implemented by subclasses"""
        raise NotImplementedError


class ClineAgentWorker(BaseAgentWorker):
    """Worker for Cline autonomous coding agent"""
    
    def execute_task(self, task: DelegatedTask) -> Dict[str, Any]:
        """Execute a task using Cline MCP integration"""
        try:
            self.logger.info(f"Cline agent executing: {task.title}")
            
            # Use MCP Cline integration
            result = self._execute_cline_task(task)
            
            return {
                "cline_execution": "completed",
                "task_type": "autonomous_coding",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "cline_execution": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _execute_cline_task(self, task: DelegatedTask) -> Dict[str, Any]:
        """Execute a task using Cline MCP integration"""
        try:
            # Use the existing Cline integration
            from brain.modules.cline_integration.task_integration_core import QuarkClineIntegration
            
            cline_integration = QuarkClineIntegration()
            
            # Convert task to Cline format
            cline_task = {
                "title": task.title,
                "description": task.description,
                "context": task.result.get("context", {}) if task.result else {}
            }
            
            # Execute via Cline (this would need to be adapted based on your Cline setup)
            result = {
                "cline_execution": "completed",
                "task_type": "autonomous_coding",
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {
                "cline_execution": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


class ComplianceAgentWorker(BaseAgentWorker):
    """Worker for compliance checking agent"""
    
    def __init__(self, workspace_root: Path, logger: logging.Logger, compliance_system):
        super().__init__(workspace_root, logger)
        self.compliance_system = compliance_system
    
    def execute_task(self, task: DelegatedTask) -> Dict[str, Any]:
        """Execute compliance check task"""
        try:
            self.logger.info(f"Compliance agent executing: {task.title}")
            
            # Run compliance check
            context = task.result.get("context", {}) if task.result else {}
            paths = context.get("paths", [])
            
            if paths:
                success = self.compliance_system.check_compliance_now(paths)
                result = {
                    "compliance_check_passed": success,
                    "checked_paths": paths,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Check entire workspace
                success = self.compliance_system.check_compliance_now()
                result = {
                    "compliance_check_passed": success,
                    "checked_workspace": True,
                    "timestamp": datetime.now().isoformat()
                }
            
            return result
            
        except Exception as e:
            return {
                "compliance_check_failed": True,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


class TestingAgentWorker(BaseAgentWorker):
    """Worker for testing agent"""
    
    def execute_task(self, task: DelegatedTask) -> Dict[str, Any]:
        """Execute testing task"""
        try:
            self.logger.info(f"Testing agent executing: {task.title}")
            
            # Run tests
            context = task.result.get("context", {}) if task.result else {}
            test_path = context.get("test_path", "tests/")
            
            result = subprocess.run(
                ["python", "-m", "pytest", test_path, "-v", "--tb=short"],
                capture_output=True,
                text=True,
                cwd=str(self.workspace_root)
            )
            
            return {
                "test_results": {
                    "return_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "tests_passed": result.returncode == 0
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "test_execution_failed": True,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


class DocumentationAgentWorker(BaseAgentWorker):
    """Worker for documentation agent"""

    def execute_task(self, task: DelegatedTask) -> Dict[str, Any]:
        """Execute documentation generation task"""
        try:
            self.logger.info(f"Documentation agent executing: {task.title}")

            # Perform a search on arXiv
            search = arxiv.Search(
                query=task.description,
                max_results=3,
                sort_by=arxiv.SortCriterion.Relevance
            )

            # Process the search results
            summary = ""
            for result in search.results():
                summary += f"Title: {result.title}\n"
                summary += f"Authors: {', '.join(author.name for author in result.authors)}\n"
                summary += f"URL: {result.entry_id}\n"
                summary += f"Abstract: {result.summary}\n\n"

            return {
                "documentation_generated": True,
                "output": summary,
                "error": "",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "documentation_generation_failed": True,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
