"""
CrewAI Adapter for Agent Hub

Provides a safe interface to CrewAI workflows with proper isolation,
resource limits, and security controls.
"""

import logging
import time
import json
import yaml
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class CrewAIAdapter:
    """Adapter for CrewAI workflows with enhanced safety and monitoring."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.module = config.get("module", "crewai")
        self.workflow = config.get("workflow", "")
        self.timeout = config.get("timeout", 300)
        self.max_output_size = config.get("max_output_size", 1024 * 1024)  # 1MB
        self.allowed_tools = config.get("allowed_tools", [])
        self.blocked_tools = config.get("blocked_tools", ["sudo", "rm", "dd", "mkfs"])
        self.resource_limits = config.get("resource_limits", {})
        self.max_agents = config.get("max_agents", 5)
        self.max_iterations = config.get("max_iterations", 10)
        
        # Validate configuration
        self._validate_config()
        
        # Load workflow if specified
        self.workflow_config = self._load_workflow_config()
    
    def _validate_config(self):
        """Validate adapter configuration."""
        if not self.module:
            raise ValueError("CrewAI module not specified")
        
        # Check if workflow file exists
        if self.workflow and not Path(self.workflow).exists():
            logger.warning(f"CrewAI workflow file not found: {self.workflow}")
    
    def _load_workflow_config(self) -> Optional[Dict[str, Any]]:
        """Load workflow configuration from file."""
        if not self.workflow or not Path(self.workflow).exists():
            return None
        
        try:
            with open(self.workflow, 'r') as f:
                if self.workflow.endswith('.yaml') or self.workflow.endswith('.yml'):
                    return yaml.safe_load(f)
                elif self.workflow.endswith('.json'):
                    return json.load(f)
                else:
                    logger.warning(f"Unsupported workflow format: {self.workflow}")
                    return None
        except Exception as e:
            logger.error(f"Failed to load workflow config: {e}")
            return None
    
    def execute(self, prompt: str, run_dir: Path, env: Dict[str, str],
                allow_shell: bool = False, sudo_ok: bool = False,
                resource_limits: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a prompt through CrewAI workflow with safety controls.
        
        Args:
            prompt: User prompt to execute
            run_dir: Isolated run directory
            env: Environment variables
            allow_shell: Whether to allow shell access
            sudo_ok: Whether to allow sudo operations
            resource_limits: Resource limits to apply
            
        Returns:
            Dict with execution results
        """
        try:
            # Apply resource limits
            if resource_limits:
                env = self._apply_resource_limits(env, resource_limits)
            
            # Set up execution environment
            execution_env = self._setup_execution_environment(run_dir, env, allow_shell, sudo_ok)
            
            # Execute with timeout
            result = self._execute_with_timeout(prompt, execution_env, run_dir)
            
            # Post-process results
            result = self._post_process_result(result, run_dir)
            
            return result
            
        except Exception as e:
            logger.error(f"CrewAI workflow execution failed: {e}")
            return {
                "cmd": f"crewai:{self.module}",
                "rc": 1,
                "stdout": "",
                "stderr": f"Execution error: {str(e)}",
                "dt": 0,
                "timeout": False,
                "error": str(e)
            }
    
    def _setup_execution_environment(self, run_dir: Path, env: Dict[str, str],
                                   allow_shell: bool, sudo_ok: bool) -> Dict[str, Any]:
        """Set up the execution environment for CrewAI."""
        execution_env = {
            "run_dir": str(run_dir),
            "env": env.copy(),
            "working_dir": str(run_dir),
            "allow_shell": allow_shell,
            "sudo_ok": sudo_ok,
            "max_agents": self.max_agents,
            "max_iterations": self.max_iterations
        }
        
        # Add CrewAI-specific environment variables
        execution_env["env"]["CREWAI_MAX_AGENTS"] = str(self.max_agents)
        execution_env["env"]["CREWAI_MAX_ITERATIONS"] = str(self.max_iterations)
        execution_env["env"]["CREWAI_ALLOW_SHELL"] = str(allow_shell).lower()
        execution_env["env"]["CREWAI_SUDO_OK"] = str(sudo_ok).lower()
        
        # Create CrewAI-specific directories
        crewai_dir = run_dir / "crewai"
        crewai_dir.mkdir(exist_ok=True)
        execution_env["crewai_dir"] = str(crewai_dir)
        
        # Copy workflow config if available
        if self.workflow_config:
            workflow_copy = crewai_dir / "workflow_config.json"
            with open(workflow_copy, 'w') as f:
                json.dump(self.workflow_config, f, indent=2)
            execution_env["workflow_config_path"] = str(workflow_copy)
        
        return execution_env
    
    def _execute_with_timeout(self, prompt: str, execution_env: Dict[str, Any], 
                             run_dir: Path) -> Dict[str, Any]:
        """Execute the CrewAI workflow with timeout protection."""
        import threading
        
        start_time = time.time()
        result = {"stdout": "", "stderr": "", "rc": 0}
        
        # Create a thread for execution
        execution_complete = threading.Event()
        
        def execute_crewai():
            try:
                # Import CrewAI components
                from crewai import Crew, Agent, Task
                
                # Create safe tools based on permissions
                tools = self._create_safe_tools(execution_env)
                
                # Create agents based on workflow config or default
                agents = self._create_agents(tools, execution_env)
                
                # Create tasks
                tasks = self._create_tasks(prompt, agents, execution_env)
                
                # Create and run crew
                crew = Crew(
                    agents=agents,
                    tasks=tasks,
                    verbose=True,
                    max_iterations=execution_env["max_iterations"]
                )
                
                # Execute workflow
                output = crew.kickoff()
                
                if isinstance(output, str):
                    result["stdout"] = output
                elif isinstance(output, dict):
                    result.update(output)
                else:
                    result["stdout"] = str(output)
                    
            except ImportError as e:
                result["stderr"] = f"CrewAI not installed: {e}"
                result["rc"] = 1
            except Exception as e:
                result["stderr"] = str(e)
                result["rc"] = 1
            finally:
                execution_complete.set()
        
        # Start execution thread
        execution_thread = threading.Thread(target=execute_crewai)
        execution_thread.daemon = True
        execution_thread.start()
        
        # Wait for completion or timeout
        if execution_complete.wait(timeout=self.timeout):
            execution_thread.join()
        else:
            # Timeout occurred
            result["stderr"] = f"Execution timed out after {self.timeout} seconds"
            result["rc"] = -1
            result["timeout"] = True
        
        result["dt"] = time.time() - start_time
        return result
    
    def _create_safe_tools(self, execution_env: Dict[str, Any]) -> List:
        """Create safe tools based on permissions."""
        tools = []
        
        try:
            # Add basic tools
            if execution_env["allow_shell"]:
                # Add shell tools with restrictions
                pass  # TODO: Implement safe shell tools
            
            # Add file system tools
            if execution_env["allow_shell"]:
                # Add safe file system tools
                pass  # TODO: Implement safe file system tools
            
            # Add Python tools
            # TODO: Implement safe Python execution tools
            
        except ImportError:
            logger.warning("CrewAI tools not available")
        
        return tools
    
    def _create_agents(self, tools: List, execution_env: Dict[str, Any]) -> List:
        """Create agents based on workflow configuration."""
        agents = []
        
        try:
            from crewai import Agent
            
            if self.workflow_config and "agents" in self.workflow_config:
                # Create agents from workflow config
                for agent_config in self.workflow_config["agents"]:
                    agent = Agent(
                        role=agent_config.get("role", "Agent"),
                        goal=agent_config.get("goal", "Complete assigned tasks"),
                        backstory=agent_config.get("backstory", ""),
                        tools=tools,
                        verbose=True
                    )
                    agents.append(agent)
            else:
                # Create default agents
                planner = Agent(
                    role="Planner",
                    goal="Plan and coordinate tasks",
                    backstory="Expert at breaking down complex problems into manageable steps",
                    tools=tools,
                    verbose=True
                )
                
                executor = Agent(
                    role="Executor",
                    goal="Execute planned tasks",
                    backstory="Skilled at implementing solutions and following plans",
                    tools=tools,
                    verbose=True
                )
                
                agents = [planner, executor]
            
            # Limit number of agents
            if len(agents) > execution_env["max_agents"]:
                logger.warning(f"Limiting agents from {len(agents)} to {execution_env['max_agents']}")
                agents = agents[:execution_env["max_agents"]]
                
        except ImportError:
            logger.error("CrewAI Agent not available")
            return []
        
        return agents
    
    def _create_tasks(self, prompt: str, agents: List, execution_env: Dict[str, Any]) -> List:
        """Create tasks based on the prompt and available agents."""
        tasks = []
        
        try:
            from crewai import Task
            
            if self.workflow_config and "tasks" in self.workflow_config:
                # Create tasks from workflow config
                for task_config in self.workflow_config["tasks"]:
                    task = Task(
                        description=task_config.get("description", prompt),
                        agent=agents[task_config.get("agent_index", 0) % len(agents)],
                        expected_output=task_config.get("expected_output", "Task completed"),
                        context=prompt
                    )
                    tasks.append(task)
            else:
                # Create default tasks
                if len(agents) >= 2:
                    planning_task = Task(
                        description=f"Plan how to address: {prompt}",
                        agent=agents[0],
                        expected_output="A detailed plan with steps",
                        context=prompt
                    )
                    
                    execution_task = Task(
                        description="Execute the planned solution",
                        agent=agents[1],
                        expected_output="Solution implemented and results documented",
                        context=prompt
                    )
                    
                    tasks = [planning_task, execution_task]
                else:
                    # Single agent task
                    task = Task(
                        description=prompt,
                        agent=agents[0],
                        expected_output="Task completed successfully",
                        context=prompt
                    )
                    tasks = [task]
                    
        except ImportError:
            logger.error("CrewAI Task not available")
            return []
        
        return tasks
    
    def _apply_resource_limits(self, env: Dict[str, str], limits: Dict[str, Any]) -> Dict[str, str]:
        """Apply resource limits to environment."""
        # CPU limits
        if "cpu_limit" in limits:
            env["SM_CPU_LIMIT"] = str(limits["cpu_limit"])
        
        # Memory limits
        if "memory_limit_gb" in limits:
            env["SM_MEMORY_LIMIT_GB"] = str(limits["memory_limit_gb"])
        
        # GPU limits
        if "gpu_limit" in limits:
            if limits["gpu_limit"] == "none":
                env["CUDA_VISIBLE_DEVICES"] = ""
                env["MPS_DEVICE"] = "none"
            else:
                env["CUDA_VISIBLE_DEVICES"] = limits["gpu_limit"]
                if limits["gpu_limit"] == "0":
                    env["MPS_DEVICE"] = "0"
        
        return env
    
    def _post_process_result(self, result: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
        """Post-process execution results for safety and consistency."""
        # Truncate output if too large
        if result.get("stdout") and len(result["stdout"]) > self.max_output_size:
            result["stdout"] = result["stdout"][:self.max_output_size] + "\n[Output truncated]"
            logger.warning("Output truncated due to size limit")
        
        if result.get("stderr") and len(result["stderr"]) > self.max_output_size:
            result["stderr"] = result["stderr"][:self.max_output_size] + "\n[Error output truncated]"
            logger.warning("Error output truncated due to size limit")
        
        # Check for potentially dangerous content in output
        dangerous_patterns = [
            r"sudo\s+", r"rm\s+-rf", r"dd\s+if=", r"mkfs\.",
            r"chmod\s+777", r"chown\s+root", r"passwd\s+",
            r"DROP\s+DATABASE", r"DELETE\s+FROM", r"rmdir\s+/"
        ]
        
        dangerous_content = []
        for pattern in dangerous_patterns:
            import re
            if re.search(pattern, result.get("stdout", ""), re.IGNORECASE):
                dangerous_content.append(pattern)
        
        if dangerous_content:
            result["dangerous_content"] = dangerous_content
            logger.warning(f"Potentially dangerous content detected: {dangerous_content}")
        
        # Save CrewAI artifacts
        crewai_dir = run_dir / "crewai"
        if crewai_dir.exists():
            artifacts = list(crewai_dir.glob("*"))
            result["artifacts"] = [str(art) for art in artifacts]
        
        # Add metadata
        result["module"] = self.module
        result["workflow"] = self.workflow
        result["max_agents"] = self.max_agents
        result["max_iterations"] = self.max_iterations
        
        return result
    
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this adapter provides."""
        capabilities = ["planning", "orchestrate", "collaboration"]
        
        # Add capabilities based on configuration
        if self.config.get("allow_shell", False):
            capabilities.extend(["shell", "fs"])
        
        if self.config.get("allow_python", True):
            capabilities.append("python")
        
        return capabilities
    
    def get_safety_info(self) -> Dict[str, Any]:
        """Get safety information about this adapter."""
        return {
            "allow_shell": self.config.get("allow_shell", False),
            "allow_sudo": self.config.get("allow_sudo", False),
            "timeout": self.timeout,
            "max_output_size": self.max_output_size,
            "allowed_tools": self.allowed_tools,
            "blocked_tools": self.blocked_tools,
            "resource_limits": bool(self.resource_limits),
            "max_agents": self.max_agents,
            "max_iterations": self.max_iterations,
            "module": self.module,
            "workflow": self.workflow
        }
    
    def validate_workflow(self) -> bool:
        """Validate that the workflow configuration is safe and complete."""
        if not self.workflow_config:
            return True  # No workflow config is fine
        
        try:
            # Check for dangerous agent configurations
            if "agents" in self.workflow_config:
                for agent in self.workflow_config["agents"]:
                    if "role" in agent and "admin" in agent["role"].lower():
                        logger.warning("Admin role detected in workflow")
                        return False
            
            # Check for excessive resource usage
            if "max_iterations" in self.workflow_config:
                if self.workflow_config["max_iterations"] > 20:
                    logger.warning("Excessive max_iterations in workflow")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Workflow validation failed: {e}")
            return False

# Factory function for backward compatibility
def create_adapter(config: Dict[str, Any]) -> CrewAIAdapter:
    """Create a CrewAI adapter instance."""
    return CrewAIAdapter(config)
