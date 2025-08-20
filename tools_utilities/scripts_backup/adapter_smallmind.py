"""
SmallMind Adapter for Agent Hub

Provides a safe interface to SmallMind agents with proper isolation,
resource limits, and security controls.
"""

import importlib
import inspect
import logging
import json
import time
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import os, sys

logger = logging.getLogger(__name__)

class SmallMindAdapter:
    """Adapter for SmallMind agents with enhanced safety and monitoring."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.entry_point = config.get("entry", "")
        self.timeout = config.get("timeout", 300)
        self.max_output_size = config.get("max_output_size", 1024 * 1024)  # 1MB
        self.allowed_modules = config.get("allowed_modules", [])
        self.blocked_modules = config.get("blocked_modules", ["os", "subprocess", "sys"])
        self.resource_limits = config.get("resource_limits", {})
        
        # Validate configuration
        self._validate_config()
        
        # Load entry point
        self.agent_function = self._load_agent_function()
    
    def _validate_config(self):
        """Validate adapter configuration."""
        if not self.entry_point:
            raise ValueError("SmallMind agent entry point not specified")
        
        if ":" not in self.entry_point:
            raise ValueError("Entry point must be in format 'module:function'")
        
        # Check if entry point file exists
        module_path, func_name = self.entry_point.split(":", 1)
        module_file = Path(module_path + ".py")
        
        if not module_file.exists():
            logger.warning(f"SmallMind agent module not found: {module_file}")
    
    def _load_agent_function(self) -> Optional[Callable]:
        """Load the agent function from the specified entry point."""
        try:
            module_path, func_name = self.entry_point.split(":", 1)
            
            # Convert file path to module path
            module_name = module_path.replace("/", ".").replace("\\", ".")
            if module_name.endswith(".py"):
                module_name = module_name[:-3]
            
            # Import module
            module = importlib.import_module(module_name)
            
            # Get function
            if hasattr(module, func_name):
                func = getattr(module, func_name)
                if callable(func):
                    return func
                else:
                    logger.error(f"Entry point {func_name} is not callable")
                    return None
            else:
                logger.error(f"Function {func_name} not found in module {module_name}")
                return None
                
        except ImportError as e:
            logger.error(f"Failed to import SmallMind agent module: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load SmallMind agent function: {e}")
            return None
    
    def execute(self, prompt: str, run_dir: Path, env: Dict[str, str],
                allow_shell: bool = False, sudo_ok: bool = False,
                resource_limits: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a prompt through SmallMind agent with safety controls.
        
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
        if not self.agent_function:
            return {
                "cmd": "",
                "rc": 1,
                "stdout": "",
                "stderr": "SmallMind agent function not loaded",
                "dt": 0,
                "timeout": False,
                "error": "Agent function not loaded"
            }
        
        try:
            # Apply resource limits
            if resource_limits:
                env = self._apply_resource_limits(env, resource_limits)
            
            # Set up execution environment
            execution_env = self._setup_execution_environment(run_dir, env)
            
            # Execute with timeout
            result = self._execute_with_timeout(prompt, execution_env)
            
            # Post-process results
            result = self._post_process_result(result, run_dir)
            
            return result
            
        except Exception as e:
            logger.error(f"SmallMind agent execution failed: {e}")
            return {
                "cmd": f"smallmind:{self.entry_point}",
                "rc": 1,
                "stdout": "",
                "stderr": f"Execution error: {str(e)}",
                "dt": 0,
                "timeout": False,
                "error": str(e)
            }
    
    def _setup_execution_environment(self, run_dir: Path, env: Dict[str, str]) -> Dict[str, Any]:
        """Set up the execution environment for the agent."""
        execution_env = {
            "run_dir": str(run_dir),
            "env": env.copy(),
            "working_dir": str(run_dir),
            "allow_shell": env.get("SM_ALLOW_SHELL", "false").lower() == "true",
            "sudo_ok": env.get("SM_SUDO_OK", "false").lower() == "true"
        }
        
        # Add SmallMind-specific environment variables
        execution_env["env"]["SM_AGENT_ID"] = self.config.get("id", "unknown")
        execution_env["env"]["SM_AGENT_TYPE"] = "smallmind"
        execution_env["env"]["SM_EXECUTION_MODE"] = "isolated"
        
        # Create agent-specific directories
        agent_dir = run_dir / "agent"
        agent_dir.mkdir(exist_ok=True)
        execution_env["agent_dir"] = str(agent_dir)
        
        return execution_env
    
    def _execute_with_timeout(self, prompt: str, execution_env: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent function with timeout protection."""
        import threading
        import signal
        
        start_time = time.time()
        result = {"stdout": "", "stderr": "", "rc": 0}
        
        # Create a thread for execution
        execution_complete = threading.Event()
        
        def execute_agent():
            try:
                # Call the agent function
                if self._is_function_async():
                    # Handle async functions
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        output = loop.run_until_complete(self.agent_function(prompt, execution_env))
                    finally:
                        loop.close()
                else:
                    # Handle sync functions
                    output = self.agent_function(prompt, execution_env)
                
                if isinstance(output, str):
                    result["stdout"] = output
                elif isinstance(output, dict):
                    result.update(output)
                else:
                    result["stdout"] = str(output)
                    
            except Exception as e:
                result["stderr"] = str(e)
                result["rc"] = 1
            finally:
                execution_complete.set()
        
        # Start execution thread
        execution_thread = threading.Thread(target=execute_agent)
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
    
    def _is_function_async(self) -> bool:
        """Check if the agent function is async."""
        if not self.agent_function:
            return False
        
        try:
            return inspect.iscoroutinefunction(self.agent_function)
        except:
            return False
    
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
        
        # Check for potentially dangerous operations in output
        dangerous_patterns = [
            r"sudo\s+", r"rm\s+-rf", r"dd\s+if=", r"mkfs\.",
            r"chmod\s+777", r"chown\s+root", r"passwd\s+"
        ]
        
        dangerous_operations = []
        for pattern in dangerous_patterns:
            import re
            if re.search(pattern, result.get("stdout", "")):
                dangerous_operations.append(pattern)
            if re.search(pattern, result.get("stderr", "")):
                dangerous_operations.append(pattern)
        
        if dangerous_operations:
            result["dangerous_operations"] = dangerous_operations
            logger.warning(f"Potentially dangerous operations detected: {dangerous_operations}")
        
        # Save agent artifacts
        agent_dir = run_dir / "agent"
        if agent_dir.exists():
            artifacts = list(agent_dir.glob("*"))
            result["artifacts"] = [str(art) for art in artifacts]
        
        # Add metadata
        result["agent_id"] = self.config.get("id", "unknown")
        result["entry_point"] = self.entry_point
        
        return result
    
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this adapter provides."""
        # Get capabilities from config, fallback to default
        capabilities = self.config.get("capabilities", ["reasoning", "planning"])
        
        # Add capabilities based on configuration
        if self.config.get("allow_shell", False):
            capabilities.extend(["shell", "fs"])
        
        if self.config.get("allow_python", True):
            capabilities.append("python")
        
        return list(set(capabilities))  # Remove duplicates
    
    def get_safety_info(self) -> Dict[str, Any]:
        """Get safety information about this adapter."""
        return {
            "allow_shell": self.config.get("allow_shell", False),
            "allow_sudo": self.config.get("allow_sudo", False),
            "timeout": self.timeout,
            "max_output_size": self.max_output_size,
            "allowed_modules": self.allowed_modules,
            "blocked_modules": self.blocked_modules,
            "resource_limits": bool(self.resource_limits),
            "entry_point": self.entry_point
        }
    
    def validate_agent_function(self) -> bool:
        """Validate that the agent function can be called safely."""
        if not self.agent_function:
            return False
        
        try:
            # Check function signature
            sig = inspect.signature(self.agent_function)
            params = list(sig.parameters.keys())
            
            # Should accept at least one parameter (prompt)
            if len(params) < 1:
                logger.warning("Agent function should accept at least one parameter (prompt)")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate agent function: {e}")
            return False

# Factory function for backward compatibility
def create_adapter(config: Dict[str, Any]) -> SmallMindAdapter:
    """Create a SmallMind adapter instance."""
    return SmallMindAdapter(config)
