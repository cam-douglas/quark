"""
Open Interpreter Adapter for Agent Hub

Provides a safe interface to Open Interpreter with proper isolation,
resource limits, and security controls.
"""

import subprocess
import shlex
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class OpenInterpreterAdapter:
    """Adapter for Open Interpreter with enhanced safety and monitoring."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bin_path = config.get("bin", "openinterpreter")
        self.flags = config.get("flags", [])
        self.timeout = config.get("timeout", 300)
        self.max_output_size = config.get("max_output_size", 1024 * 1024)  # 1MB
        self.allowed_tools = config.get("allowed_tools", [])
        self.blocked_tools = config.get("blocked_tools", ["sudo", "rm", "dd", "mkfs"])
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate adapter configuration."""
        if not self.bin_path:
            raise ValueError("Open Interpreter binary path not specified")
        
        # Check if binary exists
        try:
            result = subprocess.run(["which", self.bin_path], 
                                  capture_output=True, text=True, check=True)
            self.bin_path = result.stdout.strip()
        except subprocess.CalledProcessError:
            logger.warning(f"Open Interpreter binary not found: {self.bin_path}")
    
    def execute(self, prompt: str, run_dir: Path, env: Dict[str, str],
                allow_shell: bool = False, sudo_ok: bool = False,
                resource_limits: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a prompt through Open Interpreter with safety controls.
        
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
            # Build command with safety flags
            cmd_parts = [self.bin_path]
            
            # Add safety flags
            if not allow_shell:
                cmd_parts.extend(["--disable-shell"])
            if not sudo_ok:
                cmd_parts.extend(["--disable-sudo"])
            
            # Add custom flags
            cmd_parts.extend(self.flags)
            
            # Add prompt
            cmd_parts.extend(["--eval", shlex.quote(prompt)])
            
            # Add output directory for artifacts
            artifacts_dir = run_dir / "artifacts"
            artifacts_dir.mkdir(exist_ok=True)
            cmd_parts.extend(["--output-dir", str(artifacts_dir)])
            
            # Build final command
            cmd = " ".join(cmd_parts)
            
            # Set safety environment variables
            safe_env = env.copy()
            if not allow_shell:
                safe_env["INTERPRETER_SHELL_TOOL"] = "false"
            if not sudo_ok:
                safe_env["INTERPRETER_SUDO_OK"] = "false"
            
            # Add tool restrictions
            if self.allowed_tools:
                safe_env["INTERPRETER_ALLOWED_TOOLS"] = ",".join(self.allowed_tools)
            if self.blocked_tools:
                safe_env["INTERPRETER_BLOCKED_TOOLS"] = ",".join(self.blocked_tools)
            
            # Execute with timeout and resource limits
            result = self._run_with_limits(cmd, run_dir, safe_env, resource_limits)
            
            # Post-process results
            result = self._post_process_result(result, run_dir)
            
            return result
            
        except Exception as e:
            logger.error(f"Open Interpreter execution failed: {e}")
            return {
                "cmd": cmd if 'cmd' in locals() else "",
                "rc": 1,
                "stdout": "",
                "stderr": f"Execution error: {str(e)}",
                "dt": 0,
                "timeout": False,
                "error": str(e)
            }
    
    def _run_with_limits(self, cmd: str, run_dir: Path, env: Dict[str, str],
                         resource_limits: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Run command with resource limits and monitoring."""
        import time
        import signal
        import threading
        
        start_time = time.time()
        
        # Apply resource limits
        if resource_limits:
            env = self._apply_resource_limits(env, resource_limits)
        
        try:
            # Start process
            process = subprocess.Popen(
                cmd,
                cwd=str(run_dir),
                env=env,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )
            
            # Set up timeout
            def timeout_handler():
                try:
                    if hasattr(os, 'setsid'):
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    else:
                        process.terminate()
                except (ProcessLookupError, OSError):
                    pass
            
            timer = threading.Timer(self.timeout, timeout_handler)
            timer.start()
            
            try:
                stdout, stderr = process.communicate()
                timer.cancel()
            except Exception as e:
                timer.cancel()
                raise e
            
            duration = time.time() - start_time
            
            return {
                "cmd": cmd,
                "rc": process.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "dt": duration,
                "timeout": False
            }
            
        except subprocess.TimeoutExpired:
            timer.cancel()
            try:
                if hasattr(os, 'setsid'):
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                else:
                    process.kill()
            except (ProcessLookupError, OSError):
                pass
            
            return {
                "cmd": cmd,
                "rc": -1,
                "stdout": "",
                "stderr": f"Command timed out after {self.timeout} seconds",
                "dt": self.timeout,
                "timeout": True
            }
    
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
        
        # Check for potentially dangerous operations
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
        
        # Save artifacts metadata
        artifacts_dir = run_dir / "artifacts"
        if artifacts_dir.exists():
            artifacts = list(artifacts_dir.glob("*"))
            result["artifacts"] = [str(art) for art in artifacts]
        
        return result
    
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this adapter provides."""
        capabilities = ["code", "python", "shell", "fs"]
        
        # Remove capabilities based on configuration
        if not self.config.get("allow_shell", True):
            capabilities.remove("shell")
        
        return capabilities
    
    def get_safety_info(self) -> Dict[str, Any]:
        """Get safety information about this adapter."""
        return {
            "allow_shell": self.config.get("allow_shell", True),
            "allow_sudo": self.config.get("allow_sudo", False),
            "timeout": self.timeout,
            "max_output_size": self.max_output_size,
            "allowed_tools": self.allowed_tools,
            "blocked_tools": self.blocked_tools,
            "resource_limits": bool(self.config.get("resource_limits"))
        }

# Factory function for backward compatibility
def create_adapter(config: Dict[str, Any]) -> OpenInterpreterAdapter:
    """Create an Open Interpreter adapter instance."""
    return OpenInterpreterAdapter(config)
