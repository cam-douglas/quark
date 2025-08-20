"""
LlamaCPP Adapter for Agent Hub

Provides a safe interface to LlamaCPP models with proper isolation,
resource limits, and security controls.
"""

import logging
import time
import json
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class LlamaCPPAdapter:
    """Adapter for LlamaCPP models with enhanced safety and monitoring."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_path = config.get("model_path", "")
        self.context_length = config.get("ctx", 4096)
        self.gpu_layers = config.get("gpu_layers", 0)
        self.n_threads = config.get("n_threads", 4)
        self.n_batch = config.get("n_batch", 512)
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.9)
        self.timeout = config.get("timeout", 300)
        self.max_output_size = config.get("max_output_size", 1024 * 1024)  # 1MB
        self.resource_limits = config.get("resource_limits", {})
        
        # Model instance (lazy loaded)
        self._model = None
        self._model_info = None
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate adapter configuration."""
        if not self.model_path:
            raise ValueError("LlamaCPP model path not specified")
        
        # Check if model file exists
        if not Path(self.model_path).exists():
            logger.warning(f"LlamaCPP model file not found: {self.model_path}")
        
        # Validate parameters
        if self.context_length <= 0:
            raise ValueError("Context length must be positive")
        
        if self.gpu_layers < 0:
            raise ValueError("GPU layers cannot be negative")
        
        if self.n_threads <= 0:
            raise ValueError("Number of threads must be positive")
    
    def _load_model(self):
        """Lazy load the LlamaCPP model."""
        if self._model is not None:
            return
        
        try:
            from llama_cpp import Llama
            
            logger.info(f"Loading LlamaCPP model from {self.model_path}")
            
            # Create model instance
            self._model = Llama(
                model_path=self.model_path,
                n_ctx=self.context_length,
                n_gpu_layers=self.gpu_layers,
                n_threads=self.n_threads,
                n_batch=self.n_batch,
                verbose=False
            )
            
            # Get model info
            self._model_info = {
                "model_path": self.model_path,
                "context_length": self.context_length,
                "gpu_layers": self.gpu_layers,
                "n_threads": self.n_threads,
                "n_batch": self.n_batch,
                "model_type": "llama_cpp"
            }
            
            logger.info(f"LlamaCPP model loaded successfully")
            
        except ImportError as e:
            logger.error(f"LlamaCPP not installed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load LlamaCPP model: {e}")
            raise
    
    def execute(self, prompt: str, run_dir: Path, env: Dict[str, str],
                allow_shell: bool = False, sudo_ok: bool = False,
                resource_limits: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a prompt through LlamaCPP model with safety controls.
        
        Args:
            prompt: User prompt to execute
            run_dir: Isolated run directory
            env: Environment variables
            allow_shell: Whether to allow shell access (not applicable for LlamaCPP)
            sudo_ok: Whether to allow sudo operations (not applicable for LlamaCPP)
            resource_limits: Resource limits to apply
            
        Returns:
            Dict with execution results
        """
        try:
            # Apply resource limits
            if resource_limits:
                env = self._apply_resource_limits(env, resource_limits)
            
            # Load model if not already loaded
            self._load_model()
            
            # Execute with timeout
            result = self._execute_with_timeout(prompt, run_dir)
            
            # Post-process results
            result = self._post_process_result(result, run_dir)
            
            return result
            
        except Exception as e:
            logger.error(f"LlamaCPP model execution failed: {e}")
            return {
                "cmd": f"llamacpp:{self.model_path}",
                "rc": 1,
                "stdout": "",
                "stderr": f"Execution error: {str(e)}",
                "dt": 0,
                "timeout": False,
                "error": str(e)
            }
    
    def _execute_with_timeout(self, prompt: str, run_dir: Path) -> Dict[str, Any]:
        """Execute the model with timeout protection."""
        import threading
        
        start_time = time.time()
        result = {"stdout": "", "stderr": "", "rc": 0}
        
        # Create a thread for execution
        execution_complete = threading.Event()
        
        def execute_model():
            try:
                # Generate response
                response = self._model(
                    prompt,
                    max_tokens=self.context_length - len(prompt),
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stop=["</s>", "\n\n", "Human:", "Assistant:"],
                    echo=False
                )
                
                # Extract generated text
                if isinstance(response, dict) and "choices" in response:
                    if len(response["choices"]) > 0:
                        generated_text = response["choices"][0].get("text", "")
                        result["stdout"] = generated_text.strip()
                    else:
                        result["stdout"] = "No response generated"
                else:
                    result["stdout"] = str(response)
                    
            except Exception as e:
                result["stderr"] = str(e)
                result["rc"] = 1
            finally:
                execution_complete.set()
        
        # Start execution thread
        execution_thread = threading.Thread(target=execute_model)
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
    
    def _apply_resource_limits(self, env: Dict[str, str], limits: Dict[str, Any]) -> Dict[str, str]:
        """Apply resource limits to environment."""
        # CPU limits
        if "cpu_limit" in limits:
            env["SM_CPU_LIMIT"] = str(limits["cpu_limit"])
            # Adjust thread count if CPU limited
            if limits["cpu_limit"] < self.n_threads:
                self.n_threads = max(1, limits["cpu_limit"])
        
        # Memory limits
        if "memory_limit_gb" in limits:
            env["SM_MEMORY_LIMIT_GB"] = str(limits["memory_limit_gb"])
        
        # GPU limits
        if "gpu_limit" in limits:
            if limits["gpu_limit"] == "none":
                env["CUDA_VISIBLE_DEVICES"] = ""
                env["MPS_DEVICE"] = "none"
                # Force CPU-only mode
                self.gpu_layers = 0
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
        
        # Save model artifacts
        model_dir = run_dir / "model"
        model_dir.mkdir(exist_ok=True)
        
        # Save model info
        model_info = {
            "model_path": self.model_path,
            "context_length": self.context_length,
            "gpu_layers": self.gpu_layers,
            "n_threads": self.n_threads,
            "n_batch": self.n_batch,
            "temperature": self.temperature,
            "top_p": self.top_p
        }
        
        with open(model_dir / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        # Add metadata
        result["model_path"] = self.model_path
        result["model_info"] = model_info
        
        return result
    
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this adapter provides."""
        capabilities = ["chat", "text_generation", "reasoning"]
        
        # Add capabilities based on model type
        if "code" in self.model_path.lower() or "code" in self.config.get("capabilities", []):
            capabilities.append("code")
        
        if "instruct" in self.model_path.lower():
            capabilities.append("instruction_following")
        
        return capabilities
    
    def get_safety_info(self) -> Dict[str, Any]:
        """Get safety information about this adapter."""
        return {
            "allow_shell": False,  # LlamaCPP models don't have shell access
            "allow_sudo": False,   # LlamaCPP models don't have sudo access
            "timeout": self.timeout,
            "max_output_size": self.max_output_size,
            "resource_limits": bool(self.resource_limits),
            "model_path": self.model_path,
            "context_length": self.context_length,
            "gpu_layers": self.gpu_layers
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self._model is None:
            return {"status": "not_loaded"}
        
        try:
            info = {
                "status": "loaded",
                "model_path": self.model_path,
                "context_length": self.context_length,
                "gpu_layers": self.gpu_layers,
                "n_threads": self.n_threads,
                "n_batch": self.n_batch,
                "model_type": "llama_cpp"
            }
            
            # Add model-specific info if available
            if hasattr(self._model, 'model'):
                info["model_name"] = getattr(self._model.model, 'name', "unknown")
                info["model_size"] = getattr(self._model.model, 'size', "unknown")
            
            return info
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        try:
            import psutil
            
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
                "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
                "percent": process.memory_percent()
            }
            
        except ImportError:
            return {"error": "psutil not available"}
        except Exception as e:
            return {"error": str(e)}
    
    def cleanup(self):
        """Clean up model resources."""
        if self._model is not None:
            try:
                # LlamaCPP doesn't have explicit cleanup, but we can delete references
                del self._model
                self._model = None
                self._model_info = None
                
                # Force garbage collection
                import gc
                gc.collect()
                
                logger.info("LlamaCPP model resources cleaned up")
            except Exception as e:
                logger.warning(f"Failed to cleanup LlamaCPP model resources: {e}")
    
    def reload_model(self):
        """Reload the model (useful for configuration changes)."""
        try:
            self.cleanup()
            self._load_model()
            logger.info("LlamaCPP model reloaded successfully")
        except Exception as e:
            logger.error(f"Failed to reload LlamaCPP model: {e}")

# Factory function for backward compatibility
def create_adapter(config: Dict[str, Any]) -> LlamaCPPAdapter:
    """Create a LlamaCPP adapter instance."""
    return LlamaCPPAdapter(config)
