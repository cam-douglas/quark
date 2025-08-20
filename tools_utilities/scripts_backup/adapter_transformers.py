"""
Transformers Adapter for Agent Hub

Provides a safe interface to Hugging Face Transformers models with proper isolation,
resource limits, and security controls.
"""

import logging
import time
import json
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class TransformersAdapter:
    """Adapter for Hugging Face Transformers models with enhanced safety and monitoring."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_id = config.get("model_id", "")
        self.device = config.get("device", "cpu")
        self.dtype = config.get("dtype", "float16")
        self.max_length = config.get("max_length", 2048)
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.9)
        self.timeout = config.get("timeout", 300)
        self.max_output_size = config.get("max_output_size", 1024 * 1024)  # 1MB
        self.resource_limits = config.get("resource_limits", {})
        
        # Model instance (lazy loaded)
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate adapter configuration."""
        if not self.model_id:
            raise ValueError("Transformers model ID not specified")
        
        # Validate device
        valid_devices = ["cpu", "cuda", "mps", "auto"]
        if self.device not in valid_devices:
            logger.warning(f"Invalid device '{self.device}', using 'auto'")
            self.device = "auto"
        
        # Validate dtype
        valid_dtypes = ["float16", "float32", "bfloat16"]
        if self.dtype not in valid_dtypes:
            logger.warning(f"Invalid dtype '{self.dtype}', using 'float16'")
            self.dtype = "float16"
    
    def _load_model(self):
        """Lazy load the model and tokenizer."""
        if self._model is not None:
            return
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            
            # Set device
            if self.device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            
            logger.info(f"Loading model {self.model_id} on device {self.device}")
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=getattr(torch, self.dtype),
                device_map=self.device if self.device != "cpu" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Create pipeline
            self._pipeline = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                device=self.device if self.device != "cpu" else -1
            )
            
            logger.info(f"Model {self.model_id} loaded successfully")
            
        except ImportError as e:
            logger.error(f"Transformers not installed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load model {self.model_id}: {e}")
            raise
    
    def execute(self, prompt: str, run_dir: Path, env: Dict[str, str],
                allow_shell: bool = False, sudo_ok: bool = False,
                resource_limits: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a prompt through Transformers model with safety controls.
        
        Args:
            prompt: User prompt to execute
            run_dir: Isolated run directory
            env: Environment variables
            allow_shell: Whether to allow shell access (not applicable for Transformers)
            sudo_ok: Whether to allow sudo operations (not applicable for Transformers)
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
            logger.error(f"Transformers model execution failed: {e}")
            return {
                "cmd": f"transformers:{self.model_id}",
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
                response = self._pipeline(
                    prompt,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id
                )
                
                # Extract generated text
                if isinstance(response, list) and len(response) > 0:
                    generated_text = response[0].get("generated_text", "")
                    # Remove the input prompt from the generated text
                    if generated_text.startswith(prompt):
                        generated_text = generated_text[len(prompt):].strip()
                    result["stdout"] = generated_text
                else:
                    result["stdout"] = "No response generated"
                    
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
        
        # Memory limits
        if "memory_limit_gb" in limits:
            env["SM_MEMORY_LIMIT_GB"] = str(limits["memory_limit_gb"])
        
        # GPU limits
        if "gpu_limit" in limits:
            if limits["gpu_limit"] == "none":
                env["CUDA_VISIBLE_DEVICES"] = ""
                env["MPS_DEVICE"] = "none"
                # Force CPU usage
                self.device = "cpu"
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
            "model_id": self.model_id,
            "device": self.device,
            "dtype": self.dtype,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p
        }
        
        with open(model_dir / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        # Add metadata
        result["model_id"] = self.model_id
        result["device"] = self.device
        result["model_info"] = model_info
        
        return result
    
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this adapter provides."""
        capabilities = ["chat", "text_generation", "reasoning"]
        
        # Add capabilities based on model type
        if "code" in self.model_id.lower() or "code" in self.config.get("capabilities", []):
            capabilities.append("code")
        
        if "instruct" in self.model_id.lower():
            capabilities.append("instruction_following")
        
        return capabilities
    
    def get_safety_info(self) -> Dict[str, Any]:
        """Get safety information about this adapter."""
        return {
            "allow_shell": False,  # Transformers models don't have shell access
            "allow_sudo": False,   # Transformers models don't have sudo access
            "timeout": self.timeout,
            "max_output_size": self.max_output_size,
            "resource_limits": bool(self.resource_limits),
            "model_id": self.model_id,
            "device": self.device,
            "max_length": self.max_length
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self._model is None:
            return {"status": "not_loaded"}
        
        try:
            import torch
            
            info = {
                "status": "loaded",
                "model_id": self.model_id,
                "device": self.device,
                "dtype": self.dtype,
                "parameters": sum(p.numel() for p in self._model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self._model.parameters() if p.requires_grad)
            }
            
            # Add device-specific info
            if self.device == "cuda" and torch.cuda.is_available():
                info["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1024**3  # GB
                info["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1024**3    # GB
            
            return info
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def cleanup(self):
        """Clean up model resources."""
        if self._model is not None:
            try:
                import torch
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                del self._model
                del self._tokenizer
                del self._pipeline
                self._model = None
                self._tokenizer = None
                self._pipeline = None
                logger.info("Model resources cleaned up")
            except Exception as e:
                logger.warning(f"Failed to cleanup model resources: {e}")

# Factory function for backward compatibility
def create_adapter(config: Dict[str, Any]) -> TransformersAdapter:
    """Create a Transformers adapter instance."""
    return TransformersAdapter(config)
