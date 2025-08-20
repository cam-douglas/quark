import subprocess, shlex, os, pathlib, json, time, signal, threading
from typing import Dict, Any, List, Optional
from ................................................utils import (
    new_run_dir, save_jsonl, seed_everything, apply_resource_limits,
    capture_system_info, validate_run_environment, logger
)
import logging

class RunTracker:
    """Track and manage model runs with proper isolation and cleanup."""
    
    def __init__(self, run_dir: pathlib.Path):
        self.run_dir = run_dir
        self.start_time = time.time()
        self.process = None
        self.lock_file = run_dir / ".running"
        self.lock_file.touch()  # Mark as running
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        
    def cleanup(self):
        """Clean up run resources."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.lock_file.exists():
            self.lock_file.unlink()

def _run_subprocess(cmd: str, cwd: pathlib.Path, env: Dict[str, str], 
                   timeout: int = 300, resource_limits: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Run subprocess with enhanced safety and resource control.
    
    Args:
        cmd: Command to execute
        cwd: Working directory
        env: Environment variables
        timeout: Maximum execution time in seconds
        resource_limits: Resource limits to apply
        
    Returns:
        Dict with execution results
    """
    t0 = time.time()
    
    # Apply resource limits if specified
    if resource_limits:
        env = apply_resource_limits(
            cpu_limit=resource_limits.get("cpu_limit"),
            memory_limit_gb=resource_limits.get("memory_limit_gb"),
            gpu_limit=resource_limits.get("gpu_limit")
        )
    
    # Prepare subprocess with safety features
    try:
        p = subprocess.Popen(
            cmd, 
            cwd=str(cwd), 
            env=env, 
            shell=True,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            preexec_fn=os.setsid if os.name != 'nt' else None  # Create new process group
        )
        
        # Set up timeout handling
        def timeout_handler():
            try:
                if os.name != 'nt':
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                else:
                    p.terminate()
            except (ProcessLookupError, OSError):
                pass
        
        timer = threading.Timer(timeout, timeout_handler)
        timer.start()
        
        try:
            out, err = p.communicate()
            timer.cancel()
        except Exception as e:
            timer.cancel()
            raise e
            
        dt = time.time() - t0
        
        return {
            "cmd": cmd, 
            "rc": p.returncode, 
            "stdout": out, 
            "stderr": err, 
            "dt": dt,
            "timeout": False
        }
        
    except subprocess.TimeoutExpired:
        timer.cancel()
        try:
            if os.name != 'nt':
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            else:
                p.kill()
        except (ProcessLookupError, OSError):
            pass
        
        return {
            "cmd": cmd,
            "rc": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds",
            "dt": timeout,
            "timeout": True
        }
    except Exception as e:
        return {
            "cmd": cmd,
            "rc": -1,
            "stdout": "",
            "stderr": f"Execution failed: {str(e)}",
            "dt": time.time() - t0,
            "timeout": False
        }

def run_model(model_cfg: Dict[str, Any], prompt: str, 
              allow_shell: bool = False, sudo_ok: bool = False,
              timeout: int = 300, resource_limits: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Run a model with enhanced safety, isolation, and monitoring.
    
    Args:
        model_cfg: Model configuration from registry
        prompt: User prompt/input
        allow_shell: Whether to allow shell access
        sudo_ok: Whether to allow sudo operations
        timeout: Maximum execution time
        resource_limits: Resource limits to apply
        
    Returns:
        Dict with run results and metadata
    """
    # Create isolated run directory
    rd = new_run_dir("ask")
    
    with RunTracker(rd) as tracker:
        # Set deterministic seed
        seed = model_cfg.get("seed", 42)
        seed_everything(seed)
        
        # Capture system information for reproducibility
        system_info = capture_system_info()
        
        # Validate environment can support the model
        if not validate_run_environment(model_cfg):
            logger.error(f"Environment validation failed for model {model_cfg['id']}")
            return {
                "run_dir": str(rd),
                "error": "Environment validation failed",
                "system_info": system_info
            }
        
        # Prepare environment
        env = os.environ.copy()
        env["SM_RUN_DIR"] = str(rd)
        env["SM_MODEL_ID"] = model_cfg["id"]
        env["SM_PROMPT"] = prompt
        env["SM_ALLOW_SHELL"] = str(allow_shell).lower()
        env["SM_SUDO_OK"] = str(sudo_ok).lower()
        
        # Safety checks
        if not allow_shell and "shell" in model_cfg.get("capabilities", []):
            logger.warning(f"Shell access requested but not allowed for model {model_cfg['id']}")
        
        if not sudo_ok and "sudo" in prompt.lower():
            logger.warning("Sudo operation requested but not allowed")
            return {
                "run_dir": str(rd),
                "error": "Sudo operations not allowed",
                "system_info": system_info
            }
        
        # Initialize traces
        traces = []
        traces.append({
            "event": "run_start",
            "timestamp": time.time(),
            "model_id": model_cfg["id"],
            "prompt": prompt,
            "allow_shell": allow_shell,
            "sudo_ok": sudo_ok,
            "system_info": system_info
        })
        
        # Execute model based on type
        model_type = model_cfg["type"]
        result = None
        
        try:
            if model_type == "open_interpreter":
                result = _run_open_interpreter(model_cfg, prompt, rd, env, allow_shell, sudo_ok, timeout, resource_limits)
            elif model_type == "crewai":
                result = _run_crewai(model_cfg, prompt, rd, env, timeout, resource_limits)
            elif model_type == "llamacpp":
                result = _run_llamacpp(model_cfg, prompt, rd, env, timeout, resource_limits)
            elif model_type == "transformers":
                result = _run_transformers(model_cfg, prompt, rd, env, timeout, resource_limits)
            elif model_type == "smallmind":
                result = _run_smallmind(model_cfg, prompt, rd, env, timeout, resource_limits)
            else:
                result = {
                    "cmd": "",
                    "rc": 1,
                    "stdout": "",
                    "stderr": f"Unknown model type: {model_type}",
                    "dt": 0,
                    "timeout": False
                }
            
            traces.append({
                "event": "invoke",
                "adapter": model_type,
                "timestamp": time.time(),
                **result
            })
            
        except Exception as e:
            logger.error(f"Model execution failed: {e}")
            result = {
                "cmd": "",
                "rc": 1,
                "stdout": "",
                "stderr": f"Execution error: {str(e)}",
                "dt": 0,
                "timeout": False
            }
            traces.append({
                "event": "error",
                "adapter": model_type,
                "timestamp": time.time(),
                "error": str(e),
                **result
            })
        
        # Finalize run
        traces.append({
            "event": "run_complete",
            "timestamp": time.time(),
            "total_duration": time.time() - tracker.start_time
        })
        
        # Save traces and results
        save_jsonl(rd / "trace.jsonl", traces)
        
        # Save stdout/stderr to separate files
        if result and result.get("stdout"):
            with open(rd / "stdout" / "output.txt", "w") as f:
                f.write(result["stdout"])
        
        if result and result.get("stderr"):
            with open(rd / "stderr" / "errors.txt", "w") as f:
                f.write(result["stderr"])
        
        # Save run metadata
        metadata = {
            "model_config": model_cfg,
            "prompt": prompt,
            "allow_shell": allow_shell,
            "sudo_ok": sudo_ok,
            "system_info": system_info,
            "traces": traces,
            "result": result
        }
        
        with open(rd / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return {
            "run_dir": str(rd),
            "result": result if result else {"error": "No result generated"},
            "metadata": metadata,
            "traces_count": len(traces)
        }

def _run_open_interpreter(model_cfg: Dict[str, Any], prompt: str, run_dir: pathlib.Path, 
                         env: Dict[str, str], allow_shell: bool, sudo_ok: bool, 
                         timeout: int, resource_limits: Dict[str, Any]) -> Dict[str, Any]:
    """Run Open Interpreter with proper configuration."""
    bin_path = model_cfg.get("bin", "openinterpreter")
    flags = " ".join(model_cfg.get("flags", []))
    
    # Build command with safety flags
    cmd_parts = [bin_path]
    
    if not allow_shell:
        cmd_parts.extend(["--disable-shell"])
    if not sudo_ok:
        cmd_parts.extend(["--disable-sudo"])
    
    cmd_parts.extend(["--eval", shlex.quote(prompt)])
    if flags:
        cmd_parts.extend(shlex.split(flags))
    
    cmd = " ".join(cmd_parts)
    
    # Set safety environment variables
    if not allow_shell:
        env["INTERPRETER_SHELL_TOOL"] = "false"
    if not sudo_ok:
        env["INTERPRETER_SUDO_OK"] = "false"
    
    return _run_subprocess(cmd, run_dir, env, timeout, resource_limits)

def _run_crewai(model_cfg: Dict[str, Any], prompt: str, run_dir: pathlib.Path,
                env: Dict[str, str], timeout: int, resource_limits: Dict[str, Any]) -> Dict[str, Any]:
    """Run CrewAI workflow."""
    module = model_cfg.get("module", "crewai")
    workflow = model_cfg.get("workflow", "")
    
    if workflow and pathlib.Path(workflow).exists():
        cmd = f"python -m {module} --workflow {workflow} --prompt {shlex.quote(prompt)}"
    else:
        cmd = f"python -m {module} --prompt {shlex.quote(prompt)}"
    
    return _run_subprocess(cmd, run_dir, env, timeout, resource_limits)

def _run_llamacpp(model_cfg: Dict[str, Any], prompt: str, run_dir: pathlib.Path,
                  env: Dict[str, str], timeout: int, resource_limits: Dict[str, Any]) -> Dict[str, Any]:
    """Run LlamaCPP model."""
    model_path = model_cfg.get("model_path", "")
    ctx = model_cfg.get("ctx", 4096)
    gpu_layers = model_cfg.get("gpu_layers", 0)
    
    # For now, use a mock implementation
    # TODO: Implement actual LlamaCPP integration
    msg = f"[llamacpp:{model_path}] {prompt[:120]}..."
    return {
        "cmd": f"python -c 'print(\"{msg}\")'",
        "rc": 0,
        "stdout": msg + "\n",
        "stderr": "",
        "dt": 0.01,
        "timeout": False
    }

def _run_transformers(model_cfg: Dict[str, Any], prompt: str, run_dir: pathlib.Path,
                     env: Dict[str, str], timeout: int, resource_limits: Dict[str, Any]) -> Dict[str, Any]:
    """Run Transformers model."""
    model_id = model_cfg.get("model_id", "")
    device = model_cfg.get("device", "cpu")
    
    # For now, use a mock implementation
    # TODO: Implement actual Transformers integration
    msg = f"[transformers:{model_id}] {prompt[:120]}..."
    return {
        "cmd": f"python -c 'print(\"{msg}\")'",
        "rc": 0,
        "stdout": msg + "\n",
        "stderr": "",
        "dt": 0.01,
        "timeout": False
    }

def _run_smallmind(model_cfg: Dict[str, Any], prompt: str, run_dir: pathlib.Path,
                   env: Dict[str, str], timeout: int, resource_limits: Dict[str, Any]) -> Dict[str, Any]:
    """Run SmallMind agent."""
    entry = model_cfg.get("entry", "")
    
    if not entry:
        return {
            "cmd": "",
            "rc": 1,
            "stdout": "",
            "stderr": "No entry point specified for SmallMind model",
            "dt": 0,
            "timeout": False
        }
    
    mod, func = entry.split(":")
    code = f"import importlib; m=importlib.import_module('{mod.replace('/','.').rstrip('.py')}'); print(m.{func}({prompt!r}))"
    
    return _run_subprocess(f'python -c "{code}"', run_dir, env, timeout, resource_limits)
