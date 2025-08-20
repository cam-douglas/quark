import os, json, time, uuid, pathlib, random, subprocess, shlex
from typing import Dict, Any, Optional
import logging

ROOT = pathlib.Path("ROOT")
LOGS = ROOT / "logs"; LOGS.mkdir(exist_ok=True)
RUNS = ROOT / "runs"; RUNS.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS / "agent_hub.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def new_run_dir(prefix="run") -> pathlib.Path:
    """Create isolated run directory with timestamp and UUID for reproducibility."""
    run_id = f"{prefix}-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    rd = RUNS / run_id
    rd.mkdir(parents=True, exist_ok=False)
    
    # Create subdirectories for different outputs
    (rd / "stdout").mkdir()
    (rd / "stderr").mkdir()
    (rd / "artifacts").mkdir()
    (rd / "checkpoints").mkdir()
    
    return rd

def save_jsonl(path, records):
    """Save records as JSONL with atomic write for safety."""
    if not records:
        return
    
    # Write to temporary file first, then atomic move
    temp_path = str(path) + ".tmp"
    with open(temp_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    os.rename(temp_path, path)

def seed_everything(seed: int = 42):
    """Set deterministic seeds for all random sources."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    logger.info(f"Seeded all random sources with {seed}")

def apply_resource_limits(cpu_limit: Optional[int] = None, 
                         memory_limit_gb: Optional[int] = None,
                         gpu_limit: Optional[str] = None) -> Dict[str, str]:
    """Apply resource limits and return modified environment."""
    env = os.environ.copy()
    
    # CPU limits via ulimit (macOS) or cgroups (Linux)
    if cpu_limit:
        if os.uname().sysname == "Darwin":  # macOS
            try:
                subprocess.run(["ulimit", "-t", str(cpu_limit)], check=True)
            except subprocess.CalledProcessError:
                logger.warning("Failed to set CPU limit on macOS")
        else:  # Linux
            env["SM_CPU_LIMIT"] = str(cpu_limit)
    
    # Memory limits
    if memory_limit_gb:
        if os.uname().sysname == "Darwin":  # macOS
            try:
                subprocess.run(["ulimit", "-v", str(memory_limit_gb * 1024 * 1024)], check=True)
            except subprocess.CalledProcessError:
                logger.warning("Failed to set memory limit on macOS")
        else:  # Linux
            env["SM_MEMORY_LIMIT_GB"] = str(memory_limit_gb)
    
    # GPU limits
    if gpu_limit:
        if gpu_limit == "none":
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["MPS_DEVICE"] = "none"
        else:
            env["CUDA_VISIBLE_DEVICES"] = gpu_limit
            if gpu_limit == "0":  # Single GPU
                env["MPS_DEVICE"] = "0"
    
    return env

def capture_system_info() -> Dict[str, Any]:
    """Capture system information for reproducibility."""
    info = {
        "timestamp": time.time(),
        "platform": os.uname().sysname,
        "architecture": os.uname().machine,
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "cwd": os.getcwd(),
        "env_vars": {k: v for k, v in os.environ.items() if not k.lower().startswith(('secret', 'key', 'password', 'token'))}
    }
    
    # Try to get GPU info
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    
    return info

def validate_run_environment(model_cfg: Dict[str, Any]) -> bool:
    """Validate that the environment can support the model requirements."""
    try:
        # Check if required binaries exist
        if "bin" in model_cfg:
            result = subprocess.run(["which", model_cfg["bin"]], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Required binary {model_cfg['bin']} not found")
                return False
        
        # Check GPU requirements
        if model_cfg.get("device") == "mps" and os.uname().sysname == "Darwin":
            try:
                import torch
                if not torch.backends.mps.is_available():
                    logger.error("MPS device requested but not available")
                    return False
            except ImportError:
                logger.error("PyTorch required for MPS device")
                return False
        
        # Check memory requirements
        if "memory_gb" in model_cfg:
            import psutil
            available_memory = psutil.virtual_memory().available / (1024**3)
            if available_memory < model_cfg["memory_gb"]:
                logger.warning(f"Model requires {model_cfg['memory_gb']}GB, only {available_memory:.1f}GB available")
        
        return True
        
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        return False

def cleanup_old_runs(max_age_hours: int = 24):
    """Clean up old run directories to prevent disk space issues."""
    cutoff_time = time.time() - (max_age_hours * 3600)
    
    for run_dir in RUNS.iterdir():
        if run_dir.is_dir():
            try:
                # Check if directory is old enough to consider for cleanup
                dir_time = run_dir.stat().st_mtime
                if dir_time < cutoff_time:
                    # Check if it's safe to delete (not currently running)
                    lock_file = run_dir / ".running"
                    if not lock_file.exists():
                        import shutil
                        shutil.rmtree(run_dir)
                        logger.info(f"Cleaned up old run directory: {run_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {run_dir}: {e}")
