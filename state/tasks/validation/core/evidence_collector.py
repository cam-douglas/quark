#!/usr/bin/env python3
"""
Evidence Collector Module
=========================
Manages validation artifacts and evidence collection.
"""

import json
import shutil
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional


class EvidenceCollector:
    """Collect and manage validation evidence artifacts."""
    
    def __init__(self, validation_root: Path):
        self.validation_root = validation_root
        self.evidence_dir = validation_root / "evidence"
        self.evidence_dir.mkdir(exist_ok=True)
    
    def create_run_id(self) -> str:
        """Generate a unique run ID."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def setup_evidence_directory(self, run_id: Optional[str] = None) -> Path:
        """Create evidence directory structure for a run."""
        if run_id is None:
            run_id = self.create_run_id()
        
        run_dir = self.evidence_dir / run_id
        run_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (run_dir / "plots").mkdir(exist_ok=True)
        (run_dir / "logs").mkdir(exist_ok=True)
        (run_dir / "configs").mkdir(exist_ok=True)
        
        return run_dir
    
    def collect_metrics(self, run_dir: Path, metrics: Dict[str, Any]) -> None:
        """Save metrics to evidence directory."""
        metrics_file = run_dir / "metrics.json"
        
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"✅ Metrics saved to {metrics_file}")
    
    def collect_config(self, run_dir: Path) -> None:
        """Collect configuration files."""
        config = {
            "timestamp": datetime.now().isoformat(),
            "validation_root": str(self.validation_root),
            "python_version": self._get_python_version(),
            "git_commit": self._get_git_commit()
        }
        
        # Check for project config files
        project_root = Path(__file__).parent.parent.parent.parent.parent
        config_files = [
            "pyproject.toml",
            "requirements.txt",
            ".env"
        ]
        
        for config_file in config_files:
            path = project_root / config_file
            if path.exists():
                config[config_file] = str(path)
                # Copy to evidence
                shutil.copy2(path, run_dir / "configs" / config_file)
        
        # Save config summary
        with open(run_dir / "config.yaml", "w") as f:
            import yaml
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ Configuration collected")
    
    def collect_seeds(self, run_dir: Path, seeds: Optional[Dict[str, int]] = None) -> None:
        """Record random seeds for reproducibility."""
        if seeds is None:
            seeds = {
                "numpy": 42,
                "torch": 42,
                "random": 42,
                "tensorflow": 42
            }
        
        with open(run_dir / "seeds.txt", "w") as f:
            for name, seed in seeds.items():
                f.write(f"{name}={seed}\n")
        
        print(f"✅ Seeds recorded")
    
    def collect_environment(self, run_dir: Path) -> None:
        """Collect environment information."""
        env_info = []
        
        # OS information
        import platform
        env_info.append(f"OS: {platform.system()} {platform.release()}")
        env_info.append(f"Python: {platform.python_version()}")
        env_info.append(f"Architecture: {platform.machine()}")
        
        # Python packages
        try:
            result = subprocess.run(
                ["pip", "freeze"],
                capture_output=True,
                text=True,
                check=True
            )
            env_info.append("\n=== Python Packages ===")
            env_info.append(result.stdout)
        except subprocess.CalledProcessError:
            env_info.append("Could not collect pip packages")
        
        # CUDA information if available
        try:
            import torch
            if torch.cuda.is_available():
                env_info.append(f"\nCUDA: {torch.version.cuda}")
                env_info.append(f"GPU: {torch.cuda.get_device_name(0)}")
        except ImportError:
            pass
        
        with open(run_dir / "environment.txt", "w") as f:
            f.write("\n".join(env_info))
        
        print(f"✅ Environment information collected")
    
    def collect_dataset_hashes(self, run_dir: Path, dataset_paths: List[Path]) -> None:
        """Compute and store dataset hashes."""
        hashes = {}
        
        for path in dataset_paths:
            if path.exists():
                if path.is_file():
                    hashes[str(path)] = self._compute_file_hash(path)
                elif path.is_dir():
                    # Hash all files in directory
                    for file_path in path.rglob("*"):
                        if file_path.is_file():
                            rel_path = file_path.relative_to(path)
                            hashes[str(rel_path)] = self._compute_file_hash(file_path)
        
        with open(run_dir / "dataset_hashes.txt", "w") as f:
            for path, hash_val in sorted(hashes.items()):
                f.write(f"{hash_val}  {path}\n")
        
        print(f"✅ Dataset hashes computed: {len(hashes)} files")
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def _get_python_version(self) -> str:
        """Get Python version string."""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()[:8]
        except subprocess.CalledProcessError:
            return "unknown"
    
    def collect_logs(self, run_dir: Path, log_content: str) -> None:
        """Save execution logs."""
        with open(run_dir / "logs.txt", "w") as f:
            f.write(log_content)
        
        print(f"✅ Logs saved")
    
    def validate_evidence_completeness(self, run_dir: Path) -> Dict[str, bool]:
        """Check if all required evidence files are present."""
        required = {
            "metrics.json": (run_dir / "metrics.json").exists(),
            "config.yaml": (run_dir / "config.yaml").exists(),
            "seeds.txt": (run_dir / "seeds.txt").exists(),
            "environment.txt": (run_dir / "environment.txt").exists(),
            "dataset_hashes.txt": (run_dir / "dataset_hashes.txt").exists(),
            "logs.txt": (run_dir / "logs.txt").exists()
        }
        
        return required
    
    def generate_evidence_summary(self, run_id: str) -> str:
        """Generate a summary of collected evidence."""
        run_dir = self.evidence_dir / run_id
        
        if not run_dir.exists():
            return f"Evidence directory not found: {run_id}"
        
        summary = []
        summary.append(f"\n📁 Evidence Summary for Run: {run_id}")
        summary.append("=" * 50)
        
        # Check completeness
        completeness = self.validate_evidence_completeness(run_dir)
        
        summary.append("\n📋 Required Files:")
        for file_name, exists in completeness.items():
            status = "✅" if exists else "❌"
            summary.append(f"  {status} {file_name}")
        
        # Check metrics if available
        metrics_file = run_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
                
            summary.append("\n📊 Key Metrics:")
            if "kpis" in metrics:
                for kpi, data in list(metrics["kpis"].items())[:5]:
                    if isinstance(data, dict) and "value" in data:
                        summary.append(f"  • {kpi}: {data['value']}")
        
        return "\n".join(summary)
