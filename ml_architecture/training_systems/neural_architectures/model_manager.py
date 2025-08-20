"""
Simple Model Manager for MoE Models
"""

import os
import json
from pathlib import Path
from datetime import datetime
from 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORYclasses import 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORYclass
from typing import Dict, Any, Optional

@dataclass
class MoEModelConfig:
    """Configuration for a MoE model"""
    model_id: str
    model_name: str
    num_experts: int
    top_k: int
    max_memory_gb: float

@dataclass
class ModelInfo:
    """Information about a downloaded model"""
    model_name: str
    local_path: str
    model_size_gb: float
    download_date: str
    is_loaded: bool = False

class MoEModelManager:
    """Simple model manager for MoE models"""
    
    def __init__(self, base_dir: str = "./models"):
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "checkpoints"
        self.configs_dir = self.base_dir / "configs"
        self.registry_file = self.configs_dir / "model_registry.json"
        
        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations
        self.model_configs = self._init_default_models()
        self.model_registry = self._load_registry()
    
    def _init_default_models(self) -> Dict[str, MoEModelConfig]:
        """Initialize default model configurations"""
        return {
            "deepseek-v2": MoEModelConfig(
                model_id="deepseek-ai/DeepSeek-V2",
                model_name="DeepSeek-V2",
                num_experts=64,
                top_k=8,
                max_memory_gb=32.0
            ),
            "qwen1.5-moe": MoEModelConfig(
                model_id="Qwen/Qwen1.5-MoE-A2.7B",
                model_name="Qwen1.5-MoE",
                num_experts=8,
                top_k=2,
                max_memory_gb=8.0
            ),
            "mix-tao-moe": MoEModelConfig(
                model_id="mixtao/MixTAO-7Bx2-MoE-v8.1",
                model_name="MixTAO-MoE",
                num_experts=16,
                top_k=4,
                max_memory_gb=16.0
            )
        }
    
    def download_model(self, model_name: str, force_redownload: bool = False) -> Path:
        """Download a model from HuggingFace with maximum speed optimizations"""
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = self.model_configs[model_name]
        local_path = self.models_dir / model_name
        
        # Check if already downloaded
        if local_path.exists() and not force_redownload:
            print(f"Model {model_name} already exists at {local_path}")
            return local_path
        
        try:
            print(f"🚀 Downloading {model_name} from {config.model_id}")
            print(f"   Optimizing for maximum download speed...")
            
            # Import here to avoid dependency issues
            from huggingface_hub import snapshot_download
            
            # Speed optimizations:
            # 1. Use multiple workers for parallel downloads
            # 2. Disable progress bars for faster execution
            # 3. Use local_dir_use_symlinks=False for better performance
            # 4. Set max_workers for optimal parallelization
            local_path = snapshot_download(
                repo_id=config.model_id,
                local_dir=local_path,
                local_dir_use_symlinks=False,
                max_workers=8,  # Optimize for parallel downloads
                resume_download=True,  # Resume interrupted downloads
                allow_patterns=["*.safetensors", "*.bin", "*.json", "*.txt", "*.model"],  # Only essential files
                ignore_patterns=["*.md", "*.git*", "*.h5", "*.ckpt"]  # Skip unnecessary files
            )
            
            # Calculate size and update registry
            print(f"   Calculating model size...")
            size_gb = self._calculate_model_size(local_path)
            model_info = ModelInfo(
                model_name=model_name,
                local_path=str(local_path),
                model_size_gb=size_gb,
                download_date=datetime.now().isoformat()
            )
            
            self.model_registry[model_name] = model_info
            self._save_registry()
            
            print(f"✅ Successfully downloaded {model_name} to {local_path}")
            print(f"   Size: {size_gb:.2f} GB")
            print(f"   Download completed at: {model_info.download_date}")
            return Path(local_path)
            
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")
            raise
    
    def get_available_models(self) -> list:
        """Get list of available model names"""
        return list(self.model_configs.keys())
    
    def get_downloaded_models(self) -> list:
        """Get list of downloaded model names"""
        return list(self.model_registry.keys())
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model"""
        return self.model_registry.get(model_name)
    
    def _calculate_model_size(self, path: Path) -> float:
        """Calculate the size of a model directory in GB"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size / (1024**3)
    
    def _load_registry(self) -> Dict[str, ModelInfo]:
        """Load the model registry from file"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                    registry = {}
                    for name, info in data.items():
                        try:
                            # Validate the data before creating ModelInfo
                            if isinstance(info, dict) and all(key in info for key in ['model_name', 'local_path', 'model_size_gb', 'download_date']):
                                registry[name] = ModelInfo(**info)
                        except Exception as e:
                            print(f"Warning: Skipping invalid model info for {name}: {e}")
                            continue
                    return registry
            except Exception as e:
                print(f"Failed to load registry: {e}")
        return {}
    
    def _save_registry(self):
        """Save the model registry to file"""
        try:
            # Convert ModelInfo objects to dicts
            data = {}
            for name, info in self.model_registry.items():
                data[name] = {
                    "model_name": info.model_name,
                    "local_path": info.local_path,
                    "model_size_gb": info.model_size_gb,
                    "download_date": info.download_date,
                    "is_loaded": info.is_loaded
                }
            
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Failed to save registry: {e}")
