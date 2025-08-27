#!/usr/bin/env python3
"""
S3 Model and Dataset Manager for Quark Brain Simulation System
Manages downloading, caching, and organizing models and datasets in Tokyo S3 bucket
"""

import os
import boto3
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tqdm import tqdm

@dataclass
class ModelConfig:
    """Configuration for a model to be managed"""
    name: str
    source_url: str
    model_type: str  # 'huggingface', 'pytorch', 'braket', 'bedrock', 'custom'
    size_gb: float
    description: str
    dependencies: List[str] = None
    s3_key: str = None
    local_path: str = None
    checksum: str = None

@dataclass
class DatasetConfig:
    """Configuration for a dataset to be managed"""
    name: str
    source_url: str
    dataset_type: str  # 'training', 'validation', 'benchmark', 'brain_data'
    size_gb: float
    description: str
    format: str  # 'numpy', 'pytorch', 'hdf5', 'parquet'
    s3_key: str = None
    local_path: str = None
    checksum: str = None

class S3ModelManager:
    """Manages models and datasets in S3 bucket with local caching"""
    
    def __init__(self, bucket_name: str = "quark-tokyo-bucket", region: str = "ap-northeast-1"):
        self.bucket_name = bucket_name
        self.region = region
        self.s3_client = boto3.client('s3', region_name=region)
        self.local_cache_dir = Path.home() / ".quark" / "model_cache"
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Tokyo instance specifications
        self.instance_specs = {
            "type": "c5.xlarge",
            "vcpus": 4,
            "memory_gb": 8,
            "storage_gb": 200,
            "network": "Up to 10 Gigabit",
            "region": "ap-northeast-1"
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize model and dataset registry
        self.model_registry = self._load_registry("models")
        self.dataset_registry = self._load_registry("datasets")
        
    def _load_registry(self, registry_type: str) -> Dict[str, Any]:
        """Load model or dataset registry from S3 or create new"""
        registry_key = f"quark-registry/{registry_type}_registry.json"
        
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=registry_key)
            registry = json.loads(response['Body'].read())
            self.logger.info(f"✅ Loaded {registry_type} registry with {len(registry)} entries")
            return registry
        except Exception as e:
            self.logger.info(f"📝 Creating new {registry_type} registry")
            return {}
    
    def _save_registry(self, registry_type: str, registry: Dict[str, Any]):
        """Save model or dataset registry to S3"""
        registry_key = f"quark-registry/{registry_type}_registry.json"
        
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=registry_key,
                Body=json.dumps(registry, indent=2, default=str),
                ContentType='application/json'
            )
            self.logger.info(f"✅ Saved {registry_type} registry to S3")
        except Exception as e:
            self.logger.error(f"❌ Failed to save {registry_type} registry: {e}")
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _download_with_progress(self, url: str, local_path: Path) -> bool:
        """Download file with progress bar"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(local_path, 'wb') as f, tqdm(
                desc=local_path.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            return True
        except Exception as e:
            self.logger.error(f"❌ Download failed: {e}")
            return False
    
    def _upload_to_s3(self, local_path: Path, s3_key: str) -> bool:
        """Upload file to S3 with progress"""
        try:
            file_size = local_path.stat().st_size
            
            with tqdm(desc=f"Uploading {local_path.name}", total=file_size, unit='B', unit_scale=True) as pbar:
                def upload_callback(bytes_transferred):
                    pbar.update(bytes_transferred)
                
                self.s3_client.upload_file(
                    str(local_path),
                    self.bucket_name,
                    s3_key,
                    Callback=upload_callback
                )
            
            self.logger.info(f"✅ Uploaded {local_path.name} to S3: {s3_key}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Upload failed: {e}")
            return False
    
    def _download_from_s3(self, s3_key: str, local_path: Path) -> bool:
        """Download file from S3 with progress"""
        try:
            # Get file size
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            file_size = response['ContentLength']
            
            with tqdm(desc=f"Downloading {local_path.name}", total=file_size, unit='B', unit_scale=True) as pbar:
                def download_callback(bytes_transferred):
                    pbar.update(bytes_transferred)
                
                self.s3_client.download_file(
                    self.bucket_name,
                    s3_key,
                    str(local_path),
                    Callback=download_callback
                )
            
            self.logger.info(f"✅ Downloaded {s3_key} from S3")
            return True
        except Exception as e:
            self.logger.error(f"❌ S3 download failed: {e}")
            return False
    
    def register_model(self, model_config: ModelConfig) -> bool:
        """Register a new model for management"""
        # Set default S3 key if not provided
        if not model_config.s3_key:
            model_config.s3_key = f"models/{model_config.model_type}/{model_config.name}"
        
        # Add to registry
        self.model_registry[model_config.name] = {
            "name": model_config.name,
            "source_url": model_config.source_url,
            "model_type": model_config.model_type,
            "size_gb": model_config.size_gb,
            "description": model_config.description,
            "dependencies": model_config.dependencies or [],
            "s3_key": model_config.s3_key,
            "local_path": model_config.local_path,
            "checksum": model_config.checksum,
            "registered_at": datetime.now().isoformat(),
            "status": "registered"
        }
        
        # Save registry
        self._save_registry("models", self.model_registry)
        self.logger.info(f"✅ Registered model: {model_config.name}")
        return True
    
    def register_dataset(self, dataset_config: DatasetConfig) -> bool:
        """Register a new dataset for management"""
        # Set default S3 key if not provided
        if not dataset_config.s3_key:
            dataset_config.s3_key = f"datasets/{dataset_config.dataset_type}/{dataset_config.name}"
        
        # Add to registry
        self.dataset_registry[dataset_config.name] = {
            "name": dataset_config.name,
            "source_url": dataset_config.source_url,
            "dataset_type": dataset_config.dataset_type,
            "size_gb": dataset_config.size_gb,
            "description": dataset_config.description,
            "format": dataset_config.format,
            "s3_key": dataset_config.s3_key,
            "local_path": dataset_config.local_path,
            "checksum": dataset_config.checksum,
            "registered_at": datetime.now().isoformat(),
            "status": "registered"
        }
        
        # Save registry
        self._save_registry("datasets", self.dataset_registry)
        self.logger.info(f"✅ Registered dataset: {dataset_config.name}")
        return True
    
    def download_model(self, model_name: str, force_refresh: bool = False) -> Optional[Path]:
        """Download model to local cache and S3"""
        if model_name not in self.model_registry:
            self.logger.error(f"❌ Model not found in registry: {model_name}")
            return None
        
        model_info = self.model_registry[model_name]
        local_path = self.local_cache_dir / "models" / model_name
        local_path.mkdir(parents=True, exist_ok=True)
        
        # Check if already cached locally
        if local_path.exists() and not force_refresh:
            self.logger.info(f"✅ Model already cached locally: {local_path}")
            return local_path
        
        # Try downloading from S3 first
        s3_file_path = local_path / "model.bin"
        if self._download_from_s3(model_info["s3_key"], s3_file_path):
            return local_path
        
        # Download from source URL
        self.logger.info(f"📥 Downloading model from source: {model_info['source_url']}")
        if self._download_with_progress(model_info["source_url"], s3_file_path):
            # Upload to S3 for future use
            self._upload_to_s3(s3_file_path, model_info["s3_key"])
            
            # Update registry with checksum
            checksum = self._calculate_checksum(s3_file_path)
            self.model_registry[model_name]["checksum"] = checksum
            self.model_registry[model_name]["status"] = "available"
            self._save_registry("models", self.model_registry)
            
            return local_path
        
        return None
    
    def download_dataset(self, dataset_name: str, force_refresh: bool = False) -> Optional[Path]:
        """Download dataset to local cache and S3"""
        if dataset_name not in self.dataset_registry:
            self.logger.error(f"❌ Dataset not found in registry: {dataset_name}")
            return None
        
        dataset_info = self.dataset_registry[dataset_name]
        local_path = self.local_cache_dir / "datasets" / dataset_name
        local_path.mkdir(parents=True, exist_ok=True)
        
        # Check if already cached locally
        if local_path.exists() and not force_refresh:
            self.logger.info(f"✅ Dataset already cached locally: {local_path}")
            return local_path
        
        # Try downloading from S3 first
        s3_file_path = local_path / "dataset.bin"
        if self._download_from_s3(dataset_info["s3_key"], s3_file_path):
            return local_path
        
        # Download from source URL
        self.logger.info(f"📥 Downloading dataset from source: {dataset_info['source_url']}")
        if self._download_with_progress(dataset_info["source_url"], s3_file_path):
            # Upload to S3 for future use
            self._upload_to_s3(s3_file_path, dataset_info["s3_key"])
            
            # Update registry with checksum
            checksum = self._calculate_checksum(s3_file_path)
            self.dataset_registry[dataset_name]["checksum"] = checksum
            self.dataset_registry[dataset_name]["status"] = "available"
            self._save_registry("datasets", self.dataset_registry)
            
            return local_path
        
        return None
    
    def check_storage_capacity(self) -> Dict[str, Any]:
        """Check storage capacity and recommendations"""
        total_models_size = sum(model.get("size_gb", 0) for model in self.model_registry.values())
        total_datasets_size = sum(dataset.get("size_gb", 0) for dataset in self.dataset_registry.values())
        total_size = total_models_size + total_datasets_size
        
        storage_info = {
            "instance_storage_gb": self.instance_specs["storage_gb"],
            "total_registered_size_gb": total_size,
            "models_size_gb": total_models_size,
            "datasets_size_gb": total_datasets_size,
            "storage_utilization": (total_size / self.instance_specs["storage_gb"]) * 100,
            "available_space_gb": self.instance_specs["storage_gb"] - total_size,
            "recommendations": []
        }
        
        # Add recommendations
        if storage_info["storage_utilization"] > 80:
            storage_info["recommendations"].append("⚠️ Storage utilization > 80%. Consider removing unused models/datasets.")
        elif storage_info["storage_utilization"] > 60:
            storage_info["recommendations"].append("📊 Storage utilization > 60%. Monitor usage closely.")
        else:
            storage_info["recommendations"].append("✅ Storage utilization healthy.")
        
        if total_size > self.instance_specs["storage_gb"]:
            storage_info["recommendations"].append("🚨 Total registered size exceeds instance storage! Use S3 streaming.")
        
        return storage_info
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        storage_info = self.check_storage_capacity()
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "instance_specs": self.instance_specs,
            "s3_bucket": self.bucket_name,
            "s3_region": self.region,
            "models": {
                "total_registered": len(self.model_registry),
                "by_type": {},
                "total_size_gb": storage_info["models_size_gb"]
            },
            "datasets": {
                "total_registered": len(self.dataset_registry),
                "by_type": {},
                "total_size_gb": storage_info["datasets_size_gb"]
            },
            "storage": storage_info,
            "cache_directory": str(self.local_cache_dir)
        }
        
        # Count by type
        for model in self.model_registry.values():
            model_type = model.get("model_type", "unknown")
            status["models"]["by_type"][model_type] = status["models"]["by_type"].get(model_type, 0) + 1
        
        for dataset in self.dataset_registry.values():
            dataset_type = dataset.get("dataset_type", "unknown")
            status["datasets"]["by_type"][dataset_type] = status["datasets"]["by_type"].get(dataset_type, 0) + 1
        
        return status
    
    def setup_quark_models(self) -> bool:
        """Setup standard Quark brain simulation models and datasets"""
        self.logger.info("🧠 Setting up Quark brain simulation models and datasets...")
        
        # Standard models for brain simulation
        standard_models = [
            ModelConfig(
                name="brain_base_transformer",
                source_url="https://huggingface.co/google/flan-t5-base/resolve/main/pytorch_model.bin",
                model_type="huggingface",
                size_gb=1.2,
                description="Base transformer for brain language processing"
            ),
            ModelConfig(
                name="neural_dynamics_model",
                source_url="https://github.com/pytorch/examples/raw/main/mnist/mnist_cnn.pt",
                model_type="pytorch",
                size_gb=0.1,
                description="Neural dynamics modeling for brain simulation"
            ),
            ModelConfig(
                name="consciousness_integration_model",
                source_url="https://example.com/consciousness_model.pt",  # Placeholder
                model_type="custom",
                size_gb=2.5,
                description="Global workspace theory consciousness integration model"
            )
        ]
        
        # Standard datasets for brain simulation
        standard_datasets = [
            DatasetConfig(
                name="brain_connectivity_data",
                source_url="https://example.com/brain_connectivity.h5",  # Placeholder
                dataset_type="brain_data",
                size_gb=5.0,
                description="Human brain connectivity matrices for simulation",
                format="hdf5"
            ),
            DatasetConfig(
                name="cognitive_benchmarks",
                source_url="https://example.com/cognitive_benchmarks.tar.gz",  # Placeholder
                dataset_type="benchmark",
                size_gb=1.5,
                description="Cognitive science benchmark tasks for validation",
                format="numpy"
            ),
            DatasetConfig(
                name="neural_training_data",
                source_url="https://example.com/neural_training.pt",  # Placeholder
                dataset_type="training",
                size_gb=8.0,
                description="Training data for neural dynamics models",
                format="pytorch"
            )
        ]
        
        # Register all models and datasets
        success_count = 0
        total_count = len(standard_models) + len(standard_datasets)
        
        for model in standard_models:
            if self.register_model(model):
                success_count += 1
        
        for dataset in standard_datasets:
            if self.register_dataset(dataset):
                success_count += 1
        
        self.logger.info(f"✅ Setup complete: {success_count}/{total_count} items registered")
        return success_count == total_count

def main():
    """Main function for testing and setup"""
    print("🧠⚡ Quark S3 Model Manager - Tokyo Instance")
    print("=" * 50)
    
    # Initialize manager
    manager = S3ModelManager()
    
    # Get system status
    status = manager.get_system_status()
    
    print(f"🏢 Instance: {status['instance_specs']['type']} in {status['instance_specs']['region']}")
    print(f"💾 Storage: {status['instance_specs']['storage_gb']}GB ({status['instance_specs']['memory_gb']}GB RAM)")
    print(f"📦 S3 Bucket: {status['s3_bucket']}")
    print(f"📊 Models: {status['models']['total_registered']} registered ({status['models']['total_size_gb']:.1f}GB)")
    print(f"📊 Datasets: {status['datasets']['total_registered']} registered ({status['datasets']['total_size_gb']:.1f}GB)")
    print(f"💽 Storage Utilization: {status['storage']['storage_utilization']:.1f}%")
    
    # Setup Quark models if none exist
    if status['models']['total_registered'] == 0:
        print("\n🚀 Setting up standard Quark models and datasets...")
        manager.setup_quark_models()
    
    # Print recommendations
    print("\n💡 Recommendations:")
    for rec in status['storage']['recommendations']:
        print(f"   {rec}")
    
    return manager

if __name__ == "__main__":
    main()
