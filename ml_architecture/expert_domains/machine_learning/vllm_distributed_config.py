#!/usr/bin/env python3
"""
vLLM Distributed Training Configuration
======================================

Distributed training configuration and orchestration for vLLM-enhanced brain simulation.
Supports multi-GPU, multi-node deployment with optimal resource allocation and load balancing.

Features:
- Multi-GPU tensor parallelism configuration
- Multi-node distributed brain simulation
- Dynamic resource allocation and scaling
- Load balancing and fault tolerance
- Performance monitoring and optimization

Author: Quark Brain Team
Date: 2025-01-20
"""

import os, sys
import json
import yaml
import logging
import socket
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import psutil
import torch

# Distributed training imports
try:
    import torch.distributed as dist
    import torch.multiprocessing as mp
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False

# Ray for distributed orchestration (optional)
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DistributedConfig:
    """Configuration for distributed vLLM brain training."""
    
    # Cluster Configuration
    world_size: int = 1                    # Total number of processes
    num_nodes: int = 1                     # Number of compute nodes
    gpus_per_node: int = 1                 # GPUs per node
    cpu_cores_per_node: int = 8            # CPU cores per node
    memory_gb_per_node: int = 32           # RAM per node
    
    # vLLM Distributed Settings
    tensor_parallel_size: int = 1          # Tensor parallelism degree
    pipeline_parallel_size: int = 1        # Pipeline parallelism degree
    max_model_len: int = 32768             # Maximum sequence length
    gpu_memory_utilization: float = 0.85   # GPU memory usage
    
    # Brain Simulation Distribution
    brains_per_gpu: int = 4                # Brain simulations per GPU
    concurrent_episodes: int = 8           # Concurrent training episodes
    batch_size_per_device: int = 4         # Training batch size per device
    
    # Network Configuration
    master_addr: str = "localhost"         # Master node address
    master_port: int = 29500               # Master node port
    backend: str = "nccl"                  # Communication backend (nccl/gloo)
    timeout_minutes: int = 30              # Communication timeout
    
    # Resource Management
    enable_dynamic_scaling: bool = True    # Dynamic resource scaling
    auto_detect_hardware: bool = True      # Auto-detect available hardware
    memory_efficient_mode: bool = True     # Enable memory optimizations
    enable_checkpointing: bool = True      # Enable distributed checkpointing
    
    # Performance Optimization
    use_zero_optimization: bool = True     # ZeRO optimizer state partitioning
    gradient_accumulation_steps: int = 4   # Gradient accumulation
    mixed_precision: bool = True           # Mixed precision training
    compile_model: bool = False            # PyTorch 2.0 compilation
    
    # Monitoring and Logging
    enable_profiling: bool = False         # Performance profiling
    log_interval: int = 10                 # Logging interval
    checkpoint_interval: int = 100         # Checkpoint interval
    
    # Fault Tolerance
    enable_fault_tolerance: bool = True    # Fault tolerance mechanisms
    max_retries: int = 3                   # Maximum retry attempts
    health_check_interval: int = 60        # Health check interval (seconds)

@dataclass
class NodeInfo:
    """Information about a compute node."""
    node_id: str
    hostname: str
    ip_address: str
    gpu_count: int
    gpu_memory_gb: List[float]
    cpu_cores: int
    memory_gb: float
    status: str = "healthy"  # healthy, unhealthy, offline
    load: float = 0.0        # Current load (0-1)
    last_seen: Optional[datetime] = None

class HardwareDetector:
    """Detect and analyze available hardware resources."""
    
    @staticmethod
    def detect_gpus() -> Dict[str, Any]:
        """Detect available GPUs and their capabilities."""
        gpu_info = {
            "count": 0,
            "total_memory_gb": 0,
            "devices": []
        }
        
        if torch.cuda.is_available():
            gpu_info["count"] = torch.cuda.device_count()
            
            for i in range(gpu_info["count"]):
                props = torch.cuda.get_device_properties(i)
                device_info = {
                    "id": i,
                    "name": props.name,
                    "memory_gb": props.total_memory / (1024**3),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multiprocessor_count": props.multiprocessor_count
                }
                gpu_info["devices"].append(device_info)
                gpu_info["total_memory_gb"] += device_info["memory_gb"]
        
        return gpu_info
    
    @staticmethod
    def detect_cpu_memory() -> Dict[str, Any]:
        """Detect CPU and memory information."""
        return {
            "cpu_cores": psutil.cpu_count(logical=False),
            "cpu_threads": psutil.cpu_count(logical=True),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "available_memory_gb": psutil.virtual_memory().available / (1024**3)
        }
    
    @staticmethod
    def detect_network() -> Dict[str, Any]:
        """Detect network configuration."""
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        
        return {
            "hostname": hostname,
            "ip_address": ip_address,
            "network_interfaces": list(psutil.net_if_addrs().keys())
        }
    
    @classmethod
    def get_full_hardware_info(cls) -> Dict[str, Any]:
        """Get complete hardware information."""
        return {
            "gpu": cls.detect_gpus(),
            "cpu_memory": cls.detect_cpu_memory(),
            "network": cls.detect_network(),
            "timestamp": datetime.now().isoformat()
        }

class DistributedVLLMConfig:
    """Generate vLLM configuration for distributed deployment."""
    
    def __init__(self, distributed_config: DistributedConfig):
        self.config = distributed_config
        self.hardware_info = HardwareDetector.get_full_hardware_info()
    
    def generate_vllm_args(self, rank: int = 0) -> List[str]:
        """Generate vLLM command-line arguments for distributed deployment."""
        args = [
            "--model", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",  # Default model
            "--tensor-parallel-size", str(self.config.tensor_parallel_size),
            "--pipeline-parallel-size", str(self.config.pipeline_parallel_size),
            "--max-model-len", str(self.config.max_model_len),
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
            "--trust-remote-code",
            "--enable-prefix-caching",
        ]
        
        if self.config.mixed_precision:
            args.extend(["--dtype", "bfloat16"])
        
        if self.config.memory_efficient_mode:
            args.extend([
                "--use-v2-block-manager",
                "--enable-chunked-prefill"
            ])
        
        # Add distributed settings
        if self.config.world_size > 1:
            args.extend([
                "--distributed-executor-backend", "ray" if RAY_AVAILABLE else "mp"
            ])
        
        return args
    
    def generate_serving_config(self) -> Dict[str, Any]:
        """Generate vLLM serving configuration."""
        return {
            "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "pipeline_parallel_size": self.config.pipeline_parallel_size,
            "max_model_len": self.config.max_model_len,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 256,
            "trust_remote_code": True,
            "enable_prefix_caching": True,
            "use_v2_block_manager": self.config.memory_efficient_mode,
            "enforce_eager": not self.config.compile_model,
            "dtype": "bfloat16" if self.config.mixed_precision else "float16"
        }

class ClusterManager:
    """Manage distributed brain simulation cluster."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.nodes: Dict[str, NodeInfo] = {}
        self.is_master = False
        self.rank = 0
        self.local_rank = 0
        
    def initialize_cluster(self) -> bool:
        """Initialize distributed cluster."""
        logger.info("🚀 Initializing Distributed Cluster")
        
        try:
            # Auto-detect hardware if enabled
            if self.config.auto_detect_hardware:
                self._auto_configure_hardware()
            
            # Initialize distributed backend
            if self.config.world_size > 1:
                success = self._init_distributed_backend()
                if not success:
                    return False
            
            # Initialize Ray if available and configured
            if RAY_AVAILABLE and self.config.world_size > 1:
                self._init_ray_cluster()
            
            logger.info("✅ Cluster initialization successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Cluster initialization failed: {e}")
            return False
    
    def _auto_configure_hardware(self):
        """Auto-configure based on detected hardware."""
        hardware = HardwareDetector.get_full_hardware_info()
        gpu_count = hardware["gpu"]["count"]
        
        if gpu_count > 1:
            # Optimal tensor parallel configuration
            if gpu_count >= 8:
                self.config.tensor_parallel_size = min(8, gpu_count)
            elif gpu_count >= 4:
                self.config.tensor_parallel_size = min(4, gpu_count)
            elif gpu_count >= 2:
                self.config.tensor_parallel_size = min(2, gpu_count)
            
            self.config.gpus_per_node = gpu_count
        
        # Adjust concurrent episodes based on available memory
        total_gpu_memory = hardware["gpu"]["total_memory_gb"]
        if total_gpu_memory > 80:  # High-memory setup
            self.config.concurrent_episodes = 16
            self.config.brains_per_gpu = 8
        elif total_gpu_memory > 40:  # Medium-memory setup
            self.config.concurrent_episodes = 8
            self.config.brains_per_gpu = 4
        else:  # Low-memory setup
            self.config.concurrent_episodes = 4
            self.config.brains_per_gpu = 2
        
        logger.info(f"🔧 Auto-configured: TP={self.config.tensor_parallel_size}, "
                   f"Episodes={self.config.concurrent_episodes}")
    
    def _init_distributed_backend(self) -> bool:
        """Initialize PyTorch distributed backend."""
        if not DISTRIBUTED_AVAILABLE:
            logger.warning("⚠️  PyTorch distributed not available")
            return False
        
        try:
            # Set environment variables
            os.environ["MASTER_ADDR"] = self.config.master_addr
            os.environ["MASTER_PORT"] = str(self.config.master_port)
            os.environ["WORLD_SIZE"] = str(self.config.world_size)
            os.environ["RANK"] = str(self.rank)
            
            # Initialize process group
            dist.init_process_group(
                backend=self.config.backend,
                world_size=self.config.world_size,
                rank=self.rank,
                timeout=torch.distributed.default_pg_timeout
            )
            
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.is_master = (self.rank == 0)
            
            logger.info(f"🌐 Distributed backend initialized: rank={self.rank}/{self.config.world_size}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize distributed backend: {e}")
            return False
    
    def _init_ray_cluster(self):
        """Initialize Ray cluster for distributed orchestration."""
        try:
            if not ray.is_initialized():
                ray.init(
                    address="auto" if self.config.world_size > 1 else None,
                    num_cpus=self.config.cpu_cores_per_node,
                    num_gpus=self.config.gpus_per_node,
                    object_store_memory=int(self.config.memory_gb_per_node * 0.3 * 1024**3)
                )
            
            logger.info(f"☀️  Ray cluster initialized: {ray.cluster_resources()}")
            
        except Exception as e:
            logger.warning(f"⚠️  Ray initialization failed: {e}")
    
    def register_node(self, node_info: NodeInfo):
        """Register a new node in the cluster."""
        self.nodes[node_info.node_id] = node_info
        logger.info(f"📝 Registered node: {node_info.hostname} ({node_info.gpu_count} GPUs)")
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status."""
        total_gpus = sum(node.gpu_count for node in self.nodes.values())
        healthy_nodes = sum(1 for node in self.nodes.values() if node.status == "healthy")
        
        return {
            "total_nodes": len(self.nodes),
            "healthy_nodes": healthy_nodes,
            "total_gpus": total_gpus,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "concurrent_episodes": self.config.concurrent_episodes,
            "nodes": {node_id: {
                "hostname": node.hostname,
                "status": node.status,
                "load": node.load,
                "gpu_count": node.gpu_count
            } for node_id, node in self.nodes.items()}
        }
    
    def shutdown_cluster(self):
        """Shutdown distributed cluster."""
        logger.info("🔄 Shutting down cluster...")
        
        if DISTRIBUTED_AVAILABLE and dist.is_initialized():
            dist.destroy_process_group()
        
        if RAY_AVAILABLE and ray.is_initialized():
            ray.shutdown()
        
        logger.info("✅ Cluster shutdown complete")

class DistributedBrainTrainer:
    """Distributed brain training orchestrator."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.cluster_manager = ClusterManager(config)
        self.vllm_config_gen = DistributedVLLMConfig(config)
        self.training_workers = []
        
    async def setup_distributed_training(self) -> bool:
        """Setup distributed training environment."""
        logger.info("🎯 Setting up Distributed Brain Training")
        
        # Initialize cluster
        success = self.cluster_manager.initialize_cluster()
        if not success:
            return False
        
        # Generate deployment configurations
        self._generate_deployment_configs()
        
        # Launch training workers
        if self.config.world_size > 1:
            await self._launch_distributed_workers()
        
        logger.info("✅ Distributed training setup complete")
        return True
    
    def _generate_deployment_configs(self):
        """Generate deployment configurations for all nodes."""
        configs_dir = Path("distributed_configs")
        configs_dir.mkdir(exist_ok=True)
        
        # Generate vLLM serving config
        vllm_serving_config = self.vllm_config_gen.generate_serving_config()
        with open(configs_dir / "vllm_serving.json", 'w') as f:
            json.dump(vllm_serving_config, f, indent=2)
        
        # Generate Docker Compose configuration
        docker_compose = self._generate_docker_compose()
        with open(configs_dir / "docker-compose.yml", 'w') as f:
            yaml.dump(docker_compose, f, default_flow_style=False)
        
        # Generate Kubernetes configuration
        k8s_config = self._generate_kubernetes_config()
        with open(configs_dir / "brain-training-deployment.yaml", 'w') as f:
            yaml.dump(k8s_config, f, default_flow_style=False)
        
        logger.info(f"📋 Deployment configs generated in {configs_dir}")
    
    def _generate_docker_compose(self) -> Dict[str, Any]:
        """Generate Docker Compose configuration."""
        return {
            "version": "3.8",
            "services": {
                "vllm-server": {
                    "image": "vllm/vllm-openai:latest",
                    "command": [
                        "--model", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        "--tensor-parallel-size", str(self.config.tensor_parallel_size),
                        "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
                        "--host", "0.0.0.0",
                        "--port", "8000",
                        "--trust-remote-code"
                    ],
                    "ports": ["8000:8000"],
                    "environment": {
                        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                        "NCCL_DEBUG": "INFO"
                    },
                    "deploy": {
                        "resources": {
                            "reservations": {
                                "devices": [
                                    {
                                        "driver": "nvidia",
                                        "count": self.config.tensor_parallel_size,
                                        "capabilities": ["gpu"]
                                    }
                                ]
                            }
                        }
                    },
                    "volumes": [
                        "./models:/models",
                        "./cache:/cache"
                    ]
                },
                "brain-trainer": {
                    "build": ".",
                    "depends_on": ["vllm-server"],
                    "environment": {
                        "VLLM_ENDPOINT": "http://vllm-server:8000",
                        "WORLD_SIZE": str(self.config.world_size),
                        "MASTER_ADDR": self.config.master_addr,
                        "MASTER_PORT": str(self.config.master_port)
                    },
                    "volumes": [
                        "./training_outputs:/app/training_outputs",
                        "./src:/app/src"
                    ],
                    "command": "python -m src.core.vllm_training_pipeline"
                }
            }
        }
    
    def _generate_kubernetes_config(self) -> Dict[str, Any]:
        """Generate Kubernetes deployment configuration."""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "brain-training-deployment",
                "labels": {"app": "brain-training"}
            },
            "spec": {
                "replicas": self.config.world_size,
                "selector": {"matchLabels": {"app": "brain-training"}},
                "template": {
                    "metadata": {"labels": {"app": "brain-training"}},
                    "spec": {
                        "containers": [
                            {
                                "name": "vllm-brain-trainer",
                                "image": "quark-brain:latest",
                                "ports": [{"containerPort": 8000}],
                                "env": [
                                    {"name": "WORLD_SIZE", "value": str(self.config.world_size)},
                                    {"name": "MASTER_ADDR", "value": self.config.master_addr},
                                    {"name": "MASTER_PORT", "value": str(self.config.master_port)},
                                    {"name": "CUDA_VISIBLE_DEVICES", "value": "0,1,2,3"}
                                ],
                                "resources": {
                                    "requests": {
                                        "nvidia.com/gpu": self.config.tensor_parallel_size,
                                        "memory": f"{self.config.memory_gb_per_node}Gi",
                                        "cpu": str(self.config.cpu_cores_per_node)
                                    },
                                    "limits": {
                                        "nvidia.com/gpu": self.config.tensor_parallel_size,
                                        "memory": f"{self.config.memory_gb_per_node}Gi",
                                        "cpu": str(self.config.cpu_cores_per_node)
                                    }
                                },
                                "volumeMounts": [
                                    {"name": "training-data", "mountPath": "/app/training_outputs"}
                                ]
                            }
                        ],
                        "volumes": [
                            {
                                "name": "training-data",
                                "persistentVolumeClaim": {"claimName": "training-data-pvc"}
                            }
                        ]
                    }
                }
            }
        }
    
    async def _launch_distributed_workers(self):
        """Launch distributed training workers."""
        logger.info(f"🚀 Launching {self.config.world_size} training workers")
        
        # This would typically involve launching workers on different nodes
        # For now, we'll simulate with process spawning
        for rank in range(self.config.world_size):
            worker_cmd = [
                sys.executable, "-m", "src.core.vllm_training_pipeline",
                "--distributed",
                "--rank", str(rank),
                "--world-size", str(self.config.world_size),
                "--master-addr", self.config.master_addr,
                "--master-port", str(self.config.master_port)
            ]
            
            logger.info(f"  Worker {rank}: {' '.join(worker_cmd)}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get distributed training status."""
        cluster_status = self.cluster_manager.get_cluster_status()
        
        return {
            "cluster": cluster_status,
            "config": {
                "world_size": self.config.world_size,
                "tensor_parallel_size": self.config.tensor_parallel_size,
                "concurrent_episodes": self.config.concurrent_episodes,
                "brains_per_gpu": self.config.brains_per_gpu
            },
            "workers": len(self.training_workers),
            "status": "running" if self.training_workers else "stopped"
        }
    
    async def shutdown_training(self):
        """Shutdown distributed training."""
        logger.info("🔄 Shutting down distributed training...")
        
        # Stop training workers
        for worker in self.training_workers:
            if hasattr(worker, 'terminate'):
                worker.terminate()
        
        # Shutdown cluster
        self.cluster_manager.shutdown_cluster()
        
        logger.info("✅ Distributed training shutdown complete")


# CLI and utility functions
def create_optimal_distributed_config() -> DistributedConfig:
    """Create optimal distributed configuration based on available hardware."""
    hardware = HardwareDetector.get_full_hardware_info()
    gpu_count = hardware["gpu"]["count"]
    memory_gb = hardware["cpu_memory"]["memory_gb"]
    
    config = DistributedConfig()
    
    if gpu_count >= 8:
        # High-end multi-GPU setup
        config.tensor_parallel_size = 8
        config.world_size = 2
        config.concurrent_episodes = 16
        config.brains_per_gpu = 8
    elif gpu_count >= 4:
        # Medium multi-GPU setup
        config.tensor_parallel_size = 4
        config.world_size = 1
        config.concurrent_episodes = 8
        config.brains_per_gpu = 4
    elif gpu_count >= 2:
        # Basic multi-GPU setup
        config.tensor_parallel_size = 2
        config.world_size = 1
        config.concurrent_episodes = 4
        config.brains_per_gpu = 2
    else:
        # Single GPU or CPU-only setup
        config.tensor_parallel_size = 1
        config.world_size = 1
        config.concurrent_episodes = 2
        config.brains_per_gpu = 1
    
    # Adjust memory settings
    if memory_gb > 128:
        config.memory_gb_per_node = 128
        config.batch_size_per_device = 8
    elif memory_gb > 64:
        config.memory_gb_per_node = 64
        config.batch_size_per_device = 4
    else:
        config.memory_gb_per_node = 32
        config.batch_size_per_device = 2
    
    return config


async def demo_distributed_setup():
    """Demonstrate distributed training setup."""
    print("🌐 vLLM Distributed Brain Training Demo")
    print("=" * 60)
    
    # Detect hardware
    hardware = HardwareDetector.get_full_hardware_info()
    print(f"🔧 Hardware Detected:")
    print(f"  GPUs: {hardware['gpu']['count']} ({hardware['gpu']['total_memory_gb']:.1f} GB total)")
    print(f"  CPUs: {hardware['cpu_memory']['cpu_cores']} cores")
    print(f"  Memory: {hardware['cpu_memory']['memory_gb']:.1f} GB")
    
    # Create optimal configuration
    config = create_optimal_distributed_config()
    print(f"\n📋 Optimal Configuration:")
    print(f"  Tensor Parallel: {config.tensor_parallel_size}")
    print(f"  World Size: {config.world_size}")
    print(f"  Concurrent Episodes: {config.concurrent_episodes}")
    print(f"  Brains per GPU: {config.brains_per_gpu}")
    
    # Setup distributed trainer
    trainer = DistributedBrainTrainer(config)
    success = await trainer.setup_distributed_training()
    
    if success:
        print("\n✅ Distributed setup successful!")
        status = trainer.get_training_status()
        print(f"📊 Cluster Status: {status['cluster']['healthy_nodes']}/{status['cluster']['total_nodes']} nodes healthy")
        print(f"🔥 Total GPUs: {status['cluster']['total_gpus']}")
    else:
        print("\n❌ Distributed setup failed")
    
    await trainer.shutdown_training()


if __name__ == "__main__":
    asyncio.run(demo_distributed_setup())
