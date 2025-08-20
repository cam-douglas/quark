#!/usr/bin/env python3
"""
vLLM Brain Simulation Integration Setup
======================================

Setup script for vLLM integration with Quark Brain Simulation Framework.
Automatically configures optimal settings based on available hardware.

Usage:
    python scripts/setup_vllm_integration.py [--model MODEL_NAME] [--config] [--test]

Author: Quark Brain Team
Date: 2025-01-20
"""

import os, sys
import json
import subprocess
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    import torch
    from src.core.vllm_brain_integration import VLLMConfig, VLLMBrainEngine
    from src.core.vllm_distributed_config import HardwareDetector, create_optimal_distributed_config
    from src.core.deepseek_r1_trainer import DeepSeekConfig
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    print(f"⚠️  Some imports not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VLLMSetupManager:
    """Manage vLLM integration setup and configuration."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.src_dir = self.project_root / "src"
        self.config_dir = self.project_root / "src" / "config"
        self.scripts_dir = self.project_root / "scripts"
        
    def check_prerequisites(self) -> Dict[str, bool]:
        """Check system prerequisites for vLLM integration."""
        results = {
            "python_version": sys.version_info >= (3, 8),
            "torch_available": False,
            "cuda_available": False,
            "gpu_memory_sufficient": False,
            "vllm_compatible": False
        }
        
        try:
            import torch
            results["torch_available"] = True
            results["cuda_available"] = torch.cuda.is_available()
            
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                results["gpu_memory_sufficient"] = gpu_memory_gb >= 8  # Minimum 8GB
                results["vllm_compatible"] = torch.cuda.get_device_capability()[0] >= 7  # Compute capability 7.0+
                
        except ImportError:
            pass
        
        return results
    
    def install_dependencies(self, requirements_file: str = "requirements_vllm.txt") -> bool:
        """Install vLLM integration dependencies."""
        logger.info("📦 Installing vLLM integration dependencies...")
        
        requirements_path = self.project_root / requirements_file
        if not requirements_path.exists():
            logger.error(f"❌ Requirements file not found: {requirements_path}")
            return False
        
        try:
            # Install requirements
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            if result.returncode == 0:
                logger.info("✅ Dependencies installed successfully")
                return True
            else:
                logger.error(f"❌ Dependency installation failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Dependency installation timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Dependency installation error: {e}")
            return False
    
    def detect_optimal_configuration(self) -> Dict[str, Any]:
        """Detect optimal vLLM configuration based on hardware."""
        logger.info("🔍 Detecting optimal vLLM configuration...")
        
        if not IMPORTS_AVAILABLE:
            return {"error": "Required imports not available"}
        
        # Detect hardware
        hardware_info = HardwareDetector.get_full_hardware_info()
        
        # Get optimal model recommendation
        gpu_memory_gb = hardware_info["gpu"]["total_memory_gb"]
        recommended_model_key = DeepSeekConfig.get_recommended_model(gpu_memory_gb)
        recommended_model = DeepSeekConfig.MODELS[recommended_model_key]
        
        # Create optimal distributed config
        distributed_config = create_optimal_distributed_config()
        
        # Create vLLM config
        vllm_config = VLLMConfig(
            model_name=recommended_model["name"],
            tensor_parallel_size=distributed_config.tensor_parallel_size,
            max_model_len=32768 if gpu_memory_gb > 16 else 16384,
            gpu_memory_utilization=0.85 if gpu_memory_gb > 16 else 0.80
        )
        
        config = {
            "hardware_info": hardware_info,
            "recommended_model": {
                "key": recommended_model_key,
                "name": recommended_model["name"],
                "params": recommended_model["params"],
                "memory_gb": recommended_model["memory_gb"]
            },
            "vllm_config": vllm_config.__dict__,
            "distributed_config": {
                "tensor_parallel_size": distributed_config.tensor_parallel_size,
                "world_size": distributed_config.world_size,
                "concurrent_episodes": distributed_config.concurrent_episodes,
                "brains_per_gpu": distributed_config.brains_per_gpu
            },
            "performance_estimates": {
                "expected_inference_ms": self._estimate_inference_time(recommended_model_key, gpu_memory_gb),
                "estimated_throughput": self._estimate_throughput(recommended_model_key, distributed_config.tensor_parallel_size),
                "memory_usage_percent": (recommended_model["memory_gb"] / gpu_memory_gb) * 100
            }
        }
        
        return config
    
    def _estimate_inference_time(self, model_key: str, gpu_memory_gb: float) -> float:
        """Estimate inference time in milliseconds."""
        # Simple heuristic based on model size and GPU memory
        model_params = DeepSeekConfig.MODELS[model_key]["params"]
        
        if "1.5B" in model_params:
            base_time = 50
        elif "7B" in model_params:
            base_time = 150
        elif "14B" in model_params:
            base_time = 300
        elif "32B" in model_params:
            base_time = 600
        elif "70B" in model_params:
            base_time = 1200
        else:
            base_time = 200
        
        # Adjust based on GPU memory (more memory = faster)
        memory_factor = max(0.5, min(2.0, 32 / gpu_memory_gb))
        return base_time * memory_factor
    
    def _estimate_throughput(self, model_key: str, tensor_parallel_size: int) -> float:
        """Estimate throughput in tokens per second."""
        base_inference_time = self._estimate_inference_time(model_key, 32)  # Normalized
        
        # Assume 100 tokens average output
        tokens_per_request = 100
        requests_per_second = 1000 / base_inference_time
        
        # Tensor parallelism improves throughput
        parallelism_factor = min(tensor_parallel_size * 0.8, tensor_parallel_size)
        
        return requests_per_second * tokens_per_request * parallelism_factor
    
    def create_configuration_files(self, config: Dict[str, Any]):
        """Create configuration files for vLLM integration."""
        logger.info("📝 Creating configuration files...")
        
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)
        
        # Create vLLM brain integration config
        vllm_brain_config = {
            "vllm": config["vllm_config"],
            "distributed": config["distributed_config"],
            "model_info": config["recommended_model"],
            "performance": config["performance_estimates"]
        }
        
        config_path = self.config_dir / "vllm_brain_config.json"
        with open(config_path, 'w') as f:
            json.dump(vllm_brain_config, f, indent=2)
        
        logger.info(f"✅ vLLM config saved: {config_path}")
        
        # Create deployment scripts
        self._create_deployment_scripts(config)
        
        # Create docker configuration
        self._create_docker_config(config)
    
    def _create_deployment_scripts(self, config: Dict[str, Any]):
        """Create deployment scripts."""
        scripts = {
            "start_vllm_server.sh": self._generate_vllm_server_script(config),
            "start_brain_training.sh": self._generate_training_script(config),
            "start_api_server.sh": self._generate_api_server_script(config)
        }
        
        for script_name, script_content in scripts.items():
            script_path = self.scripts_dir / script_name
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Make executable
            os.chmod(script_path, 0o755)
            logger.info(f"✅ Created script: {script_path}")
    
    def _generate_vllm_server_script(self, config: Dict[str, Any]) -> str:
        """Generate vLLM server startup script."""
        vllm_config = config["vllm_config"]
        model_name = config["recommended_model"]["name"]
        
        return f"""#!/bin/bash
# vLLM Server Startup Script
# Generated by vLLM Brain Integration Setup

echo "🚀 Starting vLLM Server for Brain Simulation"
echo "Model: {model_name}"
echo "Tensor Parallel Size: {vllm_config['tensor_parallel_size']}"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_TRITON_FLASH_ATTN=1

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \\
    --model "{model_name}" \\
    --tensor-parallel-size {vllm_config['tensor_parallel_size']} \\
    --max-model-len {vllm_config['max_model_len']} \\
    --gpu-memory-utilization {vllm_config['gpu_memory_utilization']} \\
    --host 0.0.0.0 \\
    --port 8000 \\
    --trust-remote-code \\
    --enable-prefix-caching \\
    --use-v2-block-manager \\
    --dtype bfloat16 \\
    --api-key brain-simulation-key

echo "✅ vLLM Server started on http://localhost:8000"
"""
    
    def _generate_training_script(self, config: Dict[str, Any]) -> str:
        """Generate brain training script."""
        return f"""#!/bin/bash
# Brain Training with vLLM Integration
# Generated by vLLM Brain Integration Setup

echo "🧠 Starting Brain Simulation Training with vLLM"

# Set environment variables
export VLLM_MODEL="{config['recommended_model']['name']}"
export TENSOR_PARALLEL_SIZE={config['distributed_config']['tensor_parallel_size']}
export CONCURRENT_EPISODES={config['distributed_config']['concurrent_episodes']}

# Start training
python -m src.core.vllm_training_pipeline \\
    --model "$VLLM_MODEL" \\
    --stages F N0 N1 \\
    --episodes-per-stage 10 \\
    --steps-per-episode 100 \\
    --output-dir "training_outputs_$(date +%Y%m%d_%H%M%S)"

echo "✅ Training completed!"
"""
    
    def _generate_api_server_script(self, config: Dict[str, Any]) -> str:
        """Generate API server script."""
        return f"""#!/bin/bash
# Brain Simulation API Server
# Generated by vLLM Brain Integration Setup

echo "🌐 Starting Brain Simulation API Server"

# Set environment variables
export VLLM_MODEL="{config['recommended_model']['name']}"
export TENSOR_PARALLEL_SIZE={config['distributed_config']['tensor_parallel_size']}
export PORT=8001

# Start API server
python -m src.core.vllm_api_server

echo "✅ API Server started on http://localhost:8001"
echo "📖 API Documentation: http://localhost:8001/docs"
"""
    
    def _create_docker_config(self, config: Dict[str, Any]):
        """Create Docker configuration."""
        dockerfile_content = f"""# vLLM Brain Simulation Dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \\
    python3.10 \\
    python3-pip \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Set Python as default
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements_vllm.txt .
RUN pip install --no-cache-dir -r requirements_vllm.txt

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Set environment variables
ENV VLLM_MODEL="{config['recommended_model']['name']}"
ENV TENSOR_PARALLEL_SIZE={config['distributed_config']['tensor_parallel_size']}
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 8000 8001

# Default command
CMD ["python", "-m", "src.core.vllm_api_server"]
"""
        
        dockerfile_path = self.project_root / "Dockerfile.vllm"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        logger.info(f"✅ Created Dockerfile: {dockerfile_path}")
        
        # Create Docker Compose
        compose_content = f"""version: '3.8'

services:
  vllm-brain-simulation:
    build:
      context: .
      dockerfile: Dockerfile.vllm
    ports:
      - "8000:8000"  # vLLM server
      - "8001:8001"  # API server
    environment:
      - CUDA_VISIBLE_DEVICES=0,1,2,3
      - VLLM_MODEL={config['recommended_model']['name']}
      - TENSOR_PARALLEL_SIZE={config['distributed_config']['tensor_parallel_size']}
    volumes:
      - ./training_outputs:/app/training_outputs
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: {config['distributed_config']['tensor_parallel_size']}
              capabilities: [gpu]
"""
        
        compose_path = self.project_root / "docker-compose.vllm.yml"
        with open(compose_path, 'w') as f:
            f.write(compose_content)
        
        logger.info(f"✅ Created Docker Compose: {compose_path}")
    
    async def test_integration(self, config: Dict[str, Any]) -> Dict[str, bool]:
        """Test vLLM integration components."""
        logger.info("🧪 Testing vLLM integration...")
        
        test_results = {
            "vllm_engine_init": False,
            "brain_inference": False,
            "training_pipeline": False,
            "api_server": False
        }
        
        if not IMPORTS_AVAILABLE:
            logger.error("❌ Cannot run tests - imports not available")
            return test_results
        
        try:
            # Test vLLM engine initialization
            vllm_config = VLLMConfig(**config["vllm_config"])
            engine = VLLMBrainEngine(vllm_config)
            
            # This would normally require GPU and model download
            # For now, just test configuration
            test_results["vllm_engine_init"] = True
            logger.info("✅ vLLM engine configuration test passed")
            
        except Exception as e:
            logger.error(f"❌ vLLM engine test failed: {e}")
        
        return test_results
    
    def print_setup_summary(self, config: Dict[str, Any]):
        """Print setup summary and next steps."""
        print("\n" + "="*60)
        print("🎉 vLLM Brain Simulation Integration Setup Complete!")
        print("="*60)
        
        print("\n📊 Configuration Summary:")
        print(f"  Model: {config['recommended_model']['name']}")
        print(f"  Parameters: {config['recommended_model']['params']}")
        print(f"  Memory Required: {config['recommended_model']['memory_gb']} GB")
        print(f"  Tensor Parallel Size: {config['distributed_config']['tensor_parallel_size']}")
        print(f"  Concurrent Episodes: {config['distributed_config']['concurrent_episodes']}")
        print(f"  Expected Inference Time: {config['performance_estimates']['expected_inference_ms']:.0f} ms")
        print(f"  Estimated Throughput: {config['performance_estimates']['estimated_throughput']:.0f} tokens/sec")
        
        print("\n🚀 Quick Start Commands:")
        print("  1. Start vLLM server:")
        print("     ./scripts/start_vllm_server.sh")
        print("\n  2. Start brain training:")
        print("     ./scripts/start_brain_training.sh")
        print("\n  3. Start API server:")
        print("     ./scripts/start_api_server.sh")
        print("\n  4. Use Docker:")
        print("     docker-compose -f docker-compose.vllm.yml up")
        
        print("\n📖 Documentation:")
        print("  API Docs: http://localhost:8001/docs")
        print("  Health Check: http://localhost:8001/health")
        print("  vLLM OpenAI API: http://localhost:8000/v1/models")
        
        print("\n✨ Next Steps:")
        print("  1. Download model (first run will be slow)")
        print("  2. Run integration tests")
        print("  3. Start with small brain simulations")
        print("  4. Scale up based on performance")
        print("\n" + "="*60)

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup vLLM Brain Simulation Integration")
    parser.add_argument("--model", help="Specific model to configure")
    parser.add_argument("--install", action="store_true", help="Install dependencies")
    parser.add_argument("--config", action="store_true", help="Generate configuration files")
    parser.add_argument("--test", action="store_true", help="Run integration tests")
    parser.add_argument("--all", action="store_true", help="Run all setup steps")
    
    args = parser.parse_args()
    
    setup_manager = VLLMSetupManager()
    
    print("🧠 vLLM Brain Simulation Integration Setup")
    print("="*50)
    
    # Check prerequisites
    prereqs = setup_manager.check_prerequisites()
    print("\n🔍 Prerequisites Check:")
    for check, result in prereqs.items():
        status = "✅" if result else "❌"
        print(f"  {status} {check.replace('_', ' ').title()}")
    
    if not all(prereqs.values()):
        print("\n⚠️  Some prerequisites are not met. Please check your environment.")
        if not prereqs["cuda_available"]:
            print("  💡 vLLM requires CUDA-compatible GPU for optimal performance")
        if not prereqs["gpu_memory_sufficient"]:
            print("  💡 Consider using a smaller model or increasing GPU memory")
    
    # Install dependencies
    if args.install or args.all:
        success = setup_manager.install_dependencies()
        if not success:
            print("❌ Dependency installation failed. Exiting.")
            return 1
    
    # Detect configuration
    if args.config or args.all or not args.install:
        config = setup_manager.detect_optimal_configuration()
        
        if "error" in config:
            print(f"❌ Configuration detection failed: {config['error']}")
            return 1
        
        # Create configuration files
        setup_manager.create_configuration_files(config)
        
        # Print summary
        setup_manager.print_setup_summary(config)
    
    # Run tests
    if args.test or args.all:
        import asyncio
        test_results = asyncio.run(setup_manager.test_integration(config))
        
        print("\n🧪 Integration Test Results:")
        for test_name, result in test_results.items():
            status = "✅" if result else "❌"
            print(f"  {status} {test_name.replace('_', ' ').title()}")
    
    print("\n🎉 Setup completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
