#!/usr/bin/env python3
"""Deploy vLLM on Google Cloud Platform for Quark brain systems.

This script sets up vLLM deployment on Google Cloud using Vertex AI and Compute Engine.
"""

import json
import subprocess
from pathlib import Path

def create_vllm_deployment_config():
    """Create deployment configuration for vLLM on GCP."""
    
    config = {
        "deployment_name": "quark-vllm-brain-server",
        "project_id": "quark-469604",
        "region": "us-central1",
        "machine_type": "n1-standard-4",
        "gpu_type": "nvidia-tesla-t4",
        "gpu_count": 1,
        "disk_size": "100GB",
        "model_path": "gs://quark-brain-models/gpt2-small",
        "vllm_config": {
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.8,
            "max_model_len": 2048,
            "port": 8000
        }
    }
    
    return config

def create_dockerfile():
    """Create Dockerfile for vLLM deployment."""
    
    dockerfile_content = """
FROM vllm/vllm-openai:latest

# Install additional dependencies for Quark integration
RUN pip install google-cloud-storage transformers

# Copy Quark brain wrapper
COPY brain/externals/vllm_brain_wrapper.py /app/vllm_brain_wrapper.py

# Set environment variables
ENV VLLM_CPU_KVCACHE_SPACE=4
ENV HF_TOKEN=${HF_TOKEN}

# Expose port
EXPOSE 8000

# Start vLLM server with brain optimizations
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "/models/gpt2-small", \
     "--tensor-parallel-size", "1", \
     "--gpu-memory-utilization", "0.8", \
     "--max-model-len", "2048", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
"""
    
    return dockerfile_content

def create_deployment_script():
    """Create deployment script for GCP."""
    
    script_content = """#!/bin/bash
# Deploy vLLM to Google Cloud Platform

set -e

PROJECT_ID="quark-469604"
REGION="us-central1"
SERVICE_NAME="quark-vllm-brain"

echo "🚀 Deploying vLLM to Google Cloud..."

# Build and push Docker image
echo "📦 Building Docker image..."
docker build -t gcr.io/${PROJECT_ID}/${SERVICE_NAME} .
docker push gcr.io/${PROJECT_ID}/${SERVICE_NAME}

# Deploy to Cloud Run
echo "☁️ Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \\
    --image gcr.io/${PROJECT_ID}/${SERVICE_NAME} \\
    --platform managed \\
    --region ${REGION} \\
    --allow-unauthenticated \\
    --memory 8Gi \\
    --cpu 4 \\
    --timeout 3600 \\
    --set-env-vars HF_TOKEN=${HF_TOKEN}

echo "✅ Deployment complete!"
echo "🔗 Service URL: $(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)')"
"""
    
    return script_content

def main():
    """Main deployment setup function."""
    print("🧠 Setting up vLLM deployment for Quark on Google Cloud")
    print("=" * 60)
    
    # Create deployment directory
    deploy_dir = Path("deployment/vllm-gcp")
    deploy_dir.mkdir(parents=True, exist_ok=True)
    
    # Create configuration
    config = create_vllm_deployment_config()
    with open(deploy_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"✅ Created deployment config: {deploy_dir}/config.json")
    
    # Create Dockerfile
    dockerfile = create_dockerfile()
    with open(deploy_dir / "Dockerfile", "w") as f:
        f.write(dockerfile)
    print(f"✅ Created Dockerfile: {deploy_dir}/Dockerfile")
    
    # Create deployment script
    script = create_deployment_script()
    with open(deploy_dir / "deploy.sh", "w") as f:
        f.write(script)
    
    # Make script executable
    subprocess.run(["chmod", "+x", str(deploy_dir / "deploy.sh")])
    print(f"✅ Created deployment script: {deploy_dir}/deploy.sh")
    
    print("\n📋 Next steps:")
    print(f"1. cd {deploy_dir}")
    print("2. ./deploy.sh")
    print("3. Test the deployed service")
    
    print("\n🔗 Integration with Quark:")
    print("- Update brain systems to use deployed vLLM endpoint")
    print("- Configure load balancing for multiple brain simulations")
    print("- Set up monitoring and auto-scaling")

if __name__ == "__main__":
    main()
