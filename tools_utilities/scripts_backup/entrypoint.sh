#!/bin/bash
# Wikipedia Training Container Entrypoint
# =======================================
# 
# Handles container initialization, distributed training setup,
# and graceful shutdown for Wikipedia training pipeline

set -e

# Configuration
export PYTHONPATH="/app:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Logging setup
LOG_DIR="/app/logs"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/container.log")
exec 2>&1

echo "Starting Wikipedia Training Container at $(date)"
echo "Container ID: $(hostname)"
echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"
echo "Python Path: $PYTHONPATH"

# Function to check CUDA availability
check_cuda() {
    echo "Checking CUDA availability..."
    python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB')
"
}

# Function to setup distributed training
setup_distributed() {
    echo "Setting up distributed training..."
    
    export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
    export MASTER_PORT=${MASTER_PORT:-"23456"}
    export WORLD_SIZE=${WORLD_SIZE:-"1"}
    export RANK=${RANK:-"0"}
    export LOCAL_RANK=${LOCAL_RANK:-"0"}
    
    echo "Distributed Training Configuration:"
    echo "  Master Address: $MASTER_ADDR"
    echo "  Master Port: $MASTER_PORT"
    echo "  World Size: $WORLD_SIZE"
    echo "  Rank: $RANK"
    echo "  Local Rank: $LOCAL_RANK"
    
    # Wait for master node to be ready
    if [ "$RANK" != "0" ]; then
        echo "Waiting for master node..."
        while ! nc -z "$MASTER_ADDR" "$MASTER_PORT"; do
            sleep 5
            echo "Still waiting for master node..."
        done
        echo "Master node is ready!"
    fi
}

# Function to download Wikipedia data if needed
setup_wikipedia_data() {
    echo "Setting up Wikipedia data..."
    
    local cache_dir="/mnt/cache/wikipedia"
    local dump_date=${WIKIPEDIA_DUMP_DATE:-"20240101"}
    local dump_file="enwiki-${dump_date}-pages-articles.xml.bz2"
    local dump_path="$cache_dir/$dump_file"
    
    mkdir -p "$cache_dir"
    
    if [ ! -f "$dump_path" ]; then
        echo "Wikipedia dump not found. Downloading..."
        local dump_url="https://dumps.wikimedia.org/enwiki/${dump_date}/${dump_file}"
        
        # Download with retry logic
        local max_retries=3
        local retry=0
        
        while [ $retry -lt $max_retries ]; do
            if wget -c -O "$dump_path" "$dump_url"; then
                echo "Successfully downloaded Wikipedia dump"
                break
            else
                retry=$((retry + 1))
                echo "Download failed (attempt $retry/$max_retries), retrying..."
                sleep 10
            fi
        done
        
        if [ $retry -eq $max_retries ]; then
            echo "Failed to download Wikipedia dump after $max_retries attempts"
            exit 1
        fi
    else
        echo "Wikipedia dump already exists: $dump_path"
    fi
    
    # Verify file integrity
    if [ -f "$dump_path" ]; then
        local file_size=$(stat -c%s "$dump_path")
        echo "Wikipedia dump size: $((file_size / 1024 / 1024 / 1024)) GB"
        
        if [ $file_size -lt 1000000000 ]; then  # Less than 1GB suggests incomplete download
            echo "Warning: Wikipedia dump seems too small, may be incomplete"
        fi
    fi
}

# Function to setup cloud storage
setup_cloud_storage() {
    echo "Setting up cloud storage..."
    
    case "${CLOUD_PROVIDER:-aws}" in
        "aws")
            if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
                echo "AWS credentials found"
                aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID"
                aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
                aws configure set default.region "${AWS_REGION:-us-west-2}"
            else
                echo "Using IAM role for AWS authentication"
            fi
            ;;
        "gcp")
            if [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
                echo "GCP credentials found: $GOOGLE_APPLICATION_CREDENTIALS"
            else
                echo "Warning: No GCP credentials found"
            fi
            ;;
        "azure")
            if [ -n "$AZURE_STORAGE_CONNECTION_STRING" ]; then
                echo "Azure storage credentials found"
            else
                echo "Warning: No Azure credentials found"
            fi
            ;;
    esac
}

# Function to setup monitoring
setup_monitoring() {
    echo "Setting up monitoring..."
    
    # Initialize wandb if API key is provided
    if [ -n "$WANDB_API_KEY" ]; then
        echo "Initializing Weights & Biases..."
        wandb login "$WANDB_API_KEY"
    else
        echo "No W&B API key provided, running in offline mode"
        export WANDB_MODE=offline
    fi
    
    # Start system monitoring
    if command -v gpustat &> /dev/null; then
        echo "Starting GPU monitoring..."
        gpustat --json -i 30 > "$LOG_DIR/gpu_stats.json" &
    fi
}

# Function to cleanup on exit
cleanup() {
    echo "Cleaning up..."
    
    # Kill background processes
    jobs -p | xargs -r kill
    
    # Save final logs
    if [ -d "$LOG_DIR" ]; then
        echo "Training completed at $(date)" >> "$LOG_DIR/container.log"
    fi
    
    echo "Cleanup complete"
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# Main execution
main() {
    echo "=== Wikipedia Training Container Initialization ==="
    
    # System checks
    check_cuda
    
    # Setup components
    setup_distributed
    setup_cloud_storage
    setup_monitoring
    
    # Setup data if this is the master node or single node
    if [ "${RANK:-0}" = "0" ] || [ "${WORLD_SIZE:-1}" = "1" ]; then
        setup_wikipedia_data
    fi
    
    echo "=== Starting Training Process ==="
    
    # Execute the main command
    if [ $# -eq 0 ]; then
        # Default command
        python3 /app/training/wikipedia_cloud_training.py \
            --cloud-provider "${CLOUD_PROVIDER:-aws}" \
            --num-nodes "${WORLD_SIZE:-1}"
    else
        # Execute provided command
        exec "$@"
    fi
}

# Run main function
main "$@"
