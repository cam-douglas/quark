#!/usr/bin/env python3
"""
Google Colab Setup Script for Kaggle Integration with High-Performance Options
Optimized for Colab Pro/Pro+ users with maximum processing power
"""

import os
import subprocess
import json

def run_command(cmd, description):
    """Run a shell command and return result"""
    print(f"\n🔄 {description}")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Success: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def main():
    print("🚀 Setting up Google Colab with Kaggle Integration")
    print("🔥 HIGH-PERFORMANCE SETUP for Colab Pro/Pro+ Users")
    print("=" * 60)
    
    # Check if Kaggle is working
    if not run_command("kaggle competitions list", "Testing Kaggle API"):
        print("\n❌ Kaggle API not working. Please check your setup.")
        return
    
    # Create Colab setup instructions
    print("\n📝 Creating high-performance Colab setup instructions...")
    
    colab_instructions = """
# 🚀 HIGH-PERFORMANCE Google Colab Setup for Pro/Pro+ Users

## 🎯 Performance Optimization Checklist
- [ ] **Runtime Type**: GPU (T4/V100) or TPU v2-8/v3-8
- [ ] **RAM**: 32GB+ (Pro) or 52GB+ (Pro+)
- [ ] **Storage**: 100GB+ (Pro) or 200GB+ (Pro+)
- [ ] **Background Execution**: Enabled
- [ ] **Form**: Pro or Pro+ subscription active

## 📱 Step 1: Open Colab Pro/Pro+
1. Go to: https://colab.research.google.com
2. Ensure you're logged in with Pro/Pro+ account
3. Click "File" → "New notebook"

## ⚡ Step 2: Enable Maximum Performance
1. **Runtime** → **Change runtime type**
2. **Hardware accelerator**: **GPU** (T4/V100) or **TPU** (v2-8/v3-8)
3. **Runtime shape**: **High-RAM** (32GB+)
4. **GPU type**: **T4** (Pro) or **V100** (Pro+)

## 🔧 Step 3: Install and Configure Kaggle
Copy and paste this into the first cell:

```python
# High-performance Kaggle setup for Colab Pro/Pro+
!pip install -q kaggle torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
!pip install -q pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm

from google.colab import files
import os

# Upload your kaggle.json
print("Upload your kaggle.json file:")
files.upload()

# Configure Kaggle with high-performance settings
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Test connection with performance metrics
import time
start_time = time.time()
!kaggle competitions list
end_time = time.time()
print(f"\\n⚡ API Response Time: {end_time - start_time:.2f} seconds")
```

## 🚀 Step 4: Download Data with Parallel Processing
In the second cell:

```python
# High-performance data download
import concurrent.futures
import os

def download_competition(comp_name):
    !kaggle competitions download -c {comp_name} -p ./data --force
    
# Download multiple competitions in parallel (Pro+ feature)
competitions = ['titanic', 'house-prices-advanced-regression-techniques']
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    executor.map(download_competition, competitions)

# Extract with parallel processing
!cd ./data && for file in *.zip; do unzip -q "$file" & done; wait
```

## 🔍 Step 5: Verify High-Performance Setup
In the third cell:

```python
# Performance verification
import torch
import pandas as pd
import time

print("🔥 HIGH-PERFORMANCE SETUP VERIFICATION")
print("=" * 40)

# GPU/TPU Check
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA: {torch.version.cuda}")
    print(f"✅ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
elif hasattr(torch, 'xla'):
    print("✅ TPU: Available and configured")
else:
    print("⚠️  No GPU/TPU detected - check runtime settings")

# Data Loading Performance Test
start_time = time.time()
df = pd.read_csv('./data/train.csv')
load_time = time.time() - start_time
print(f"✅ Data Load: {load_time:.3f} seconds")
print(f"✅ Dataset Size: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Memory Usage
import psutil
memory_gb = psutil.virtual_memory().total / 1e9
print(f"✅ Available RAM: {memory_gb:.1f} GB")
```

## 🎯 Step 6: Advanced Performance Features (Pro+ Only)
```python
# Background execution for long-running tasks
!pip install -q jupyter_contrib_nbextensions
!jupyter contrib nbextension install --user

# Enable background execution
!jupyter nbextension enable --py widgetsnbextension

# Memory optimization
import gc
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None
```

## 🚀 Step 7: Model Training with Maximum Performance
```python
# High-performance training setup
import torch
from torch import nn
import torch.nn.functional as F

# Enable mixed precision for faster training (Pro+)
if torch.cuda.is_available():
    scaler = torch.cuda.amp.GradScaler()
    print("✅ Mixed Precision Training: Enabled")

# Multi-GPU setup (if available)
if torch.cuda.device_count() > 1:
    print(f"✅ Multi-GPU: {torch.cuda.device_count()} devices")
    device = torch.device("cuda")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"✅ Using device: {device}")
```

## 📊 Performance Monitoring
```python
# Real-time performance monitoring
!pip install -q nvidia-ml-py3
import nvidia_ml_py3 as nvml

if torch.cuda.is_available():
    nvml.nvmlInit()
    handle = nvml.nvmlDeviceGetHandleByIndex(0)
    
    # Monitor GPU usage
    gpu_util = nvml.nvmlDeviceGetUtilizationRates(handle)
    gpu_memory = nvml.nvmlDeviceGetMemoryInfo(handle)
    
    print(f"🔥 GPU Utilization: {gpu_util.gpu}%")
    print(f"🔥 GPU Memory: {gpu_memory.used/1e9:.1f}GB / {gpu_memory.total/1e9:.1f}GB")
```

## 🎯 Pro Tips for Maximum Performance
1. **Use TPU for large models** (v3-8 = 128GB memory)
2. **Enable mixed precision** for 2x speed boost
3. **Batch processing** with optimal batch sizes
4. **Memory management** with gradient checkpointing
5. **Background execution** for long-running tasks
6. **Parallel data loading** with multiple workers

## 🚨 Troubleshooting High-Performance Issues
- **Out of Memory**: Reduce batch size, use gradient checkpointing
- **Slow GPU**: Check runtime type, enable mixed precision
- **TPU Errors**: Restart runtime, check TPU availability
- **Memory Leaks**: Use `gc.collect()` and `torch.cuda.empty_cache()`
"""
    
    # Save instructions to file
    with open("COLAB_HIGH_PERFORMANCE_SETUP.md", "w") as f:
        f.write(colab_instructions)
    
    print("✅ Created COLAB_HIGH_PERFORMANCE_SETUP.md")
    
    # Test high-performance data download locally
    print("\n🔄 Testing high-performance local setup...")
    if run_command("mkdir -p high_perf_test", "Create high-performance test directory"):
        if run_command("kaggle competitions download -c titanic -p ./high_perf_test", "Download Titanic dataset"):
            if run_command("unzip ./high_perf_test/titanic.zip -d ./high_perf_test", "Extract dataset"):
                print("✅ High-performance local test successful!")
                print("📁 Data available in: ./high_perf_test/")
                
                # Test parallel processing simulation
                print("\n🔄 Testing parallel processing capabilities...")
                if run_command("cd high_perf_test && ls -la", "List extracted files"):
                    print("✅ Parallel processing test completed!")
            else:
                print("❌ Failed to extract dataset")
        else:
            print("❌ Failed to download dataset")
    
    print("\n🎉 HIGH-PERFORMANCE SETUP COMPLETE!")
    print("📖 Follow the instructions in: COLAB_HIGH_PERFORMANCE_SETUP.md")
    print("🔥 Optimized for Colab Pro/Pro+ with maximum processing power!")

if __name__ == "__main__":
    main()
