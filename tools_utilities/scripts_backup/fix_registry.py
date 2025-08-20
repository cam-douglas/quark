#!/usr/bin/env python3
"""
Fix Registry Script

This script manually updates the model registry to include
already downloaded models.
"""

import json
import os
from pathlib import Path
from datetime import datetime

def fix_registry():
    """Fix the model registry by adding already downloaded models."""
    
    registry_file = Path("models/configs/model_registry.json")
    deepseek_path = Path("models/checkpoints/deepseek-v2")
    
    if not deepseek_path.exists():
        print("❌ DeepSeek-V2 model not found")
        return
    
    # Create registry entry for DeepSeek-V2
    registry = {
        "deepseek-v2": {
            "model_id": "deepseek-ai/DeepSeek-V2",
            "local_path": str(deepseek_path.absolute()),
            "model_size_gb": 216.0,  # From du command
            "download_date": datetime.now().isoformat(),
            "checksum": "manual_fix",  # Placeholder
            "config": {
                "model_type": "deepseek",
                "num_experts": 16,
                "top_k": 2
            },
            "is_loaded": False,
            "memory_usage_mb": 0.0
        }
    }
    
    # Save registry
    with open(registry_file, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print("✅ Registry updated with DeepSeek-V2")
    print(f"   - Path: {deepseek_path}")
    print(f"   - Size: 216.0 GB")
    print(f"   - Registry file: {registry_file}")

if __name__ == "__main__":
    fix_registry()
