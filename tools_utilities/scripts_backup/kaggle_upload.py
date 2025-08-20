#!/usr/bin/env python3
"""Kaggle Model Upload Tool"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime

def setup_kaggle():
    """Setup Kaggle credentials"""
    print("🔐 Setting up Kaggle...")
    print("1. Go to https://www.kaggle.com/settings/account")
    print("2. Click 'Create New API Token'")
    print("3. Download kaggle.json")
    print("4. Place in ~/.kaggle/kaggle.json")
    print("5. Run: chmod 600 ~/.kaggle/kaggle.json")
    print("6. Install: pip install kaggle")

def prepare_model(model_name, project_root="."):
    """Prepare model for Kaggle upload"""
    models_dir = Path(project_root) / "src" / "models" / "models" / "checkpoints"
    export_dir = Path(project_root) / "kaggle_export" / model_name
    
    model_path = models_dir / model_name
    if not model_path.exists():
        print(f"❌ Model {model_name} not found")
        return None
    
    # Create export directory
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy model files
    shutil.copytree(model_path, export_dir / "model", dirs_exist_ok=True)
    
    # Create metadata
    metadata = {
        "model_name": model_name,
        "upload_date": datetime.now().isoformat(),
        "description": f"Trained {model_name} from SmallMind brain development project",
        "tags": ["brain-development", "neuroscience", "ml"],
        "license": "MIT"
    }
    
    with open(export_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ {model_name} prepared at: {export_dir}")
    return export_dir

def main():
    print("🚀 Kaggle Model Upload Tool")
    print("=" * 40)
    
    # Setup Kaggle
    setup_kaggle()
    
    # Prepare models
    models = ["qwen1.5-moe", "deepseek-v2"]
    
    for model in models:
        print(f"\n📦 Preparing {model}...")
        export_path = prepare_model(model)
        if export_path:
            print(f"   Ready for upload: {export_path}")
    
    print("\n📋 Next steps:")
    print("1. Set up Kaggle credentials (see above)")
    print("2. Run: kaggle datasets create -p kaggle_export/qwen1.5-moe")
    print("3. Run: kaggle datasets create -p kaggle_export/deepseek-v2")

if __name__ == "__main__":
    main()
