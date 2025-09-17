#!/usr/bin/env python3
"""
Integrate Google Colab Trained Model Back to Quark
==================================================

This script integrates the trained model from Google Colab back into the Quark system.
Run this after downloading best_model_colab.pth from Colab.
"""

import os
import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import shutil


def integrate_colab_model():
    """Integrate the Colab-trained model into Quark."""
    
    print("=" * 60)
    print("🔄 Integrating Colab Model into Quark")
    print("=" * 60)
    
    # Define paths
    base_dir = Path("/Users/camdouglas/quark/brain/modules/brainstem_segmentation")
    model_dir = Path("/Users/camdouglas/quark/data/models/brainstem")
    
    # Create model directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Check for downloaded model
    print("\n1️⃣ Checking for downloaded model...")
    
    possible_locations = [
        Path.home() / "Downloads" / "best_model_colab.pth",
        base_dir / "best_model_colab.pth",
        Path.cwd() / "best_model_colab.pth",
        base_dir / "colab_upload" / "best_model_colab.pth"
    ]
    
    model_path = None
    for location in possible_locations:
        if location.exists():
            model_path = location
            print(f"   ✅ Found model at: {model_path}")
            break
    
    if model_path is None:
        print("   ❌ Model not found. Please download best_model_colab.pth from Colab")
        print("   Expected locations:")
        for loc in possible_locations:
            print(f"      - {loc}")
        return False
    
    # 2. Load and verify the model
    print("\n2️⃣ Loading and verifying model...")
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract information
        if isinstance(checkpoint, dict):
            model_state = checkpoint.get('model_state_dict', checkpoint)
            best_dice = checkpoint.get('best_dice', 'Unknown')
            epoch = checkpoint.get('epoch', 'Unknown')
            config = checkpoint.get('config', None)
        else:
            model_state = checkpoint
            best_dice = 'Unknown'
            epoch = 'Unknown'
            config = None
        
        print(f"   ✅ Model loaded successfully")
        print(f"   Best Dice: {best_dice}")
        print(f"   Training epochs: {epoch}")
        
        # Check if target achieved
        if isinstance(best_dice, (float, int)) and best_dice >= 0.87:
            print(f"   🎯 TARGET ACHIEVED! Dice: {best_dice:.4f} ≥ 0.87")
        elif isinstance(best_dice, (float, int)):
            print(f"   ⚠️ Target not yet achieved. Dice: {best_dice:.4f} < 0.87")
            print(f"   Consider training for more epochs on Colab")
        
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return False
    
    # 3. Copy model to Quark model directory
    print("\n3️⃣ Installing model in Quark...")
    
    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = model_dir / f"brainstem_colab_{timestamp}.pth"
    best_model_path = model_dir / "best_model.pth"
    
    # Copy model
    shutil.copy2(model_path, final_model_path)
    print(f"   ✅ Saved to: {final_model_path}")
    
    # Also save as best_model.pth for easy access
    shutil.copy2(model_path, best_model_path)
    print(f"   ✅ Linked as: {best_model_path}")
    
    # 4. Create metadata
    print("\n4️⃣ Creating integration metadata...")
    
    metadata = {
        "integration_date": datetime.now().isoformat(),
        "source": "Google Colab GPU Training",
        "model_path": str(final_model_path),
        "best_dice": float(best_dice) if isinstance(best_dice, (float, int)) else best_dice,
        "training_epochs": int(epoch) if isinstance(epoch, (int, float)) else epoch,
        "target_achieved": isinstance(best_dice, (float, int)) and best_dice >= 0.87,
        "architecture": "Enhanced U-Net 3D with Attention Gates",
        "input_channels": 4,
        "num_classes": 6,
        "notes": "Trained on Google Colab with GPU acceleration"
    }
    
    metadata_path = model_dir / f"metadata_{timestamp}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✅ Metadata saved: {metadata_path}")
    
    # 5. Update ONNX export if needed
    print("\n5️⃣ Preparing for ONNX export...")
    
    if isinstance(best_dice, (float, int)) and best_dice >= 0.87:
        print("   ✅ Model ready for ONNX export")
        print("   Run: python brain/modules/brainstem_segmentation/export_final_onnx.py")
    else:
        print("   ⚠️ Continue training to reach target before ONNX export")
    
    # 6. Create validation script
    print("\n6️⃣ Creating validation script...")
    
    validation_script = base_dir / "validate_integrated_model.py"
    validation_code = '''#!/usr/bin/env python3
"""Validate the integrated model."""

import torch
import numpy as np
from pathlib import Path

# Load model
model_path = Path("/Users/camdouglas/quark/data/models/brainstem/best_model.pth")
checkpoint = torch.load(model_path, map_location='cpu')

if isinstance(checkpoint, dict):
    print(f"✅ Model validation:")
    print(f"   Best Dice: {checkpoint.get('best_dice', 'Unknown')}")
    print(f"   Epochs trained: {checkpoint.get('epoch', 'Unknown')}")
    print(f"   Target achieved: {checkpoint.get('best_dice', 0) >= 0.87}")
else:
    print("✅ Model loaded successfully (weights only)")

print(f"\\n📍 Model location: {model_path}")
print(f"   Ready for inference and ONNX export")
'''
    
    with open(validation_script, "w") as f:
        f.write(validation_code)
    print(f"   ✅ Created: {validation_script.name}")
    
    print("\n" + "=" * 60)
    print("✅ INTEGRATION COMPLETE")
    print("=" * 60)
    print(f"\n📦 Model installed at:")
    print(f"   {best_model_path}")
    print(f"\n🔍 To validate:")
    print(f"   python {validation_script}")
    print(f"\n🚀 Next steps:")
    
    if isinstance(best_dice, (float, int)) and best_dice >= 0.87:
        print(f"   1. Export to ONNX for deployment")
        print(f"   2. Run inference on test data")
        print(f"   3. Deploy to production")
    else:
        print(f"   1. Continue training on Colab to reach Dice ≥ 0.87")
        print(f"   2. Re-integrate improved model")
        print(f"   3. Then export to ONNX")
    
    print("\n" + "=" * 60)
    
    return True


if __name__ == "__main__":
    success = integrate_colab_model()
    if not success:
        print("\n❌ Integration failed. Please check the errors above.")
        print("Make sure you've downloaded best_model_colab.pth from Google Colab")
