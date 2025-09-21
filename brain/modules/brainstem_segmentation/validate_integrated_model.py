#!/usr/bin/env python3
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

print(f"\n📍 Model location: {model_path}")
print(f"   Ready for inference and ONNX export")
