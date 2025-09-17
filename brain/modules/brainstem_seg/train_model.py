#!/usr/bin/env python3
"""
Brainstem Segmentation Model Training Script

Conducts actual training of the ViT-GNN hybrid model for brainstem segmentation.
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging
import json
from datetime import datetime
import sys

# Add pipeline to path
sys.path.append(str(Path(__file__).parent))
from pipeline import BrainstemTrainer, TrainingConfig

def setup_training_environment():
    """Setup training environment and check prerequisites."""
    
    print("🔧 SETTING UP TRAINING ENVIRONMENT")
    print("=" * 50)
    
    # Check hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   GPU: {gpu_name}")
        print(f"   GPU Memory: {gpu_memory:.1f} GB")
    else:
        print(f"   Device: CPU (CUDA not available)")
    
    # Check data availability
    data_dir = Path("/Users/camdouglas/quark/data/datasets/brainstem_segmentation")
    nextbrain_dir = data_dir / "nextbrain"
    
    data_available = {
        "nextbrain_volume": (nextbrain_dir / "T2w.nii.gz").exists(),
        "nextbrain_labels": (nextbrain_dir / "manual_segmentation.nii.gz").exists(),
        "label_schema": (data_dir / "metadata" / "brainstem_labels_schema.json").exists()
    }
    
    print(f"\n📁 Data Availability:")
    for dataset, available in data_available.items():
        status = "✅" if available else "❌"
        print(f"   {status} {dataset}")
    
    all_data_available = all(data_available.values())
    
    return device, all_data_available, data_dir

def create_training_config(device: str, fast_mode: bool = True) -> TrainingConfig:
    """Create training configuration."""
    
    if fast_mode:
        # Fast training for demonstration/testing
        config = TrainingConfig(
            # Model parameters (reduced for speed)
            embed_dim=384,      # Reduced from 768
            vit_layers=4,       # Reduced from 8  
            gnn_layers=2,       # Reduced from 3
            num_heads=6,        # Reduced from 8
            
            # Training parameters (fast)
            batch_size=2,       # Small batch for memory
            learning_rate=5e-4, # Higher LR for faster convergence
            num_epochs=20,      # Reduced from 100
            warmup_epochs=2,    # Reduced warmup
            
            # Validation parameters
            val_interval=2,     # More frequent validation
            save_interval=5,    # Less frequent saves
            early_stopping_patience=8,
            
            # Hardware
            device=device,
            mixed_precision=device == "cuda"
        )
        print(f"🚀 Fast Training Mode: {config.num_epochs} epochs, {config.embed_dim}d embedding")
    else:
        # Full training configuration
        config = TrainingConfig(device=device)
        print(f"🎯 Full Training Mode: {config.num_epochs} epochs, {config.embed_dim}d embedding")
    
    return config

def conduct_training(config: TrainingConfig, data_dir: Path, output_dir: Path):
    """Conduct the actual training."""
    
    print(f"\n🚀 STARTING TRAINING")
    print("=" * 30)
    
    try:
        # Initialize trainer
        print(f"🔧 Initializing trainer...")
        trainer = BrainstemTrainer(config, data_dir, output_dir)
        
        print(f"✅ Trainer initialized successfully")
        print(f"   Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
        print(f"   Device: {config.device}")
        print(f"   Batch size: {config.batch_size}")
        
        # Start training
        print(f"\n🎯 Starting training loop...")
        trainer.train()
        
        print(f"\n✅ Training completed successfully!")
        
        # Load and display best results
        best_model_path = output_dir / "best_model.pth"
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=config.device)
            best_metrics = checkpoint.get('metrics', {})
            
            print(f"\n📊 Best Model Performance:")
            print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"   Validation Dice: {best_metrics.get('dice', 0):.4f}")
            print(f"   Validation Accuracy: {best_metrics.get('accuracy', 0):.4f}")
            print(f"   Validation Loss: {best_metrics.get('loss', 0):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        logging.error(f"Training error: {e}", exc_info=True)
        return False

def main():
    """Main training execution."""
    
    print("🧠 BRAINSTEM SEGMENTATION MODEL TRAINING")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup environment
    device, data_available, data_dir = setup_training_environment()
    
    if not data_available:
        print(f"\n❌ Missing required data files!")
        print(f"   Please ensure NextBrain data is available in:")
        print(f"   {data_dir}/nextbrain/")
        return False
    
    # Create output directory
    output_dir = Path("/Users/camdouglas/quark/data/models/brainstem_segmentation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    print(f"\n📝 Logging to: {log_file}")
    
    # Ask user for training mode
    print(f"\n🎯 Training Mode Selection:")
    print(f"   1. Fast Mode (20 epochs, reduced model, ~10-15 min)")
    print(f"   2. Full Mode (100 epochs, full model, ~2-4 hours)")
    
    # For automation, default to fast mode
    fast_mode = True
    print(f"   Selected: Fast Mode (for demonstration)")
    
    # Create configuration
    config = create_training_config(device, fast_mode)
    
    # Save configuration
    config_file = output_dir / "training_config.json"
    config_dict = {
        "device": config.device,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "num_epochs": config.num_epochs,
        "embed_dim": config.embed_dim,
        "vit_layers": config.vit_layers,
        "gnn_layers": config.gnn_layers,
        "mixed_precision": config.mixed_precision,
        "training_start": datetime.now().isoformat()
    }
    
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"💾 Configuration saved: {config_file}")
    
    # Conduct training
    success = conduct_training(config, data_dir, output_dir)
    
    if success:
        print(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"   Output directory: {output_dir}")
        print(f"   Best model: {output_dir}/best_model.pth")
        print(f"   Training log: {log_file}")
        
        # List generated files
        print(f"\n📁 Generated Files:")
        for file in output_dir.glob("*"):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"   {file.name} ({size_mb:.1f} MB)")
        
        return True
    else:
        print(f"\n❌ TRAINING FAILED!")
        print(f"   Check log file: {log_file}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
