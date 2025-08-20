#!/usr/bin/env python3
"""
Start Continuous Training Script

Simple script to start infinite training loop for your SmallMind model.
"""

import os, sys
import argparse
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    parser = argparse.ArgumentParser(description="Start continuous training for SmallMind")
    parser.add_argument("--model-path", required=True, help="Path to your model")
    parser.add_argument("--output-dir", default="./continuous_training", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1000, help="Steps per epoch")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"❌ Model path not found: {args.model_path}")
        sys.exit(1)
    
    print("🚀 Starting SmallMind Continuous Training")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")
    print(f"Steps per epoch: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print()
    print("⚠️  This will run FOREVER until you stop it!")
    print("Press Ctrl+C to stop training gracefully")
    print("=" * 50)
    
    try:
        from smallmind.models.continuous_trainer import ContinuousTrainer
        
        # Create trainer
        trainer = ContinuousTrainer(args.model_path, args.output_dir)
        
        # Override default config
        trainer.config = type('Config', (), {
            'steps_per_epoch': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate
        })()
        
        # Start training
        trainer.train_forever()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Training stopped by user")
        print("Training completed gracefully!")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
