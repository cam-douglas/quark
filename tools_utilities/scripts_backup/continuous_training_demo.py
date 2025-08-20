#!/usr/bin/env python3
"""
Continuous Training Demo for SmallMind

Shows how to run training in an infinite loop until manually stopped.
"""

import os, sys
import logging
import time
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from smallmind.models.continuous_trainer import ContinuousTrainer, train_forever
    from smallmind.models import get_dataset_integrator
    CONTINUOUS_TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"Error importing continuous training: {e}")
    CONTINUOUS_TRAINING_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_continuous_training():
    """Demo the continuous training system"""
    if not CONTINUOUS_TRAINING_AVAILABLE:
        print("❌ Continuous training not available")
        return
    
    print("🚀 SmallMind Continuous Training Demo")
    print("=" * 50)
    print("This demo shows how to train your model continuously")
    print("in an infinite loop until manually stopped.")
    print()
    
    # Check if we have a model to train
    model_path = input("Enter the path to your model (or press Enter for demo): ").strip()
    
    if not model_path:
        # Demo mode - show how it would work
        print("\n📚 Demo Mode - Showing how continuous training works:")
        print("1. Training runs in an infinite loop")
        print("2. Rotates through different dataset mixtures")
        print("3. Saves checkpoints after each epoch")
        print("4. Continues until manually stopped (Ctrl+C)")
        print("5. Automatically resumes from last checkpoint")
        
        print("\n🔧 To run actual continuous training:")
        print("python src/smallmind/demos/continuous_training_demo.py")
        print("Then provide a real model path when prompted.")
        
        return
    
    # Validate model path
    if not os.path.exists(model_path):
        print(f"❌ Model path not found: {model_path}")
        return
    
    print(f"\n✅ Model found: {model_path}")
    
    # Setup output directory
    output_dir = f"./continuous_training_{int(time.time())}"
    
    print(f"📁 Output directory: {output_dir}")
    print(f"🔄 Training mixtures: balanced, code_focused, reasoning_focused")
    print(f"⏱️  Epochs: Infinite (until stopped)")
    print(f"💾 Checkpoints: Saved after each epoch")
    
    # Confirm before starting
    print("\n⚠️  WARNING: This will start infinite training!")
    print("Press Ctrl+C to stop when you want to finish.")
    
    confirm = input("\nStart continuous training? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    print("\n🚀 Starting continuous training...")
    print("Press Ctrl+C to stop training gracefully")
    print("=" * 50)
    
    try:
        # Start continuous training
        trainer = ContinuousTrainer(model_path, output_dir)
        trainer.train_forever()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Training stopped by user")
        print("Training will complete the current epoch and save progress.")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("Check the logs for more details.")

def demo_dataset_rotation():
    """Demo how dataset rotation works"""
    print("\n🔄 Dataset Rotation Demo:")
    print("-" * 30)
    
    try:
        integrator = get_dataset_integrator()
        mixtures = ["balanced", "code_focused", "reasoning_focused"]
        
        print("Available training mixtures:")
        for i, mixture in enumerate(mixtures):
            info = integrator.get_mixture_info(mixture)
            if info:
                print(f"  {i+1}. {mixture}: {', '.join(info['datasets'][:3])}...")
        
        print("\nRotation pattern:")
        print("  Epoch 1: balanced (general skills)")
        print("  Epoch 2: code_focused (code emphasis)")
        print("  Epoch 3: reasoning_focused (reasoning skills)")
        print("  Epoch 4: balanced (back to general)")
        print("  ... and so on")
        
        print("\nThis ensures your model gets balanced training across all domains!")
        
    except Exception as e:
        print(f"❌ Failed to show dataset rotation: {e}")

def demo_checkpointing():
    """Demo how checkpointing works"""
    print("\n💾 Checkpointing Demo:")
    print("-" * 30)
    
    print("After each epoch, the system:")
    print("  ✅ Saves the trained model")
    print("  ✅ Saves training metrics")
    print("  ✅ Updates training state")
    print("  ✅ Creates epoch summary")
    
    print("\nDirectory structure:")
    print("  continuous_training/")
    print("  ├── epoch_0001_balanced/")
    print("  │   ├── pytorch_model.bin")
    print("  │   ├── config.json")
    print("  │   └── training_results.json")
    print("  ├── epoch_0002_code_focused/")
    print("  │   ├── pytorch_model.bin")
    print("  │   └── ...")
    print("  └── training_summary.json")
    
    print("\nYou can resume training from any checkpoint!")

def demo_control_methods():
    """Demo different ways to control continuous training"""
    print("\n🎮 Control Methods Demo:")
    print("-" * 30)
    
    print("1. **Graceful Stop (Ctrl+C)**")
    print("   - Completes current epoch")
    print("   - Saves all progress")
    print("   - Clean shutdown")
    
    print("\n2. **Programmatic Stop**")
    print("   - trainer.stop_training()")
    print("   - Clean shutdown from code")
    
    print("\n3. **Signal Stop (SIGTERM)**")
    print("   - System signal handling")
    print("   - Graceful shutdown")
    
    print("\n4. **Auto-restart on Failure**")
    print("   - Automatically retries failed epochs")
    print("   - Configurable retry limits")
    print("   - Continues training seamlessly")

def main():
    """Main demo function"""
    print("🎯 SmallMind Continuous Training Demo")
    print("This demo shows how to train your model continuously")
    print("in an infinite loop until manually stopped.\n")
    
    try:
        # Demo 1: Dataset rotation
        demo_dataset_rotation()
        
        # Demo 2: Checkpointing
        demo_checkpointing()
        
        # Demo 3: Control methods
        demo_control_methods()
        
        # Demo 4: Actual continuous training
        demo_continuous_training()
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("\nThis might be due to missing dependencies.")
        print("Install with: pip install datasets transformers torch accelerate")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
