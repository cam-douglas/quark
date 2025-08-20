#!/usr/bin/env python3
"""
Start Multi-Model Training Script

Trains all 3 SmallMind models simultaneously with proper resource management.
"""

import os, sys
import argparse
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    parser = argparse.ArgumentParser(description="Start multi-model training for SmallMind")
    parser.add_argument("--output-dir", default="./multi_model_training", help="Output directory")
    parser.add_argument("--check-models", action="store_true", help="Check available models before starting")
    
    args = parser.parse_args()
    
    print("🚀 SmallMind Multi-Model Training")
    print("=" * 50)
    print("This will train ALL available models simultaneously:")
    print("  • DeepSeek V2 (64 experts, 32GB memory)")
    print("  • Qwen 1.5 MoE (8 experts, 8GB memory)")
    print("  • MixTAO MoE (16 experts, 16GB memory)")
    print()
    print("⚠️  This will run FOREVER until you stop it!")
    print("Press Ctrl+C to stop all training gracefully")
    print("=" * 50)
    
    # Check available models
    if args.check_models:
        print("\n🔍 Checking available models...")
        models = [
            ("DeepSeek V2", "src/smallmind/models/models/checkpoints/deepseek-v2"),
            ("Qwen 1.5 MoE", "src/smallmind/models/models/checkpoints/qwen1.5-moe"),
            ("MixTAO MoE", "src/smallmind/models/models/checkpoints/mix-tao-moe")
        ]
        
        available_models = []
        for name, path in models:
            if os.path.exists(path):
                print(f"✅ {name}: {path}")
                available_models.append(name)
            else:
                print(f"❌ {name}: {path} (not found)")
        
        if not available_models:
            print("\n❌ No models found! Please check your model paths.")
            return
        
        print(f"\n📊 Found {len(available_models)} models: {', '.join(available_models)}")
    
    # Confirm before starting
    confirm = input("\nStart multi-model training? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    try:
        from smallmind.models.multi_model_trainer import start_multi_model_training
        
        print(f"\n🚀 Starting multi-model training...")
        print(f"📁 Output directory: {args.output_dir}")
        print("=" * 50)
        
        # Start training all models
        trainer = start_multi_model_training(args.output_dir)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install datasets transformers torch accelerate")
        return 1
    except Exception as e:
        print(f"❌ Multi-model training failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
