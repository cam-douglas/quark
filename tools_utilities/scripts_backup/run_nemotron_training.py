#!/usr/bin/env python3
"""
Nemotron Training Runner Script
==============================

Convenient script to run Nemotron dataset training with different configurations.
Integrates with the Quark brain simulation framework.

Usage:
    python scripts/run_nemotron_training.py --model-config small --splits chat code math
    python scripts/run_nemotron_training.py --model-config medium --max-samples 10000
"""

import os, sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.nemotron_trainer import NemotronTrainer, NemotronConfig

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('nemotron_training.log'),
            logging.StreamHandler()
        ]
    )

def check_gpu_memory():
    """Check available GPU memory and recommend model config."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🖥️  GPU Memory: {gpu_memory:.1f} GB")
            
            if gpu_memory >= 32:
                return "large"
            elif gpu_memory >= 16:
                return "medium"
            else:
                return "small"
        else:
            print("⚠️  No GPU detected, using CPU")
            return "small"
    except Exception as e:
        print(f"⚠️  Error checking GPU: {e}")
        return "small"

def show_dataset_info():
    """Display Nemotron dataset information."""
    print("\n📊 Nemotron Dataset Information:")
    print("=" * 50)
    print(f"Dataset: {NemotronConfig.DATASET_INFO['name']}")
    print(f"License: {NemotronConfig.DATASET_INFO['license']}")
    print(f"Total Examples: {NemotronConfig.DATASET_INFO['total']:,}")
    print("\nCategories:")
    for category, count in NemotronConfig.DATASET_INFO['categories'].items():
        print(f"  - {category}: {count:,} examples")
    print(f"\nModels Used: {', '.join(NemotronConfig.DATASET_INFO['models_used'])}")

def show_model_configs():
    """Display available model configurations."""
    print("\n🤖 Available Model Configurations:")
    print("=" * 50)
    for config_name, config in NemotronConfig.MODEL_CONFIGS.items():
        print(f"\n{config_name.upper()}:")
        print(f"  Model: {config['model_name']}")
        print(f"  Memory Required: {config['memory_gb']} GB")
        print(f"  Batch Size: {config['batch_size']}")
        print(f"  Gradient Accumulation: {config['gradient_accumulation']}")

def validate_splits(splits):
    """Validate that requested splits exist in the dataset."""
    available_splits = list(NemotronConfig.DATASET_INFO['categories'].keys())
    invalid_splits = [split for split in splits if split not in available_splits]
    
    if invalid_splits:
        print(f"❌ Invalid splits: {invalid_splits}")
        print(f"Available splits: {available_splits}")
        return False
    
    return True

def run_training(args):
    """Run Nemotron training with specified configuration."""
    print("\n🚀 Starting Nemotron Training Pipeline")
    print("=" * 60)
    
    # Validate splits
    if not validate_splits(args.splits):
        return False
    
    # Show configuration
    print(f"Model Config: {args.model_config}")
    print(f"Splits: {', '.join(args.splits)}")
    print(f"Max Samples per Split: {args.max_samples:,}")
    print(f"Output Directory: {args.output_dir}")
    
    # Initialize trainer
    try:
        trainer = NemotronTrainer(model_config=args.model_config)
    except Exception as e:
        print(f"❌ Error initializing trainer: {e}")
        return False
    
    # Run training
    result = trainer.run_training(
        splits=args.splits,
        max_samples_per_split=args.max_samples
    )
    
    if result["success"]:
        print("\n🎉 Training Completed Successfully!")
        print("=" * 40)
        print(f"📁 Model Path: {result['model_path']}")
        print(f"📊 Total Samples: {result['total_samples']:,}")
        print(f"📈 Training Loss: {result['training_loss']:.4f}")
        print(f"⏱️  Training Time: {result['training_time']}")
        print(f"🔄 Total Steps: {result['total_steps']}")
        
        # Test the model
        print("\n🧪 Testing Fine-tuned Model...")
        try:
            test_prompt = "Explain how neural networks can simulate consciousness."
            response = trainer.generate_response(test_prompt, max_length=256)
            print(f"Test Prompt: {test_prompt}")
            print(f"Response: {response[:200]}...")
        except Exception as e:
            print(f"⚠️  Model testing failed: {e}")
        
        return True
    else:
        print(f"\n❌ Training Failed: {result['error']}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Nemotron Training Pipeline for Quark Brain Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with small model and limited samples
  python scripts/run_nemotron_training.py --model-config small --splits chat code --max-samples 1000
  
  # Full training with medium model
  python scripts/run_nemotron_training.py --model-config medium --max-samples 10000
  
  # Large model training with specific splits
  python scripts/run_nemotron_training.py --model-config large --splits math stem tool_calling --max-samples 5000
        """
    )
    
    parser.add_argument(
        "--model-config", 
        choices=["small", "medium", "large"], 
        default=None,
        help="Model configuration based on available GPU memory"
    )
    
    parser.add_argument(
        "--splits", 
        nargs="+", 
        default=["chat", "code", "math", "stem", "tool_calling"],
        help="Dataset splits to use for training"
    )
    
    parser.add_argument(
        "--max-samples", 
        type=int, 
        default=5000,
        help="Maximum samples per split (default: 5000)"
    )
    
    parser.add_argument(
        "--output-dir", 
        default="./nemotron_fine_tuned",
        help="Output directory for fine-tuned model"
    )
    
    parser.add_argument(
        "--info", 
        action="store_true",
        help="Show dataset and model information"
    )
    
    parser.add_argument(
        "--check-gpu", 
        action="store_true",
        help="Check GPU memory and recommend model config"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Show information if requested
    if args.info:
        show_dataset_info()
        show_model_configs()
        return
    
    # Check GPU if requested
    if args.check_gpu:
        recommended_config = check_gpu_memory()
        print(f"💡 Recommended model config: {recommended_config}")
        return
    
    # Auto-detect model config if not specified
    if args.model_config is None:
        args.model_config = check_gpu_memory()
        print(f"🤖 Auto-selected model config: {args.model_config}")
    
    # Show initial information
    show_dataset_info()
    print(f"\n🎯 Training Configuration:")
    print(f"  Model Config: {args.model_config}")
    print(f"  Splits: {', '.join(args.splits)}")
    print(f"  Max Samples per Split: {args.max_samples:,}")
    print(f"  Estimated Total Samples: {len(args.splits) * args.max_samples:,}")
    
    # Confirm before starting
    response = input("\n🤔 Proceed with training? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Training cancelled.")
        return
    
    # Run training
    success = run_training(args)
    
    if success:
        print("\n✅ Nemotron training completed successfully!")
        print("🎉 Your brain simulation now has enhanced reasoning capabilities!")
    else:
        print("\n❌ Training failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
