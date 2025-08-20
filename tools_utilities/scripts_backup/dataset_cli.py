#!/usr/bin/env python3
"""
Dataset Integration CLI for SmallMind

Command-line interface for managing and exploring high-quality open LLM datasets.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from smallmind.models import (
        get_dataset_integrator, get_trainer,
        DATASET_INTEGRATION_AVAILABLE, TRAINING_PIPELINE_AVAILABLE
    )
except ImportError as e:
    print(f"Error importing SmallMind modules: {e}")
    print("Make sure you have installed the required dependencies:")
    print("pip install datasets transformers torch accelerate")
    sys.exit(1)

def list_mixtures(args):
    """List available training mixtures"""
    if not DATASET_INTEGRATION_AVAILABLE:
        print("❌ Dataset integration not available")
        return
    
    integrator = get_dataset_integrator()
    mixtures = integrator.list_available_mixtures()
    
    print("📊 Available Training Mixtures:")
    print("=" * 40)
    
    for mixture in mixtures:
        info = integrator.get_mixture_info(mixture)
        if info:
            print(f"\n🎯 {mixture.upper()}")
            print(f"   Datasets: {', '.join(info['datasets'])}")
            print(f"   Weights: {info['weights']}")
            print(f"   Seed: {info['seed']}")
            if info.get('max_samples'):
                print(f"   Max Samples: {info['max_samples']:,}")

def list_datasets(args):
    """List available datasets"""
    if not DATASET_INTEGRATION_AVAILABLE:
        print("❌ Dataset integration not available")
        return
    
    integrator = get_dataset_integrator()
    
    print("🔍 Available Datasets:")
    print("=" * 40)
    
    for name, config in integrator.dataset_configs.items():
        print(f"\n📁 {name}")
        print(f"   Name: {config.name}")
        print(f"   ID: {config.dataset_id}")
        print(f"   Split: {config.split}")
        if config.subset:
            print(f"   Subset: {config.subset}")
        print(f"   Weight: {config.weight}")
        print(f"   Streaming: {config.streaming}")
        if config.max_samples:
            print(f"   Max Samples: {config.max_samples:,}")

def explore_mixture(args):
    """Explore a specific training mixture"""
    if not DATASET_INTEGRATION_AVAILABLE:
        print("❌ Dataset integration not available")
        return
    
    integrator = get_dataset_integrator()
    
    if args.mixture not in integrator.training_mixtures:
        print(f"❌ Unknown mixture: {args.mixture}")
        print(f"Available mixtures: {', '.join(integrator.list_available_mixtures())}")
        return
    
    print(f"🔍 Exploring mixture: {args.mixture}")
    print("=" * 50)
    
    try:
        mixture = integrator.create_training_mixture(args.mixture)
        
        # Sample examples
        sample_count = 0
        for example in mixture:
            if sample_count >= args.samples:
                break
            
            print(f"\n📝 Example {sample_count + 1}:")
            
            # Find text content
            text_content = None
            for key, value in example.items():
                if isinstance(value, str) and len(value) > 50:
                    text_content = value
                    break
            
            if text_content:
                print(f"   Text: {text_content[:args.max_length]}...")
                print(f"   Length: {len(text_content):,} characters")
            
            # Show other fields
            for key, value in example.items():
                if key != 'text' and isinstance(value, str) and len(value) < 100:
                    print(f"   {key}: {value}")
            
            sample_count += 1
            
    except Exception as e:
        print(f"❌ Failed to explore mixture: {e}")

def explore_dataset(args):
    """Explore a specific dataset"""
    if not DATASET_INTEGRATION_AVAILABLE:
        print("❌ Dataset integration not available")
        return
    
    integrator = get_dataset_integrator()
    
    if args.dataset not in integrator.dataset_configs:
        print(f"❌ Unknown dataset: {args.dataset}")
        print(f"Available datasets: {', '.join(integrator.dataset_configs.keys())}")
        return
    
    print(f"🔍 Exploring dataset: {args.dataset}")
    print("=" * 50)
    
    try:
        dataset = integrator.load_dataset(integrator.dataset_configs[args.dataset])
        
        # Sample examples
        sample_count = 0
        for example in dataset:
            if sample_count >= args.samples:
                break
            
            print(f"\n📝 Example {sample_count + 1}:")
            
            # Find text content
            text_content = None
            for key, value in example.items():
                if isinstance(value, str) and len(value) > 50:
                    text_content = value
                    break
            
            if text_content:
                print(f"   Text: {text_content[:args.max_length]}...")
                print(f"   Length: {len(text_content):,} characters")
            
            # Show other fields
            for key, value in example.items():
                if key != 'text' and isinstance(value, str) and len(value) < 100:
                    print(f"   {key}: {value}")
            
            sample_count += 1
            
    except Exception as e:
        print(f"❌ Failed to explore dataset: {e}")

def create_custom_mixture(args):
    """Create a custom training mixture"""
    if not DATASET_INTEGRATION_AVAILABLE:
        print("❌ Dataset integration not available")
        return
    
    integrator = get_dataset_integrator()
    
    # Parse datasets and weights
    datasets = args.datasets.split(',')
    weights = [float(w) for w in args.weights.split(',')]
    
    if len(datasets) != len(weights):
        print("❌ Number of datasets must match number of weights")
        return
    
    try:
        mixture = integrator.create_custom_mixture(
            name=args.name,
            datasets=datasets,
            weights=weights,
            seed=args.seed
        )
        
        print(f"✅ Created custom mixture: {mixture.name}")
        print(f"   Datasets: {[d.name for d in mixture.datasets]}")
        print(f"   Weights: {mixture.interleave_weights}")
        print(f"   Seed: {mixture.seed}")
        
    except Exception as e:
        print(f"❌ Failed to create custom mixture: {e}")

def validate_training(args):
    """Validate a training configuration"""
    if not TRAINING_PIPELINE_AVAILABLE:
        print("❌ Training pipeline not available")
        return
    
    trainer = get_trainer()
    
    # Create a sample config for validation
    from smallmind.models import TrainingConfig
    
    config = TrainingConfig(
        model_name_or_path=args.model_path,
        output_dir=args.output_dir,
        mixture_name=args.mixture,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        learning_rate=args.learning_rate
    )
    
    print(f"🔍 Validating training configuration...")
    print("=" * 50)
    
    issues = trainer.validate_training_config(config)
    
    if issues:
        print("❌ Configuration issues found:")
        for issue in issues:
            print(f"   ⚠️  {issue}")
    else:
        print("✅ Configuration validation passed!")
    
    # Show mixture info
    mixture_info = trainer.get_mixture_info(args.mixture)
    if mixture_info:
        print(f"\n📊 Mixture Information:")
        print(f"   Name: {mixture_info['name']}")
        print(f"   Datasets: {', '.join(mixture_info['datasets'])}")
        print(f"   Weights: {mixture_info['weights']}")

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="SmallMind Dataset Integration CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available mixtures
  python dataset_cli.py list-mixtures
  
  # Explore a specific mixture
  python dataset_cli.py explore-mixture --mixture balanced --samples 3
  
  # Explore a specific dataset
  python dataset_cli.py explore-dataset --dataset fineweb --samples 2
  
  # Create custom mixture
  python dataset_cli.py create-mixture --name my_mix --datasets fineweb,stack_v2 --weights 0.6,0.4
  
  # Validate training config
  python dataset_cli.py validate-training --model-path ./models/model --mixture balanced
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List mixtures command
    list_mixtures_parser = subparsers.add_parser('list-mixtures', help='List available training mixtures')
    list_mixtures_parser.set_defaults(func=list_mixtures)
    
    # List datasets command
    list_datasets_parser = subparsers.add_parser('list-datasets', help='List available datasets')
    list_datasets_parser.set_defaults(func=list_datasets)
    
    # Explore mixture command
    explore_mixture_parser = subparsers.add_parser('explore-mixture', help='Explore a specific training mixture')
    explore_mixture_parser.add_argument('--mixture', required=True, help='Mixture name to explore')
    explore_mixture_parser.add_argument('--samples', type=int, default=3, help='Number of samples to show')
    explore_mixture_parser.add_argument('--max-length', type=int, default=200, help='Maximum text length to show')
    explore_mixture_parser.set_defaults(func=explore_mixture)
    
    # Explore dataset command
    explore_dataset_parser = subparsers.add_parser('explore-dataset', help='Explore a specific dataset')
    explore_dataset_parser.add_argument('--dataset', required=True, help='Dataset name to explore')
    explore_dataset_parser.add_argument('--samples', type=int, default=3, help='Number of samples to show')
    explore_dataset_parser.add_argument('--max-length', type=int, default=200, help='Maximum text length to show')
    explore_dataset_parser.set_defaults(func=explore_dataset)
    
    # Create custom mixture command
    create_mixture_parser = subparsers.add_parser('create-mixture', help='Create a custom training mixture')
    create_mixture_parser.add_argument('--name', required=True, help='Name for the custom mixture')
    create_mixture_parser.add_argument('--datasets', required=True, help='Comma-separated list of dataset names')
    create_mixture_parser.add_argument('--weights', required=True, help='Comma-separated list of weights')
    create_mixture_parser.add_argument('--seed', type=int, default=42, help='Random seed for interleaving')
    create_mixture_parser.set_defaults(func=create_custom_mixture)
    
    # Validate training command
    validate_parser = subparsers.add_parser('validate-training', help='Validate a training configuration')
    validate_parser.add_argument('--model-path', required=True, help='Path to the model')
    validate_parser.add_argument('--output-dir', default='./output', help='Output directory for training')
    validate_parser.add_argument('--mixture', default='balanced', help='Training mixture to use')
    validate_parser.add_argument('--max-steps', type=int, default=1000, help='Maximum training steps')
    validate_parser.add_argument('--batch-size', type=int, default=4, help='Batch size per device')
    validate_parser.add_argument('--grad-accumulation', type=int, default=4, help='Gradient accumulation steps')
    validate_parser.add_argument('--learning-rate', type=float, default=5e-5, help='Learning rate')
    validate_parser.set_defaults(func=validate_training)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute the command
    try:
        args.func(args)
    except Exception as e:
        print(f"❌ Command failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
