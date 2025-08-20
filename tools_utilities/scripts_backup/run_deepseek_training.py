#!/usr/bin/env python3
"""
DeepSeek-R1 Training Runner Script
=================================

Quick script to run DeepSeek-R1 fine-tuning with various configurations.

Usage:
    python scripts/run_deepseek_training.py --model qwen-7b --epochs 3 --output ./my_model
    python scripts/run_deepseek_training.py --auto --dataset custom_brain_data.json
    python scripts/run_deepseek_training.py --evaluate ./my_fine_tuned_model
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.deepseek_r1_trainer import DeepSeekR1Trainer, DeepSeekConfig, create_deployment_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DeepSeek-R1 Training Runner")
    
    # Model selection
    parser.add_argument(
        "--model", 
        type=str, 
        choices=list(DeepSeekConfig.MODELS.keys()),
        help="Model variant to use (auto-selects if not specified)"
    )
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--output", type=str, default="./fine_tuned_deepseek_r1", help="Output directory")
    
    # Dataset options
    parser.add_argument("--dataset", type=str, help="Custom dataset JSON file")
    parser.add_argument("--auto", action="store_true", help="Auto-select model and run with defaults")
    
    # Modes
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--evaluate", type=str, help="Evaluate model at path")
    parser.add_argument("--test", action="store_true", help="Test model generation")
    parser.add_argument("--info", action="store_true", help="Show model information")
    
    # Hardware options
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--cache-dir", type=str, default="./model_cache", help="Model cache directory")
    
    return parser.parse_args()


def load_custom_dataset(dataset_path: str):
    """Load custom dataset from JSON file."""
    try:
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        # Validate dataset format
        required_keys = ["question", "reasoning", "answer"]
        for i, example in enumerate(dataset):
            for key in required_keys:
                if key not in example:
                    raise ValueError(f"Missing key '{key}' in example {i}")
        
        logger.info(f"📊 Loaded {len(dataset)} examples from {dataset_path}")
        return dataset
    
    except Exception as e:
        logger.error(f"❌ Error loading dataset: {e}")
        return None


def show_model_info():
    """Display available models and system information."""
    print("🧠 DeepSeek-R1 Model Information")
    print("=" * 60)
    
    # Available models
    DeepSeekConfig.print_models()
    
    # System info
    import torch
    print(f"🖥️  System Information:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        print(f"   GPU Memory: {gpu_memory} GB")
        
        # Recommend model
        recommended = DeepSeekConfig.get_recommended_model(gpu_memory)
        print(f"   Recommended model: {recommended}")
    else:
        print("   Running on CPU")
        print("   Recommended model: deepseek-r1-distill-qwen-1.5b")


def run_training(args):
    """Run model training."""
    logger.info("🚀 Starting DeepSeek-R1 training...")
    
    # Initialize trainer
    trainer = DeepSeekR1Trainer(
        model_key=args.model,
        device=args.device,
        cache_dir=args.cache_dir
    )
    
    # Prepare dataset
    if args.dataset:
        custom_examples = load_custom_dataset(args.dataset)
        if custom_examples is None:
            return False
        train_dataset = trainer.prepare_training_dataset(custom_examples)
    else:
        train_dataset = trainer.prepare_training_dataset()
    
    # Setup training
    hf_trainer, training_args = trainer.setup_fine_tuning(
        train_dataset,
        output_dir=args.output
    )
    
    # Update training arguments with CLI options
    training_args.num_train_epochs = args.epochs
    training_args.per_device_train_batch_size = args.batch_size
    training_args.learning_rate = args.learning_rate
    
    # Run training
    result = trainer.run_fine_tuning(hf_trainer)
    
    if result["success"]:
        logger.info(f"✅ Training completed successfully!")
        logger.info(f"📁 Model saved to: {result['model_path']}")
        
        # Create deployment config
        deployment_config = create_deployment_config(trainer, result['model_path'])
        config_file = os.path.join(result['model_path'], "deployment_config.json")
        
        with open(config_file, 'w') as f:
            json.dump(deployment_config, f, indent=2)
        
        logger.info(f"⚙️  Deployment config saved to: {config_file}")
        return True
    else:
        logger.error(f"❌ Training failed: {result['error']}")
        return False


def run_evaluation(model_path: str):
    """Evaluate a trained model."""
    logger.info(f"🧪 Evaluating model: {model_path}")
    
    # Initialize trainer (for comparison)
    trainer = DeepSeekR1Trainer()
    
    # Run evaluation
    result = trainer.evaluate_model(model_path)
    
    if result.get("success", False):
        logger.info("📊 Evaluation Results:")
        for i, eval_result in enumerate(result["evaluation_results"], 1):
            print(f"\n🔍 Test {i}: {eval_result['prompt']}")
            print(f"📝 Original: {eval_result['original_response'][:200]}...")
            print(f"🎯 Fine-tuned: {eval_result['fine_tuned_response'][:200]}...")
    else:
        logger.error(f"❌ Evaluation failed: {result.get('error', 'Unknown error')}")


def run_test(args):
    """Test model generation."""
    logger.info("🧪 Testing model generation...")
    
    # Initialize trainer
    trainer = DeepSeekR1Trainer(
        model_key=args.model,
        device=args.device,
        cache_dir=args.cache_dir
    )
    
    # Test prompts
    test_prompts = [
        "How does consciousness emerge in neural networks?",
        "What is the relationship between brain complexity and intelligence?",
        "Explain the role of attention in cognitive processing.",
        "How can we measure awareness in artificial systems?"
    ]
    
    print("\n🧠 Model Test Results:")
    print("=" * 60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}: {prompt}")
        print("-" * 40)
        
        response = trainer.generate_reasoning_response(
            prompt,
            max_length=512,
            temperature=0.6
        )
        
        print(f"💭 Response: {response}")


def main():
    """Main function."""
    args = parse_arguments()
    
    # Show model info
    if args.info:
        show_model_info()
        return
    
    # Auto mode
    if args.auto:
        args.train = True
        logger.info("🤖 Auto mode: Running training with default settings")
    
    # Training mode
    if args.train:
        success = run_training(args)
        if not success:
            sys.exit(1)
    
    # Evaluation mode
    elif args.evaluate:
        run_evaluation(args.evaluate)
    
    # Test mode
    elif args.test:
        run_test(args)
    
    # Default: show help
    else:
        logger.info("🔍 No mode specified. Use --help for options or --info for model information.")
        show_model_info()


if __name__ == "__main__":
    main()
