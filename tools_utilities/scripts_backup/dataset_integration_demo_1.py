#!/usr/bin/env python3
"""
Dataset Integration Demo for SmallMind

Demonstrates how to use the high-quality open LLM datasets
for training and fine-tuning your small_mind model.
"""

import os, sys
import logging
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from smallmind.models.dataset_integration import (
    DatasetIntegrator, get_dataset_integrator, 
    create_balanced_mixture, create_code_focused_mixture
)
from smallmind.models.training_pipeline import (
    SmallMindTrainer, TrainingConfig, get_trainer, quick_train
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_dataset_integration():
    """Demo the dataset integration capabilities"""
    print("🚀 SmallMind Dataset Integration Demo")
    print("=" * 50)
    
    # Initialize the dataset integrator
    integrator = get_dataset_integrator()
    
    # Show available mixtures
    print("\n📊 Available Training Mixtures:")
    mixtures = integrator.list_available_mixtures()
    for mixture in mixtures:
        info = integrator.get_mixture_info(mixture)
        if info:
            print(f"  • {mixture}: {info['datasets']}")
    
    # Show dataset information
    print("\n🔍 Dataset Information:")
    for dataset_name in ["fineweb", "stack_v2", "opencode_reasoning"]:
        info = integrator.get_dataset_info(dataset_name)
        if info:
            print(f"  • {info['name']}: {info['dataset_id']}")
            if info.get('available_configs'):
                print(f"    Configs: {info['available_configs'][:3]}...")
    
    return integrator

def demo_training_mixtures(integrator):
    """Demo creating and exploring training mixtures"""
    print("\n🎯 Training Mixture Demo:")
    print("-" * 30)
    
    # Create a balanced mixture
    print("Creating balanced training mixture...")
    balanced_mixture = integrator.create_training_mixture("balanced")
    
    # Sample some examples
    print("\nSampling examples from balanced mixture:")
    sample_count = 0
    for example in balanced_mixture:
        if sample_count >= 3:
            break
        
        # Find the text content
        text_content = None
        for key, value in example.items():
            if isinstance(value, str) and len(value) > 50:
                text_content = value
                break
        
        if text_content:
            print(f"\nExample {sample_count + 1}:")
            print(f"  Source: {example.get('source', 'unknown')}")
            print(f"  Text: {text_content[:200]}...")
        
        sample_count += 1
    
    return balanced_mixture

def demo_custom_mixture(integrator):
    """Demo creating a custom training mixture"""
    print("\n🔧 Custom Mixture Demo:")
    print("-" * 30)
    
    # Create a custom code-focused mixture
    custom_mixture = integrator.create_custom_mixture(
        name="my_code_mixture",
        datasets=["stack_v2", "opencode_reasoning", "fineweb"],
        weights=[0.5, 0.3, 0.2],
        seed=123
    )
    
    print(f"Created custom mixture: {custom_mixture.name}")
    print(f"Datasets: {[d.name for d in custom_mixture.datasets]}")
    print(f"Weights: {custom_mixture.interleave_weights}")
    
    return custom_mixture

def demo_training_pipeline():
    """Demo the training pipeline (without actually training)"""
    print("\n🏋️ Training Pipeline Demo:")
    print("-" * 30)
    
    # Initialize trainer
    trainer = get_trainer()
    
    # Show available mixtures
    print("Available training mixtures:")
    for mixture in trainer.list_available_mixtures():
        info = trainer.get_mixture_info(mixture)
        if info:
            print(f"  • {mixture}: {info['datasets']}")
    
    # Create a sample training config
    sample_config = TrainingConfig(
        model_name_or_path="./models/checkpoints/deepseek-v2",  # Example path
        output_dir="./trained_models/sample_run",
        mixture_name="balanced",
        max_steps=100,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4
    )
    
    # Validate the config
    print("\nValidating sample training config...")
    issues = trainer.validate_training_config(sample_config)
    if issues:
        print("Config issues found:")
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print("✅ Config validation passed!")
    
    return trainer, sample_config

def demo_quick_functions():
    """Demo the quick convenience functions"""
    print("\n⚡ Quick Functions Demo:")
    print("-" * 30)
    
    # Quick balanced mixture
    print("Creating balanced mixture with convenience function...")
    try:
        balanced = create_balanced_mixture()
        print("✅ Balanced mixture created successfully")
    except Exception as e:
        print(f"❌ Failed to create balanced mixture: {e}")
    
    # Quick code-focused mixture
    print("Creating code-focused mixture with convenience function...")
    try:
        code_focused = create_code_focused_mixture()
        print("✅ Code-focused mixture created successfully")
    except Exception as e:
        print(f"❌ Failed to create code-focused mixture: {e}")

def demo_data_exploration(integrator):
    """Demo exploring the actual data"""
    print("\n🔍 Data Exploration Demo:")
    print("-" * 30)
    
    # Explore FineWeb data
    print("Exploring FineWeb dataset...")
    try:
        fineweb = integrator.load_dataset(integrator.dataset_configs["fineweb"])
        
        # Sample a few examples
        sample_count = 0
        for example in fineweb:
            if sample_count >= 2:
                break
            
            # Find text content
            text_content = None
            for key, value in example.items():
                if isinstance(value, str) and len(value) > 100:
                    text_content = value
                    break
            
            if text_content:
                print(f"\nFineWeb Example {sample_count + 1}:")
                print(f"  Text: {text_content[:300]}...")
                print(f"  Length: {len(text_content)} characters")
            
            sample_count += 1
            
    except Exception as e:
        print(f"❌ Failed to explore FineWeb: {e}")
    
    # Explore The Stack v2 data
    print("\nExploring The Stack v2 dataset...")
    try:
        stack_v2 = integrator.load_dataset(integrator.dataset_configs["stack_v2"])
        
        # Sample a few examples
        sample_count = 0
        for example in stack_v2:
            if sample_count >= 2:
                break
            
            # Look for code content
            code_content = None
            for key, value in example.items():
                if isinstance(value, str) and any(keyword in key.lower() for keyword in ['code', 'content', 'text']):
                    if len(value) > 100:
                        code_content = value
                        break
            
            if code_content:
                print(f"\nStack v2 Example {sample_count + 1}:")
                print(f"  Content: {code_content[:300]}...")
                print(f"  Length: {len(code_content)} characters")
            
            sample_count += 1
            
    except Exception as e:
        print(f"❌ Failed to explore Stack v2: {e}")

def main():
    """Main demo function"""
    print("🎯 SmallMind Dataset Integration Demo")
    print("This demo showcases the integration of high-quality open LLM datasets")
    print("for enhanced natural language understanding while maintaining code capabilities.\n")
    
    try:
        # Demo 1: Basic dataset integration
        integrator = demo_dataset_integration()
        
        # Demo 2: Training mixtures
        balanced_mixture = demo_training_mixtures(integrator)
        
        # Demo 3: Custom mixtures
        custom_mixture = demo_custom_mixture(integrator)
        
        # Demo 4: Training pipeline
        trainer, sample_config = demo_training_pipeline()
        
        # Demo 5: Quick functions
        demo_quick_functions()
        
        # Demo 6: Data exploration
        demo_data_exploration(integrator)
        
        print("\n🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("1. Install required dependencies: pip install datasets transformers torch accelerate")
        print("2. Use the integrator to create training mixtures")
        print("3. Use the trainer to fine-tune your models")
        print("4. Customize mixtures for your specific use case")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("\nThis might be due to missing dependencies.")
        print("Install with: pip install datasets transformers torch accelerate")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
