#!/usr/bin/env python3
"""
Demo script for the Unified Super Mind

This demonstrates how to:
1. Create and configure the unified model
2. Train it on sample data
3. Generate responses
4. Show the integrated capabilities
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.smallmind.unified.super_mind import UnifiedSuperMind, SuperMindConfig
from src.smallmind.unified.trainer import SuperMindTrainer
from src.smallmind.unified.dataset import create_sample_dataset

def main():
    """Main demo function."""
    print("🧠 Unified Super Mind - Complete Integration Demo")
    print("=" * 60)
    print("This demo shows how all Small-Mind components work together")
    print("in one unified, trainable model.")
    print("=" * 60)
    
    # Configuration
    config = SuperMindConfig(
        base_model="microsoft/DialoGPT-medium",
        num_experts=4,
        learning_rate=1e-4,
        batch_size=2,
        max_steps=100,
        hidden_size=256,  # Smaller for demo
        num_layers=4
    )
    
    print(f"\n🔧 Configuration:")
    print(f"   • Base model: {config.base_model}")
    print(f"   • Number of experts: {config.num_experts}")
    print(f"   • Hidden size: {config.hidden_size}")
    print(f"   • Learning rate: {config.learning_rate}")
    print(f"   • Max training steps: {config.max_steps}")
    
    # Create model
    print(f"\n🚀 Creating Unified Super Mind...")
    try:
        model = UnifiedSuperMind(config)
        print("✅ Model created successfully!")
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return
    
    # Show model architecture
    print(f"\n🏗️  Model Architecture:")
    print(f"   • Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   • MOE experts: {len(model.experts)}")
    print(f"   • Brain development stages: {len(model.brain_development_modules)}")
    print(f"   • Device: {model.device}")
    
    # Create sample dataset
    print(f"\n📚 Creating sample dataset...")
    try:
        dataset = create_sample_dataset()
        print(f"✅ Dataset created with {len(dataset)} examples")
    except Exception as e:
        print(f"❌ Failed to create dataset: {e}")
        return
    
    # Create trainer
    print(f"\n🎯 Creating trainer...")
    try:
        trainer = SuperMindTrainer(model, config)
        print("✅ Trainer created successfully!")
    except Exception as e:
        print(f"❌ Failed to create trainer: {e}")
        return
    
    # Test generation before training
    print(f"\n🧪 Testing generation before training...")
    try:
        prompt = "The Unified Super Mind can"
        response = model.generate_response(prompt, max_length=20)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
    except Exception as e:
        print(f"⚠️  Generation failed: {e}")
    
    # Train the model
    print(f"\n🚀 Starting training...")
    try:
        trainer.train(dataset, save_dir=None)
        print("✅ Training completed!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Test generation after training
    print(f"\n🧪 Testing generation after training...")
    try:
        prompt = "Curiosity-driven learning"
        response = model.generate_response(prompt, max_length=30)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        
        prompt = "Neuroscience-inspired"
        response = model.generate_response(prompt, max_length=30)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
    except Exception as e:
        print(f"⚠️  Generation failed: {e}")
    
    # Show learning progress
    print(f"\n📊 Learning Progress:")
    print(f"   • Training steps completed: {model.training_step}")
    print(f"   • Brain development stage: {model.brain_development_stage}")
    print(f"   • Curiosity patterns recorded: {len(model.curiosity_patterns)}")
    
    if model.curiosity_patterns:
        recent_curiosity = sum(model.curiosity_patterns[-10:]) / min(10, len(model.curiosity_patterns))
        print(f"   • Recent curiosity level: {recent_curiosity:.3f}")
    
    print(f"\n🎉 Demo completed!")
    print(f"💡 The Unified Super Mind has successfully:")
    print(f"   • Combined MOE architecture with multiple experts")
    print(f"   • Integrated child-like learning mechanisms")
    print(f"   • Applied neuroscience-inspired processing")
    print(f"   • Enabled continuous learning and adaptation")
    print(f"   • Learned from training data")
    
    print(f"\n🔧 Next steps:")
    print(f"   • Use your own training data")
    print(f"   • Adjust configuration parameters")
    print(f"   • Train for more steps")
    print(f"   • Save and load checkpoints")
    print(f"   • Deploy for inference")
    
    return model, trainer

if __name__ == "__main__":
    model, trainer = main()
