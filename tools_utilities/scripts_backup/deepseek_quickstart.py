#!/usr/bin/env python3
"""
DeepSeek-R1 Quick Start Example
===============================

This example demonstrates how to:
1. Get started with DeepSeek-R1 models
2. Create and use datasets for fine-tuning
3. Integrate with your brain simulation framework

Run this example:
    python examples/deepseek_quickstart.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from development.src.core.deepseek_r1_trainer import DeepSeekR1Trainer, DeepSeekConfig
import json

def demo_model_selection():
    """Demonstrate automatic model selection."""
    print("🧠 DeepSeek-R1 Quick Start Demo")
    print("=" * 50)
    
    # Show available models
    print("\n📋 Available Models:")
    DeepSeekConfig.print_models()
    
    # Auto-select model
    print("\n🤖 Auto-selecting model for your hardware...")
    trainer = DeepSeekR1Trainer()
    print(f"✅ Selected: {trainer.model_name}")
    print(f"🔧 Device: {trainer.device}")
    
    return trainer

def demo_basic_generation(trainer):
    """Demonstrate basic text generation."""
    print("\n🧪 Testing Basic Generation")
    print("-" * 30)
    
    test_prompts = [
        "What is consciousness?",
        "How do neural networks learn?",
        "Explain brain plasticity in simple terms."
    ]
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: {prompt}")
        response = trainer.generate_reasoning_response(
            prompt, 
            max_length=300, 
            temperature=0.6
        )
        print(f"💭 Response: {response[:200]}...")

def demo_dataset_creation():
    """Demonstrate creating custom datasets."""
    print("\n📊 Creating Custom Dataset")
    print("-" * 30)
    
    # Create sample brain simulation dataset
    brain_simulation_dataset = [
        {
            "question": "How do you model neural spike trains?",
            "reasoning": "<think>\nModeling neural spike trains requires several considerations:\n\n1. Temporal Dynamics:\n- Spikes occur at discrete time points\n- Inter-spike intervals follow specific distributions\n- Refractory periods limit spike frequency\n\n2. Mathematical Models:\n- Poisson processes for random firing\n- Gamma processes for more realistic timing\n- Hawkes processes for spike history dependence\n\n3. Implementation:\n- Time-binned spike counts\n- Point process models\n- Continuous rate functions\n\n4. Validation:\n- Compare with experimental data\n- Check statistical properties\n- Verify biological plausibility\n</think>\n\nNeural spike trains can be modeled using point processes like Poisson or Hawkes processes, considering temporal dynamics, refractory periods, and spike history dependencies.",
            "answer": "Use point processes (Poisson, Hawkes) with temporal dynamics, refractory periods, and spike history dependencies for realistic neural spike train modeling."
        },
        {
            "question": "What are the key components of a brain simulation architecture?",
            "reasoning": "<think>\nA comprehensive brain simulation architecture needs several key components:\n\n1. Neural Models:\n- Individual neuron models (integrate-and-fire, Hodgkin-Huxley)\n- Population dynamics\n- Synaptic transmission models\n\n2. Connectivity:\n- Anatomical connectivity matrices\n- Synaptic weights and delays\n- Plasticity rules (STDP, homeostasis)\n\n3. Brain Regions:\n- Cortical areas (V1, PFC, etc.)\n- Subcortical structures (thalamus, hippocampus)\n- Inter-regional connections\n\n4. Simulation Engine:\n- Numerical integration methods\n- Parallel computing support\n- Real-time constraints\n\n5. Analysis Tools:\n- Signal processing\n- Connectivity analysis\n- Behavioral metrics\n</think>\n\nBrain simulation architecture requires neural models, connectivity matrices, brain region organization, simulation engines, and analysis tools for comprehensive modeling.",
            "answer": "Key components include neural models, connectivity matrices, brain region organization, simulation engines with parallel computing, and analysis tools for signal processing and behavioral metrics."
        }
    ]
    
    # Save dataset
    dataset_file = "brain_simulation_dataset.json"
    with open(dataset_file, 'w') as f:
        json.dump(brain_simulation_dataset, f, indent=2)
    
    print(f"💾 Created dataset: {dataset_file}")
    print(f"📊 Examples: {len(brain_simulation_dataset)}")
    
    # Show first example
    print(f"\n📝 Sample Example:")
    example = brain_simulation_dataset[0]
    print(f"Question: {example['question']}")
    print(f"Reasoning: {example['reasoning'][:150]}...")
    
    return dataset_file

def demo_training_setup(trainer, dataset_file):
    """Demonstrate training setup."""
    print("\n🔧 Training Setup Demo")
    print("-" * 30)
    
    # Load custom dataset
    with open(dataset_file, 'r') as f:
        custom_examples = json.load(f)
    
    # Prepare training dataset
    train_dataset = trainer.prepare_training_dataset(custom_examples)
    print(f"📊 Training dataset size: {len(train_dataset)}")
    
    # Setup training configuration
    hf_trainer, training_args = trainer.setup_fine_tuning(
        train_dataset,
        output_dir="./demo_fine_tuned_model"
    )
    
    print(f"⚙️  Training Configuration:")
    print(f"   - Epochs: {training_args.num_train_epochs}")
    print(f"   - Batch size: {training_args.per_device_train_batch_size}")
    print(f"   - Learning rate: {training_args.learning_rate}")
    print(f"   - Output: {training_args.output_dir}")
    
    # Estimate training time
    estimated_steps = len(train_dataset) // (
        training_args.per_device_train_batch_size * 
        training_args.gradient_accumulation_steps
    )
    total_steps = estimated_steps * training_args.num_train_epochs
    print(f"   - Estimated steps: {total_steps}")
    
    print("\n💡 To run training:")
    print("   result = trainer.run_fine_tuning(hf_trainer)")
    
    return hf_trainer

def demo_integration():
    """Demonstrate integration with brain simulation."""
    print("\n🧠 Brain Simulation Integration")
    print("-" * 30)
    
    # Example brain state data
    brain_state = {
        "timestamp": "2025-01-20T10:30:00",
        "neural_activity": {
            "cortex": 0.75,
            "hippocampus": 0.68,
            "thalamus": 0.82
        },
        "consciousness_metrics": {
            "awareness_score": 0.71,
            "integration_index": 0.66,
            "global_workspace_activity": 0.79
        },
        "memory_systems": {
            "working_memory_load": 0.58,
            "episodic_memory_activity": 0.42,
            "semantic_memory_access": 0.73
        }
    }
    
    print(f"🧪 Example Brain State Analysis:")
    print(json.dumps(brain_state, indent=2))
    
    # Show how to analyze with DeepSeek-R1
    print(f"\n🔍 Analysis Prompt:")
    analysis_prompt = f"""
    Analyze this brain simulation state data:
    
    {json.dumps(brain_state, indent=2)}
    
    Provide insights on:
    1. Overall neural activity patterns
    2. Consciousness indicators
    3. Memory system performance
    4. Recommendations for optimization
    """
    
    print(analysis_prompt[:200] + "...")
    print("\n💡 Use trainer.generate_reasoning_response(analysis_prompt) to get AI analysis")

def demo_deployment_options():
    """Show deployment options."""
    print("\n🚀 Deployment Options")
    print("-" * 30)
    
    print("1. 🖥️  Local Development:")
    print("   trainer = DeepSeekR1Trainer()")
    print("   response = trainer.generate_reasoning_response(prompt)")
    
    print("\n2. 🌐 API Server (vLLM):")
    print("   vllm serve ./my_fine_tuned_model --port 8000")
    print("   curl -X POST http://localhost:8000/v1/completions \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"model\": \"./my_fine_tuned_model\", \"prompt\": \"...\", \"max_tokens\": 512}'")
    
    print("\n3. ☁️  Cloud Deployment:")
    print("   # Use SkyPilot for cloud deployment")
    print("   sky launch skypilot_deepseek.yaml")
    
    print("\n4. 🧠 Brain Simulation Integration:")
    print("   from development.src.core.deepseek_r1_trainer import BrainSimulationIntegration")
    print("   integration = BrainSimulationIntegration(trainer)")
    print("   insights = integration.generate_simulation_insights(metrics)")

def main():
    """Main demo function."""
    print("🚀 Starting DeepSeek-R1 Quick Start Demo...")
    
    try:
        # 1. Model selection
        trainer = demo_model_selection()
        
        # 2. Basic generation
        demo_basic_generation(trainer)
        
        # 3. Dataset creation
        dataset_file = demo_dataset_creation()
        
        # 4. Training setup
        demo_training_setup(trainer, dataset_file)
        
        # 5. Integration demo
        demo_integration()
        
        # 6. Deployment options
        demo_deployment_options()
        
        print("\n✅ Demo completed successfully!")
        print("\n🎯 Next Steps:")
        print("1. Run actual training: python scripts/run_deepseek_training.py --auto")
        print("2. Check the full notebook: notebooks/training/deepseek_r1_training.ipynb")
        print("3. Read the guide: DEEPSEEK_R1_SETUP_GUIDE.md")
        print("4. Integrate with your brain simulation")
        
        # Clean up demo files
        if os.path.exists(dataset_file):
            os.remove(dataset_file)
            print(f"\n🧹 Cleaned up demo file: {dataset_file}")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        print("💡 Make sure you have the required dependencies installed:")
        print("   pip install torch transformers datasets accelerate")

if __name__ == "__main__":
    main()
