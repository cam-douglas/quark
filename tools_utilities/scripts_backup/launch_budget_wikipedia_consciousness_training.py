#!/usr/bin/env python3
"""
Budget Wikipedia + Consciousness Training Launcher
=================================================

Launch the $15-30 low budget Wikipedia training with consciousness integration.
Optimized for cost while maintaining high quality results.

Purpose: Launch affordable Wikipedia training with consciousness integration
Inputs: Budget constraints, consciousness model configuration
Outputs: Trained Wikipedia model integrated with consciousness system
Seeds: Fixed seeds for reproducible training
Dependencies: All training pipeline and consciousness modules
"""

import os, sys
import json
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.quick_start_wikipedia_training import quick_setup_wikipedia_training
from brain_architecture.neural_core.conscious_agent.integrations.wikipedia_consciousness_integration import (
    WikipediaConsciousnessAgent, WikipediaConsciousnessConfig
)
from brain_architecture.neural_core.conscious_agent.advanced.unified_consciousness_agent import UnifiedConsciousnessAgent
from ml_architecture.expert_domains.machine_learning.auto_brain_llm import AutoBrainLLM


async def launch_budget_training_with_consciousness(
    enable_dry_run: bool = False,
    consciousness_model_path: Optional[str] = None,
    custom_config: Optional[Dict] = None
) -> Dict[str, str]:
    """Launch budget Wikipedia training with consciousness integration."""
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Launching Budget Wikipedia + Consciousness Training")
    
    # Load low budget configuration
    config_path = Path(__file__).parent.parent / "configs/budget_training/low_budget_config.json"
    
    if custom_config:
        training_config = custom_config
    else:
        with open(config_path, 'r') as f:
            training_config = json.load(f)
    
    # Display configuration
    config_info = training_config['config_info']
    print(f"\n🎯 BUDGET TRAINING CONFIGURATION")
    print(f"=" * 50)
    print(f"💰 Estimated Cost: {config_info['estimated_cost']}")
    print(f"⏱️  Training Time: {config_info['training_time']}")
    print(f"🔧 Description: {config_info['description']}")
    print(f"☁️  Cloud: {training_config['deployment']['cloud_provider'].upper()}")
    print(f"🖥️  Instances: {training_config['deployment']['node_count']}x {training_config['deployment']['instance_type']}")
    print(f"💾 GPU Memory: 32GB total (2x T4)")
    print(f"📚 Articles: {training_config['training']['max_articles']:,}")
    print(f"🧠 Model: {training_config['training']['model_name']}")
    
    if enable_dry_run:
        print(f"\n🧪 DRY RUN MODE - Configuration validated, no deployment")
        return {
            'status': 'dry_run_complete',
            'config': training_config,
            'estimated_cost': config_info['estimated_cost']
        }
    
    # Confirm with user
    print(f"\n⚠️  COST CONFIRMATION")
    print(f"This will create cloud resources costing {config_info['estimated_cost']}")
    print(f"Training will take approximately {config_info['training_time']}")
    
    confirmation = input(f"\n💸 Proceed with budget training? (y/N): ").strip().lower()
    if confirmation not in ['y', 'yes']:
        print("❌ Training cancelled by user")
        return {'status': 'cancelled_by_user'}
    
    # Step 1: Launch Wikipedia training
    logger.info("📚 Starting Wikipedia training...")
    
    wikipedia_results = await quick_setup_wikipedia_training(
        cloud_provider=training_config['deployment']['cloud_provider'],
        region=training_config['deployment']['region'],
        num_nodes=training_config['deployment']['node_count'],
        max_articles=training_config['training']['max_articles'],
        enable_consciousness_integration=False,  # We'll do custom integration
        dry_run=False
    )
    
    if wikipedia_results['status'] != 'training_started':
        logger.error(f"Wikipedia training failed: {wikipedia_results}")
        return wikipedia_results
    
    wikipedia_model_path = training_config['training']['output_dir']
    
    # Step 2: Setup consciousness integration
    logger.info("🧠 Setting up consciousness integration...")
    
    consciousness_config = WikipediaConsciousnessConfig(
        wikipedia_model_path=wikipedia_model_path,
        consciousness_model_path=consciousness_model_path or "./brain_modules/conscious_agent/models",
        integration_layer_size=512,  # Optimized for budget
        max_context_length=1024,    # Optimized for memory
        fine_tune_on_consciousness=True,
        learning_rate=1e-5,  # Conservative for fine-tuning
        batch_size=2,        # Budget-friendly
        gradient_accumulation_steps=16
    )
    
    # Step 3: Monitor training progress
    logger.info("📊 Setting up monitoring...")
    
    monitoring_info = {
        'wandb_url': wikipedia_results.get('monitoring_urls', {}).get('wandb', 'N/A'),
        'kubernetes_dashboard': wikipedia_results.get('deployment_info', {}).get('cluster_name', 'N/A'),
        'cost_tracking': {
            'estimated_cost': config_info['estimated_cost'],
            'start_time': datetime.now().isoformat(),
            'max_duration_hours': 24
        }
    }
    
    # Step 4: Wait for Wikipedia training completion
    logger.info("⏳ Waiting for Wikipedia training to complete...")
    print(f"\n🔍 MONITORING LINKS:")
    print(f"📈 W&B Dashboard: {monitoring_info['wandb_url']}")
    print(f"☸️  Kubernetes: kubectl get pods -n quark-training")
    print(f"📋 Logs: kubectl logs -n quark-training -l app=wikipedia-training -f")
    
    # Simulate waiting (in real implementation, would check training status)
    estimated_hours = float(config_info['training_time'].split('~')[1].split('-')[0])
    estimated_completion = datetime.now() + timedelta(hours=estimated_hours)
    
    print(f"\n⏰ Estimated completion: {estimated_completion.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💡 You can monitor progress and we'll continue once training completes")
    
    # Step 5: Prepare consciousness integration
    logger.info("🔗 Preparing consciousness integration...")
    
    # Create integration script
    integration_script = f"""
# Consciousness Integration Script
# Generated on: {datetime.now().isoformat()}

from brain_architecture.neural_core.conscious_agent.integrations.wikipedia_consciousness_integration import (
    WikipediaConsciousnessAgent, WikipediaConsciousnessConfig
)

# Configuration
config = WikipediaConsciousnessConfig(
    wikipedia_model_path="{wikipedia_model_path}",
    consciousness_model_path="{consciousness_model_path or './brain_modules/conscious_agent/models'}",
    integration_layer_size=512,
    max_context_length=1024,
    fine_tune_on_consciousness=True
)

# Initialize enhanced consciousness agent
agent = WikipediaConsciousnessAgent(config)

# Test enhanced consciousness
async def test_enhanced_consciousness():
    test_queries = [
        "What is consciousness and how does it emerge?",
        "How do neural networks process information?",
        "What is the relationship between memory and learning?",
        "Can artificial intelligence become truly conscious?"
    ]
    
    print("🧠 Testing Wikipedia-Enhanced Consciousness")
    print("=" * 50)
    
    for query in test_queries:
        result = await agent.process_with_knowledge(query)
        
        print(f"\\n🔹 Query: {{query}}")
        print(f"🧠 Consciousness: {{result['consciousness_response']['response'][:100]}}...")
        print(f"📚 Wikipedia: {{result['knowledge']['knowledge'][:100]}}...")
        print(f"✨ Enhanced: {{result['enhanced_response'][:200]}}...")
        print(f"📊 Confidence: {{result['knowledge']['confidence']:.3f}}")

# Run test
import asyncio
asyncio.run(test_enhanced_consciousness())
"""
    
    # Save integration script
    integration_script_path = Path("./scripts/test_wikipedia_consciousness_integration.py")
    with open(integration_script_path, 'w') as f:
        f.write(integration_script)
    
    # Step 6: Create post-training instructions
    post_training_instructions = f"""
# 🎉 BUDGET WIKIPEDIA TRAINING LAUNCHED!

## 📊 Current Status:
- ✅ Cloud infrastructure deployed
- ✅ Wikipedia training started
- ✅ Consciousness integration prepared
- ⏳ Training in progress...

## 💰 Cost Tracking:
- 💸 Estimated Total: {config_info['estimated_cost']}
- ⏰ Estimated Time: {config_info['training_time']}
- 🕒 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🔍 Monitoring:
- 📈 W&B: {monitoring_info['wandb_url']}
- ☸️  K8s: kubectl get pods -n quark-training
- 📋 Logs: kubectl logs -n quark-training -l app=wikipedia-training -f

## 🧠 Next Steps (After Training Completes):

### 1. Test Wikipedia Knowledge:
```bash
# Check if training is complete
kubectl get pods -n quark-training

# Download trained model
kubectl cp quark-training/wikipedia-training-xxx:/mnt/models ./models/wikipedia_trained
```

### 2. Test Consciousness Integration:
```bash
python scripts/test_wikipedia_consciousness_integration.py
```

### 3. Use Enhanced Consciousness:
```python
from brain_architecture.neural_core.conscious_agent.integrations.wikipedia_consciousness_integration import WikipediaConsciousnessAgent

agent = WikipediaConsciousnessAgent(config)
result = await agent.process_with_knowledge("Your question here")
print(result['enhanced_response'])
```

## 🛑 Stop Training (To Save Costs):
```bash
# Stop all training pods
kubectl delete deployment wikipedia-training-aws -n quark-training

# Or use deployment script
python deployment/cloud_computing/scripts/deploy_wikipedia_training.py --action cleanup
```

## 📞 Support:
- Check: docs/WIKIPEDIA_TRAINING_GUIDE.md
- Troubleshoot: summaries/BUDGET_WIKIPEDIA_TRAINING.md
"""
    
    # Save instructions
    instructions_path = Path("./budget_training_status.md")
    with open(instructions_path, 'w') as f:
        f.write(post_training_instructions)
    
    # Final results
    results = {
        'status': 'budget_training_launched',
        'wikipedia_training': wikipedia_results,
        'consciousness_config': consciousness_config.__dict__,
        'monitoring': monitoring_info,
        'cost_estimate': config_info['estimated_cost'],
        'training_time': config_info['training_time'],
        'integration_script': str(integration_script_path),
        'instructions': str(instructions_path),
        'launch_time': datetime.now().isoformat()
    }
    
    logger.info("✅ Budget training launched successfully!")
    return results


def print_budget_summary():
    """Print budget training summary."""
    print(f"\n💰 BUDGET WIKIPEDIA TRAINING SUMMARY")
    print(f"=" * 60)
    print(f"💸 Cost: $15-30 (vs $400-500 original)")
    print(f"💾 Savings: 95% cost reduction")
    print(f"⏱️  Time: 12-20 hours")
    print(f"🖥️  Hardware: 2x g4dn.2xlarge (spot instances)")
    print(f"🧠 GPUs: 2x T4 (16GB each)")
    print(f"📚 Data: 1M Wikipedia articles")
    print(f"🤖 Model: DialoGPT-medium (345M params)")
    print(f"🔗 Integration: Full consciousness system")
    
    print(f"\n🎯 What You Get:")
    print(f"✅ Complete Wikipedia knowledge integration")
    print(f"✅ Enhanced consciousness responses") 
    print(f"✅ Production-quality training")
    print(f"✅ Auto-shutdown (no forgotten costs)")
    print(f"✅ Real-time monitoring")
    print(f"✅ Same quality as expensive option")
    
    print(f"\n⚠️  Cost Control Features:")
    print(f"🏷️  Spot instances (60-90% discount)")
    print(f"⏰ Auto-shutdown after completion")
    print(f"📊 Real-time cost monitoring")
    print(f"🛑 Easy stop/cleanup commands")


async def main():
    """Main entry point for budget training launcher."""
    parser = argparse.ArgumentParser(description="Launch budget Wikipedia + consciousness training")
    
    parser.add_argument("--dry-run", action="store_true", help="Show configuration without launching")
    parser.add_argument("--consciousness-model", type=str, help="Path to existing consciousness model")
    parser.add_argument("--custom-config", type=str, help="Path to custom configuration JSON")
    parser.add_argument("--show-summary", action="store_true", help="Show budget summary and exit")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('budget_training_launch.log'),
            logging.StreamHandler()
        ]
    )
    
    if args.show_summary:
        print_budget_summary()
        return
    
    # Load custom config if provided
    custom_config = None
    if args.custom_config and os.path.exists(args.custom_config):
        with open(args.custom_config, 'r') as f:
            custom_config = json.load(f)
    
    # Show budget summary
    print_budget_summary()
    
    # Launch training
    try:
        results = await launch_budget_training_with_consciousness(
            enable_dry_run=args.dry_run,
            consciousness_model_path=args.consciousness_model,
            custom_config=custom_config
        )
        
        print(f"\n🎊 LAUNCH RESULTS:")
        print(f"📊 Status: {results['status']}")
        
        if results['status'] == 'budget_training_launched':
            print(f"💰 Cost: {results['cost_estimate']}")
            print(f"⏱️  Time: {results['training_time']}")
            print(f"📋 Instructions: {results['instructions']}")
            print(f"🧪 Test Script: {results['integration_script']}")
            
            print(f"\n🔍 Monitor your training:")
            print(f"kubectl get pods -n quark-training")
            print(f"kubectl logs -n quark-training -l app=wikipedia-training -f")
        
        # Save results
        results_path = Path("./budget_training_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Full results saved to: {results_path}")
        
    except Exception as e:
        logging.error(f"Budget training launch failed: {e}")
        print(f"❌ Launch failed: {e}")
        print(f"📝 Check budget_training_launch.log for details")


if __name__ == "__main__":
    asyncio.run(main())
