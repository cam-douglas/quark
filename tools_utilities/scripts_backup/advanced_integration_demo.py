#!/usr/bin/env python3
"""
🚀 Small-Mind Advanced Integration Demo

Comprehensive demonstration of all integrated optimization and simulation features:
- Advanced model optimization with vLLM, FlashAttention, and quantization
- Neural network optimization with SAM (Sharpness-Aware Minimization)
- Brain development simulation with physics engines
- Comprehensive pipeline execution
"""

import os, sys
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add the smallmind package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_advanced_optimization():
    """Demonstrate advanced model optimization"""
    logger.info("🚀 Demo: Advanced Model Optimization")
    
    try:
        from smallmind.ml_optimization.advanced_optimizer import SmallMindAdvancedOptimizer
        
        # Create optimizer with custom config
        config = {
            "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
            "quantization": "awq",
            "flash_attention": True,
            "torch_compile": True,
            "memory_budget_gb": 8
        }
        
        optimizer = SmallMindAdvancedOptimizer(config)
        
        # Run optimization
        logger.info("Running advanced optimization...")
        results = optimizer.optimize_model()
        
        # Display results
        print("\n" + "="*60)
        print("ADVANCED OPTIMIZATION RESULTS")
        print("="*60)
        print(f"Model: {results['model_name']}")
        print(f"Optimizations Applied: {len(results['optimizations'])}")
        
        for opt_name, opt_details in results['optimizations'].items():
            print(f"\n🔧 {opt_name.upper()}:")
            print(f"  Status: {opt_details['status']}")
            if 'method' in opt_details:
                print(f"  Method: {opt_details['method']}")
            print(f"  Benefits: {', '.join(opt_details['benefits'])}")
        
        # Generate and display report
        print("\n" + optimizer.generate_optimization_report())
        
        return True
        
    except ImportError as e:
        logger.warning(f"Advanced optimization not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Advanced optimization failed: {e}")
        return False

def demo_neural_optimization():
    """Demonstrate neural network optimization with SAM"""
    logger.info("🧠 Demo: Neural Network Optimization with SAM")
    
    try:
        from smallmind.ml_optimization.neural_optimizer import (
            SmallMindNeuralOptimizer, 
            SmallMindNeuralModel,
            create_synthetic_brain_data
        )
        
        # Create synthetic brain development data
        logger.info("Creating synthetic brain development data...")
        X, y = create_synthetic_brain_data(
            input_size=100,
            output_size=50,
            num_samples=2000
        )
        
        # Create model
        model = SmallMindNeuralModel(
            input_size=100,
            hidden_size=256,
            output_size=50,
            dropout=0.1,
            num_layers=3
        )
        
        # Create optimizer with SAM
        optimizer = SmallMindNeuralOptimizer(
            model,
            learning_rate=0.001,
            rho=0.05  # SAM radius
        )
        
        # Train model
        logger.info("Training neural network with SAM...")
        history = optimizer.train_with_sam(
            X, y,
            epochs=50,
            batch_size=32,
            validation_split=0.2
        )
        
        # Evaluate model
        logger.info("Evaluating model performance...")
        metrics = optimizer.evaluate_model(X, y)
        
        print("\n" + "="*60)
        print("NEURAL NETWORK OPTIMIZATION RESULTS")
        print("="*60)
        print(f"Training Epochs: {len(history['train_loss'])}")
        print(f"Final Training Loss: {history['train_loss'][-1]:.6f}")
        print(f"Final Validation Loss: {history['val_loss'][-1]:.6f}")
        print(f"\nModel Performance:")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  R²: {metrics['r2']:.6f}")
        
        # Hyperparameter optimization
        logger.info("Running hyperparameter optimization...")
        opt_results = optimizer.optimize_hyperparameters(X, y, n_trials=20)
        
        if opt_results:
            print(f"\nHyperparameter Optimization Results:")
            print(f"  Best Parameters: {opt_results['best_params']}")
            print(f"  Best Validation Loss: {opt_results['best_value']:.6f}")
        
        # Model compression
        logger.info("Demonstrating model compression...")
        compressed_model = optimizer.compress_model(
            compression_ratio=0.7,
            method='pruning'
        )
        
        # Save model
        model_file = "demo_neural_model.pth"
        optimizer.save_model(model_file)
        logger.info(f"Model saved to {model_file}")
        
        return True
        
    except ImportError as e:
        logger.warning(f"Neural optimization not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Neural optimization failed: {e}")
        return False

def demo_brain_simulation():
    """Demonstrate brain development simulation"""
    logger.info("🧠 Demo: Brain Development Simulation")
    
    try:
        from smallmind.simulation.simulation_runner import (
            BrainDevelopmentSimulation,
            create_simulation_config
        )
        
        # Create simulation configuration
        config = create_simulation_config(
            simulation_type="brain_development",
            duration_steps=500,  # Shorter for demo
            physics_engine="pybullet"
        )
        
        # Override for demo
        config.update({
            "output_directory": "demo_simulation_results",
            "visualization": True,
            "save_results": True
        })
        
        # Create simulation
        simulation = BrainDevelopmentSimulation(config)
        
        # Run simulation
        logger.info("Running brain development simulation...")
        results = simulation.run_simulation()
        
        if results:
            print("\n" + "="*60)
            print("BRAIN DEVELOPMENT SIMULATION RESULTS")
            print("="*60)
            print(f"Total Steps: {results['total_steps']}")
            print(f"Execution Time: {results['execution_time']:.2f} seconds")
            print(f"Final Development: {results['final_development']['overall_development']:.3f}")
            
            # Show brain region development
            brain_regions = results['final_development']['brain_regions']
            print(f"\nBrain Region Development:")
            for region, progress in brain_regions.items():
                print(f"  {region.replace('_', ' ').title()}: {progress:.3f}")
            
            print(f"\nResults saved to: {config['output_directory']}")
        
        return True
        
    except ImportError as e:
        logger.warning(f"Brain simulation not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Brain simulation failed: {e}")
        return False

def demo_comprehensive_pipeline():
    """Demonstrate comprehensive pipeline execution"""
    logger.info("🚀 Demo: Comprehensive Small-Mind Pipeline")
    
    try:
        from smallmind.cli.advanced_cli import SmallMindAdvancedCLI
        
        # Create CLI instance
        cli = SmallMindAdvancedCLI()
        
        # Create mock args for pipeline
        class MockArgs:
            def __init__(self):
                self.config = None
        
        args = MockArgs()
        
        # Run pipeline
        logger.info("Executing comprehensive pipeline...")
        cli.run_comprehensive_pipeline(args)
        
        logger.info("✅ Comprehensive pipeline completed")
        return True
        
    except ImportError as e:
        logger.warning(f"Comprehensive pipeline not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Comprehensive pipeline failed: {e}")
        return False

def demo_configuration_management():
    """Demonstrate configuration management"""
    logger.info("📝 Demo: Configuration Management")
    
    try:
        from smallmind.cli.advanced_cli import SmallMindAdvancedCLI
        
        # Create CLI instance
        cli = SmallMindAdvancedCLI()
        
        # Create mock args
        class MockArgs:
            pass
        
        args = MockArgs()
        
        # Create configuration templates
        logger.info("Creating configuration templates...")
        cli.create_config_template(args)
        
        # Load and display one of the configs
        try:
            with open("advanced_optimization_config.json", 'r') as f:
                config = json.load(f)
            
            print("\n" + "="*60)
            print("CONFIGURATION TEMPLATE EXAMPLE")
            print("="*60)
            print("Advanced Optimization Configuration:")
            for key, value in config.items():
                print(f"  {key}: {value}")
        
        except FileNotFoundError:
            logger.warning("Configuration file not found")
        
        return True
        
    except ImportError as e:
        logger.warning(f"Configuration management not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Configuration management failed: {e}")
        return False

def main():
    """Main demo execution"""
    logger.info("🚀 Starting Small-Mind Advanced Integration Demo")
    logger.info("=" * 60)
    
    # Track demo results
    demo_results = {}
    
    # Demo 1: Advanced Optimization
    logger.info("\n" + "="*60)
    demo_results['advanced_optimization'] = demo_advanced_optimization()
    
    # Demo 2: Neural Optimization
    logger.info("\n" + "="*60)
    demo_results['neural_optimization'] = demo_neural_optimization()
    
    # Demo 3: Brain Simulation
    logger.info("\n" + "="*60)
    demo_results['brain_simulation'] = demo_brain_simulation()
    
    # Demo 4: Configuration Management
    logger.info("\n" + "="*60)
    demo_results['configuration_management'] = demo_configuration_management()
    
    # Demo 5: Comprehensive Pipeline (optional)
    logger.info("\n" + "="*60)
    demo_results['comprehensive_pipeline'] = demo_comprehensive_pipeline()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("DEMO SUMMARY")
    logger.info("="*60)
    
    successful_demos = sum(demo_results.values())
    total_demos = len(demo_results)
    
    for demo_name, success in demo_results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{demo_name.replace('_', ' ').title()}: {status}")
    
    logger.info(f"\nOverall Success Rate: {successful_demos}/{total_demos} ({successful_demos/total_demos*100:.1f}%)")
    
    if successful_demos == total_demos:
        logger.info("🎉 All demos completed successfully!")
    else:
        logger.info("⚠️  Some demos failed. Check the logs for details.")
    
    logger.info("\n📚 Next Steps:")
    logger.info("1. Review the generated configuration files")
    logger.info("2. Run individual demos with custom parameters")
    logger.info("3. Explore the CLI interface: python -m smallmind.cli.advanced_cli --help")
    logger.info("4. Check the documentation for advanced usage")

if __name__ == "__main__":
    main()
