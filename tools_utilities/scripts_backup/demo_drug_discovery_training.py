#!/usr/bin/env python3
"""
Drug Discovery Training Demo for Quark Brain Simulation Framework
Demonstrates training on Kaggle drug discovery dataset with brain integration
"""

import os, sys
import json
import time
from datetime import datetime

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'database', 'training_scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'database'))

def demo_drug_discovery_training():
    """Demonstrate drug discovery training capabilities"""
    
    print("🧬 QUARK DRUG DISCOVERY TRAINING DEMO")
    print("=" * 50)
    print()
    
    try:
        # Import the trainer
        from drug_discovery_trainer import DrugDiscoveryTrainer
        
        print("✅ Successfully imported DrugDiscoveryTrainer")
        print()
        
        # Initialize trainer
        print("🔧 Initializing Drug Discovery Trainer...")
        trainer = DrugDiscoveryTrainer(database_path="database")
        print("✅ Trainer initialized successfully")
        print()
        
        # Print configuration summary
        print("📋 TRAINING CONFIGURATION:")
        print("-" * 30)
        for key, value in trainer.config.items():
            print(f"  {key}: {value}")
        print()
        
        # Print brain features info
        print("🧠 BRAIN INTEGRATION FEATURES:")
        print("-" * 30)
        if trainer.brain_features:
            for region, features in trainer.brain_features.items():
                print(f"  {region}: {len(features)} feature types")
        print()
        
        # Generate summary
        print(trainer.generate_training_summary())
        print()
        
        # Ask user if they want to run full training
        print("🤔 Would you like to run the full training pipeline?")
        print("This will:")
        print("  1. Download Kaggle dataset (or create dummy data)")
        print("  2. Train molecular property predictor")
        print("  3. Train brain-drug integration model")
        print("  4. Generate visualizations and reports")
        print()
        
        response = input("Run full training? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            print("\n🚀 Starting full training pipeline...")
            start_time = time.time()
            
            # Run training
            results = trainer.run_full_training_pipeline()
            
            end_time = time.time()
            training_duration = end_time - start_time
            
            if results:
                print(f"\n🎉 Training completed successfully!")
                print(f"⏱️  Total training time: {training_duration:.2f} seconds")
                print()
                
                # Print results summary
                print("📊 TRAINING RESULTS SUMMARY:")
                print("-" * 30)
                
                if results.get('molecular_model'):
                    mol_eval = results['molecular_model']['evaluation_results']
                    print(f"  Molecular Model Accuracy: {mol_eval['accuracy']:.4f}")
                
                if results.get('integration_model'):
                    int_eval = results['integration_model']['evaluation_results']
                    print(f"  Integration Model Accuracy: {int_eval['accuracy']:.4f}")
                
                print(f"  Dataset Size: {results['dataset_info']['samples']} samples")
                print(f"  Feature Dimensions: {results['dataset_info']['features']}")
                print(f"  Brain Feature Dimensions: {results['brain_features_shape']}")
                print()
                
                # Show file locations
                print("📁 OUTPUT FILES:")
                print("-" * 30)
                print(f"  Results: database/drug_discovery_training_results.json")
                print(f"  Models: database/best_molecular_predictor.pth")
                print(f"  Models: database/best_brain_drug_integration.pth")
                print(f"  Plots: database/*_training_results.png")
                print()
                
            else:
                print("\n❌ Training failed. Check logs for details.")
                
        else:
            print("\n👍 Demo completed without full training.")
            print("To run training later, use:")
            print("  python demo_drug_discovery_training.py")
        
        print("\n🔗 NEXT STEPS:")
        print("-" * 30)
        print("1. Explore the generated visualizations")
        print("2. Examine the training results JSON")
        print("3. Modify configuration for different experiments")
        print("4. Integrate with other Quark brain components")
        print("5. Try different datasets or model architectures")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all requirements are installed:")
        print("  pip install -r database/training_scripts/drug_discovery_requirements.txt")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("Check the logs and requirements for troubleshooting.")

def quick_test():
    """Quick test to verify system readiness"""
    
    print("🔍 QUICK SYSTEM TEST")
    print("=" * 20)
    print()
    
    # Test imports
    test_results = {
        "torch": False,
        "sklearn": False,
        "pandas": False,
        "numpy": False,
        "matplotlib": False,
        "kaggle": False
    }
    
    # Test core libraries
    try:
        import torch
        test_results["torch"] = True
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not available")
    
    try:
        import sklearn
        test_results["sklearn"] = True
        print(f"✅ Scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("❌ Scikit-learn not available")
    
    try:
        import pandas
        test_results["pandas"] = True
        print(f"✅ Pandas: {pandas.__version__}")
    except ImportError:
        print("❌ Pandas not available")
    
    try:
        import numpy
        test_results["numpy"] = True
        print(f"✅ NumPy: {numpy.__version__}")
    except ImportError:
        print("❌ NumPy not available")
    
    try:
        import matplotlib
        test_results["matplotlib"] = True
        print(f"✅ Matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("❌ Matplotlib not available")
    
    try:
        import kaggle
        test_results["kaggle"] = True
        print(f"✅ Kaggle API available")
    except ImportError:
        print("❌ Kaggle API not available")
    
    # GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  GPU: Not available (CPU training will be slower)")
    except:
        print("❌ Cannot check GPU availability")
    
    print()
    
    # Overall readiness
    required_libs = ["torch", "sklearn", "pandas", "numpy", "matplotlib"]
    all_required = all(test_results[lib] for lib in required_libs)
    
    if all_required:
        print("🎉 System ready for drug discovery training!")
        return True
    else:
        print("⚠️  Some required libraries are missing.")
        print("Install requirements:")
        print("  pip install -r database/training_scripts/drug_discovery_requirements.txt")
        return False

def main():
    """Main demo function"""
    
    print("🧬 QUARK DRUG DISCOVERY TRAINING SYSTEM")
    print("=" * 60)
    print(f"Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Quick system test
    if quick_test():
        print()
        print("🚀 Proceeding with full demo...")
        print()
        demo_drug_discovery_training()
    else:
        print()
        print("⚠️  Please install missing requirements first.")
    
    print()
    print("=" * 60)
    print("Demo completed.")

if __name__ == "__main__":
    main()
