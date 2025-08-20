#!/usr/bin/env python3
"""
BALSA Training Demo Script
Demonstrates the neuroimaging training pipeline for consciousness agent

Purpose: Show BALSA training capabilities and usage examples
Inputs: Demo datasets, training configurations
Outputs: Training demonstrations, sample results
Seeds: Demo session IDs, sample neuroimaging data
Dependencies: balsa_training, matplotlib, pandas
"""

import os, sys
import json
import time
from datetime import datetime

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def demo_balsa_training():
    """Demonstrate BALSA training pipeline"""
    print("🧠🔬 BALSA Training Pipeline Demo")
    print("=" * 50)
    print("Demonstrating neuroimaging data training for consciousness agent")
    print("Dataset: Washington University BALSA (https://balsa.wustl.edu/)")
    print()
    
    try:
        # Import BALSA trainer
        from balsa_training import BALSATrainer
        
        # Initialize trainer
        print("📊 Initializing BALSA Trainer...")
        trainer = BALSATrainer(database_path="database")
        print("✅ Trainer initialized successfully")
        print()
        
        # Show BALSA information
        print("📋 BALSA Dataset Information:")
        print(f"  Name: {trainer.balsa_info['name']}")
        print(f"  Institution: {trainer.balsa_info['institution']}")
        print(f"  URL: {trainer.balsa_info['url']}")
        print(f"  Available Datasets: {len(trainer.balsa_info['datasets'])}")
        print()
        
        # Show training configuration
        print("⚙️ Training Configuration:")
        config = trainer.training_config
        print(f"  Model Type: {config['model_type']}")
        print(f"  Learning Rate: {config['learning_rate']}")
        print(f"  Batch Size: {config['batch_size']}")
        print(f"  Epochs: {config['epochs']}")
        print(f"  Validation Split: {config['validation_split']}")
        print()
        
        # Show processing pipeline
        print("🔧 Processing Pipeline:")
        pipeline = trainer.processing_pipeline
        print(f"  Structural Processing Steps: {len(pipeline['structural_processing'])}")
        print(f"  Functional Processing Steps: {len(pipeline['functional_processing'])}")
        print(f"  Connectivity Analysis Steps: {len(pipeline['connectivity_analysis'])}")
        print()
        
        # Demo individual processing steps
        print("🔄 Demonstrating Individual Processing Steps...")
        print()
        
        # 1. Fetch datasets
        print("1️⃣ Fetching BALSA Datasets...")
        datasets = trainer.fetch_balsa_datasets()
        print(f"   Found {len(datasets)} datasets")
        for name, info in datasets.items():
            print(f"   - {name}: {info['subjects']} subjects, {info['access_level']}")
        print()
        
        # 2. Process structural data
        print("2️⃣ Processing Structural MRI Data...")
        dataset_name = "HCP-Young Adult 2025"
        structural_data = trainer.process_structural_data(dataset_name)
        print(f"   Processed {len(structural_data['brain_regions'])} brain regions")
        print(f"   Generated {len(structural_data['outputs'])} output types")
        print()
        
        # 3. Process functional data
        print("3️⃣ Processing Functional MRI Data...")
        functional_data = trainer.process_functional_data(dataset_name)
        print(f"   Processed {len(functional_data['functional_networks'])} functional networks")
        print(f"   Computed {len(functional_data['connectivity_metrics'])} connectivity metrics")
        print()
        
        # 4. Analyze connectivity
        print("4️⃣ Analyzing Connectivity Patterns...")
        connectivity_data = trainer.analyze_connectivity_patterns(dataset_name)
        print(f"   Analyzed {len(connectivity_data['connectivity_measures'])} connectivity types")
        print(f"   Computed {len(connectivity_data['network_metrics'])} network metrics")
        print()
        
        # 5. Integrate brain atlas
        print("5️⃣ Integrating Brain Atlas...")
        atlas_data = trainer.integrate_brain_atlas(dataset_name)
        print(f"   Integrated {len(atlas_data['atlas_systems'])} atlas systems")
        print(f"   Mapped {len(atlas_data['region_mapping'])} region types")
        print()
        
        # 6. Train consciousness model
        print("6️⃣ Training Consciousness Model...")
        processed_data = {
            f"{dataset_name}_structural": structural_data,
            f"{dataset_name}_functional": functional_data,
            f"{dataset_name}_connectivity": connectivity_data,
            f"{dataset_name}_atlas": atlas_data
        }
        
        training_results = trainer.train_consciousness_model(processed_data)
        print(f"   Model Type: {training_results['model_type']}")
        print(f"   Training Data: {training_results['training_data']['total_samples']} samples")
        
        # Show performance metrics
        metrics = training_results['performance_metrics']
        print(f"   Training Accuracy: {metrics['training_accuracy']:.3f}")
        print(f"   Validation Accuracy: {metrics['validation_accuracy']:.3f}")
        print(f"   Consciousness Correlation: {metrics['consciousness_correlation']:.3f}")
        print()
        
        # 7. Update brain regions and extract knowledge
        print("7️⃣ Updating Brain Regions and Extracting Knowledge...")
        brain_regions_updated = trainer.update_brain_regions(processed_data)
        knowledge_extracted = trainer.extract_knowledge(processed_data)
        
        print(f"   Brain Regions Updated: {brain_regions_updated}")
        print(f"   Knowledge Points Extracted: {knowledge_extracted}")
        print()
        
        # 8. Run complete pipeline
        print("8️⃣ Running Complete Training Pipeline...")
        print("   This will process all datasets and train the complete model...")
        
        # For demo purposes, we'll simulate the pipeline run
        print("   ⏳ Simulating pipeline execution...")
        time.sleep(2)  # Simulate processing time
        
        # Update training session
        trainer.training_session.update({
            "datasets_processed": list(datasets.keys()),
            "knowledge_extracted": knowledge_extracted,
            "brain_regions_updated": brain_regions_updated,
            "training_iterations": 1,
            "model_performance": training_results,
            "neuroimaging_metrics": {
                "total_datasets": len(datasets),
                "processed_modalities": len(processed_data),
                "connectivity_analyses": len([k for k in processed_data.keys() if "connectivity" in k]),
                "atlas_integrations": len([k for k in processed_data.keys() if "atlas" in k])
            }
        })
        
        print("   ✅ Pipeline completed successfully!")
        print()
        
        # Show final results
        print("📊 Final Training Results:")
        summary = trainer.get_training_summary()
        print(f"   Session ID: {summary['session_id']}")
        print(f"   Datasets Processed: {len(summary['datasets_processed'])}")
        print(f"   Knowledge Extracted: {summary['knowledge_extracted']}")
        print(f"   Brain Regions Updated: {summary['brain_regions_updated']}")
        print()
        
        # Show neuroimaging metrics
        if "neuroimaging_metrics" in summary:
            metrics = summary["neuroimaging_metrics"]
            print("🔬 Neuroimaging Metrics:")
            print(f"   Total Datasets: {metrics['total_datasets']}")
            print(f"   Processed Modalities: {metrics['processed_modalities']}")
            print(f"   Connectivity Analyses: {metrics['connectivity_analyses']}")
            print(f"   Atlas Integrations: {metrics['atlas_integrations']}")
            print()
        
        # Show model performance
        if "model_performance" in summary and "performance_metrics" in summary["model_performance"]:
            perf_metrics = summary["model_performance"]["performance_metrics"]
            print("🎯 Model Performance:")
            print(f"   Training Accuracy: {perf_metrics['training_accuracy']:.3f}")
            print(f"   Validation Accuracy: {perf_metrics['validation_accuracy']:.3f}")
            print(f"   Consciousness Correlation: {perf_metrics['consciousness_correlation']:.3f}")
            print()
        
        # Save demo results
        print("💾 Saving Demo Results...")
        demo_results = {
            "demo_timestamp": datetime.now().isoformat(),
            "demo_type": "balsa_training_pipeline",
            "summary": summary,
            "sample_data": {
                "structural_sample": list(structural_data.keys())[:3],
                "functional_sample": list(functional_data.keys())[:3],
                "connectivity_sample": list(connectivity_data.keys())[:3],
                "atlas_sample": list(atlas_data.keys())[:3]
            }
        }
        
        demo_file = os.path.join("database", "balsa_outputs", f"demo_results_{int(time.time())}.json")
        os.makedirs(os.path.dirname(demo_file), exist_ok=True)
        
        with open(demo_file, 'w') as f:
            json.dump(demo_results, f, indent=2)
        
        print(f"   ✅ Demo results saved to: {demo_file}")
        print()
        
        # Final summary
        print("🎉 BALSA Training Demo Completed Successfully!")
        print("=" * 50)
        print("The consciousness agent has been trained on:")
        print("  • Structural MRI data from HCP datasets")
        print("  • Functional MRI connectivity patterns")
        print("  • Brain atlas integration")
        print("  • Consciousness correlation mapping")
        print()
        print("Next steps:")
        print("  1. Review training results in database/balsa_outputs/")
        print("  2. Integrate trained model with consciousness agent")
        print("  3. Run full training pipeline on complete datasets")
        print("  4. Validate model performance on test data")
        print()
        print("For more information, see BALSA_TRAINING_README.md")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure the BALSA training module is available")
        return False
        
    except Exception as e:
        print(f"❌ Demo Error: {e}")
        print("Check the error logs for more details")
        return False

def demo_individual_components():
    """Demonstrate individual BALSA training components"""
    print("🔧 Individual Component Demo")
    print("=" * 30)
    
    try:
        from balsa_training import BALSATrainer
        
        trainer = BALSATrainer(database_path="database")
        
        # Demo brain region mapping
        print("🧠 Brain Region Mapping Demo:")
        regions = ["prefrontal_cortex", "temporal_lobe", "parietal_lobe"]
        print(f"   Sample regions: {', '.join(regions)}")
        
        # Demo functional networks
        print("🔄 Functional Networks Demo:")
        networks = ["default_mode_network", "salience_network", "executive_control"]
        print(f"   Sample networks: {', '.join(networks)}")
        
        # Demo connectivity analysis
        print("🔗 Connectivity Analysis Demo:")
        connectivity_types = ["structural", "functional", "effective"]
        print(f"   Connectivity types: {', '.join(connectivity_types)}")
        
        print("✅ Component demo completed")
        
    except Exception as e:
        print(f"❌ Component demo error: {e}")

def main():
    """Main demo function"""
    print("🚀 Starting BALSA Training Demo...")
    print()
    
    # Run main demo
    success = demo_balsa_training()
    
    if success:
        print()
        print("🔧 Running individual component demo...")
        demo_individual_components()
    
    print()
    print("Demo completed!")

if __name__ == "__main__":
    main()
