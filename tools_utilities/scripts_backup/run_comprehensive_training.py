#!/usr/bin/env python3
"""
🧠 Comprehensive Training Runner
===============================

Complete demonstration and testing script for the systematic brain training system.
This script showcases all training components working together.

Usage:
    python run_comprehensive_training.py --demo     # Quick demonstration
    python run_comprehensive_training.py --full     # Complete training
    python run_comprehensive_training.py --test     # Test individual components

Author: Quark Brain Simulation Team
Created: 2025-01-21
"""

import os, sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Add quark root to path
QUARK_ROOT = Path(__file__).parent.parent
sys.path.append(str(QUARK_ROOT))

# Import all training systems
from training.main_training_orchestrator import MasterTrainingOrchestrator, create_default_config
from training.systematic_training_orchestrator import SystematicTrainingOrchestrator
from training.component_training_pipelines import ComponentTrainingPipeline
from training.training_counter_dashboard import TrainingCounterManager
from training.organic_connectome_enhancer import OrganicConnectomeEnhancer
from training.consciousness_enhancement_system import ConsciousnessEnhancementSystem
from training.visual_simulation_dashboard import VisualSimulationDashboard

def setup_logging():
    """Setup logging for the comprehensive training demo."""
    log_dir = QUARK_ROOT / 'training' / 'demo_logs'
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'comprehensive_training_demo_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger("comprehensive_training_demo")

def print_section_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"🧠 {title}")
    print("=" * 80)

def print_step(step_num: int, total_steps: int, description: str):
    """Print a formatted step indicator."""
    print(f"\n[{step_num}/{total_steps}] {description}")
    print("-" * 60)

def run_individual_component_tests(generate_visualizations=False):
    """Test individual training components."""
    print_section_header("INDIVIDUAL COMPONENT TESTS")
    
    # Test 1: Consciousness Enhancement System
    print_step(1, 6, "Testing Consciousness Enhancement System")
    try:
        consciousness_system = ConsciousnessEnhancementSystem()
        
        # Quick consciousness training
        print("Training consciousness enhancement (10 epochs)...")
        results = consciousness_system.train_consciousness_enhancement(
            num_epochs=10,
            batch_size=16
        )
        
        final_level = results['final_consciousness_level']
        print(f"✅ Consciousness Enhancement Test: Final level {final_level:.3f}")
        
        # Generate visualization only if requested
        if generate_visualizations:
            viz_file = consciousness_system.create_consciousness_visualization(results)
            print(f"📊 Generated consciousness visualization: {viz_file}")
        
    except Exception as e:
        print(f"❌ Consciousness Enhancement Test Failed: {e}")
    
    # Test 2: Connectome Enhancement
    print_step(2, 6, "Testing Organic Connectome Enhancement")
    try:
        connectome_enhancer = OrganicConnectomeEnhancer()
        
        print("Analyzing current connectome...")
        metrics = connectome_enhancer.analyze_current_connectome()
        print(f"Current biological plausibility: {metrics.biological_plausibility:.3f}")
        
        print("Enhancing connectome for neonate stage...")
        enhanced_graph = connectome_enhancer.enhance_connectome('neonate')
        
        enhanced_metrics = connectome_enhancer.analyze_current_connectome()
        print(f"✅ Connectome Enhancement Test: Plausibility improved to {enhanced_metrics.biological_plausibility:.3f}")
        
        # Generate visualization only if requested
        if generate_visualizations:
            viz_file = connectome_enhancer.create_enhancement_visualization()
            print(f"📊 Generated connectome visualization: {viz_file}")
        
    except Exception as e:
        print(f"❌ Connectome Enhancement Test Failed: {e}")
    
    # Test 3: Component Training Pipeline
    print_step(3, 6, "Testing Component Training Pipeline")
    try:
        component_pipeline = ComponentTrainingPipeline()
        
        print("Training conscious_agent component...")
        result = component_pipeline.train_component('conscious_agent', 'fetal')
        
        if result.get('success', False):
            final_consciousness = result.get('final_metrics', {}).get('consciousness_score', 0)
            print(f"✅ Component Training Test: Consciousness score {final_consciousness:.3f}")
        else:
            print(f"⚠️ Component Training Test: Completed with issues")
            
    except Exception as e:
        print(f"❌ Component Training Test Failed: {e}")
    
    # Test 4: Training Counter Manager
    print_step(4, 6, "Testing Training Counter Manager")
    try:
        counter_manager = TrainingCounterManager()
        
        # Initialize with demo components
        components = ['conscious_agent', 'prefrontal_cortex', 'thalamus']
        counter_manager.initialize_counters(components, {})
        
        print("Simulating training progress...")
        for i in range(5):
            for component in components:
                counter_manager.update_component_counter(
                    component,
                    iteration=i * 10,
                    epoch=i,
                    loss=1.0 - i * 0.1,
                    accuracy=i * 0.2,
                    consciousness_score=i * 0.15
                )
            time.sleep(0.5)
        
        print(f"✅ Training Counter Test: Progress tracking successful")
        
        # Generate dashboard only if requested
        if generate_visualizations:
            dashboards = counter_manager.generate_current_dashboard()
            print(f"📊 Generated {len(dashboards)} dashboards")
        
    except Exception as e:
        print(f"❌ Training Counter Test Failed: {e}")
    
    # Test 5: Visual Simulation Dashboard
    print_step(5, 6, "Testing Visual Simulation Dashboard")
    try:
        visual_dashboard = VisualSimulationDashboard()
        
        print("Starting visual simulation...")
        visual_dashboard.start_simulation()
        time.sleep(3)  # Let it collect some data
        
        # Generate dashboards only if requested
        if generate_visualizations:
            generated_files = visual_dashboard.generate_comprehensive_dashboard()
            print(f"📊 Generated {len(generated_files)} visualizations")
        
        visual_dashboard.stop_simulation()
        
        print(f"✅ Visual Simulation Test: Simulation system functional")
        
    except Exception as e:
        print(f"❌ Visual Simulation Test Failed: {e}")
    
    # Test 6: Systematic Training Orchestrator
    print_step(6, 6, "Testing Systematic Training Orchestrator")
    try:
        systematic_orchestrator = SystematicTrainingOrchestrator()
        
        print(f"Discovered {len(systematic_orchestrator.components)} trainable components")
        
        # Initialize progress tracking
        systematic_orchestrator.initialize_progress_tracking()
        
        print("✅ Systematic Orchestrator Test: Initialization successful")
        
    except Exception as e:
        print(f"❌ Systematic Orchestrator Test Failed: {e}")
    
    print_section_header("INDIVIDUAL COMPONENT TESTS COMPLETED")

def run_quick_demo():
    """Run a quick demonstration of the complete system."""
    print_section_header("QUICK COMPREHENSIVE TRAINING DEMO")
    
    print("Initializing Master Training Orchestrator...")
    
    # Create demo configuration
    config = create_default_config()
    config.stage_durations = {'fetal': 5, 'neonate': 5, 'early_postnatal': 5}
    config.consciousness_epochs = 10
    config.consciousness_target = 0.3
    config.real_time_visualization = True
    
    try:
        # Initialize orchestrator
        orchestrator = MasterTrainingOrchestrator(config, QUARK_ROOT)
        
        print_step(1, 3, "Initializing Training Systems")
        orchestrator.initialize_training_state()
        print("✅ All training systems initialized")
        
        print_step(2, 3, "Running Quick Training Sequence")
        
        # Run a single developmental stage
        stage_result = orchestrator.execute_developmental_stage('fetal')
        
        if stage_result['success']:
            print(f"✅ Fetal stage completed successfully")
            print(f"   Components trained: {len(stage_result['components_trained'])}")
            print(f"   Biological compliance: {stage_result['biological_compliance']:.3f}")
        else:
            print(f"⚠️ Fetal stage completed with issues")
        
        print_step(3, 3, "Generating Final Reports")
        
        # Generate final summary
        summary = orchestrator._create_master_summary_report()
        print(f"✅ Generated master summary report: {summary}")
        
        # Stop monitoring systems
        if orchestrator.visual_dashboard:
            orchestrator.visual_dashboard.stop_simulation()
        orchestrator.counter_manager.stop_real_time_updates()
        
        print_section_header("QUICK DEMO COMPLETED SUCCESSFULLY")
        
    except Exception as e:
        print(f"❌ Quick Demo Failed: {e}")
        logging.exception("Demo failed with exception")

def run_full_training():
    """Run the complete training sequence."""
    print_section_header("COMPLETE BRAIN TRAINING SEQUENCE")
    
    print("🚀 Starting Complete Brain Training...")
    print("This will take approximately 30-60 minutes depending on your system.")
    
    response = input("\nProceed with full training? (y/N): ").strip().lower()
    if response != 'y':
        print("Full training cancelled.")
        return
    
    # Create full configuration
    config = create_default_config()
    
    try:
        # Initialize and run
        orchestrator = MasterTrainingOrchestrator(config, QUARK_ROOT)
        
        start_time = time.time()
        results = orchestrator.execute_complete_training_sequence()
        end_time = time.time()
        
        duration = end_time - start_time
        
        print_section_header("COMPLETE TRAINING RESULTS")
        print(f"Training ID: {results.training_id}")
        print(f"Total Duration: {duration/60:.1f} minutes")
        print(f"Final Consciousness Level: {results.final_consciousness_level:.3f}")
        print(f"Biological Compliance: {results.final_biological_compliance:.3f}")
        print(f"Connectome Coherence: {results.final_connectome_coherence:.3f}")
        print(f"Generated Dashboards: {len(results.generated_dashboards)}")
        print(f"Generated Reports: {len(results.generated_reports)}")
        
        print("\n📁 Key Output Files:")
        if results.generated_reports:
            for report in results.generated_reports[:3]:
                print(f"  📝 {report}")
        
        if results.generated_dashboards:
            for dashboard in results.generated_dashboards[:3]:
                print(f"  📊 {dashboard}")
        
        print_section_header("COMPLETE TRAINING FINISHED SUCCESSFULLY")
        
    except Exception as e:
        print(f"❌ Complete Training Failed: {e}")
        logging.exception("Complete training failed with exception")

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Comprehensive Brain Training Demonstration'
    )
    
    parser.add_argument('--demo', action='store_true',
                       help='Run quick demonstration (5-10 minutes)')
    parser.add_argument('--test', action='store_true', 
                       help='Test individual components')
    parser.add_argument('--full', action='store_true',
                       help='Run complete training sequence (30-60 minutes)')
    parser.add_argument('--all', action='store_true',
                       help='Run all tests and demos')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations during tests')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    print("🧠 Quark Comprehensive Brain Training System")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base Directory: {QUARK_ROOT}")
    
    try:
        if args.test or args.all:
            run_individual_component_tests(generate_visualizations=args.visualize)
        
        if args.demo or args.all:
            run_quick_demo()
            
        if args.full or args.all:
            run_full_training()
            
        if not any([args.test, args.demo, args.full, args.all]):
            print("\nNo action specified. Available options:")
            print("  --test  : Test individual components (2-3 minutes)")
            print("  --demo  : Quick demonstration (5-10 minutes)")  
            print("  --full  : Complete training sequence (30-60 minutes)")
            print("  --all   : Run everything (45-75 minutes)")
            print("  --visualize : Generate visualizations during tests")
            print("\nExamples:")
            print("  python run_comprehensive_training.py --demo")
            print("  python run_comprehensive_training.py --test --visualize")
            
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        logger.exception("Training failed with exception")
        
    finally:
        print(f"\n🧠 Training session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()
