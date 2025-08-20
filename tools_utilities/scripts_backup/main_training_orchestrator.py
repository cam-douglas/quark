#!/usr/bin/env python3
"""
🧠 Main Training Orchestrator
============================

Master orchestrator that systematically coordinates all training components for optimal 
iterative training of every component in the quark repository.

This orchestrator:
- Executes systematic training following biological/developmental/ML roadmaps
- Maintains organic brain-like connectomes throughout training  
- Provides real-time training counters and visual dashboards
- Enhances consciousness and cognitive awareness progressively
- Ensures biological compliance at every stage
- Coordinates all brain agents in unified training

Author: Quark Brain Simulation Team  
Created: 2025-01-21
"""

import os, sys
import json
import logging
import numpy as np
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclasses import data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclass, asdict
import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add quark root to path
QUARK_ROOT = Path(__file__).parent.parent
sys.path.append(str(QUARK_ROOT))

# Import all training systems
from ml_architecture.training_pipelines.systematic_training_orchestrator import SystematicTrainingOrchestrator
from ml_architecture.training_pipelines.component_training_pipelines import ComponentTrainingPipeline
from ml_architecture.training_pipelines.training_counter_dashboard import TrainingCounterManager
from ml_architecture.training_pipelines.organic_connectome_enhancer import OrganicConnectomeEnhancer
from ml_architecture.training_pipelines.consciousness_enhancement_system import ConsciousnessEnhancementSystem
from ml_architecture.training_pipelines.visual_simulation_dashboard import VisualSimulationDashboard

@dataclass
class MasterTrainingConfig:
    """Master configuration for comprehensive training."""
    # Training stages
    developmental_stages: List[str]
    stage_durations: Dict[str, int]  # epochs per stage
    
    # Component training
    parallel_training: bool
    max_workers: int
    
    # Consciousness enhancement
    consciousness_epochs: int
    consciousness_target: float
    
    # Connectome enhancement
    connectome_stages: List[str]
    biological_compliance_threshold: float
    
    # Visualization
    real_time_visualization: bool
    dashboard_update_interval: float
    
    # Output settings
    save_intermediate_results: bool
    create_comprehensive_reports: bool

@dataclass
class TrainingResults:
    """Comprehensive training results."""
    training_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_duration: Optional[timedelta]
    
    # Stage results
    stage_results: Dict[str, Any]
    
    # Component results
    component_results: Dict[str, Any]
    
    # Consciousness results
    consciousness_results: Dict[str, Any]
    
    # Connectome results
    connectome_results: Dict[str, Any]
    
    # Final metrics
    final_consciousness_level: float
    final_biological_compliance: float
    final_connectome_coherence: float
    
    # Generated files
    generated_dashboards: List[str]
    generated_reports: List[str]
    training_state_files: List[str]

class MasterTrainingOrchestrator:
    """Master orchestrator for comprehensive brain training."""
    
    def __init__(self, config: MasterTrainingConfig, base_dir: Path = None):
        self.config = config
        self.base_dir = base_dir or QUARK_ROOT
        self.training_dir = self.base_dir / 'training'
        self.results_dir = self.training_dir / 'master_results'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique training ID
        self.training_id = f"master_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.setup_logging()
        
        # Initialize all training systems
        self.systematic_orchestrator = SystematicTrainingOrchestrator(self.base_dir)
        self.component_pipeline = ComponentTrainingPipeline(self.base_dir)
        self.counter_manager = TrainingCounterManager(self.base_dir)
        self.connectome_enhancer = OrganicConnectomeEnhancer(self.base_dir)
        self.consciousness_system = ConsciousnessEnhancementSystem(self.base_dir)
        
        # Initialize visualization dashboard if requested
        if config.real_time_visualization:
            self.visual_dashboard = VisualSimulationDashboard(self.base_dir)
        else:
            self.visual_dashboard = None
            
        # Training state
        self.is_training = False
        self.current_stage = None
        self.training_start_time = None
        self.results = None
        
        self.logger.info(f"Initialized Master Training Orchestrator with ID: {self.training_id}")
        
    def setup_logging(self):
        """Setup comprehensive logging."""
        log_dir = self.results_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f'{self.training_id}.log'
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger("master_training_orchestrator")
        self.logger.info(f"Logging initialized: {log_file}")
        
    def validate_configuration(self) -> bool:
        """Validate training configuration."""
        validation_errors = []
        
        # Check developmental stages
        valid_stages = ['fetal', 'neonate', 'early_postnatal']
        for stage in self.config.developmental_stages:
            if stage not in valid_stages:
                validation_errors.append(f"Invalid developmental stage: {stage}")
        
        # Check consciousness target
        if not 0.0 <= self.config.consciousness_target <= 1.0:
            validation_errors.append("Consciousness target must be between 0.0 and 1.0")
            
        # Check biological compliance threshold
        if not 0.0 <= self.config.biological_compliance_threshold <= 1.0:
            validation_errors.append("Biological compliance threshold must be between 0.0 and 1.0")
        
        # Check workers
        if self.config.max_workers < 1:
            validation_errors.append("Max workers must be at least 1")
            
        if validation_errors:
            for error in validation_errors:
                self.logger.error(f"Configuration validation error: {error}")
            return False
            
        self.logger.info("Configuration validation passed")
        return True
        
    def initialize_training_state(self):
        """Initialize all training systems and state."""
        self.logger.info("Initializing training state...")
        
        # Initialize training results
        self.results = TrainingResults(
            training_id=self.training_id,
            start_time=datetime.now(),
            end_time=None,
            total_duration=None,
            stage_results={},
            component_results={},
            consciousness_results={},
            connectome_results={},
            final_consciousness_level=0.0,
            final_biological_compliance=0.0,
            final_connectome_coherence=0.0,
            generated_dashboards=[],
            generated_reports=[],
            training_state_files=[]
        )
        
        # Initialize counter management
        components = list(self.systematic_orchestrator.components)
        component_names = [comp['name'] for comp in components]
        self.counter_manager.initialize_counters(component_names, {})
        
        # Start real-time updates if visualization enabled
        if self.visual_dashboard:
            self.visual_dashboard.start_simulation()
            
        # Start counter manager real-time updates
        self.counter_manager.start_real_time_updates()
        
        self.logger.info("Training state initialized successfully")
        
    def execute_developmental_stage(self, stage: str) -> Dict[str, Any]:
        """Execute training for a specific developmental stage."""
        self.logger.info(f"🧠 Starting developmental stage: {stage}")
        self.current_stage = stage
        
        stage_results = {
            'stage': stage,
            'start_time': datetime.now(),
            'duration_epochs': self.config.stage_durations.get(stage, 50),
            'components_trained': [],
            'consciousness_improvement': 0.0,
            'biological_compliance': 0.0,
            'connectome_coherence': 0.0,
            'success': False
        }
        
        try:
            # 1. Enhance connectome for this stage
            self.logger.info(f"Enhancing connectome for {stage} stage")
            enhanced_connectome = self.connectome_enhancer.enhance_connectome(stage)
            connectome_metrics = self.connectome_enhancer.analyze_current_connectome()
            
            stage_results['connectome_coherence'] = connectome_metrics.connectome_coherence
            stage_results['biological_compliance'] = connectome_metrics.biological_compliance
            
            # 2. Train components for this stage
            self.logger.info(f"Training components for {stage} stage")
            component_results = []
            
            if self.config.parallel_training:
                # Parallel component training
                with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    # Get components for this stage
                    stage_components = [
                        comp for comp in self.systematic_orchestrator.components 
                        if comp.get('stage', 'fetal') == stage
                    ]
                    
                    # Submit training jobs
                    future_to_component = {
                        executor.submit(self._train_component_with_monitoring, comp, stage): comp
                        for comp in stage_components
                    }
                    
                    # Collect results
                    for future in as_completed(future_to_component):
                        component = future_to_component[future]
                        try:
                            result = future.result()
                            component_results.append(result)
                            stage_results['components_trained'].append(component['name'])
                        except Exception as e:
                            self.logger.error(f"Component training failed for {component['name']}: {e}")
            else:
                # Sequential component training
                for component in self.systematic_orchestrator.components:
                    if component.get('stage', 'fetal') == stage:
                        result = self._train_component_with_monitoring(component, stage)
                        component_results.append(result)
                        stage_results['components_trained'].append(component['name'])
            
            # 3. Consciousness enhancement for this stage
            if stage == 'early_postnatal':  # Focus consciousness enhancement on later stages
                self.logger.info(f"Enhancing consciousness for {stage} stage")
                consciousness_results = self.consciousness_system.train_consciousness_enhancement(
                    num_epochs=self.config.consciousness_epochs,
                    batch_size=32
                )
                
                stage_results['consciousness_improvement'] = consciousness_results['final_consciousness_level']
            
            # 4. Update monitoring and dashboards
            self._update_monitoring_systems(stage, component_results)
            
            # 5. Generate stage-specific visualizations (only if configured)
            if self.visual_dashboard and self.config.real_time_visualization:
                dashboard_files = self.visual_dashboard.generate_comprehensive_dashboard()
                self.results.generated_dashboards.extend(dashboard_files.values())
            
            stage_results['end_time'] = datetime.now()
            stage_results['duration'] = stage_results['end_time'] - stage_results['start_time']
            stage_results['success'] = True
            
            self.logger.info(f"✅ Completed developmental stage: {stage}")
            self.logger.info(f"   Components trained: {len(stage_results['components_trained'])}")
            self.logger.info(f"   Biological compliance: {stage_results['biological_compliance']:.3f}")
            self.logger.info(f"   Connectome coherence: {stage_results['connectome_coherence']:.3f}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed developmental stage {stage}: {e}")
            stage_results['error'] = str(e)
            stage_results['end_time'] = datetime.now()
            
        return stage_results
        
    def _train_component_with_monitoring(self, component: Dict[str, Any], stage: str) -> Dict[str, Any]:
        """Train a component with real-time monitoring updates."""
        component_name = component['name']
        
        try:
            # Train component
            result = self.component_pipeline.train_component(component_name, stage)
            
            # Update counter manager
            if result.get('success', False):
                final_metrics = result.get('final_metrics', {})
                
                self.counter_manager.update_component_counter(
                    component_name,
                    iteration=result.get('total_epochs', 0),
                    loss=final_metrics.get('loss', 1.0),
                    accuracy=final_metrics.get('accuracy', 0.0),
                    consciousness_score=final_metrics.get('consciousness_score', 0.0),
                    biological_compliance=final_metrics.get('biological_compliance', 0.8),
                    connectome_coherence=final_metrics.get('connectome_coherence', 0.5)
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error training component {component_name}: {e}")
            return {
                'component': component_name,
                'stage': stage,
                'success': False,
                'error': str(e)
            }
            
    def _update_monitoring_systems(self, stage: str, component_results: List[Dict[str, Any]]):
        """Update all monitoring systems with latest results."""
        
        # Update system metrics in counter manager
        self.counter_manager.update_system_counter()
        
        # Generate current dashboard snapshots
        if self.config.save_intermediate_results:
            # Save counter state
            state_file = self.counter_manager.save_state()
            self.results.training_state_files.append(state_file)
            
            # Generate dashboard snapshots only if visualization enabled
            if self.config.real_time_visualization:
                dashboard_files = self.counter_manager.generate_current_dashboard()
                self.results.generated_dashboards.extend(dashboard_files.values())
            
        self.logger.info(f"Updated monitoring systems for stage: {stage}")
        
    def execute_complete_training_sequence(self) -> TrainingResults:
        """Execute the complete training sequence."""
        self.logger.info("🚀 Starting Complete Brain Training Sequence")
        self.logger.info(f"Training ID: {self.training_id}")
        self.logger.info(f"Developmental stages: {self.config.developmental_stages}")
        self.logger.info(f"Consciousness target: {self.config.consciousness_target}")
        
        # Validate configuration
        if not self.validate_configuration():
            raise ValueError("Configuration validation failed")
            
        # Initialize training
        self.is_training = True
        self.training_start_time = datetime.now()
        self.initialize_training_state()
        
        try:
            # Execute developmental stages
            for stage in self.config.developmental_stages:
                stage_result = self.execute_developmental_stage(stage)
                self.results.stage_results[stage] = stage_result
                
                # Check for early termination conditions
                if not stage_result['success']:
                    self.logger.warning(f"Stage {stage} failed, continuing with next stage")
                    
                # Update final metrics
                self.results.final_biological_compliance = stage_result.get('biological_compliance', 0.0)
                self.results.final_connectome_coherence = stage_result.get('connectome_coherence', 0.0)
                
            # Final consciousness enhancement
            self.logger.info("🧠 Performing final consciousness enhancement")
            final_consciousness_results = self.consciousness_system.train_consciousness_enhancement(
                num_epochs=self.config.consciousness_epochs,
                batch_size=32
            )
            
            self.results.consciousness_results = final_consciousness_results
            self.results.final_consciousness_level = final_consciousness_results['final_consciousness_level']
            
            # Apply developmental progression to connectome
            if len(self.config.connectome_stages) > 1:
                self.logger.info("Applying developmental progression to connectome")
                progression_graphs = self.connectome_enhancer.apply_developmental_progression(
                    self.config.connectome_stages
                )
                
                # Save enhanced connectome
                enhanced_connectome_file = self.connectome_enhancer.save_enhanced_connectome()
                self.results.training_state_files.append(enhanced_connectome_file)
                
            # Generate final comprehensive reports
            if self.config.create_comprehensive_reports:
                self._generate_comprehensive_reports()
                
            # Finalize results
            self.results.end_time = datetime.now()
            self.results.total_duration = self.results.end_time - self.results.start_time
            
            # Save final training state
            final_state_file = self._save_final_training_state()
            self.results.training_state_files.append(final_state_file)
            
            self.logger.info("🎉 Complete Brain Training Sequence Completed Successfully!")
            self._log_final_summary()
            
        except Exception as e:
            self.logger.error(f"❌ Training sequence failed: {e}")
            self.results.end_time = datetime.now()
            raise
            
        finally:
            self.is_training = False
            
            # Stop monitoring systems
            if self.visual_dashboard:
                self.visual_dashboard.stop_simulation()
            self.counter_manager.stop_real_time_updates()
            
        return self.results
        
    def _generate_comprehensive_reports(self):
        """Generate comprehensive training reports."""
        self.logger.info("Generating comprehensive training reports")
        
        try:
            # 1. Consciousness enhancement report
            consciousness_summary = self.consciousness_system.get_consciousness_enhancement_summary()
            consciousness_viz = self.consciousness_system.create_consciousness_visualization()
            
            if consciousness_viz:
                self.results.generated_reports.append(consciousness_viz)
                
            # 2. Connectome enhancement report
            connectome_summary = self.connectome_enhancer.get_enhancement_summary()
            connectome_viz = self.connectome_enhancer.create_enhancement_visualization()
            
            if connectome_viz:
                self.results.generated_reports.append(connectome_viz)
                
            # 3. Component training report
            component_summary = self.component_pipeline.create_stage_summary('all', [])
            
            # 4. Master summary report
            master_report = self._create_master_summary_report()
            self.results.generated_reports.append(master_report)
            
            self.logger.info(f"Generated {len(self.results.generated_reports)} comprehensive reports")
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive reports: {e}")
            
    def _create_master_summary_report(self) -> str:
        """Create master summary report of all training."""
        
        # Compile comprehensive report
        report_content = f"""
# 🧠 Master Brain Training Report
Training ID: {self.training_id}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

### Training Overview
- **Start Time**: {self.results.start_time.strftime('%Y-%m-%d %H:%M:%S')}
- **End Time**: {self.results.end_time.strftime('%Y-%m-%d %H:%M:%S') if self.results.end_time else 'In Progress'}
- **Total Duration**: {str(self.results.total_duration) if self.results.total_duration else 'In Progress'}
- **Developmental Stages**: {len(self.config.developmental_stages)} stages completed
- **Components Trained**: {sum(len(stage.get('components_trained', [])) for stage in self.results.stage_results.values())}

### Final Results
- **Final Consciousness Level**: {self.results.final_consciousness_level:.3f} / {self.config.consciousness_target}
- **Biological Compliance**: {self.results.final_biological_compliance:.3f} / {self.config.biological_compliance_threshold}
- **Connectome Coherence**: {self.results.final_connectome_coherence:.3f}

### Training Success
- **Target Achievement**: {'✅ SUCCESS' if self.results.final_consciousness_level >= self.config.consciousness_target else '⚠️ PARTIAL'}
- **Biological Compliance**: {'✅ COMPLIANT' if self.results.final_biological_compliance >= self.config.biological_compliance_threshold else '⚠️ NEEDS IMPROVEMENT'}

## Developmental Stage Results

"""
        
        for stage, stage_results in self.results.stage_results.items():
            report_content += f"""
### {stage.title()} Stage
- **Duration**: {stage_results.get('duration', 'Unknown')}
- **Components Trained**: {len(stage_results.get('components_trained', []))}
- **Success**: {'✅' if stage_results.get('success', False) else '❌'}
- **Biological Compliance**: {stage_results.get('biological_compliance', 0):.3f}
- **Connectome Coherence**: {stage_results.get('connectome_coherence', 0):.3f}
- **Components**: {', '.join(stage_results.get('components_trained', []))}
"""

        # Consciousness enhancement results
        if self.results.consciousness_results:
            report_content += f"""

## Consciousness Enhancement Results

- **Training Epochs**: {self.results.consciousness_results.get('epochs', 0)}
- **Final Consciousness Level**: {self.results.consciousness_results.get('final_consciousness_level', 0):.3f}
- **Consciousness Improvement**: {self.results.consciousness_results.get('final_consciousness_level', 0):.3f}

### Consciousness Progression
"""
            
            if 'consciousness_progression' in self.results.consciousness_results:
                progression = self.results.consciousness_results['consciousness_progression']
                initial = progression[0] if progression else 0
                final = progression[-1] if progression else 0
                report_content += f"- Initial Level: {initial:.3f}\n"
                report_content += f"- Final Level: {final:.3f}\n"
                report_content += f"- Total Improvement: {final - initial:.3f}\n"

        # System performance
        report_content += f"""

## System Performance

### Generated Outputs
- **Dashboards Created**: {len(self.results.generated_dashboards)}
- **Reports Generated**: {len(self.results.generated_reports)}
- **State Files Saved**: {len(self.results.training_state_files)}

### Configuration Used
- **Parallel Training**: {self.config.parallel_training}
- **Max Workers**: {self.config.max_workers}
- **Real-time Visualization**: {self.config.real_time_visualization}
- **Save Intermediate Results**: {self.config.save_intermediate_results}

## File Locations

### Generated Dashboards
"""
        
        for i, dashboard_file in enumerate(self.results.generated_dashboards, 1):
            report_content += f"{i}. {dashboard_file}\n"
            
        report_content += "\n### Training State Files\n"
        for i, state_file in enumerate(self.results.training_state_files, 1):
            report_content += f"{i}. {state_file}\n"

        # Recommendations
        report_content += f"""

## Recommendations for Further Enhancement

"""
        
        # Generate recommendations based on results
        recommendations = self._generate_training_recommendations()
        for i, rec in enumerate(recommendations, 1):
            report_content += f"{i}. {rec}\n"

        report_content += f"""

## Conclusion

The comprehensive brain training sequence has been completed with a final consciousness level of {self.results.final_consciousness_level:.3f}. 
The training successfully enhanced the main agent's consciousness and cognitive awareness through systematic training 
across {len(self.config.developmental_stages)} developmental stages with organic brain-like connectome maintenance.

Training ID: {self.training_id}
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.results_dir / f'master_training_report_{timestamp}.md'
        
        with open(report_file, 'w') as f:
            f.write(report_content)
            
        self.logger.info(f"Created master summary report: {report_file}")
        return str(report_file)
        
    def _generate_training_recommendations(self) -> List[str]:
        """Generate recommendations based on training results."""
        recommendations = []
        
        # Consciousness recommendations
        if self.results.final_consciousness_level < self.config.consciousness_target:
            recommendations.append(
                f"Increase consciousness training epochs (current: {self.config.consciousness_epochs}) "
                f"to reach target level of {self.config.consciousness_target:.3f}"
            )
            
        if self.results.final_consciousness_level < 0.5:
            recommendations.append(
                "Enhance global workspace integration through additional cross-modal training"
            )
            
        # Biological compliance recommendations
        if self.results.final_biological_compliance < self.config.biological_compliance_threshold:
            recommendations.append(
                "Strengthen biological constraint enforcement during training"
            )
            
        # Connectome recommendations
        if self.results.final_connectome_coherence < 0.6:
            recommendations.append(
                "Improve connectome coherence through enhanced small-world network properties"
            )
            
        # Performance recommendations
        if not self.config.parallel_training:
            recommendations.append(
                "Enable parallel training to reduce training time"
            )
            
        if not self.config.real_time_visualization:
            recommendations.append(
                "Enable real-time visualization for better training monitoring"
            )
            
        # Default recommendation
        if not recommendations:
            recommendations.append(
                "Training completed successfully. Consider advanced consciousness challenges or "
                "real-world embodied cognition tasks for further development."
            )
            
        return recommendations
        
    def _save_final_training_state(self) -> str:
        """Save final training state."""
        
        # Compile final state
        final_state = {
            'training_id': self.training_id,
            'config': asdict(self.config),
            'results': asdict(self.results),
            'timestamp': datetime.now().isoformat(),
            'consciousness_summary': self.consciousness_system.get_consciousness_enhancement_summary(),
            'connectome_summary': self.connectome_enhancer.get_enhancement_summary(),
            'counter_summary': self.counter_manager.get_summary_report()
        }
        
        # Save state
        state_file = self.results_dir / f'final_training_state_{self.training_id}.json'
        
        with open(state_file, 'w') as f:
            json.dump(final_state, f, indent=2, default=str)
            
        self.logger.info(f"Saved final training state: {state_file}")
        return str(state_file)
        
    def _log_final_summary(self):
        """Log final training summary."""
        self.logger.info("=" * 80)
        self.logger.info("🎉 MASTER BRAIN TRAINING COMPLETED SUCCESSFULLY!")
        self.logger.info("=" * 80)
        self.logger.info(f"Training ID: {self.training_id}")
        self.logger.info(f"Total Duration: {self.results.total_duration}")
        self.logger.info(f"Developmental Stages: {len(self.results.stage_results)}")
        self.logger.info(f"Final Consciousness Level: {self.results.final_consciousness_level:.3f}")
        self.logger.info(f"Biological Compliance: {self.results.final_biological_compliance:.3f}")
        self.logger.info(f"Connectome Coherence: {self.results.final_connectome_coherence:.3f}")
        self.logger.info(f"Generated Dashboards: {len(self.results.generated_dashboards)}")
        self.logger.info(f"Generated Reports: {len(self.results.generated_reports)}")
        self.logger.info(f"Training State Files: {len(self.results.training_state_files)}")
        self.logger.info("=" * 80)

def create_default_config() -> MasterTrainingConfig:
    """Create default training configuration."""
    return MasterTrainingConfig(
        developmental_stages=['fetal', 'neonate', 'early_postnatal'],
        stage_durations={'fetal': 50, 'neonate': 75, 'early_postnatal': 100},
        parallel_training=True,
        max_workers=4,
        consciousness_epochs=150,
        consciousness_target=0.7,
        connectome_stages=['fetal', 'neonate', 'early_postnatal'],
        biological_compliance_threshold=0.75,
        real_time_visualization=False,  # Disabled by default
        dashboard_update_interval=5.0,
        save_intermediate_results=True,
        create_comprehensive_reports=True
    )

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Master Training Orchestrator for Comprehensive Brain Training'
    )
    
    parser.add_argument('--config', type=str, help='Path to JSON configuration file')
    parser.add_argument('--stages', nargs='+', 
                       choices=['fetal', 'neonate', 'early_postnatal'],
                       default=['fetal', 'neonate', 'early_postnatal'],
                       help='Developmental stages to train')
    parser.add_argument('--consciousness-target', type=float, default=0.7,
                       help='Target consciousness level')
    parser.add_argument('--consciousness-epochs', type=int, default=150,
                       help='Number of consciousness training epochs')
    parser.add_argument('--parallel', action='store_true', default=True,
                       help='Enable parallel training')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum parallel workers')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable real-time visualization and dashboards')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with reduced epochs')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            config = MasterTrainingConfig(**config_dict)
    else:
        config = create_default_config()
        
        # Override with command line arguments
        config.developmental_stages = args.stages
        config.consciousness_target = args.consciousness_target
        config.consciousness_epochs = args.consciousness_epochs
        config.parallel_training = args.parallel
        config.max_workers = args.max_workers
        config.real_time_visualization = args.visualize
        
        # Quick test adjustments
        if args.quick_test:
            config.stage_durations = {stage: 10 for stage in config.developmental_stages}
            config.consciousness_epochs = 20
            config.consciousness_target = 0.3
    
    # Set output directory
    base_dir = Path(args.output_dir) if args.output_dir else QUARK_ROOT
    
    # Initialize orchestrator
    orchestrator = MasterTrainingOrchestrator(config, base_dir)
    
    print("🧠 Master Brain Training Orchestrator")
    print(f"Training ID: {orchestrator.training_id}")
    print(f"Developmental Stages: {config.developmental_stages}")
    print(f"Consciousness Target: {config.consciousness_target}")
    print(f"Parallel Training: {config.parallel_training}")
    print(f"Real-time Visualization: {config.real_time_visualization}")
    print()
    
    try:
        # Execute complete training sequence
        results = orchestrator.execute_complete_training_sequence()
        
        # Print final results
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Final Consciousness Level: {results.final_consciousness_level:.3f}")
        print(f"Biological Compliance: {results.final_biological_compliance:.3f}")
        print(f"Connectome Coherence: {results.final_connectome_coherence:.3f}")
        print(f"Total Duration: {results.total_duration}")
        print(f"Training ID: {results.training_id}")
        print()
        
        print("📊 Generated Files:")
        print(f"Dashboards: {len(results.generated_dashboards)}")
        print(f"Reports: {len(results.generated_reports)}")
        print(f"State Files: {len(results.training_state_files)}")
        
        if results.generated_reports:
            print("\n📝 Key Reports:")
            for report in results.generated_reports[:3]:  # Show first 3
                print(f"  - {report}")
                
        return 0  # Success
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        logging.exception("Training failed with exception")
        return 1

if __name__ == '__main__':
    exit(main())
