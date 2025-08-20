#!/usr/bin/env python3
"""
Unified Intelligence System - Small-Mind

This system combines all agents, models, and training capabilities into one
comprehensive AI that can learn, adapt, and evolve continuously.
"""

import os, sys
import json
import time
import logging
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import hashlib
import uuid

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

class UnifiedIntelligenceSystem:
    """
    Unified system that combines all agents, models, and training capabilities.
    """
    
    def __init__(self):
        self.setup_logging()
        self.system_id = str(uuid.uuid4())
        self.start_time = datetime.utcnow()
        
        # Core components
        self.agents = {}
        self.models = {}
        self.training_pipelines = {}
        self.knowledge_base = {}
        self.learning_history = []
        
        # System state
        self.is_running = False
        self.current_task = None
        self.performance_metrics = {}
        
        # Initialize all components
        self.initialize_system()
        
    def setup_logging(self):
        """Setup comprehensive logging system."""
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - [UNIFIED-INTELLIGENCE] - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "unified_intelligence.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 Unified Intelligence System Initializing...")
    
    def initialize_system(self):
        """Initialize all system components."""
        try:
            self.load_models()
            self.load_agents()
            self.setup_training_pipelines()
            self.initialize_knowledge_base()
            self.setup_learning_system()
            
            self.logger.info("✅ System initialization complete")
            self.logger.info(f"📊 Loaded: {len(self.models)} models, {len(self.agents)} agents")
            
        except Exception as e:
            self.logger.error(f"❌ System initialization failed: {e}")
            raise
    
    def load_models(self):
        """Load all available models from the registry."""
        try:
            # Load models.yaml
            models_file = project_root / "models" / "models.yaml"
            if models_file.exists():
                import yaml
                with open(models_file, 'r') as f:
                    models_config = yaml.safe_load(f)
                
                # Process each model type
                for model_type, models in models_config.items():
                    if model_type in ['routing', 'neuro_system']:
                        continue
                    
                    for model in models:
                        model_id = model['id']
                        self.models[model_id] = {
                            'config': model,
                            'type': model_type,
                            'status': 'available',
                            'last_used': None,
                            'performance_score': 0.0
                        }
                        
                        self.logger.info(f"📦 Loaded model: {model_id} ({model_type})")
            
            # Load checkpoint models
            checkpoints_dir = project_root / "src" / "smallmind" / "models" / "models" / "checkpoints"
            if checkpoints_dir.exists():
                for checkpoint_dir in checkpoints_dir.iterdir():
                    if checkpoint_dir.is_dir():
                        model_id = f"checkpoint.{checkpoint_dir.name}"
                        self.models[model_id] = {
                            'config': {
                                'id': model_id,
                                'type': 'checkpoint',
                                'path': str(checkpoint_dir),
                                'capabilities': ['reasoning', 'code', 'chat']
                            },
                            'type': 'checkpoint',
                            'status': 'available',
                            'last_used': None,
                            'performance_score': 0.0
                        }
                        self.logger.info(f"📦 Loaded checkpoint: {model_id}")
                        
        except Exception as e:
            self.logger.error(f"❌ Failed to load models: {e}")
            # Create fallback models
            self.models['fallback.generic'] = {
                'config': {'id': 'fallback.generic', 'type': 'generic', 'capabilities': ['chat']},
                'type': 'generic',
                'status': 'available',
                'last_used': None,
                'performance_score': 0.0
            }
    
    def load_agents(self):
        """Load all available agents."""
        try:
            # Load SmallMind agents
            sm_agents_dir = project_root / "src" / "smallmind"
            if sm_agents_dir.exists():
                agent_modules = [
                    'baby_agi',
                    'core',
                    'ml_optimization',
                    'neurodata',
                    'physics_simulation',
                    'simulation'
                ]
                
                for module in agent_modules:
                    module_path = sm_agents_dir / module
                    if module_path.exists():
                        agent_id = f"sm.{module}"
                        self.agents[agent_id] = {
                            'type': 'smallmind',
                            'module': module,
                            'path': str(module_path),
                            'status': 'available',
                            'capabilities': self._infer_capabilities(module),
                            'last_used': None,
                            'performance_score': 0.0
                        }
                        self.logger.info(f"🤖 Loaded agent: {agent_id}")
            
            # Load neuro system agents
            neuro_dir = project_root / ".neuro"
            if neuro_dir.exists():
                self.agents['neuro.scanner'] = {
                    'type': 'neuro',
                    'module': 'scanner',
                    'path': str(neuro_dir),
                    'status': 'available',
                    'capabilities': ['file_analysis', 'dependency_scan'],
                    'last_used': None,
                    'performance_score': 0.0
                }
                self.logger.info("🤖 Loaded neuro.scanner agent")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load agents: {e}")
            # Create fallback agent
            self.agents['fallback.agent'] = {
                'type': 'generic',
                'module': 'fallback',
                'path': str(project_root),
                'status': 'available',
                'capabilities': ['basic_reasoning'],
                'last_used': None,
                'performance_score': 0.0
            }
    
    def _infer_capabilities(self, module: str) -> List[str]:
        """Infer capabilities based on module name."""
        capability_map = {
            'baby_agi': ['planning', 'execution', 'learning'],
            'core': ['reasoning', 'coordination', 'memory'],
            'ml_optimization': ['optimization', 'training', 'analysis'],
            'neurodata': ['data_processing', 'neural_analysis'],
            'physics_simulation': ['simulation', 'physics', 'modeling'],
            'simulation': ['simulation', 'modeling', 'prediction']
        }
        return capability_map.get(module, ['basic_reasoning'])
    
    def setup_training_pipelines(self):
        """Setup training pipelines for continuous learning."""
        try:
            # Multi-model training
            multi_model_dir = project_root / "multi_model_training"
            if multi_model_dir.exists():
                training_scripts = [
                    'train_deepseek_v2.py',
                    'train_mixtao_moe.py', 
                    'train_qwen1_5_moe.py'
                ]
                
                for script in training_scripts:
                    script_path = multi_model_dir / script
                    if script_path.exists():
                        pipeline_id = f"training.{script.replace('.py', '')}"
                        self.training_pipelines[pipeline_id] = {
                            'script': str(script_path),
                            'type': 'multi_model',
                            'status': 'ready',
                            'last_run': None,
                            'success_rate': 0.0
                        }
                        self.logger.info(f"🔧 Loaded training pipeline: {pipeline_id}")
            
            # Continuous training
            continuous_dir = project_root / "my_continuous_training"
            if continuous_dir.exists():
                self.training_pipelines['continuous.learning'] = {
                    'script': str(continuous_dir),
                    'type': 'continuous',
                    'status': 'ready',
                    'last_run': None,
                    'success_rate': 0.0
                }
                self.logger.info("🔧 Loaded continuous learning pipeline")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to setup training pipelines: {e}")
    
    def initialize_knowledge_base(self):
        """Initialize the unified knowledge base."""
        try:
            # Load existing knowledge
            knowledge_file = project_root / "knowledge_base.json"
            if knowledge_file.exists():
                with open(knowledge_file, 'r') as f:
                    self.knowledge_base = json.load(f)
            else:
                self.knowledge_base = {
                    'concepts': {},
                    'relationships': {},
                    'experiences': [],
                    'learned_patterns': [],
                    'meta_knowledge': {
                        'created': datetime.utcnow().isoformat(),
                        'version': '1.0.0',
                        'total_concepts': 0,
                        'total_relationships': 0
                    }
                }
            
            self.logger.info("🧠 Knowledge base initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize knowledge base: {e}")
            self.knowledge_base = {'concepts': {}, 'relationships': {}, 'experiences': []}
    
    def setup_learning_system(self):
        """Setup the continuous learning system."""
        try:
            # Learning configuration
            self.learning_config = {
                'auto_learn': True,
                'learning_rate': 0.01,
                'memory_consolidation_interval': 300,  # 5 minutes
                'knowledge_synthesis_interval': 600,   # 10 minutes
                'performance_evaluation_interval': 1200, # 20 minutes
                'training_trigger_threshold': 0.7
            }
            
            # Start learning threads
            self.start_learning_threads()
            
            self.logger.info("🎓 Learning system initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup learning system: {e}")
    
    def start_learning_threads(self):
        """Start background learning threads."""
        def memory_consolidation_worker():
            while self.is_running:
                try:
                    time.sleep(self.learning_config['memory_consolidation_interval'])
                    self.consolidate_memories()
                except Exception as e:
                    self.logger.error(f"Memory consolidation error: {e}")
                    time.sleep(60)
        
        def knowledge_synthesis_worker():
            while self.is_running:
                try:
                    time.sleep(self.learning_config['knowledge_synthesis_interval'])
                    self.synthesize_knowledge()
                except Exception as e:
                    self.logger.error(f"Knowledge synthesis error: {e}")
                    time.sleep(60)
        
        def performance_evaluation_worker():
            while self.is_running:
                try:
                    time.sleep(self.learning_config['performance_evaluation_interval'])
                    self.evaluate_performance()
                except Exception as e:
                    self.logger.error(f"Performance evaluation error: {e}")
                    time.sleep(60)
        
        # Start threads
        threading.Thread(target=memory_consolidation_worker, daemon=True).start()
        threading.Thread(target=knowledge_synthesis_worker, daemon=True).start()
        threading.Thread(target=performance_evaluation_worker, daemon=True).start()
        
        self.logger.info("🔄 Learning threads started")
    
    def start_system(self):
        """Start the unified intelligence system."""
        if self.is_running:
            self.logger.warning("⚠️ System is already running")
            return
        
        self.is_running = True
        self.logger.info("🚀 Starting Unified Intelligence System...")
        
        # Initialize learning threads
        self.start_learning_threads()
        
        # Start autonomous operation
        self.start_autonomous_operation()
        
        self.logger.info("🎉 Unified Intelligence System is now running!")
        self.logger.info("🧠 It will continuously learn, adapt, and evolve")
        self.logger.info("🔧 All agents and models are integrated and available")
    
    def start_autonomous_operation(self):
        """Start autonomous operation mode."""
        def autonomous_worker():
            while self.is_running:
                try:
                    # Generate autonomous thoughts
                    thoughts = self.generate_autonomous_thoughts()
                    
                    # Process thoughts
                    for thought in thoughts:
                        self.process_thought(thought)
                    
                    # Check if training is needed
                    if self.should_trigger_training():
                        self.trigger_training()
                    
                    # Sleep between cycles
                    time.sleep(30)
                    
                except Exception as e:
                    self.logger.error(f"Autonomous operation error: {e}")
                    time.sleep(60)
        
        threading.Thread(target=autonomous_worker, daemon=True).start()
        self.logger.info("🔄 Autonomous operation started")
    
    def generate_autonomous_thoughts(self) -> List[Dict[str, Any]]:
        """Generate autonomous thoughts for continuous learning."""
        thoughts = []
        
        # Generate thoughts about system state
        thoughts.append({
            'type': 'system_analysis',
            'content': f"System has {len(self.models)} models and {len(self.agents)} agents available",
            'timestamp': datetime.utcnow().isoformat(),
            'priority': 'medium'
        })
        
        # Generate thoughts about learning opportunities
        if len(self.learning_history) > 0:
            thoughts.append({
                'type': 'learning_opportunity',
                'content': f"Analyzed {len(self.learning_history)} learning events",
                'timestamp': datetime.utcnow().isoformat(),
                'priority': 'high'
            })
        
        # Generate thoughts about performance
        if self.performance_metrics:
            thoughts.append({
                'type': 'performance_analysis',
                'content': f"Current performance metrics: {len(self.performance_metrics)} tracked",
                'timestamp': datetime.utcnow().isoformat(),
                'priority': 'medium'
            })
        
        return thoughts
    
    def process_thought(self, thought: Dict[str, Any]):
        """Process an autonomous thought."""
        try:
            # Add to learning history
            self.learning_history.append(thought)
            
            # Update knowledge base
            self.update_knowledge_base(thought)
            
            # Log the thought
            self.logger.info(f"💭 Processed thought: {thought['type']} - {thought['content']}")
            
        except Exception as e:
            self.logger.error(f"Error processing thought: {e}")
    
    def update_knowledge_base(self, thought: Dict[str, Any]):
        """Update the knowledge base with new information."""
        try:
            # Extract concepts from thought
            concepts = self.extract_concepts(thought['content'])
            
            # Add to knowledge base
            for concept in concepts:
                if concept not in self.knowledge_base['concepts']:
                    self.knowledge_base['concepts'][concept] = {
                        'first_seen': datetime.utcnow().isoformat(),
                        'occurrences': 0,
                        'related_thoughts': []
                    }
                
                self.knowledge_base['concepts'][concept]['occurrences'] += 1
                self.knowledge_base['concepts'][concept]['related_thoughts'].append(thought['id'] if 'id' in thought else str(uuid.uuid4()))
            
            # Update meta knowledge
            self.knowledge_base['meta_knowledge']['total_concepts'] = len(self.knowledge_base['concepts'])
            
        except Exception as e:
            self.logger.error(f"Error updating knowledge base: {e}")
    
    def extract_concepts(self, text: str) -> List[str]:
        """Extract concepts from text."""
        # Simple concept extraction (can be enhanced with NLP)
        words = text.lower().split()
        concepts = []
        
        for word in words:
            if len(word) > 3 and word.isalpha():
                concepts.append(word)
        
        return list(set(concepts))[:10]  # Limit to 10 concepts
    
    def should_trigger_training(self) -> bool:
        """Determine if training should be triggered."""
        if not self.learning_history:
            return False
        
        # Check performance threshold
        recent_performance = self.calculate_recent_performance()
        return recent_performance < self.learning_config['training_trigger_threshold']
    
    def calculate_recent_performance(self) -> float:
        """Calculate recent performance score."""
        if not self.performance_metrics:
            return 0.5
        
        # Calculate average of recent metrics
        recent_metrics = list(self.performance_metrics.values())[-10:]  # Last 10
        return sum(recent_metrics) / len(recent_metrics) if recent_metrics else 0.5
    
    def trigger_training(self):
        """Trigger training process."""
        try:
            self.logger.info("🎯 Triggering training process...")
            
            # Select best training pipeline
            best_pipeline = self.select_best_training_pipeline()
            
            if best_pipeline:
                self.run_training_pipeline(best_pipeline)
            else:
                self.logger.warning("⚠️ No suitable training pipeline available")
                
        except Exception as e:
            self.logger.error(f"Error triggering training: {e}")
    
    def select_best_training_pipeline(self) -> Optional[str]:
        """Select the best training pipeline based on current needs."""
        if not self.training_pipelines:
            return None
        
        # Simple selection based on success rate
        best_pipeline = None
        best_score = -1
        
        for pipeline_id, pipeline in self.training_pipelines.items():
            if pipeline['status'] == 'ready':
                score = pipeline['success_rate']
                if score > best_score:
                    best_score = score
                    best_pipeline = pipeline_id
        
        return best_pipeline
    
    def run_training_pipeline(self, pipeline_id: str):
        """Run a training pipeline."""
        try:
            pipeline = self.training_pipelines[pipeline_id]
            self.logger.info(f"🔧 Running training pipeline: {pipeline_id}")
            
            # Update pipeline status
            pipeline['status'] = 'running'
            pipeline['last_run'] = datetime.utcnow().isoformat()
            
            # Simulate training process
            self.simulate_training(pipeline_id)
            
            # Update pipeline status
            pipeline['status'] = 'ready'
            pipeline['success_rate'] = min(1.0, pipeline['success_rate'] + 0.1)
            
            self.logger.info(f"✅ Training pipeline {pipeline_id} completed")
            
        except Exception as e:
            self.logger.error(f"Error running training pipeline {pipeline_id}: {e}")
            if pipeline_id in self.training_pipelines:
                self.training_pipelines[pipeline_id]['status'] = 'error'
    
    def simulate_training(self, pipeline_id: str):
        """Simulate training process."""
        # This would be replaced with actual training execution
        time.sleep(5)  # Simulate training time
        
        # Update performance metrics
        self.performance_metrics[f"training_{pipeline_id}"] = 0.8
    
    def consolidate_memories(self):
        """Consolidate short-term memories into long-term knowledge."""
        try:
            if len(self.learning_history) > 100:
                # Keep only recent and important memories
                important_memories = [m for m in self.learning_history if m.get('priority') == 'high']
                recent_memories = self.learning_history[-50:]  # Last 50
                
                self.learning_history = important_memories + recent_memories
                self.logger.info("🧠 Memory consolidation completed")
                
        except Exception as e:
            self.logger.error(f"Memory consolidation error: {e}")
    
    def synthesize_knowledge(self):
        """Synthesize new knowledge from existing information."""
        try:
            # Analyze concept relationships
            concept_count = len(self.knowledge_base['concepts'])
            if concept_count > 0:
                # Create new insights
                new_insights = self.generate_insights()
                
                # Add to knowledge base
                if 'insights' not in self.knowledge_base:
                    self.knowledge_base['insights'] = []
                
                self.knowledge_base['insights'].extend(new_insights)
                
                self.logger.info(f"🧠 Knowledge synthesis: {len(new_insights)} new insights")
                
        except Exception as e:
            self.logger.error(f"Knowledge synthesis error: {e}")
    
    def generate_insights(self) -> List[Dict[str, Any]]:
        """Generate new insights from existing knowledge."""
        insights = []
        
        # Simple insight generation
        if len(self.knowledge_base['concepts']) > 5:
            insights.append({
                'type': 'pattern_recognition',
                'content': f"System has learned {len(self.knowledge_base['concepts'])} concepts",
                'timestamp': datetime.utcnow().isoformat(),
                'confidence': 0.8
            })
        
        if len(self.learning_history) > 10:
            insights.append({
                'type': 'learning_analysis',
                'content': f"Learning rate: {len(self.learning_history)} thoughts processed",
                'timestamp': datetime.utcnow().isoformat(),
                'confidence': 0.7
            })
        
        return insights
    
    def evaluate_performance(self):
        """Evaluate overall system performance."""
        try:
            # Calculate performance metrics
            total_models = len(self.models)
            available_models = len([m for m in self.models.values() if m['status'] == 'available'])
            total_agents = len(self.agents)
            available_agents = len([a for a in self.agents.values() if a['status'] == 'available'])
            
            # Calculate performance score
            model_score = available_models / total_models if total_models > 0 else 0
            agent_score = available_agents / total_agents if total_agents > 0 else 0
            knowledge_score = min(1.0, len(self.knowledge_base['concepts']) / 100)  # Normalize to 100 concepts
            
            overall_score = (model_score + agent_score + knowledge_score) / 3
            
            # Update performance metrics
            self.performance_metrics['overall'] = overall_score
            self.performance_metrics['models'] = model_score
            self.performance_metrics['agents'] = agent_score
            self.performance_metrics['knowledge'] = knowledge_score
            
            self.logger.info(f"📊 Performance evaluation: Overall score: {overall_score:.3f}")
            
        except Exception as e:
            self.logger.error(f"Performance evaluation error: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'system_id': self.system_id,
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat(),
            'uptime': (datetime.utcnow() - self.start_time).total_seconds(),
            'models': {
                'total': len(self.models),
                'available': len([m for m in self.models.values() if m['status'] == 'available']),
                'types': list(set(m['type'] for m in self.models.values()))
            },
            'agents': {
                'total': len(self.agents),
                'available': len([a for a in self.agents.values() if a['status'] == 'available']),
                'types': list(set(a['type'] for a in self.agents.values()))
            },
            'training_pipelines': {
                'total': len(self.training_pipelines),
                'ready': len([p for p in self.training_pipelines.values() if p['status'] == 'ready']),
                'running': len([p for p in self.training_pipelines.values() if p['status'] == 'running'])
            },
            'knowledge_base': {
                'concepts': len(self.knowledge_base.get('concepts', {})),
                'insights': len(self.knowledge_base.get('insights', [])),
                'learning_history': len(self.learning_history)
            },
            'performance': self.performance_metrics,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def stop_system(self):
        """Stop the unified intelligence system."""
        self.logger.info("🛑 Stopping Unified Intelligence System...")
        self.is_running = False
        
        # Save state
        self.save_system_state()
        
        self.logger.info("✅ System stopped successfully")
    
    def save_system_state(self):
        """Save current system state."""
        try:
            state = {
                'knowledge_base': self.knowledge_base,
                'learning_history': self.learning_history[-100:],  # Last 100 entries
                'performance_metrics': self.performance_metrics,
                'models': {k: {'status': v['status'], 'performance_score': v['performance_score']} for k, v in self.models.items()},
                'agents': {k: {'status': v['status'], 'performance_score': v['performance_score']} for k, v in self.agents.items()},
                'timestamp': datetime.utcnow().isoformat()
            }
            
            state_file = project_root / "unified_intelligence_state.json"
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.info("💾 System state saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save system state: {e}")

def main():
    """Main entry point."""
    print("🧠 Small-Mind Unified Intelligence System")
    print("=" * 60)
    print("This system combines ALL agents, models, and training capabilities")
    print("into one comprehensive AI that learns and evolves continuously.")
    print("=" * 60)
    
    try:
        # Create and start the system
        unified_system = UnifiedIntelligenceSystem()
        
        # Start the system
        unified_system.start_system()
        
        # Keep running and show status
        try:
            while unified_system.is_running:
                time.sleep(10)
                
                # Show status
                status = unified_system.get_system_status()
                print(f"\n📊 System Status:")
                print(f"   Models: {status['models']['available']}/{status['models']['total']} available")
                print(f"   Agents: {status['agents']['available']}/{status['agents']['total']} available")
                print(f"   Training: {status['training_pipelines']['ready']} pipelines ready")
                print(f"   Knowledge: {status['knowledge_base']['concepts']} concepts learned")
                print(f"   Performance: {status['performance'].get('overall', 0):.3f}")
                
        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested...")
            unified_system.stop_system()
            
    except Exception as e:
        print(f"❌ Failed to start Unified Intelligence System: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
