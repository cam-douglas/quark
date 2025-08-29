"""
LLM-Guided Training Pipeline for Quark
Integrates datasets from LLM-IK and LLM Articulated Manipulation repositories
to create a comprehensive training system that follows developmental principles.
"""

import os
import sys
import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging
from datetime import datetime

# Import the integrated systems
from brain.architecture.neural_core.learning.dataset_integration import dataset_integration
from brain.architecture.neural_core.motor_control.llm_inverse_kinematics import LLMInverseKinematics
from brain.architecture.neural_core.planning.llm_manipulation_planner import LLMManipulationPlanner

class LLMGuidedTrainingPipeline:
    """
    A comprehensive training pipeline that leverages LLM-generated data to train Quark's brain.
    
    This pipeline implements a developmentally-inspired curriculum that progresses from:
    1. Basic proprioception and joint control
    2. LLM-guided inverse kinematics learning
    3. Object manipulation from human demonstrations
    4. Complex multi-step task planning
    
    The key innovation is using LLMs not just for inference, but as training data generators
    that provide rich, structured learning experiences.
    """
    
    def __init__(self, brain_simulator=None):
        """
        Initialize the LLM-guided training pipeline.
        
        Args:
            brain_simulator: Reference to the main brain simulator
        """
        self.brain_simulator = brain_simulator
        self.dataset_integration = dataset_integration
        
        # Initialize LLM modules
        self.llm_ik = LLMInverseKinematics()
        self.llm_manipulation_planner = LLMManipulationPlanner()
        
        # Training state
        self.current_phase = 0
        self.training_history = []
        self.performance_metrics = {}
        self.curriculum_progress = {}
        
        # Load training data
        self.training_datasets = self._load_all_training_data()
        
        # Define developmental curriculum
        self.curriculum_phases = self._define_curriculum()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        print("🎓 LLM-Guided Training Pipeline initialized")
        print(f"   Curriculum phases: {len(self.curriculum_phases)}")
        print(f"   Available IK solutions: {len(self.training_datasets['ik_solutions']['solutions'])}")
        print(f"   Available manipulation demos: {len(self.training_datasets['manipulation_demos']['demonstrations'])}")
    
    def _load_all_training_data(self) -> Dict[str, Any]:
        """Load all available training datasets."""
        return {
            'ik_solutions': self.dataset_integration.load_ik_training_data(),
            'manipulation_demos': self.dataset_integration.load_manipulation_training_data(),
            'unified_dataset': self.dataset_integration.create_unified_training_dataset(),
            'training_recommendations': self.dataset_integration.get_training_recommendations()
        }
    
    def _define_curriculum(self) -> List[Dict[str, Any]]:
        """
        Define the developmental curriculum based on neuroscience insights and available data.
        """
        return [
            {
                'phase': 0,
                'name': 'Proprioceptive Foundation',
                'description': 'Basic body awareness and joint control',
                'duration_episodes': 1000,
                'learning_objectives': [
                    'Stable joint position control',
                    'Proprioceptive feedback integration',
                    'Basic balance maintenance'
                ],
                'data_sources': ['basic_motor_control'],
                'success_criteria': {
                    'stability_threshold': 0.9,
                    'position_accuracy': 0.1,
                    'fall_rate': 0.1
                },
                'llm_integration': 'none'
            },
            {
                'phase': 1,
                'name': 'LLM-IK Learning',
                'description': 'Learn inverse kinematics through LLM guidance',
                'duration_episodes': 5000,
                'learning_objectives': [
                    'Solve simple IK problems',
                    'Understand joint space relationships',
                    'Learn from LLM reasoning examples'
                ],
                'data_sources': ['llm_ik_normal_mode', 'llm_ik_extend_mode'],
                'success_criteria': {
                    'ik_success_rate': 0.7,
                    'reasoning_quality': 0.8,
                    'solution_diversity': 0.6
                },
                'llm_integration': 'teacher'
            },
            {
                'phase': 2,
                'name': 'Basic Object Manipulation',
                'description': 'Learn from human manipulation demonstrations',
                'duration_episodes': 10000,
                'learning_objectives': [
                    'Imitate human demonstrations',
                    'Understand object affordances',
                    'Basic grasp and release'
                ],
                'data_sources': ['manipulation_demos_basic', 'prompt_templates'],
                'success_criteria': {
                    'imitation_accuracy': 0.6,
                    'task_completion_rate': 0.5,
                    'safety_compliance': 0.95
                },
                'llm_integration': 'collaborator'
            },
            {
                'phase': 3,
                'name': 'Complex Manipulation Planning',
                'description': 'Multi-step task planning with LLM guidance',
                'duration_episodes': 20000,
                'learning_objectives': [
                    'Plan multi-step manipulations',
                    'Adapt to novel objects',
                    'Generate own solutions'
                ],
                'data_sources': ['manipulation_demos_complex', 'llm_planning', 'cross_modal_mappings'],
                'success_criteria': {
                    'planning_success_rate': 0.7,
                    'adaptation_rate': 0.5,
                    'novel_solution_generation': 0.3
                },
                'llm_integration': 'partner'
            },
            {
                'phase': 4,
                'name': 'Autonomous Learning',
                'description': 'Self-directed learning with LLM consultation',
                'duration_episodes': float('inf'),  # Continuous
                'learning_objectives': [
                    'Generate own training data',
                    'Seek LLM guidance when needed',
                    'Continuous skill refinement'
                ],
                'data_sources': ['self_generated', 'llm_consultation'],
                'success_criteria': {
                    'autonomy_level': 0.8,
                    'learning_efficiency': 0.6,
                    'skill_transfer': 0.7
                },
                'llm_integration': 'consultant'
            }
        ]
    
    def train_phase(self, phase_id: int, num_episodes: int = None) -> Dict[str, Any]:
        """
        Train Quark for a specific curriculum phase.
        
        Args:
            phase_id: Which curriculum phase to train
            num_episodes: Override default episode count
            
        Returns:
            Training results and performance metrics
        """
        if phase_id >= len(self.curriculum_phases):
            raise ValueError(f"Phase {phase_id} not defined. Available phases: 0-{len(self.curriculum_phases)-1}")
        
        phase = self.curriculum_phases[phase_id]
        episodes = num_episodes or phase['duration_episodes']
        
        self.logger.info(f"🎯 Starting Phase {phase_id}: {phase['name']}")
        self.logger.info(f"   Duration: {episodes} episodes")
        self.logger.info(f"   LLM Integration: {phase['llm_integration']}")
        
        # Prepare phase-specific training data
        phase_data = self._prepare_phase_data(phase)
        
        # Execute training based on LLM integration level
        training_results = self._execute_phase_training(phase, phase_data, episodes)
        
        # Evaluate phase completion
        evaluation_results = self._evaluate_phase_completion(phase, training_results)
        
        # Update curriculum progress
        self.curriculum_progress[phase_id] = {
            'completed': evaluation_results['passed'],
            'performance': evaluation_results['metrics'],
            'completion_time': datetime.now().isoformat(),
            'episodes_trained': episodes
        }
        
        # Record in training history
        self.training_history.append({
            'phase_id': phase_id,
            'phase_name': phase['name'],
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'timestamp': datetime.now().isoformat()
        })
        
        if evaluation_results['passed']:
            self.logger.info(f"✅ Phase {phase_id} completed successfully!")
            self.current_phase = phase_id + 1
        else:
            self.logger.warning(f"⚠️ Phase {phase_id} did not meet success criteria")
        
        return {
            'phase_id': phase_id,
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'next_phase': self.current_phase
        }
    
    def _prepare_phase_data(self, phase: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare training data for a specific phase."""
        phase_data = {
            'training_samples': [],
            'demonstration_data': [],
            'llm_guidance': [],
            'evaluation_benchmarks': []
        }
        
        for data_source in phase['data_sources']:
            if 'llm_ik' in data_source:
                # Prepare LLM-IK data
                ik_solutions = self.training_datasets['ik_solutions']['solutions']
                mode_filter = data_source.split('_')[-2]  # Extract mode (normal, extend, etc.)
                
                relevant_solutions = [
                    sol for sol in ik_solutions 
                    if sol['solving_mode'].lower() == mode_filter.lower()
                ]
                
                phase_data['training_samples'].extend(relevant_solutions[:50])  # Limit for training
                
            elif 'manipulation_demos' in data_source:
                # Prepare manipulation demonstration data
                demos = self.training_datasets['manipulation_demos']['demonstrations']
                
                if 'basic' in data_source:
                    # Filter for basic tasks (open, close, simple movements)
                    basic_tasks = ['open', 'close', 'turn']
                    relevant_demos = [
                        demo for demo in demos 
                        if any(task in demo['task_type'] for task in basic_tasks)
                    ]
                else:
                    # All demonstrations for complex training
                    relevant_demos = demos
                
                phase_data['demonstration_data'].extend(relevant_demos[:30])  # Limit for training
                
            elif data_source == 'prompt_templates':
                # Prepare LLM prompt templates
                templates = self.training_datasets['manipulation_demos']['prompt_templates']
                phase_data['llm_guidance'].extend(list(templates.values()))
        
        self.logger.info(f"   Prepared phase data:")
        self.logger.info(f"     Training samples: {len(phase_data['training_samples'])}")
        self.logger.info(f"     Demonstration data: {len(phase_data['demonstration_data'])}")
        self.logger.info(f"     LLM guidance items: {len(phase_data['llm_guidance'])}")
        
        return phase_data
    
    def _execute_phase_training(self, phase: Dict, phase_data: Dict, episodes: int) -> Dict[str, Any]:
        """Execute training for a specific phase."""
        training_results = {
            'episodes_completed': 0,
            'performance_trajectory': [],
            'llm_interactions': [],
            'skill_acquisitions': [],
            'errors_encountered': []
        }
        
        integration_mode = phase['llm_integration']
        
        # Handle infinite duration for autonomous learning phase
        if episodes == float('inf'):
            # For autonomous phase, run for a fixed number of episodes as a demonstration
            # In a real continuous run, this would be handled by an external process
            episodes = 20000 
            self.logger.info(f"     Running autonomous phase for a fixed {episodes} episodes for demonstration.")

        for episode in range(episodes):
            # In this simplified training simulation, we don't have a full
            # embodiment, so we create mock sensory data.
            mock_qpos = np.random.rand(32)
            mock_qvel = np.random.rand(32)
            mock_sensory_data = {
                "sensory_inputs": {
                    "qpos": mock_qpos,
                    "qvel": mock_qvel,
                    "reward": 0.0,
                    "is_fallen": False
                }
            }

            # We simulate a fixed number of steps per episode
            for step in range(200): # e.g., 200 steps per episode
                is_done = (step == 199)
                mock_sensory_data["sensory_inputs"]["is_fallen"] = is_done
                
                brain_output = self.brain.step(mock_sensory_data, stage=phase_index)

                # Track performance metric (e.g., reduction in error signal)
                error_signal = brain_output.get("limbic_system", {}).get("error_signal", 1.0)
                performance_metric = 1.0 - error_signal
                performance_history.append(performance_metric)

            if (episode + 1) % 1000 == 0:
                avg_performance = np.mean(performance_history[-1000:])
        
        return training_results
    
    def _run_training_episode(self, phase: Dict, phase_data: Dict, 
                            episode: int, integration_mode: str) -> Dict[str, Any]:
        """Run a single training episode."""
        episode_result = {
            'performance': 0.0,
            'actions_taken': 0,
            'llm_interaction': None,
            'skill_acquired': None,
            'error': None
        }
        
        try:
            if integration_mode == 'none':
                # Basic training without LLM
                episode_result['performance'] = self._basic_training_step()
                
            elif integration_mode == 'teacher':
                # LLM as teacher - providing solutions and guidance
                episode_result = self._llm_teacher_training_step(phase_data, episode)
                
            elif integration_mode == 'collaborator':
                # LLM as collaborator - working together on tasks
                episode_result = self._llm_collaborator_training_step(phase_data, episode)
                
            elif integration_mode == 'partner':
                # LLM as partner - equal collaboration in problem solving
                episode_result = self._llm_partner_training_step(phase_data, episode)
                
            elif integration_mode == 'consultant':
                # LLM as consultant - consulted when needed
                episode_result = self._llm_consultant_training_step(phase_data, episode)
                
        except Exception as e:
            episode_result['error'] = str(e)
            episode_result['performance'] = 0.0
        
        return episode_result
    
    def _basic_training_step(self) -> float:
        """Basic training step without LLM guidance."""
        # Simulate basic proprioceptive training
        return np.random.uniform(0.1, 0.9)  # Simulated performance
    
    def _llm_teacher_training_step(self, phase_data: Dict, episode: int) -> Dict[str, Any]:
        """Training step with LLM as teacher."""
        result = {
            'performance': 0.0,
            'actions_taken': 0,
            'llm_interaction': None,
            'skill_acquired': None
        }
        
        # Select a training sample (IK solution)
        if phase_data['training_samples']:
            sample = np.random.choice(phase_data['training_samples'])
            
            # Have LLM explain the solution
            llm_explanation = self._get_llm_explanation(sample)
            
            # Simulate learning from explanation
            learning_success = np.random.uniform(0.0, 1.0)
            
            result['performance'] = learning_success
            result['actions_taken'] = np.random.randint(5, 20)
            result['llm_interaction'] = {
                'type': 'teaching',
                'content': llm_explanation,
                'effectiveness': learning_success
            }
            
            if learning_success > 0.7:
                result['skill_acquired'] = f"IK solving mode: {sample.get('solving_mode', 'unknown')}"
        
        return result
    
    def _llm_collaborator_training_step(self, phase_data: Dict, episode: int) -> Dict[str, Any]:
        """Training step with LLM as collaborator."""
        result = {
            'performance': 0.0,
            'actions_taken': 0,
            'llm_interaction': None,
            'skill_acquired': None
        }
        
        # Select a demonstration to learn from
        if phase_data['demonstration_data']:
            demo = np.random.choice(phase_data['demonstration_data'])
            
            # Collaborate with LLM on understanding the task
            collaboration_result = self._collaborate_with_llm(demo)
            
            result['performance'] = collaboration_result['success_rate']
            result['actions_taken'] = len(demo.get('action_sequence', []))
            result['llm_interaction'] = {
                'type': 'collaboration',
                'task': demo['task_type'],
                'effectiveness': collaboration_result['success_rate']
            }
            
            if collaboration_result['success_rate'] > 0.6:
                result['skill_acquired'] = f"Manipulation: {demo['task_type']}"
        
        return result
    
    def _llm_partner_training_step(self, phase_data: Dict, episode: int) -> Dict[str, Any]:
        """Training step with LLM as equal partner."""
        result = {
            'performance': 0.0,
            'actions_taken': 0,
            'llm_interaction': None,
            'skill_acquired': None
        }
        
        # Generate a complex task together
        partnership_result = self._partner_with_llm(phase_data)
        
        result['performance'] = partnership_result['task_success']
        result['actions_taken'] = partnership_result['actions_count']
        result['llm_interaction'] = {
            'type': 'partnership',
            'task_complexity': partnership_result['complexity'],
            'effectiveness': partnership_result['task_success']
        }
        
        if partnership_result['task_success'] > 0.5:
            result['skill_acquired'] = partnership_result['new_skill']
        
        return result
    
    def _llm_consultant_training_step(self, phase_data: Dict, episode: int) -> Dict[str, Any]:
        """Training step with LLM as consultant."""
        result = {
            'performance': 0.0,
            'actions_taken': 0,
            'llm_interaction': None,
            'skill_acquired': None
        }
        
        # Autonomous action with occasional LLM consultation
        autonomous_performance = np.random.uniform(0.3, 0.8)
        
        # Consult LLM if performance is low
        if autonomous_performance < 0.5:
            consultation_result = self._consult_llm()
            result['performance'] = max(autonomous_performance, consultation_result['improved_performance'])
            result['llm_interaction'] = {
                'type': 'consultation',
                'reason': 'low_performance',
                'effectiveness': consultation_result['improved_performance'] - autonomous_performance
            }
        else:
            result['performance'] = autonomous_performance
        
        result['actions_taken'] = np.random.randint(10, 30)
        
        return result
    
    def _get_llm_explanation(self, sample: Dict) -> str:
        """Get LLM explanation for a training sample."""
        return f"This IK solution uses {sample.get('solving_mode', 'unknown')} mode to solve for {sample.get('dof', 'unknown')} degrees of freedom."
    
    def _collaborate_with_llm(self, demo: Dict) -> Dict[str, Any]:
        """Collaborate with LLM on understanding a demonstration."""
        return {
            'success_rate': np.random.uniform(0.4, 0.8),
            'insights_gained': np.random.randint(1, 5)
        }
    
    def _partner_with_llm(self, phase_data: Dict) -> Dict[str, Any]:
        """Partner with LLM on complex task generation."""
        return {
            'task_success': np.random.uniform(0.3, 0.7),
            'actions_count': np.random.randint(15, 40),
            'complexity': np.random.uniform(0.5, 1.0),
            'new_skill': f"Complex planning skill {np.random.randint(1, 100)}"
        }
    
    def _consult_llm(self) -> Dict[str, Any]:
        """Consult LLM for guidance."""
        return {
            'improved_performance': np.random.uniform(0.5, 0.9),
            'guidance_quality': np.random.uniform(0.6, 1.0)
        }
    
    def _evaluate_phase_completion(self, phase: Dict, training_results: Dict) -> Dict[str, Any]:
        """Evaluate whether a phase has been completed successfully."""
        success_criteria = phase['success_criteria']
        performance_trajectory = training_results['performance_trajectory']
        
        # Calculate metrics
        if not performance_trajectory:
            return {'passed': False, 'metrics': {}, 'reasons': ['No training data']}
        
        avg_performance = np.mean(performance_trajectory[-1000:])  # Last 1000 episodes
        final_performance = np.mean(performance_trajectory[-100:])   # Last 100 episodes
        improvement_rate = (final_performance - np.mean(performance_trajectory[:100])) if len(performance_trajectory) > 100 else 0
        
        metrics = {
            'average_performance': avg_performance,
            'final_performance': final_performance,
            'improvement_rate': improvement_rate,
            'episodes_trained': len(performance_trajectory),
            'llm_interactions_count': len(training_results['llm_interactions']),
            'skills_acquired_count': len(training_results['skill_acquisitions'])
        }
        
        # Check success criteria
        passed_criteria = []
        failed_criteria = []
        
        for criterion, threshold in success_criteria.items():
            if criterion in metrics:
                if metrics[criterion] >= threshold:
                    passed_criteria.append(criterion)
                else:
                    failed_criteria.append(f"{criterion}: {metrics[criterion]:.3f} < {threshold}")
        
        passed = len(failed_criteria) == 0
        
        return {
            'passed': passed,
            'metrics': metrics,
            'passed_criteria': passed_criteria,
            'failed_criteria': failed_criteria,
            'reasons': failed_criteria if not passed else []
        }
    
    def run_full_curriculum(self, start_phase: int = 0) -> Dict[str, Any]:
        """
        Run the complete curriculum from start_phase to completion.
        """
        self.logger.info(f"🚀 Starting full curriculum from phase {start_phase}")
        
        curriculum_results = {
            'phases_completed': [],
            'phases_failed': [],
            'total_episodes': 0,
            'total_training_time': None,
            'final_capabilities': []
        }
        
        start_time = datetime.now()
        
        for phase_id in range(start_phase, len(self.curriculum_phases)):
            phase_result = self.train_phase(phase_id)
            
            curriculum_results['total_episodes'] += phase_result['training_results']['episodes_completed']
            
            if phase_result['evaluation_results']['passed']:
                curriculum_results['phases_completed'].append(phase_id)
                
                # Record capabilities gained
                skills = phase_result['training_results']['skill_acquisitions']
                curriculum_results['final_capabilities'].extend(skills)
                
            else:
                curriculum_results['phases_failed'].append(phase_id)
                self.logger.warning(f"Phase {phase_id} failed. Stopping curriculum.")
                break
        
        curriculum_results['total_training_time'] = (datetime.now() - start_time).total_seconds()
        
        self.logger.info(f"🎓 Curriculum completed!")
        self.logger.info(f"   Phases completed: {len(curriculum_results['phases_completed'])}")
        self.logger.info(f"   Total episodes: {curriculum_results['total_episodes']}")
        self.logger.info(f"   Training time: {curriculum_results['total_training_time']:.1f} seconds")
        
        return curriculum_results
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status and progress."""
        return {
            'current_phase': self.current_phase,
            'curriculum_progress': self.curriculum_progress,
            'total_training_episodes': sum(
                progress.get('episodes_trained', 0) 
                for progress in self.curriculum_progress.values()
            ),
            'phases_available': len(self.curriculum_phases),
            'training_history_length': len(self.training_history),
            'datasets_loaded': {
                'ik_solutions': len(self.training_datasets['ik_solutions']['solutions']),
                'manipulation_demos': len(self.training_datasets['manipulation_demos']['demonstrations'])
            }
        }

# Factory function for brain integration
def create_llm_training_pipeline(brain_simulator=None) -> LLMGuidedTrainingPipeline:
    """Create LLM training pipeline for brain integration."""
    return LLMGuidedTrainingPipeline(brain_simulator)

if __name__ == "__main__":
    # Demonstration
    print("🎓 LLM-Guided Training Pipeline Demonstration")
    print("="*60)
    
    pipeline = LLMGuidedTrainingPipeline()
    
    # Show training status
    status = pipeline.get_training_status()
    print(f"\n📊 Training Status:")
    print(f"   Current phase: {status['current_phase']}")
    print(f"   Available phases: {status['phases_available']}")
    print(f"   IK solutions loaded: {status['datasets_loaded']['ik_solutions']}")
    print(f"   Manipulation demos loaded: {status['datasets_loaded']['manipulation_demos']}")
    
    # Demonstrate phase training
    print(f"\n🎯 Demonstrating Phase 1 Training (LLM-IK Learning):")
    phase_result = pipeline.train_phase(1, num_episodes=100)  # Short demo
    
    print(f"   Phase completed: {phase_result['evaluation_results']['passed']}")
    print(f"   Episodes trained: {phase_result['training_results']['episodes_completed']}")
    print(f"   LLM interactions: {len(phase_result['training_results']['llm_interactions'])}")
    print(f"   Skills acquired: {len(phase_result['training_results']['skill_acquisitions'])}")
    
    print(f"\n✅ Training pipeline demonstration complete!")
