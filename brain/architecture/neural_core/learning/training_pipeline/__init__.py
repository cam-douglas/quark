#!/usr/bin/env python3
"""Training Pipeline Module - Main interface for LLM-guided training system.

Provides unified interface to training pipeline components with preserved integrations.

Integration: Main training pipeline interface for neural learning workflows.
Rationale: Clean API abstraction maintaining all existing functionality.
"""

from typing import Dict, List, Any, Optional
import logging

from .curriculum_manager import CurriculumManager
from .training_executor import TrainingExecutor

# Import dependencies
from brain.architecture.neural_core.learning.dataset_integration import DatasetIntegration

logger = logging.getLogger(__name__)

class LLMGuidedTrainingPipeline:
    """
    A comprehensive training pipeline that leverages LLM-generated data to train Quark's brain.
    
    This pipeline implements a developmentally-inspired curriculum that progresses from:
    1. Basic proprioception and joint control
    2. LLM-guided inverse kinematics learning
    3. Object manipulation from human demonstrations
    4. Complex multi-step task planning
    """

    def __init__(self, brain_simulator=None):
        """Initialize the LLM-guided training pipeline."""

        self.brain_simulator = brain_simulator
        self.dataset_integration = DatasetIntegration()

        # Initialize modular components
        self.curriculum_manager = CurriculumManager()
        self.training_executor = TrainingExecutor(brain_simulator)

        # Load training data
        self.training_data = self._load_all_training_data()

        logger.info("LLM-Guided Training Pipeline initialized with modular components")
        print(f"📚 Loaded training data: {len(self.training_data)} datasets")
        print(f"🎯 Curriculum: {self.curriculum_manager.get_curriculum_summary()['total_phases']} phases")

    def _load_all_training_data(self) -> Dict[str, Any]:
        """Load all available training datasets."""
        return {
            'ik_solutions': self.dataset_integration.load_ik_training_data(),
            'manipulation_demos': self.dataset_integration.load_manipulation_training_data(),
            'dataset_summary': self.dataset_integration.get_integration_summary()
        }

    def train_phase(self, phase_id: int, num_episodes: int = None) -> Dict[str, Any]:
        """Train a specific curriculum phase."""

        # Get phase configuration
        if phase_id <= len(self.curriculum_manager.curriculum_phases):
            phase = self.curriculum_manager.curriculum_phases[phase_id - 1]
        else:
            return {"error": f"Phase {phase_id} not found"}

        episodes_to_run = num_episodes or phase.get("episodes", 100)

        print(f"🚀 Starting Phase {phase_id}: {phase['name']}")
        print(f"   Episodes: {episodes_to_run}")
        print(f"   LLM Role: {phase['llm_involvement']}")

        # Prepare phase data
        phase_data = self._prepare_phase_data(phase)

        # Execute training episodes
        phase_results = self._execute_phase_training(phase, phase_data, episodes_to_run)

        # Check completion criteria
        completion_check = self.curriculum_manager.check_phase_completion(phase_results)

        return {
            "phase_id": phase_id,
            "phase_name": phase["name"],
            "episodes_completed": episodes_to_run,
            "results": phase_results,
            "phase_complete": completion_check,
            "can_advance": completion_check
        }

    def _prepare_phase_data(self, phase: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for phase training."""

        data_sources = phase.get("data_sources", [])
        phase_data = {"sources": data_sources}

        # Add relevant training data based on sources
        for source in data_sources:
            if source in self.training_data:
                phase_data[source] = self.training_data[source]

        return phase_data

    def _execute_phase_training(self, phase: Dict, phase_data: Dict, episodes: int) -> Dict[str, Any]:
        """Execute training for a complete phase."""

        episode_results = []
        total_reward = 0.0
        success_count = 0

        for episode_num in range(episodes):
            episode_result = self.training_executor.execute_training_episode(
                phase, phase_data, episode_num
            )

            episode_results.append(episode_result)

            if episode_result.get("success", False):
                success_count += 1

            total_reward += episode_result.get("reward", 0.0)

            # Progress logging
            if episode_num % 100 == 0:
                avg_reward = total_reward / (episode_num + 1)
                success_rate = success_count / (episode_num + 1)
                print(f"   Episode {episode_num}: Avg Reward {avg_reward:.3f}, Success Rate {success_rate:.2f}")

        # Calculate phase metrics
        avg_reward = total_reward / episodes
        success_rate = success_count / episodes

        return {
            "episodes": episodes,
            "average_reward": avg_reward,
            "success_rate": success_rate,
            "total_reward": total_reward,
            "episode_results": episode_results[-10:],  # Keep last 10 for analysis
            "reward_threshold": phase.get("success_criteria", {}).get("reward_threshold", 0.5)
        }

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        curriculum_status = self.curriculum_manager.get_curriculum_summary()
        training_status = self.training_executor.get_training_summary()

        return {
            "curriculum": curriculum_status,
            "training": training_status,
            "total_datasets": len(self.training_data),
            "pipeline_active": True
        }

# Export for backward compatibility
__all__ = ['LLMGuidedTrainingPipeline', 'CurriculumManager', 'TrainingExecutor']
