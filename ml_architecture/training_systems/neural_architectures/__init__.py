"""
MoE Neuroscience Expert System Package

Contains the core MoE routing, management, and neuroscience expert components.
"""

# Import only the core components to avoid circular imports
from ...........................................................neuroscience_experts import NeuroscienceExpertManager, NeuroscienceTask, NeuroscienceTaskType

# Import MoE components separately to avoid circular dependencies
# These can be imported directly when needed

# Dataset integration components
try:
    from ...........................................................dataset_integration import (
        DatasetIntegrator, DatasetConfig, TrainingMixture,
        get_dataset_integrator, create_balanced_mixture, create_code_focused_mixture
    )
    DATASET_INTEGRATION_AVAILABLE = True
except ImportError:
    DATASET_INTEGRATION_AVAILABLE = False
    print("Warning: Dataset integration components not available")

# Training pipeline components
try:
    from ...........................................................training_pipeline import (
        SmallMindTrainer, TrainingConfig, get_trainer, quick_train
    )
    TRAINING_PIPELINE_AVAILABLE = True
except ImportError:
    TRAINING_PIPELINE_AVAILABLE = False
    print("Warning: Training pipeline components not available")

__all__ = [
    "NeuroscienceExpertManager",
    "NeuroscienceTask",
    "NeuroscienceTaskType",
    # Dataset integration
    "DatasetIntegrator",
    "DatasetConfig", 
    "TrainingMixture",
    "get_dataset_integrator",
    "create_balanced_mixture",
    "create_code_focused_mixture",
    # Training pipeline
    "SmallMindTrainer",
    "TrainingConfig",
    "get_trainer",
    "quick_train",
    # Availability flags
    "DATASET_INTEGRATION_AVAILABLE",
    "TRAINING_PIPELINE_AVAILABLE"
]
