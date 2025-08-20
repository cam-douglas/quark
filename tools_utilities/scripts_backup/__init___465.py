"""
SmallMind: Integrated Computational Neuroscience & AI Platform

A comprehensive system for brain development simulation, AI model training,
and multi-modal data integration across neuroscience and machine learning.
"""

# Core modules
from ........................................................... import baby_agi
from ........................................................... import cli
from ........................................................... import config
from ........................................................... import data_knowledge.datasets_knowledge.datasets_knowledge.datasets
from ........................................................... import demos
from ........................................................... import integration
from ........................................................... import ml_optimization
from ........................................................... import models
from ........................................................... import neurodata
from ........................................................... import physics_simulation
from ........................................................... import requirements
from ........................................................... import scripts
from ........................................................... import simulation
from ........................................................... import testing.test_suites
from ........................................................... import tools
from ........................................................... import visualization

# Optional dependency handling with silent fallbacks
def _init_optional_dependencies():
    """Initialize optional dependencies with silent fallbacks"""
    global NEST_AVAILABLE, VISIT_AVAILABLE, MUJOCO_AVAILABLE
    
    # NEST Simulator
    try:
        import nest
        NEST_AVAILABLE = True
    except ImportError:
        NEST_AVAILABLE = False
    
    # VisIt Visualization
    try:
        import visit
        VISIT_AVAILABLE = True
    except ImportError:
        VISIT_AVAILABLE = False
    
    # MuJoCo Physics
    try:
        import mujoco
        MUJOCO_AVAILABLE = True
    except ImportError:
        MUJOCO_AVAILABLE = False

# Initialize optional dependencies
_init_optional_dependencies()

# These can be imported directly when needed

# Dataset integration components
try:
    from .....................................................models.dataset_integration import (
        DatasetIntegrator, DatasetConfig, TrainingMixture,
        get_dataset_integrator, create_balanced_mixture, create_code_focused_mixture
    )
    DATASET_INTEGRATION_AVAILABLE = True
except ImportError:
    DATASET_INTEGRATION_AVAILABLE = False

# Training pipeline components
try:
    from .....................................................models.training_pipeline import (
        SmallMindTrainer, TrainingConfig, get_trainer, quick_train
    )
    TRAINING_PIPELINE_AVAILABLE = True
except ImportError:
    TRAINING_PIPELINE_AVAILABLE = False

# Cloud integration components
try:
    from .....................................................models.cloud_integration import (
        CloudTrainer, CloudConfig, CloudInstance,
        create_aws_trainer, create_gcp_trainer
    )
    CLOUD_INTEGRATION_AVAILABLE = True
except ImportError:
    CLOUD_INTEGRATION_AVAILABLE = False

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
    # Cloud integration
    "CloudTrainer",
    "CloudConfig", 
    "CloudInstance",
    "create_aws_trainer",
    "create_gcp_trainer",
    # Availability flags
    "DATASET_INTEGRATION_AVAILABLE",
    "TRAINING_PIPELINE_AVAILABLE",
    "CLOUD_INTEGRATION_AVAILABLE",
    # Optional dependencies
    "NEST_AVAILABLE",
    "VISIT_AVAILABLE", 
    "MUJOCO_AVAILABLE"
]

# Version info
__version__ = "1.0.0"
__author__ = "SmallMind Team"
__description__ = "Integrated Computational Neuroscience & AI Platform"
