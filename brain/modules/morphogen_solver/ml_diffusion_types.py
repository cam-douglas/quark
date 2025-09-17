#!/usr/bin/env python3
"""ML Diffusion Model Type Definitions.

Type definitions and parameter structures for diffusion-based generative
models for morphogen concentration prediction and enhancement.

Integration: Type definitions for ML diffusion model system
Rationale: Centralized type definitions for diffusion model components
"""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

class DiffusionModelType(Enum):
    """Types of diffusion models."""
    DDPM = "ddpm"                    # Denoising Diffusion Probabilistic Model
    DDIM = "ddim"                    # Denoising Diffusion Implicit Model
    SCORE_SDE = "score_sde"          # Score-based SDE

class UNetArchitecture(Enum):
    """UNet3D architecture variants."""
    STANDARD = "standard"            # Standard UNet3D
    ATTENTION = "attention"          # UNet3D with attention
    RESIDUAL = "residual"           # ResUNet3D

@dataclass
class DiffusionTrainingConfig:
    """Configuration for diffusion model training."""
    model_type: DiffusionModelType
    unet_architecture: UNetArchitecture
    timesteps: int = 1000            # Number of diffusion timesteps
    beta_start: float = 0.0001       # Starting noise schedule
    beta_end: float = 0.02           # Ending noise schedule
    learning_rate: float = 1e-4      # Training learning rate
    batch_size: int = 4              # Training batch size
    num_epochs: int = 100            # Training epochs
    input_channels: int = 4          # SHH, BMP, WNT, FGF channels
    output_channels: int = 4         # Same as input
    spatial_resolution: int = 64     # Spatial resolution (voxels per dim)

@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic embryo data generation."""
    num_samples: int = 1000          # Number of synthetic samples
    noise_level: float = 0.1         # Gaussian noise level
    morphogen_range: Tuple[float, float] = (0.0, 1.0)  # Concentration range
    spatial_variations: bool = True   # Include spatial variations
    temporal_dynamics: bool = True    # Include temporal evolution
    biological_constraints: bool = True  # Apply biological constraints

@dataclass
class TrainingResult:
    """Result of diffusion model training."""
    model_path: str                  # Path to saved model
    training_loss: List[float]       # Training loss history
    validation_loss: List[float]     # Validation loss history
    final_loss: float               # Final training loss
    convergence_achieved: bool       # Whether training converged
    training_time_seconds: float     # Total training time
    model_parameters: int            # Number of model parameters
