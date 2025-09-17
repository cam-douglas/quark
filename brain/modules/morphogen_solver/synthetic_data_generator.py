#!/usr/bin/env python3
"""Synthetic Embryo Data Generator.

Generates synthetic embryo data for diffusion model training including
morphogen concentration fields with biological constraints and variations.

Integration: Data generation component for ML diffusion system
Rationale: Focused synthetic data generation separated from training logic
"""

from typing import Dict, List, Tuple
import numpy as np
import logging

from .spatial_grid import SpatialGrid, GridDimensions
from .morphogen_solver import MorphogenSolver
from .ml_diffusion_types import SyntheticDataConfig

logger = logging.getLogger(__name__)

class SyntheticEmbryoDataGenerator:
    """Generator for synthetic embryo morphogen data.
    
    Creates realistic synthetic embryo data by running morphogen simulations
    with varied parameters and adding controlled noise for ML training.
    """
    
    def __init__(self, base_grid_dimensions: GridDimensions,
                 data_config: SyntheticDataConfig):
        """Initialize synthetic data generator.
        
        Args:
            base_grid_dimensions: Base grid dimensions for simulations
            data_config: Data generation configuration
        """
        self.base_dims = base_grid_dimensions
        self.config = data_config
        
        # Parameter variation ranges
        self.parameter_ranges = self._initialize_parameter_ranges()
        
        logger.info("Initialized SyntheticEmbryoDataGenerator")
        logger.info(f"Target samples: {data_config.num_samples}")
        logger.info(f"Spatial resolution: {base_grid_dimensions.resolution} µm")
    
    def _initialize_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Initialize parameter variation ranges for synthetic data."""
        return {
            "shh_diffusion": (50.0, 150.0),      # µm²/s
            "shh_degradation": (0.001, 0.01),    # s⁻¹
            "bmp_diffusion": (30.0, 100.0),      # µm²/s
            "bmp_degradation": (0.002, 0.015),   # s⁻¹
            "wnt_diffusion": (20.0, 80.0),       # µm²/s
            "wnt_degradation": (0.0015, 0.008),  # s⁻¹
            "fgf_diffusion": (40.0, 120.0),      # µm²/s
            "fgf_degradation": (0.003, 0.012),   # s⁻¹
        }
    
    def generate_synthetic_dataset(self) -> Dict[str, np.ndarray]:
        """Generate complete synthetic dataset.
        
        Returns:
            Dictionary with synthetic morphogen concentration data
        """
        logger.info(f"Generating {self.config.num_samples} synthetic samples")
        
        # Initialize arrays for storing data
        spatial_dims = (self.config.num_samples, 4,  # 4 morphogens
                       self.base_dims.x_size, self.base_dims.y_size, self.base_dims.z_size)
        
        morphogen_data = np.zeros(spatial_dims)
        parameter_data = []
        
        # Generate samples
        for i in range(self.config.num_samples):
            if i % 100 == 0:
                logger.info(f"Generating sample {i+1}/{self.config.num_samples}")
            
            # Generate varied parameters
            varied_params = self._generate_varied_parameters()
            parameter_data.append(varied_params)
            
            # Run morphogen simulation with varied parameters
            morphogen_fields = self._simulate_morphogen_sample(varied_params)
            
            # Apply noise and biological constraints
            if self.config.noise_level > 0:
                morphogen_fields = self._apply_noise(morphogen_fields)
            
            if self.config.biological_constraints:
                morphogen_fields = self._apply_biological_constraints(morphogen_fields)
            
            # Store sample
            morphogen_data[i, 0, :, :, :] = morphogen_fields['SHH']
            morphogen_data[i, 1, :, :, :] = morphogen_fields['BMP']
            morphogen_data[i, 2, :, :, :] = morphogen_fields['WNT']
            morphogen_data[i, 3, :, :, :] = morphogen_fields['FGF']
        
        # Normalize data to specified range
        morphogen_data = self._normalize_data(morphogen_data)
        
        dataset = {
            "morphogen_concentrations": morphogen_data,
            "parameters": parameter_data,
            "metadata": {
                "num_samples": self.config.num_samples,
                "spatial_resolution": self.base_dims.resolution,
                "grid_dimensions": (self.base_dims.x_size, self.base_dims.y_size, self.base_dims.z_size),
                "morphogen_order": ["SHH", "BMP", "WNT", "FGF"],
                "concentration_range": self.config.morphogen_range,
                "noise_level": self.config.noise_level
            }
        }
        
        logger.info(f"Generated synthetic dataset: {morphogen_data.shape}")
        
        return dataset
    
    def _generate_varied_parameters(self) -> Dict[str, float]:
        """Generate varied parameters for single sample."""
        varied_params = {}
        
        for param_name, (min_val, max_val) in self.parameter_ranges.items():
            # Random sampling within range
            varied_params[param_name] = np.random.uniform(min_val, max_val)
        
        return varied_params
    
    def _simulate_morphogen_sample(self, varied_params: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Simulate morphogen sample with varied parameters."""
        # Create morphogen solver with varied parameters
        solver = MorphogenSolver(self.base_dims, species="mouse", stage="E8.5-E10.5")
        
        # Configure with varied parameters (simplified approach)
        # In practice, would modify solver parameters based on varied_params
        
        # Configure neural tube
        neural_tube_config = {
            'length_um': self.base_dims.y_size * self.base_dims.resolution,
            'width_um': self.base_dims.x_size * self.base_dims.resolution,
            'height_um': self.base_dims.z_size * self.base_dims.resolution
        }
        
        solver.configure_neural_tube(neural_tube_config)
        
        # Run simulation for short time to establish gradients
        simulation_time = 3600.0  # 1 hour simulation
        dt = 10.0  # 10 second timesteps
        
        solver.simulate_morphogen_dynamics(simulation_time, dt)
        
        # Extract concentration fields
        morphogen_fields = {
            'SHH': solver.spatial_grid.get_morphogen_concentration('SHH'),
            'BMP': solver.spatial_grid.get_morphogen_concentration('BMP'),
            'WNT': np.zeros_like(solver.spatial_grid.get_morphogen_concentration('SHH')),  # Placeholder
            'FGF': np.zeros_like(solver.spatial_grid.get_morphogen_concentration('SHH'))   # Placeholder
        }
        
        # Add WNT/FGF fields (simplified - would use actual WNT/FGF systems)
        dims = self.base_dims
        
        # Simple WNT gradient (high posterior)
        wnt_field = np.zeros((dims.x_size, dims.y_size, dims.z_size))
        for y in range(dims.y_size):
            wnt_field[:, y, :] = (y / dims.y_size) ** 2  # Quadratic posterior increase
        morphogen_fields['WNT'] = wnt_field * varied_params.get('wnt_diffusion', 50.0) / 50.0
        
        # Simple FGF gradient (isthmus peak)
        fgf_field = np.zeros((dims.x_size, dims.y_size, dims.z_size))
        isthmus_pos = int(0.4 * dims.y_size)
        for y in range(dims.y_size):
            # Gaussian peak at isthmus
            fgf_field[:, y, :] = np.exp(-((y - isthmus_pos) / (0.1 * dims.y_size))**2)
        morphogen_fields['FGF'] = fgf_field * varied_params.get('fgf_diffusion', 80.0) / 80.0
        
        return morphogen_fields
    
    def _apply_noise(self, morphogen_fields: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply Gaussian noise to morphogen fields."""
        noisy_fields = {}
        
        for morphogen, field in morphogen_fields.items():
            # Add Gaussian noise
            noise = np.random.normal(0, self.config.noise_level * np.std(field), field.shape)
            noisy_field = field + noise
            
            # Ensure non-negative concentrations
            noisy_field = np.maximum(noisy_field, 0.0)
            
            noisy_fields[morphogen] = noisy_field
        
        return noisy_fields
    
    def _apply_biological_constraints(self, morphogen_fields: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply biological constraints to morphogen fields."""
        constrained_fields = {}
        
        for morphogen, field in morphogen_fields.items():
            constrained_field = field.copy()
            
            # Apply morphogen-specific constraints
            if morphogen == 'SHH':
                # SHH should be highest ventrally (bottom of grid)
                ventral_region = constrained_field[:, :, :constrained_field.shape[2]//3]
                if np.max(ventral_region) < 0.1:
                    # Ensure minimum ventral SHH
                    constrained_field[:, :, :constrained_field.shape[2]//3] += 0.1
                    
            elif morphogen == 'BMP':
                # BMP should be highest dorsally (top of grid)
                dorsal_region = constrained_field[:, :, 2*constrained_field.shape[2]//3:]
                if np.max(dorsal_region) < 0.1:
                    # Ensure minimum dorsal BMP
                    constrained_field[:, :, 2*constrained_field.shape[2]//3:] += 0.1
                    
            elif morphogen == 'WNT':
                # WNT should be highest posteriorly
                posterior_region = constrained_field[:, 2*constrained_field.shape[1]//3:, :]
                if np.max(posterior_region) < 0.1:
                    # Ensure minimum posterior WNT
                    constrained_field[:, 2*constrained_field.shape[1]//3:, :] += 0.1
                    
            elif morphogen == 'FGF':
                # FGF should have isthmic peak
                isthmus_region = constrained_field[:, 
                                 int(0.35*constrained_field.shape[1]):int(0.45*constrained_field.shape[1]), :]
                if np.max(isthmus_region) < 0.2:
                    # Ensure isthmic FGF peak
                    constrained_field[:, 
                                    int(0.35*constrained_field.shape[1]):int(0.45*constrained_field.shape[1]), :] += 0.2
            
            # Apply concentration limits
            constrained_field = np.clip(constrained_field, 0.0, 2.0)  # Max 2 nM
            
            constrained_fields[morphogen] = constrained_field
        
        return constrained_fields
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to specified range."""
        min_val, max_val = self.config.morphogen_range
        
        # Normalize each morphogen channel separately
        normalized_data = np.zeros_like(data)
        
        for i in range(data.shape[1]):  # For each morphogen channel
            channel_data = data[:, i, :, :, :]
            
            # Min-max normalization
            data_min = np.min(channel_data)
            data_max = np.max(channel_data)
            
            if data_max > data_min:
                normalized_channel = (channel_data - data_min) / (data_max - data_min)
                normalized_channel = normalized_channel * (max_val - min_val) + min_val
            else:
                normalized_channel = np.full_like(channel_data, min_val)
            
            normalized_data[:, i, :, :, :] = normalized_channel
        
        return normalized_data
    
    def create_train_val_split(self, dataset: Dict[str, np.ndarray], 
                              val_fraction: float = 0.2) -> Tuple[Dict, Dict]:
        """Create training and validation splits.
        
        Args:
            dataset: Complete dataset
            val_fraction: Fraction for validation set
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        morphogen_data = dataset["morphogen_concentrations"]
        parameter_data = dataset["parameters"]
        
        n_samples = morphogen_data.shape[0]
        n_val = int(n_samples * val_fraction)
        n_train = n_samples - n_val
        
        # Random shuffle
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # Create splits
        train_dataset = {
            "morphogen_concentrations": morphogen_data[train_indices],
            "parameters": [parameter_data[i] for i in train_indices],
            "metadata": dataset["metadata"].copy()
        }
        train_dataset["metadata"]["num_samples"] = n_train
        
        val_dataset = {
            "morphogen_concentrations": morphogen_data[val_indices],
            "parameters": [parameter_data[i] for i in val_indices],
            "metadata": dataset["metadata"].copy()
        }
        val_dataset["metadata"]["num_samples"] = n_val
        
        logger.info(f"Created train/val split: {n_train} train, {n_val} validation")
        
        return train_dataset, val_dataset
