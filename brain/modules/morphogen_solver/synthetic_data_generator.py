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
from .parameter_types import DiffusionParameters
from .parameter_calculator import ParameterCalculator

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
        self.calculator = ParameterCalculator()
        
        # Parameter variation ranges
        self.parameter_ranges = self._initialize_parameter_ranges()
        
        logger.info("Initialized SyntheticEmbryoDataGenerator")
        logger.info(f"Target samples: {data_config.num_samples}")
        logger.info(f"Spatial resolution: {base_grid_dimensions.resolution} µm")
    
    def _initialize_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Initialize parameter variation ranges for synthetic data.
        
        Note: Ranges are +/- 50% of the validated defaults to ensure stability.
        """
        return {
            # SHH: default diffusion=0.1, degradation=0.000128
            "shh_diffusion": (0.05, 0.15),
            "shh_degradation": (0.000064, 0.000192),
            # BMP: default diffusion=0.5, degradation=0.002
            "bmp_diffusion": (0.25, 0.75),
            "bmp_degradation": (0.001, 0.003),
            # WNT: default diffusion=0.3, degradation=0.0015
            "wnt_diffusion": (0.15, 0.45),
            "wnt_degradation": (0.00075, 0.00225),
            # FGF: default diffusion=1.0, degradation=0.003
            "fgf_diffusion": (0.5, 1.5),
            "fgf_degradation": (0.0015, 0.0045),
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
        # Create morphogen solver
        solver = MorphogenSolver(self.base_dims, species="mouse", stage="E8.5-E10.5")

        # Override parameters for this specific simulation run
        for param_name, value in varied_params.items():
            morphogen, param_type = param_name.split('_')
            morphogen = morphogen.upper()
            
            # Create a new DiffusionParameters object with the varied value
            original_params = solver.bio_params.get_diffusion_parameters(morphogen)
            new_params_dict = original_params.__dict__.copy()
            
            if param_type == 'diffusion':
                new_params_dict['diffusion_coefficient'] = value
            elif param_type == 'degradation':
                new_params_dict['degradation_rate'] = value
                # Recalculate half-life to maintain consistency
                if value > 0:
                    new_params_dict['half_life'] = np.log(2) / value
                else:
                    new_params_dict['half_life'] = float('inf')
            
            new_diffusion_params = DiffusionParameters(**new_params_dict)
            solver.override_diffusion_parameters(morphogen, new_diffusion_params)

        # Configure neural tube geometry
        neural_tube_config = {
            'neural_tube_length': self.base_dims.y_size * self.base_dims.resolution,
            'neural_tube_width': self.base_dims.x_size * self.base_dims.resolution,
            'neural_tube_height': self.base_dims.z_size * self.base_dims.resolution
        }
        solver.configure_neural_tube(**neural_tube_config)
        
        # Calculate a conservative, fixed timestep based on the fastest diffusing morphogen (FGF)
        fgf_params = solver.bio_params.get_diffusion_parameters('FGF')
        dt = self.calculator.optimize_time_step(
            diffusion_coefficient=fgf_params.diffusion_coefficient,
            grid_spacing=self.base_dims.resolution,
            degradation_rate=fgf_params.degradation_rate,
            safety_factor=0.1  # Use a very conservative safety factor
        )
        
        # Run simulation for a short time to establish gradients
        simulation_time = 3600.0  # 1 hour simulation
        solver.simulate_morphogen_dynamics(simulation_time, dt)
        
        # Extract all four simulated concentration fields
        morphogen_fields = {
            'SHH': solver.spatial_grid.get_morphogen_concentration('SHH'),
            'BMP': solver.spatial_grid.get_morphogen_concentration('BMP'),
            'WNT': solver.spatial_grid.get_morphogen_concentration('WNT'),
            'FGF': solver.spatial_grid.get_morphogen_concentration('FGF')
        }
        
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
