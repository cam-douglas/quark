#!/usr/bin/env python3
"""Inference Predictor for Morphogen Gradients.

Core prediction engine for morphogen gradient inference including tensor
preparation, model execution, and result conversion for real-time prediction.

Integration: Prediction engine component for inference pipeline
Rationale: Focused prediction logic separated from main inference coordinator
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
import logging

from .spatial_grid import SpatialGrid
from .unet3d_backbone import UNet3DBackbone
from .ddpm_scheduler import DDPMScheduler

logger = logging.getLogger(__name__)

class InferencePredictor:
    """Core prediction engine for morphogen gradient inference.
    
    Handles tensor preparation, model execution, and result conversion
    for real-time morphogen concentration prediction using trained
    diffusion models.
    """
    
    def __init__(self, model: UNet3DBackbone, scheduler: DDPMScheduler,
                 spatial_grid: SpatialGrid, device: torch.device):
        """Initialize inference predictor.
        
        Args:
            model: Trained UNet3D model
            scheduler: DDPM scheduler
            spatial_grid: Spatial grid for morphogen storage
            device: Computation device
        """
        self.model = model
        self.scheduler = scheduler
        self.grid = spatial_grid
        self.device = device
        
        logger.info("Initialized InferencePredictor")
    
    def predict_morphogen_concentrations(self, input_conditions: Dict[str, Any],
                                        target_resolution: int) -> torch.Tensor:
        """Predict morphogen concentrations using diffusion model.
        
        Args:
            input_conditions: Input conditions for prediction
            target_resolution: Target spatial resolution
            
        Returns:
            Predicted concentration tensor
        """
        # Prepare input tensor
        input_tensor = self._prepare_input_tensor(input_conditions, target_resolution)
        
        # Run inference
        with torch.no_grad():
            prediction = self._run_fast_inference(input_tensor)
        
        return prediction
    
    def _prepare_input_tensor(self, input_conditions: Dict[str, Any], 
                             resolution: int) -> torch.Tensor:
        """Prepare input tensor for inference."""
        # Create input tensor with current morphogen concentrations
        input_data = np.zeros((1, 4, resolution, resolution, resolution))  # Batch size 1
        
        # Fill with current concentrations (resized to target resolution)
        morphogens = ['SHH', 'BMP', 'WNT', 'FGF']
        for i, morphogen in enumerate(morphogens):
            if self.grid.has_morphogen(morphogen):
                concentration = self.grid.get_morphogen_concentration(morphogen)
                # Resize to target resolution
                resized = self._resize_3d(concentration, (resolution, resolution, resolution))
                input_data[0, i, :, :, :] = resized
        
        return torch.from_numpy(input_data).float().to(self.device)
    
    def _resize_3d(self, data: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """Resize 3D data to target shape."""
        # Convert to torch tensor for interpolation
        data_tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float()
        
        # Interpolate
        resized_tensor = F.interpolate(
            data_tensor, size=target_shape, mode='trilinear', align_corners=False
        )
        
        return resized_tensor.squeeze(0).squeeze(0).numpy()
    
    def _run_fast_inference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run fast inference with reduced timesteps."""
        # Generate prediction using fewer diffusion steps for speed
        fast_timesteps = 50  # Reduced from 1000 for real-time inference
        
        # Start from input as initial guess (not pure noise)
        sample = input_tensor.clone()
        
        # Add some noise for diversity
        noise_level = 0.1
        noise = torch.randn_like(sample) * noise_level
        sample = sample + noise
        
        # Fast sampling loop
        timestep_skip = self.scheduler.timesteps // fast_timesteps
        
        for i in range(fast_timesteps):
            t = (fast_timesteps - i - 1) * timestep_skip
            timestep_tensor = torch.full((sample.shape[0],), t, device=self.device, dtype=torch.long)
            
            # Model prediction
            model_output = self.model(sample, timestep_tensor)
            
            # Denoising step
            sample = self.scheduler.step(model_output, t, sample)
        
        return sample
    
    def convert_to_morphogen_maps(self, prediction: torch.Tensor) -> Dict[str, np.ndarray]:
        """Convert model prediction to morphogen concentration maps."""
        prediction_np = prediction.cpu().numpy()[0]  # Remove batch dimension
        
        # Resize back to grid resolution if needed
        grid_shape = (self.grid.dimensions.x_size, self.grid.dimensions.y_size, self.grid.dimensions.z_size)
        
        morphogen_maps = {}
        morphogens = ['SHH', 'BMP', 'WNT', 'FGF']
        
        for i, morphogen in enumerate(morphogens):
            concentration = prediction_np[i]
            
            # Resize to grid resolution
            if concentration.shape != grid_shape:
                concentration = self._resize_3d(concentration, grid_shape)
            
            # Ensure valid concentration range
            concentration = np.clip(concentration, 0.0, 2.0)
            
            morphogen_maps[morphogen] = concentration
        
        return morphogen_maps
    
    def validate_prediction_quality(self, prediction: torch.Tensor,
                                   ground_truth: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Validate quality of morphogen predictions.
        
        Args:
            prediction: Predicted concentrations
            ground_truth: Ground truth concentrations (optional)
            
        Returns:
            Quality metrics
        """
        quality_metrics = {}
        
        # Basic prediction statistics
        prediction_np = prediction.cpu().numpy()
        
        quality_metrics.update({
            "prediction_mean": float(np.mean(prediction_np)),
            "prediction_std": float(np.std(prediction_np)),
            "prediction_min": float(np.min(prediction_np)),
            "prediction_max": float(np.max(prediction_np)),
            "non_negative_fraction": float(np.sum(prediction_np >= 0) / prediction_np.size)
        })
        
        # Comparison with ground truth if available
        if ground_truth is not None:
            gt_np = ground_truth.cpu().numpy()
            
            # Calculate error metrics
            mse = np.mean((prediction_np - gt_np)**2)
            mae = np.mean(np.abs(prediction_np - gt_np))
            
            # Calculate correlation
            correlation = np.corrcoef(prediction_np.flatten(), gt_np.flatten())[0, 1]
            
            quality_metrics.update({
                "mse_vs_ground_truth": float(mse),
                "mae_vs_ground_truth": float(mae),
                "correlation_vs_ground_truth": float(correlation) if not np.isnan(correlation) else 0.0
            })
        
        return quality_metrics
