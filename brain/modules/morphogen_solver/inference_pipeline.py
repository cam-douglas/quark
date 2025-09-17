#!/usr/bin/env python3
"""Diffusion Model Inference Pipeline.

Main coordinator for optimized inference pipeline for real-time morphogen
gradient prediction including multi-scale resolution handling and GPU
acceleration for enhanced morphogen solver performance.

Integration: Inference component for ML diffusion system
Rationale: Main inference coordinator with focused responsibilities
"""

from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn.functional as F
import numpy as np
import logging
import time
from pathlib import Path

from .spatial_grid import SpatialGrid
from .diffusion_model_trainer import DiffusionModelTrainer
from .unet3d_backbone import UNet3DBackbone
from .ddpm_scheduler import DDPMScheduler
from .ml_diffusion_types import DiffusionTrainingConfig
from .inference_optimizer import InferenceOptimizer
from .inference_predictor import InferencePredictor

logger = logging.getLogger(__name__)

class InferencePipeline:
    """Optimized inference pipeline for morphogen gradient prediction.
    
    Main coordinator for real-time morphogen gradient prediction with
    multi-scale resolution handling, GPU acceleration, and integration
    with the existing morphogen solver system.
    """
    
    def __init__(self, model_path: str, spatial_grid: SpatialGrid,
                 enable_gpu_acceleration: bool = True):
        """Initialize inference pipeline.
        
        Args:
            model_path: Path to trained diffusion model
            spatial_grid: Spatial grid for morphogen storage
            enable_gpu_acceleration: Whether to enable GPU acceleration
        """
        self.model_path = Path(model_path)
        self.grid = spatial_grid
        
        # Initialize optimizer
        self.optimizer = InferenceOptimizer(enable_gpu_acceleration)
        self.device = self.optimizer.device
        
        # Initialize components
        self.model: Optional[UNet3DBackbone] = None
        self.scheduler: Optional[DDPMScheduler] = None
        self.predictor: Optional[InferencePredictor] = None
        
        # Performance optimization state
        self.prediction_cache = self.optimizer.create_inference_cache()
        self.multi_scale_models: Dict[int, UNet3DBackbone] = {}
        
        # Load model
        self._load_trained_model()
        
        logger.info("Initialized InferencePipeline")
        logger.info(f"Device: {self.device}")
        logger.info(f"GPU acceleration: {self.optimizer.enable_gpu}")
    
    def _load_trained_model(self) -> None:
        """Load trained diffusion model."""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract config
            config = checkpoint["config"]
            
            # Initialize model
            self.model = UNet3DBackbone(
                input_channels=config.input_channels,
                output_channels=config.output_channels,
                architecture=config.unet_architecture
            ).to(self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint["model_state_dict"])
            
            # Optimize model
            optimization_results = self.optimizer.optimize_model_for_inference(self.model)
            self.model = optimization_results["optimized_model"]
            
            # Initialize scheduler
            self.scheduler = DDPMScheduler(config)
            
            # Initialize predictor
            self.predictor = InferencePredictor(self.model, self.scheduler, self.grid, self.device)
            
            logger.info(f"Loaded and optimized model from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict_morphogen_gradients(self, input_conditions: Dict[str, Any],
                                   target_resolution: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Predict morphogen gradients using trained diffusion model."""
        if self.predictor is None:
            raise ValueError("Predictor not initialized")
        
        logger.info("Predicting morphogen gradients")
        start_time = time.time()
        
        # Determine target resolution
        if target_resolution is None:
            target_resolution = self._determine_optimal_resolution(input_conditions)
        
        # Check cache first
        cache_key = self._generate_cache_key(str(input_conditions), target_resolution)
        if cache_key in self.prediction_cache["cache_storage"]:
            logger.debug("Using cached prediction")
            morphogen_predictions = self.prediction_cache["cache_storage"][cache_key]
            self.optimizer.update_cache_statistics(self.prediction_cache, cache_hit=True)
        else:
            # Run prediction
            prediction = self.predictor.predict_morphogen_concentrations(input_conditions, target_resolution)
            
            # Convert to morphogen maps
            morphogen_predictions = self.predictor.convert_to_morphogen_maps(prediction)
            
            # Cache result
            self._update_cache(cache_key, morphogen_predictions)
            self.optimizer.update_cache_statistics(self.prediction_cache, cache_hit=False)
        
        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.3f}s")
        
        return morphogen_predictions
    
    def _determine_optimal_resolution(self, input_conditions: Dict[str, Any]) -> int:
        """Determine optimal resolution for inference."""
        # Base resolution on grid size and available GPU memory
        base_resolution = 64
        
        if self.optimizer.enable_gpu:
            # Check GPU memory
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                if gpu_memory > 8e9:  # >8GB
                    base_resolution = 128
                elif gpu_memory > 4e9:  # >4GB
                    base_resolution = 96
        
        # Adjust based on grid dimensions
        grid_size = max(self.grid.dimensions.x_size, self.grid.dimensions.y_size, self.grid.dimensions.z_size)
        
        if grid_size < 64:
            target_resolution = min(base_resolution, grid_size)
        else:
            target_resolution = base_resolution
        
        return target_resolution
    
    def _generate_cache_key(self, input_str: str, resolution: int) -> str:
        """Generate cache key for input conditions."""
        # Simple hash based on input string and resolution
        cache_hash = hash((input_str, resolution))
        return f"prediction_{abs(cache_hash)}"
    
    def _update_cache(self, cache_key: str, prediction: Dict[str, np.ndarray]) -> None:
        """Update prediction cache."""
        cache_storage = self.prediction_cache["cache_storage"]
        
        # Remove oldest entries if cache is full
        if len(cache_storage) >= self.prediction_cache["max_size"]:
            oldest_key = next(iter(cache_storage))
            del cache_storage[oldest_key]
        
        cache_storage[cache_key] = prediction.copy()
        self.prediction_cache["current_size"] = len(cache_storage)
    
    def create_multi_scale_predictions(self, input_conditions: Dict[str, Any],
                                     scales: List[int] = [32, 64, 128]) -> Dict[int, Dict[str, np.ndarray]]:
        """Create predictions at multiple spatial scales."""
        logger.info(f"Creating multi-scale predictions: {scales}")
        
        multi_scale_predictions = {}
        
        for scale in scales:
            try:
                # Predict at this scale
                predictions = self.predict_morphogen_gradients(
                    input_conditions, target_resolution=scale
                )
                multi_scale_predictions[scale] = predictions
                
                logger.debug(f"Scale {scale}: prediction completed")
                
            except Exception as e:
                logger.warning(f"Failed to predict at scale {scale}: {e}")
                continue
        
        logger.info(f"Multi-scale predictions completed: {len(multi_scale_predictions)} scales")
        
        return multi_scale_predictions
    
    def enhance_morphogen_solver(self, morphogen_solver, enhancement_strength: float = 0.1) -> Dict[str, Any]:
        """Enhance morphogen solver predictions with ML model."""
        logger.info("Enhancing morphogen solver with ML predictions")
        
        # Get current morphogen concentrations
        current_concentrations = {}
        for morphogen in ['SHH', 'BMP', 'WNT', 'FGF']:
            if morphogen_solver.spatial_grid.has_morphogen(morphogen):
                current_concentrations[morphogen] = morphogen_solver.spatial_grid.get_morphogen_concentration(morphogen)
        
        # Get ML predictions
        ml_predictions = self.predict_morphogen_gradients(current_concentrations)
        
        # Blend current and predicted concentrations
        enhanced_concentrations = {}
        for morphogen in current_concentrations:
            if morphogen in ml_predictions:
                current = current_concentrations[morphogen]
                predicted = ml_predictions[morphogen]
                
                # Weighted blend
                enhanced = (1.0 - enhancement_strength) * current + enhancement_strength * predicted
                enhanced_concentrations[morphogen] = enhanced
                
                # Update solver
                morphogen_solver.spatial_grid.set_morphogen_concentration(morphogen, enhanced)
        
        # Calculate enhancement metrics
        enhancement_metrics = {}
        for morphogen in enhanced_concentrations:
            current = current_concentrations[morphogen]
            enhanced = enhanced_concentrations[morphogen]
            
            # Calculate improvement metrics
            mse_improvement = np.mean((current - enhanced)**2)
            gradient_enhancement = np.mean(np.gradient(enhanced)) / np.mean(np.gradient(current)) if np.mean(np.gradient(current)) > 0 else 1.0
            
            enhancement_metrics[morphogen] = {
                "mse_change": float(mse_improvement),
                "gradient_enhancement": float(gradient_enhancement),
                "concentration_range": (float(np.min(enhanced)), float(np.max(enhanced)))
            }
        
        results = {
            "enhanced_morphogens": list(enhanced_concentrations.keys()),
            "enhancement_strength": enhancement_strength,
            "enhancement_metrics": enhancement_metrics,
            "solver_enhanced": True
        }
        
        logger.info(f"Enhanced {len(enhanced_concentrations)} morphogen fields")
        
        return results
    
    def export_pipeline_analysis(self) -> Dict[str, Any]:
        """Export comprehensive pipeline analysis."""
        # Benchmark performance
        input_shape = (1, 4, 64, 64, 64)  # Standard test shape
        performance_metrics = self.optimizer.benchmark_inference_performance(
            self.model, input_shape, num_runs=10)
        
        # Memory optimization
        memory_results = self.optimizer.optimize_memory_usage(self.model)
        
        # Model information
        model_info = {}
        if self.model is not None:
            model_summary = self.model.get_model_summary()
            model_info = {
                "model_loaded": True,
                "model_parameters": model_summary["total_parameters"],
                "input_channels": model_summary["input_channels"],
                "output_channels": model_summary["output_channels"],
                "has_attention": model_summary["has_attention"]
            }
        
        pipeline_analysis = {
            "model_path": str(self.model_path),
            "device": str(self.device),
            "gpu_acceleration": self.optimizer.enable_gpu,
            "model_info": model_info,
            "performance_metrics": performance_metrics,
            "memory_optimization": memory_results,
            "cache_configuration": {
                "cache_enabled": True,
                "cache_max_size": self.prediction_cache["max_size"],
                "current_cache_size": self.prediction_cache["current_size"],
                "cache_hit_rate": self.prediction_cache["hit_rate"]
            },
            "multi_scale_support": {
                "available_scales": list(range(32, 129, 32)),
                "optimal_resolution": self._determine_optimal_resolution({})
            }
        }
        
        return pipeline_analysis