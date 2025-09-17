#!/usr/bin/env python3
"""DDPM Scheduler Implementation.

Implements Denoising Diffusion Probabilistic Model (DDPM) scheduler for
diffusion-based morphogen concentration generation with noise scheduling
and sampling algorithms.

Integration: Scheduler component for ML diffusion system
Rationale: Focused DDPM scheduler implementation separated from training logic
"""

from typing import Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
import logging

from .ml_diffusion_types import DiffusionTrainingConfig

logger = logging.getLogger(__name__)

class DDPMScheduler:
    """DDPM scheduler for diffusion model training and inference.
    
    Implements noise scheduling, forward diffusion process, and
    reverse sampling for denoising diffusion probabilistic models
    applied to morphogen concentration generation.
    """
    
    def __init__(self, config: DiffusionTrainingConfig):
        """Initialize DDPM scheduler.
        
        Args:
            config: Diffusion training configuration
        """
        self.config = config
        self.timesteps = config.timesteps
        
        # Initialize noise schedule
        self.betas = self._create_noise_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Precompute values for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)
        
        # Posterior variance
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        logger.info("Initialized DDPMScheduler")
        logger.info(f"Timesteps: {self.timesteps}, beta range: {config.beta_start:.4f} to {config.beta_end:.4f}")
    
    def _create_noise_schedule(self) -> torch.Tensor:
        """Create noise schedule (beta values)."""
        if self.config.model_type.value == "ddpm":
            # Linear schedule
            return torch.linspace(self.config.beta_start, self.config.beta_end, self.timesteps)
        else:
            # Cosine schedule (alternative)
            steps = torch.arange(self.timesteps + 1, dtype=torch.float32) / self.timesteps
            alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2) ** 2
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(self, x_start: torch.Tensor, noise: torch.Tensor, 
                  timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to clean data (forward diffusion process).
        
        Args:
            x_start: Clean data
            noise: Random noise
            timesteps: Diffusion timesteps
            
        Returns:
            Noisy data at specified timesteps
        """
        # Get noise schedule values for timesteps
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
        
        # Apply noise: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps for training.
        
        Args:
            batch_size: Batch size
            device: Device for tensor
            
        Returns:
            Random timesteps tensor
        """
        return torch.randint(0, self.timesteps, (batch_size,), device=device)
    
    def step(self, model_output: torch.Tensor, timestep: int, 
             sample: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Perform one denoising step.
        
        Args:
            model_output: Model prediction
            timestep: Current timestep
            sample: Current sample
            generator: Random generator
            
        Returns:
            Denoised sample
        """
        t = timestep
        
        # Get schedule values
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]
        
        # Predict original sample from noise
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        
        # Compute mean of reverse process
        pred_original_sample = (sample - sqrt_one_minus_alpha_bar_t * model_output) / sqrt_alpha_bar_t
        
        # Compute coefficients for mean
        pred_sample_direction = (1 - alpha_t) / sqrt_one_minus_alpha_bar_t
        prev_sample_mean = (sample - pred_sample_direction * model_output) / torch.sqrt(alpha_t)
        
        # Add noise if not final step
        if t > 0:
            variance = self.posterior_variance[t]
            noise = torch.randn_like(sample, generator=generator)
            prev_sample = prev_sample_mean + torch.sqrt(variance) * noise
        else:
            prev_sample = prev_sample_mean
        
        return prev_sample
    
    def sample(self, model: torch.nn.Module, shape: Tuple[int, ...], 
               device: torch.device, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Generate samples using reverse diffusion process.
        
        Args:
            model: Trained diffusion model
            shape: Shape of samples to generate
            device: Device for computation
            generator: Random generator
            
        Returns:
            Generated samples
        """
        logger.info(f"Generating samples with shape {shape}")
        
        # Start from pure noise
        sample = torch.randn(shape, device=device, generator=generator)
        
        # Reverse diffusion process
        model.eval()
        with torch.no_grad():
            for t in reversed(range(self.timesteps)):
                # Model prediction
                timestep_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
                model_output = model(sample, timestep_tensor)
                
                # Denoising step
                sample = self.step(model_output, t, sample, generator)
                
                if t % 100 == 0:
                    logger.debug(f"Sampling step {t}, sample range: [{sample.min():.3f}, {sample.max():.3f}]")
        
        return sample
    
    def get_velocity(self, model_output: torch.Tensor, sample: torch.Tensor, 
                    timesteps: torch.Tensor) -> torch.Tensor:
        """Get velocity prediction for velocity parameterization.
        
        Args:
            model_output: Model prediction
            sample: Current sample
            timesteps: Timesteps tensor
            
        Returns:
            Velocity prediction
        """
        # Convert epsilon prediction to velocity
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
        
        # v = alpha * noise - sigma * x_start
        velocity = sqrt_alphas_cumprod_t * model_output - sqrt_one_minus_alphas_cumprod_t * sample
        
        return velocity
    
    def compute_loss(self, model: torch.nn.Module, x_start: torch.Tensor, 
                    noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute training loss for DDPM.
        
        Args:
            model: Model to train
            x_start: Clean data
            noise: Noise tensor (generated if None)
            
        Returns:
            Training loss
        """
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # Sample noise
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Sample timesteps
        timesteps = self.sample_timesteps(batch_size, device)
        
        # Add noise to data
        x_noisy = self.add_noise(x_start, noise, timesteps)
        
        # Model prediction
        model_output = model(x_noisy, timesteps)
        
        # Compute loss (MSE between predicted and actual noise)
        loss = F.mse_loss(model_output, noise)
        
        return loss
