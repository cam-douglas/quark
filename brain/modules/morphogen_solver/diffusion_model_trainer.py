#!/usr/bin/env python3
"""Diffusion Model Training System.

Main trainer for diffusion models applied to morphogen concentration
prediction including training loop, validation, and model persistence.

Integration: Training coordinator for ML diffusion system
Rationale: Main training coordinator with focused responsibilities
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from pathlib import Path
import time

from .ml_diffusion_types import DiffusionTrainingConfig, TrainingResult
from .synthetic_data_generator import SyntheticEmbryoDataGenerator
from .unet3d_backbone import UNet3DBackbone
from .ddpm_scheduler import DDPMScheduler

logger = logging.getLogger(__name__)

class DiffusionModelTrainer:
    """Trainer for diffusion models on morphogen data.
    
    Coordinates diffusion model training including data generation,
    model initialization, training loop execution, and validation
    for morphogen concentration prediction enhancement.
    """
    
    def __init__(self, training_config: DiffusionTrainingConfig,
                 model_save_dir: str = "/Users/camdouglas/quark/data/models/diffusion"):
        """Initialize diffusion model trainer.
        
        Args:
            training_config: Training configuration
            model_save_dir: Directory to save trained models
        """
        self.config = training_config
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model: Optional[UNet3DBackbone] = None
        self.scheduler: Optional[DDPMScheduler] = None
        self.optimizer: Optional[optim.Optimizer] = None
        
        # Training state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        
        logger.info("Initialized DiffusionModelTrainer")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model save directory: {self.model_save_dir}")
    
    def initialize_model(self) -> None:
        """Initialize diffusion model and scheduler."""
        # Initialize UNet3D backbone
        self.model = UNet3DBackbone(
            input_channels=self.config.input_channels,
            output_channels=self.config.output_channels,
            architecture=self.config.unet_architecture
        ).to(self.device)
        
        # Initialize DDPM scheduler
        self.scheduler = DDPMScheduler(self.config)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=1e-4
        )
        
        # Log model summary
        model_summary = self.model.get_model_summary()
        logger.info(f"Model initialized: {model_summary['total_parameters']:,} parameters")
        logger.info(f"Architecture: {self.config.unet_architecture.value}")
    
    def train_model(self, train_dataset: Dict[str, Any], 
                   val_dataset: Optional[Dict[str, Any]] = None) -> TrainingResult:
        """Train diffusion model on synthetic embryo data.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            
        Returns:
            Training result with metrics and model path
        """
        if self.model is None or self.scheduler is None:
            self.initialize_model()
        
        logger.info("Starting diffusion model training")
        start_time = time.time()
        
        # Prepare data loaders
        train_loader = self._create_dataloader(train_dataset, shuffle=True)
        val_loader = self._create_dataloader(val_dataset, shuffle=False) if val_dataset else None
        
        # Training loop
        best_val_loss = float('inf')
        convergence_patience = 10
        no_improvement_count = 0
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader, epoch)
            self.training_history["train_loss"].append(train_loss)
            
            # Validation phase
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader, epoch)
                self.training_history["val_loss"].append(val_loss)
                
                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improvement_count = 0
                    # Save best model
                    self._save_model("best_model.pth")
                else:
                    no_improvement_count += 1
                
                # Early stopping
                if no_improvement_count >= convergence_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                val_info = f", val_loss: {val_loss:.4f}" if val_loader else ""
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}, "
                           f"train_loss: {train_loss:.4f}{val_info}")
        
        training_time = time.time() - start_time
        
        # Save final model
        final_model_path = self._save_model("final_model.pth")
        
        # Create training result
        result = TrainingResult(
            model_path=str(final_model_path),
            training_loss=self.training_history["train_loss"],
            validation_loss=self.training_history["val_loss"],
            final_loss=self.training_history["train_loss"][-1],
            convergence_achieved=no_improvement_count < convergence_patience,
            training_time_seconds=training_time,
            model_parameters=self.model.get_model_summary()["total_parameters"]
        )
        
        logger.info(f"Training completed in {training_time:.1f}s")
        logger.info(f"Final loss: {result.final_loss:.4f}")
        logger.info(f"Convergence: {'Yes' if result.convergence_achieved else 'No'}")
        
        return result
    
    def _create_dataloader(self, dataset: Optional[Dict[str, Any]], 
                          shuffle: bool = True) -> Optional[DataLoader]:
        """Create PyTorch DataLoader from dataset."""
        if dataset is None:
            return None
        
        # Convert to tensors
        morphogen_data = torch.from_numpy(dataset["morphogen_concentrations"]).float()
        
        # Create dataset and loader
        torch_dataset = TensorDataset(morphogen_data)
        dataloader = DataLoader(
            torch_dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0  # Avoid multiprocessing issues
        )
        
        return dataloader
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (batch_data,) in enumerate(train_loader):
            batch_data = batch_data.to(self.device)
            
            # Compute loss
            loss = self.scheduler.compute_loss(self.model, batch_data)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return avg_loss
    
    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (batch_data,) in enumerate(val_loader):
                batch_data = batch_data.to(self.device)
                
                # Compute loss
                loss = self.scheduler.compute_loss(self.model, batch_data)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return avg_loss
    
    def _save_model(self, filename: str) -> Path:
        """Save model checkpoint."""
        model_path = self.model_save_dir / filename
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "training_history": self.training_history,
            "scheduler_state": {
                "betas": self.scheduler.betas,
                "alphas_cumprod": self.scheduler.alphas_cumprod
            }
        }
        
        torch.save(checkpoint, model_path)
        logger.info(f"Model saved: {model_path}")
        
        return model_path
    
    def load_model(self, model_path: str) -> bool:
        """Load trained model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
            
        Returns:
            True if model loaded successfully
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Initialize model if not already done
            if self.model is None:
                self.initialize_model()
            
            # Load state
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.training_history = checkpoint.get("training_history", {"train_loss": [], "val_loss": []})
            
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            return False
    
    def generate_morphogen_sample(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Generate morphogen concentration sample.
        
        Args:
            shape: Shape of sample to generate (B, C, D, H, W)
            
        Returns:
            Generated morphogen concentrations as numpy array
        """
        if self.model is None or self.scheduler is None:
            raise ValueError("Model not initialized")
        
        # Generate sample using scheduler
        with torch.no_grad():
            sample = self.scheduler.sample(self.model, shape, self.device)
            
            # Convert to numpy and ensure valid concentrations
            sample_np = sample.cpu().numpy()
            sample_np = np.clip(sample_np, 0.0, 2.0)  # Clip to reasonable concentration range
        
        logger.info(f"Generated morphogen sample: {sample_np.shape}")
        
        return sample_np
