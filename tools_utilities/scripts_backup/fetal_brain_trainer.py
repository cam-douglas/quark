#!/usr/bin/env python3
"""
🧠 Fetal Brain Development Trainer

Specialized training script for fetal brain development simulation models.
This trainer handles:
- Fetal anatomical development simulation
- Morphogen gradient modeling
- Tissue mechanics and growth
- Neural migration patterns
- Cortical folding simulation

INTEGRATED WITH EXISTING SIMULATION CAPABILITIES:
- Uses optimized_brain_physics.py for physics simulation
- Integrates fetal_anatomical_simulation.py for anatomical modeling
- Incorporates morphogen_physics.py for gradient modeling
- Utilizes tissue_mechanics.py for mechanical properties
- Connects with dual_mode_simulator.py for hybrid simulation
"""

import os, sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
import nibabel as nib
import pandas as pd

# Add simulation frameworks to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../simulation_frameworks'))

# Import existing simulation capabilities
try:
    from optimized_brain_physics import OptimizedBrainPhysics
    from fetal_anatomical_simulation import FetalAnatomicalSimulator
    from morphogen_physics import MorphogenPhysics
    from tissue_mechanics import TissueMechanics
    from dual_mode_simulator import DualModeSimulator
    from neural_simulator import NeuralSimulator
    from enhanced_data_resources import EnhancedDataResources
    SIMULATION_AVAILABLE = True
    print("✅ Successfully imported existing simulation capabilities")
except ImportError as e:
    print(f"⚠️  Warning: Could not import simulation modules: {e}")
    print("   Using simplified simulation for training")
    SIMULATION_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class FetalBrainConfig:
    """Configuration for fetal brain development training"""
    # Model parameters
    model_type: str = "fetal_brain_simulator"
    hidden_dim: int = 512
    num_layers: int = 8
    dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 1e-4
    epochs: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Simulation parameters
    simulation_steps: int = 1000
    time_resolution: float = 0.01
    spatial_resolution: float = 1.0  # mm
    
    # Brain development parameters
    gestational_weeks: Tuple[int, int] = (8, 40)
    brain_regions: List[str] = None
    morphogen_types: List[str] = None
    
    # Data parameters
    dataset_path: str = "data/fetal_brain/"
    save_path: str = "models/fetal_brain/"
    checkpoint_interval: int = 10
    validation_interval: int = 5
    
    # Simulation integration
    use_existing_simulations: bool = True
    physics_engine: str = "mujoco"  # mujoco, nest, hybrid
    
    def __post_init__(self):
        if self.brain_regions is None:
            self.brain_regions = [
                "cerebral_cortex", "hippocampus", "cerebellum", 
                "brainstem", "thalamus", "basal_ganglia"
            ]
        if self.morphogen_types is None:
            self.morphogen_types = [
                "shh", "wnt", "bmp", "fgf", "retinoic_acid"
            ]

class IntegratedFetalBrainSimulator:
    """Integrated simulator that combines existing simulation capabilities"""
    
    def __init__(self, config: FetalBrainConfig):
        self.config = config
        self.simulators = {}
        
        if SIMULATION_AVAILABLE and config.use_existing_simulations:
            logger.info("Initializing integrated simulation framework...")
            
            # Initialize existing simulators
            try:
                self.simulators['brain_physics'] = OptimizedBrainPhysics()
                self.simulators['anatomical'] = FetalAnatomicalSimulator()
                self.simulators['morphogen'] = MorphogenPhysics()
                self.simulators['tissue'] = TissueMechanics()
                self.simulators['dual_mode'] = DualModeSimulator()
                self.simulators['neural'] = NeuralSimulator()
                self.simulators['data_resources'] = EnhancedDataResources()
                
                logger.info("✅ All simulation modules initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize simulation modules: {e}")
                self.simulators = {}
        else:
            logger.info("Using simplified simulation for training")
    
    def simulate_brain_development(self, gestational_week: int, region: str) -> Dict:
        """Simulate brain development using integrated simulation framework"""
        
        if self.simulators and SIMULATION_AVAILABLE:
            # Use existing simulation capabilities
            try:
                # Get anatomical simulation
                anatomical_data = self.simulators['anatomical'].simulate_week(gestational_week, region)
                
                # Get morphogen gradients
                morphogen_data = self.simulators['morphogen'].calculate_gradients(gestational_week, region)
                
                # Get tissue properties
                tissue_data = self.simulators['tissue'].get_properties(gestational_week, region)
                
                # Get neural simulation data
                neural_data = self.simulators['neural'].simulate_activity(gestational_week, region)
                
                # Combine all simulation data
                combined_data = {
                    'anatomical': anatomical_data,
                    'morphogen': morphogen_data,
                    'tissue': tissue_data,
                    'neural': neural_data,
                    'week': gestational_week,
                    'region': region
                }
                
                return combined_data
                
            except Exception as e:
                logger.warning(f"Simulation failed, using fallback: {e}")
                return self._fallback_simulation(gestational_week, region)
        else:
            return self._fallback_simulation(gestational_week, region)
    
    def _fallback_simulation(self, gestational_week: int, region: str) -> Dict:
        """Fallback simulation when existing simulators are not available"""
        # Generate simplified simulation data
        brain_volume = self._generate_brain_volume(gestational_week, region)
        morphogen_gradients = self._generate_morphogen_gradients(gestational_week, region)
        tissue_properties = self._generate_tissue_properties(gestational_week, region)
        
        return {
            'anatomical': {'volume': brain_volume},
            'morphogen': {'gradients': morphogen_gradients},
            'tissue': {'properties': tissue_properties},
            'neural': {'activity': np.random.randn(128, 128, 128) * 0.1},
            'week': gestational_week,
            'region': region
        }
    
    def _generate_brain_volume(self, week: int, region: str) -> np.ndarray:
        """Generate brain volume data (fallback method)"""
        volume = np.zeros((128, 128, 128))
        
        if region == "cerebral_cortex":
            center = np.array([64, 64, 64])
            radius = 20 + (week - 8) * 2
            x, y, z = np.ogrid[:128, :128, :128]
            mask = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2
            volume[mask] = 1.0
        elif region == "hippocampus":
            center = np.array([64, 64, 60])
            radius = 8 + (week - 8) * 0.5
            x, y, z = np.ogrid[:128, :128, :128]
            mask = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2
            volume[mask] = 0.8
        
        return volume
    
    def _generate_morphogen_gradients(self, week: int, region: str) -> np.ndarray:
        """Generate morphogen gradient data (fallback method)"""
        gradients = np.zeros((5, 128, 128, 128))
        
        for i, morphogen in enumerate(["shh", "wnt", "bmp", "fgf", "retinoic_acid"]):
            gradient = np.zeros((128, 128, 128))
            x, y, z = np.ogrid[:128, :128, :128]
            
            if morphogen == "shh":
                gradient = np.exp(-((z - 64) / 20)**2) * 0.8
            elif morphogen == "wnt":
                gradient = np.exp(-((z - 64) / 20)**2) * 0.6
            elif morphogen == "bmp":
                gradient = np.exp(-((y - 64) / 30)**2) * 0.7
            elif morphogen == "fgf":
                gradient = np.exp(-((x - 64) / 25)**2) * 0.5
            elif morphogen == "retinoic_acid":
                gradient = np.exp(-((x - 32) / 40)**2) * 0.9
            
            gradients[i] = gradient
        
        return gradients
    
    def _generate_tissue_properties(self, week: int, region: str) -> np.ndarray:
        """Generate tissue properties data (fallback method)"""
        properties = np.zeros((4, 128, 128, 128))
        
        base_elasticity = 1000 + week * 50
        base_viscosity = 100 + week * 10
        base_density = 1000 + week * 5
        base_growth_rate = 0.1 - week * 0.002
        
        x, y, z = np.ogrid[:128, :128, :128]
        spatial_factor = np.exp(-((x - 64)**2 + (y - 64)**2 + (z - 64)**2) / (50**2))
        
        properties[0] = base_elasticity * spatial_factor
        properties[1] = base_viscosity * spatial_factor
        properties[2] = base_density * spatial_factor
        properties[3] = base_growth_rate * spatial_factor
        
        return properties

class FetalBrainDataset(Dataset):
    """Dataset for fetal brain development data with integrated simulation"""
    
    def __init__(self, data_path: str, gestational_weeks: Tuple[int, int], config: FetalBrainConfig, transform=None):
        self.data_path = Path(data_path)
        self.gestational_weeks = gestational_weeks
        self.transform = transform
        self.config = config
        
        # Initialize integrated simulator
        self.simulator = IntegratedFetalBrainSimulator(config)
        
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Dict]:
        """Load sample metadata"""
        samples = []
        
        # Generate samples for each gestational week and brain region
        for week in range(self.gestational_weeks[0], self.gestational_weeks[1] + 1):
            for region in self.config.brain_regions:
                sample = {
                    "week": week,
                    "region": region,
                    "file_path": f"week_{week:02d}_{region}.nii.gz",
                    "metadata": {
                        "gestational_age": week,
                        "brain_region": region,
                        "voxel_size": [1.0, 1.0, 1.0],
                        "dimensions": [128, 128, 128]
                    }
                }
                samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Use integrated simulation to generate data
        simulation_data = self.simulator.simulate_brain_development(
            sample["week"], 
            sample["region"]
        )
        
        # Extract data from simulation
        brain_volume = simulation_data['anatomical']['volume']
        morphogen_gradients = simulation_data['morphogen']['gradients']
        tissue_properties = simulation_data['tissue']['properties']
        
        data = {
            "brain_volume": torch.FloatTensor(brain_volume),
            "morphogen_gradients": torch.FloatTensor(morphogen_gradients),
            "tissue_properties": torch.FloatTensor(tissue_properties),
            "metadata": sample["metadata"],
            "simulation_data": simulation_data
        }
        
        if self.transform:
            data = self.transform(data)
        
        return data

class FetalBrainSimulator(nn.Module):
    """Neural network model for fetal brain development simulation"""
    
    def __init__(self, config: FetalBrainConfig):
        super().__init__()
        self.config = config
        
        # Encoder for brain volume
        self.volume_encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((8, 8, 8))
        )
        
        # Encoder for morphogen gradients
        self.morphogen_encoder = nn.Sequential(
            nn.Conv3d(5, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((8, 8, 8))
        )
        
        # Encoder for tissue properties
        self.tissue_encoder = nn.Sequential(
            nn.Conv3d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((8, 8, 8))
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(128 * 8 * 8 * 8 * 3, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Development prediction heads
        self.volume_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, 128 * 64 * 64 * 64),
            nn.ReLU(),
            nn.Unflatten(1, (128, 64, 64, 64)),
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv3d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        self.morphogen_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, 5 * 64 * 64 * 64),
            nn.ReLU(),
            nn.Unflatten(1, (5, 64, 64, 64)),
            nn.ConvTranspose3d(5, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv3d(16, 5, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        self.tissue_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * 64 * 64 * 64),
            nn.ReLU(),
            nn.Unflatten(1, (4, 64, 64, 64)),
            nn.ConvTranspose3d(4, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv3d(16, 4, kernel_size=3, padding=1)
        )
    
    def forward(self, brain_volume, morphogen_gradients, tissue_properties):
        """Forward pass through the fetal brain simulator"""
        
        # Encode inputs
        volume_features = self.volume_encoder(brain_volume)
        morphogen_features = self.morphogen_encoder(morphogen_gradients)
        tissue_features = self.tissue_encoder(tissue_properties)
        
        # Flatten and concatenate features
        volume_flat = volume_features.flatten(1)
        morphogen_flat = morphogen_features.flatten(1)
        tissue_flat = tissue_features.flatten(1)
        
        combined_features = torch.cat([volume_flat, morphogen_flat, tissue_flat], dim=1)
        
        # Fusion
        fused_features = self.fusion(combined_features)
        
        # Predict development
        predicted_volume = self.volume_predictor(fused_features)
        predicted_morphogens = self.morphogen_predictor(fused_features)
        predicted_tissue = self.tissue_predictor(fused_features)
        
        return {
            "predicted_volume": predicted_volume,
            "predicted_morphogens": predicted_morphogens,
            "predicted_tissue": predicted_tissue,
            "fused_features": fused_features
        }

class FetalBrainTrainer:
    """Trainer for fetal brain development models"""
    
    def __init__(self, config: FetalBrainConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize model
        self.model = FetalBrainSimulator(config).to(self.device)
        
        # Initialize optimizer and loss functions
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs)
        
        # Loss functions
        self.volume_loss = nn.BCELoss()
        self.morphogen_loss = nn.MSELoss()
        self.tissue_loss = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Create save directory
        os.makedirs(config.save_path, exist_ok=True)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """Train the fetal brain development model"""
        logger.info("Starting fetal brain development training...")
        
        for epoch in range(self.config.epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            # Validation phase
            if val_loader and epoch % self.config.validation_interval == 0:
                val_loss = self._validate_epoch(val_loader, epoch)
                self.val_losses.append(val_loss)
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step()
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            # Checkpointing
            if epoch % self.config.checkpoint_interval == 0:
                self._save_checkpoint(epoch)
        
        # Save final model
        self._save_final_model()
        
        # Plot training curves
        self._plot_training_curves()
        
        logger.info("Fetal brain development training completed!")
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            brain_volume = batch["brain_volume"].to(self.device)
            morphogen_gradients = batch["morphogen_gradients"].to(self.device)
            tissue_properties = batch["tissue_properties"].to(self.device)
            
            # Forward pass
            predictions = self.model(brain_volume, morphogen_gradients, tissue_properties)
            
            # Calculate losses
            volume_loss = self.volume_loss(predictions["predicted_volume"], brain_volume)
            morphogen_loss = self.morphogen_loss(predictions["predicted_morphogens"], morphogen_gradients)
            tissue_loss = self.tissue_loss(predictions["predicted_tissue"], tissue_properties)
            
            # Combined loss
            total_batch_loss = volume_loss + 0.5 * morphogen_loss + 0.3 * tissue_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {total_batch_loss.item():.4f}")
        
        return total_loss / num_batches
    
    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                brain_volume = batch["brain_volume"].to(self.device)
                morphogen_gradients = batch["morphogen_gradients"].to(self.device)
                tissue_properties = batch["tissue_properties"].to(self.device)
                
                # Forward pass
                predictions = self.model(brain_volume, morphogen_gradients, tissue_properties)
                
                # Calculate losses
                volume_loss = self.volume_loss(predictions["predicted_volume"], brain_volume)
                morphogen_loss = self.morphogen_loss(predictions["predicted_morphogens"], morphogen_gradients)
                tissue_loss = self.tissue_loss(predictions["predicted_tissue"], tissue_properties)
                
                # Combined loss
                total_batch_loss = volume_loss + 0.5 * morphogen_loss + 0.3 * tissue_loss
                
                total_loss += total_batch_loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates
        }
        
        checkpoint_path = os.path.join(self.config.save_path, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_final_model(self):
        """Save the final trained model"""
        model_path = os.path.join(self.config.save_path, "fetal_brain_simulator_final.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "training_history": {
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "learning_rates": self.learning_rates
            }
        }, model_path)
        logger.info(f"Final model saved: {model_path}")
    
    def _plot_training_curves(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        axes[0, 0].plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            axes[0, 0].plot(range(0, len(self.train_losses), self.config.validation_interval), 
                           self.val_losses, label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Learning rate
        axes[0, 1].plot(self.learning_rates)
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].grid(True)
        
        # Loss components (if available)
        if len(self.train_losses) > 1:
            axes[1, 0].plot(np.diff(self.train_losses))
            axes[1, 0].set_title('Loss Change Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss Change')
            axes[1, 0].grid(True)
        
        # Training progress
        axes[1, 1].plot(range(len(self.train_losses)), self.train_losses, 'b-', alpha=0.7)
        axes[1, 1].set_title('Training Progress')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.config.save_path, "training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Training curves saved: {plot_path}")

def main():
    """Main entry point for fetal brain development training"""
    parser = argparse.ArgumentParser(description="Fetal Brain Development Trainer")
    parser.add_argument("--config", type=str, default="fetal_brain_config.json",
                       help="Path to configuration file")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_data = json.load(f)
        config = FetalBrainConfig(**config_data)
    else:
        config = FetalBrainConfig()
    
    # Update config with command line arguments
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    if args.device != "auto":
        config.device = args.device
    
    # Create datasets
    train_dataset = FetalBrainDataset(
        config.dataset_path, 
        config.gestational_weeks, 
        config
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=4
    )
    
    # Initialize trainer
    trainer = FetalBrainTrainer(config)
    
    # Start training
    trainer.train(train_loader)
    
    logger.info("Fetal brain development training completed successfully!")

if __name__ == "__main__":
    main()
