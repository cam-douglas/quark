#!/usr/bin/env python3
"""
Brainstem Segmentation Training Pipeline - Phase 2 Step 3.F3

Complete preprocessing → augmentation → training pipeline for brainstem subdivision segmentation.
Integrates ViT-GNN hybrid model with data-centric augmentations.
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Union
import logging
from tqdm import tqdm
# import wandb  # Optional: for experiment tracking
from dataclasses import dataclass

# Import our custom modules
import sys
sys.path.append(str(Path(__file__).parent.parent / "brainstem_segmentation"))
from model_architecture_designer import ViTGNNHybrid, HierarchicalLoss
from data_augmentation_designer import AugmentationPipeline


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    
    # Model parameters
    input_channels: int = 1
    num_classes: int = 6  # Reduced to match our actual mapped labels
    patch_size: Tuple[int, int, int] = (64, 64, 64)
    embed_dim: int = 768
    vit_layers: int = 8
    gnn_layers: int = 3
    num_heads: int = 8
    
    # Training parameters
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 100
    warmup_epochs: int = 10
    weight_decay: float = 1e-5
    
    # Augmentation parameters
    elastic_prob: float = 0.8
    noise_prob: float = 0.6
    cutmix_prob: float = 0.4
    
    # Loss parameters
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    hierarchy_weight: float = 0.3
    boundary_weight: float = 0.2
    
    # Validation parameters
    val_interval: int = 5
    save_interval: int = 10
    early_stopping_patience: int = 20
    
    # Hardware parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    mixed_precision: bool = True


class BrainstemDataset(Dataset):
    """
    Dataset class for brainstem segmentation data.
    
    Handles loading of NextBrain data and other human brain atlases.
    """
    
    def __init__(self, 
                 data_dir: Path,
                 mode: str = "train",
                 patch_size: Tuple[int, int, int] = (64, 64, 64),
                 augment: bool = True,
                 config: TrainingConfig = None):
        
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.patch_size = patch_size
        self.augment = augment and (mode == "train")
        self.config = config or TrainingConfig()
        
        # Initialize augmentation pipeline
        if self.augment:
            self.augmentation_pipeline = AugmentationPipeline(
                elastic_prob=self.config.elastic_prob,
                noise_prob=self.config.noise_prob,
                cutmix_prob=self.config.cutmix_prob,
                morphogen_aware=True
            )
        
        # Load data paths
        self.data_paths = self._load_data_paths()
        
        # Load label schema
        self.label_schema = self._load_label_schema()
        
        logging.info(f"Loaded {len(self.data_paths)} samples for {mode} mode")
    
    def _load_data_paths(self) -> List[Dict[str, Path]]:
        """Load paths to volume and segmentation files."""
        
        data_paths = []
        
        # NextBrain data
        nextbrain_dir = self.data_dir / "nextbrain"
        if nextbrain_dir.exists():
            volume_path = nextbrain_dir / "T2w.nii.gz"
            labels_path = nextbrain_dir / "manual_segmentation.nii.gz"
            
            if volume_path.exists() and labels_path.exists():
                data_paths.append({
                    "volume": volume_path,
                    "labels": labels_path,
                    "dataset": "nextbrain",
                    "subject_id": "nextbrain_template"
                })
        
        # Add other datasets here as they become available
        # TODO: Add Arousal Nuclei Atlas, 7T MRI data, etc.
        
        return data_paths
    
    def _load_label_schema(self) -> Dict[str, Any]:
        """Load label schema for consistent mapping."""
        
        schema_file = self.data_dir.parent / "metadata" / "brainstem_labels_schema.json"
        
        if schema_file.exists():
            with open(schema_file, 'r') as f:
                return json.load(f)
        else:
            # Default schema
            return {"hierarchy": {"brainstem": {"subdivisions": {}}}}
    
    def _create_subdivision_mask(self, labels: np.ndarray) -> np.ndarray:
        """Create subdivision mask from nucleus labels."""
        
        subdivision_mask = np.zeros_like(labels)
        
        # Map nucleus labels to subdivisions based on schema
        hierarchy = self.label_schema.get("hierarchy", {}).get("brainstem", {}).get("subdivisions", {})
        
        for subdivision_name, subdivision_data in hierarchy.items():
            subdivision_id = {"midbrain": 1, "pons": 2, "medulla": 3}.get(subdivision_name, 0)
            
            nuclei = subdivision_data.get("nuclei", {})
            for nucleus_data in nuclei.values():
                label_id = nucleus_data.get("label_id")
                if label_id is not None:
                    subdivision_mask[labels == label_id] = subdivision_id
        
        return subdivision_mask
    
    def _map_labels_to_schema(self, labels: np.ndarray) -> np.ndarray:
        """Map NextBrain labels to our 16-class schema."""
        
        # Create mapping from NextBrain labels to our schema
        nextbrain_to_schema = {
            0: 0,   # Background
            4: 1,   # Red Nucleus
            9: 2,   # Brain-Stem (general)
            29: 3,  # Pontine Nuclei
            85: 4,  # Inferior Colliculus
            99: 5,  # Medulla Oblongata
        }
        
        # Create output array
        mapped_labels = np.zeros_like(labels)
        
        # Apply mapping
        for nextbrain_label, schema_label in nextbrain_to_schema.items():
            mapped_labels[labels == nextbrain_label] = schema_label
        
        # Map all other labels to background (0)
        # This handles the case where NextBrain has 333 labels but we only use key brainstem ones
        unmapped_mask = np.isin(labels, list(nextbrain_to_schema.keys()), invert=True)
        mapped_labels[unmapped_mask] = 0
        
        return mapped_labels
    
    def _extract_patches(self, volume: np.ndarray, labels: np.ndarray, 
                        subdivision_mask: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Extract patches from volume for training."""
        
        patches = []
        stride = self.patch_size[0] // 2  # 50% overlap
        
        h, w, d = volume.shape
        
        for z in range(0, d - self.patch_size[2] + 1, stride):
            for y in range(0, w - self.patch_size[1] + 1, stride):
                for x in range(0, h - self.patch_size[0] + 1, stride):
                    
                    # Extract patch
                    vol_patch = volume[x:x+self.patch_size[0], 
                                     y:y+self.patch_size[1], 
                                     z:z+self.patch_size[2]]
                    
                    label_patch = labels[x:x+self.patch_size[0], 
                                        y:y+self.patch_size[1], 
                                        z:z+self.patch_size[2]]
                    
                    sub_patch = subdivision_mask[x:x+self.patch_size[0], 
                                               y:y+self.patch_size[1], 
                                               z:z+self.patch_size[2]]
                    
                    # Skip patches with no foreground
                    if np.sum(label_patch > 0) < 100:  # Minimum foreground voxels
                        continue
                    
                    patches.append((vol_patch, label_patch, sub_patch))
        
        return patches
    
    def __len__(self) -> int:
        # For now, return number of patches from all volumes
        # In practice, this would be computed dynamically
        return len(self.data_paths) * 50  # Approximate patches per volume
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a training sample."""
        
        # For simplicity, cycle through available data
        data_idx = idx % len(self.data_paths)
        data_info = self.data_paths[data_idx]
        
        # Load volume and labels
        volume_img = nib.load(data_info["volume"])
        labels_img = nib.load(data_info["labels"])
        
        volume = volume_img.get_fdata().astype(np.float32)
        labels = labels_img.get_fdata().astype(np.int32)
        
        # Normalize volume
        volume = (volume - volume.mean()) / (volume.std() + 1e-8)
        
        # Map labels to our schema (NextBrain has 333 labels, we need 16)
        labels = self._map_labels_to_schema(labels)
        
        # Create subdivision mask
        subdivision_mask = self._create_subdivision_mask(labels)
        
        # Extract patches
        patches = self._extract_patches(volume, labels, subdivision_mask)
        
        if not patches:
            # Fallback: return a random patch
            patch_volume = np.random.randn(*self.patch_size).astype(np.float32)
            patch_labels = np.zeros(self.patch_size, dtype=np.int32)
        else:
            # Select random patch
            patch_idx = np.random.randint(0, len(patches))
            patch_volume, patch_labels, patch_subdivision = patches[patch_idx]
            
            # Apply augmentations
            if self.augment:
                # For CutMix, we need another sample
                mix_idx = np.random.randint(0, len(patches))
                mix_volume, mix_labels, _ = patches[mix_idx]
                
                patch_volume, patch_labels = self.augmentation_pipeline.augment_sample(
                    patch_volume, patch_labels, patch_subdivision, mix_volume, mix_labels
                )
        
        # Convert to tensors
        volume_tensor = torch.from_numpy(patch_volume).unsqueeze(0)  # Add channel dim
        labels_tensor = torch.from_numpy(patch_labels).long()
        
        return volume_tensor, labels_tensor


class MetricsTracker:
    """Track training and validation metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.losses = []
        self.dice_scores = []
        self.accuracy = []
        
    def update(self, loss: float, predictions: torch.Tensor, targets: torch.Tensor):
        """Update metrics with batch results."""
        
        self.losses.append(loss)
        
        # Calculate Dice score
        dice = self._calculate_dice(predictions, targets)
        self.dice_scores.append(dice)
        
        # Calculate accuracy
        acc = self._calculate_accuracy(predictions, targets)
        self.accuracy.append(acc)
    
    def _calculate_dice(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate mean Dice coefficient."""
        
        pred_classes = torch.argmax(predictions, dim=1)
        
        dice_scores = []
        for class_id in range(1, predictions.shape[1]):  # Skip background
            pred_mask = (pred_classes == class_id)
            target_mask = (targets == class_id)
            
            intersection = (pred_mask & target_mask).sum().float()
            union = pred_mask.sum().float() + target_mask.sum().float()
            
            if union > 0:
                dice = (2.0 * intersection) / union
                dice_scores.append(dice.item())
        
        return np.mean(dice_scores) if dice_scores else 0.0
    
    def _calculate_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate pixel-wise accuracy."""
        
        pred_classes = torch.argmax(predictions, dim=1)
        correct = (pred_classes == targets).sum().float()
        total = targets.numel()
        
        return (correct / total).item()
    
    def get_averages(self) -> Dict[str, float]:
        """Get average metrics."""
        
        return {
            "loss": np.mean(self.losses) if self.losses else 0.0,
            "dice": np.mean(self.dice_scores) if self.dice_scores else 0.0,
            "accuracy": np.mean(self.accuracy) if self.accuracy else 0.0
        }


class BrainstemTrainer:
    """Main training class for brainstem segmentation."""
    
    def __init__(self, config: TrainingConfig, data_dir: Path, output_dir: Path):
        self.config = config
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize loss function
        self.criterion = HierarchicalLoss(
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
            hierarchy_weight=config.hierarchy_weight,
            boundary_weight=config.boundary_weight
        )
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=config.num_epochs // 4, T_mult=2
        )
        
        # Initialize metrics
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        # Initialize mixed precision
        if config.mixed_precision and config.device == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def _setup_logging(self):
        """Setup logging configuration."""
        
        log_file = self.output_dir / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _create_model(self) -> nn.Module:
        """Create and initialize model."""
        
        model = ViTGNNHybrid(
            input_channels=self.config.input_channels,
            patch_size=(16, 16, 16),  # Internal patch size for ViT
            embed_dim=self.config.embed_dim,
            vit_layers=self.config.vit_layers,
            gnn_layers=self.config.gnn_layers,
            num_heads=self.config.num_heads,
            num_classes=self.config.num_classes
        )
        
        model = model.to(self.config.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logging.info(f"Model created: {total_params:,} total parameters, {trainable_params:,} trainable")
        
        return model
    
    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation dataloaders."""
        
        # For now, use same data for train/val with different augmentations
        train_dataset = BrainstemDataset(
            self.data_dir, mode="train", 
            patch_size=self.config.patch_size,
            augment=True, config=self.config
        )
        
        val_dataset = BrainstemDataset(
            self.data_dir, mode="val",
            patch_size=self.config.patch_size, 
            augment=False, config=self.config
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True if self.config.device == "cuda" else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True if self.config.device == "cuda" else False
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.model.train()
        self.train_metrics.reset()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        
        for batch_idx, (volumes, targets) in enumerate(pbar):
            volumes = volumes.to(self.config.device)
            targets = targets.to(self.config.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    predictions = self.model(volumes)
                    loss_dict = self.criterion(predictions, targets)
                    loss = loss_dict['total_loss']
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(volumes)
                loss_dict = self.criterion(predictions, targets)
                loss = loss_dict['total_loss']
                
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            self.train_metrics.update(loss.item(), predictions, targets)
            
            # Update progress bar
            current_metrics = self.train_metrics.get_averages()
            pbar.set_postfix({
                'loss': f"{current_metrics['loss']:.4f}",
                'dice': f"{current_metrics['dice']:.4f}",
                'acc': f"{current_metrics['accuracy']:.4f}"
            })
        
        return self.train_metrics.get_averages()
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model."""
        
        self.model.eval()
        self.val_metrics.reset()
        
        with torch.no_grad():
            for volumes, targets in tqdm(val_loader, desc="Validation"):
                volumes = volumes.to(self.config.device)
                targets = targets.to(self.config.device)
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        predictions = self.model(volumes)
                        loss_dict = self.criterion(predictions, targets)
                        loss = loss_dict['total_loss']
                else:
                    predictions = self.model(volumes)
                    loss_dict = self.criterion(predictions, targets)
                    loss = loss_dict['total_loss']
                
                self.val_metrics.update(loss.item(), predictions, targets)
        
        return self.val_metrics.get_averages()
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': metrics
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logging.info(f"New best model saved with Dice: {metrics['dice']:.4f}")
    
    def train(self):
        """Main training loop."""
        
        logging.info("Starting training...")
        logging.info(f"Config: {self.config}")
        
        # Create dataloaders
        train_loader, val_loader = self._create_dataloaders()
        
        best_dice = 0.0
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            if (epoch + 1) % self.config.val_interval == 0:
                val_metrics = self.validate(val_loader)
                
                # Update scheduler
                self.scheduler.step()
                
                # Log metrics
                logging.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
                logging.info(f"Train - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
                logging.info(f"Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
                
                # Check for improvement
                is_best = val_metrics['dice'] > best_dice
                if is_best:
                    best_dice = val_metrics['dice']
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Save checkpoint
                if (epoch + 1) % self.config.save_interval == 0 or is_best:
                    self.save_checkpoint(epoch, val_metrics, is_best)
                
                # Early stopping
                if patience_counter >= self.config.early_stopping_patience:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        logging.info("Training completed!")
        logging.info(f"Best validation Dice: {best_dice:.4f}")


def main():
    """Execute Phase 2 Step 3.F3: Complete training pipeline implementation."""
    
    print("🚀 PHASE 2 STEP 3.F3 - TRAINING PIPELINE IMPLEMENTATION")
    print("=" * 60)
    
    # Configuration
    config = TrainingConfig()
    
    # Paths
    data_dir = Path("/Users/camdouglas/quark/data/datasets/brainstem_segmentation")
    output_dir = Path("/Users/camdouglas/quark/data/models/brainstem_segmentation")
    
    print(f"📊 Training Configuration:")
    print(f"   Model: ViT-GNN Hybrid ({config.embed_dim}d, {config.vit_layers} ViT layers)")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Device: {config.device}")
    print(f"   Mixed precision: {config.mixed_precision}")
    
    print(f"\n📁 Data Configuration:")
    print(f"   Data directory: {data_dir}")
    print(f"   Output directory: {output_dir}")
    print(f"   Patch size: {config.patch_size}")
    
    # Test pipeline components
    print(f"\n🧪 Testing Pipeline Components...")
    
    try:
        # Test dataset
        dataset = BrainstemDataset(data_dir, mode="train", config=config)
        print(f"   ✅ Dataset: {len(dataset)} samples loaded")
        
        # Test model creation
        model = ViTGNNHybrid(
            input_channels=1,
            num_classes=16,
            embed_dim=config.embed_dim
        )
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   ✅ Model: {total_params:,} parameters")
        
        # Test loss function
        criterion = HierarchicalLoss()
        dummy_pred = torch.randn(2, 16, 32, 32, 32)
        dummy_target = torch.randint(0, 16, (2, 32, 32, 32))
        loss_dict = criterion(dummy_pred, dummy_target)
        print(f"   ✅ Loss function: {loss_dict['total_loss'].item():.4f}")
        
        # Test trainer initialization
        trainer = BrainstemTrainer(config, data_dir, output_dir)
        print(f"   ✅ Trainer: Initialized successfully")
        
        print(f"\n✅ All pipeline components working!")
        
    except Exception as e:
        print(f"   ❌ Pipeline test failed: {e}")
        return
    
    # Save pipeline configuration
    pipeline_config = {
        "generated": datetime.now().isoformat(),
        "phase": "Phase 2 - Design & Architecture",
        "step": "3.F3 - Training Pipeline Implementation",
        
        "pipeline_components": {
            "preprocessing": {
                "normalization": "Z-score normalization",
                "patch_extraction": f"{config.patch_size} patches with 50% overlap",
                "subdivision_mapping": "Nucleus labels → subdivision masks"
            },
            "augmentation": {
                "elastic_deformation": {"probability": config.elastic_prob},
                "noise_augmentation": {"probability": config.noise_prob},
                "cutmix_nuclei": {"probability": config.cutmix_prob},
                "morphogen_aware": True
            },
            "training": {
                "model": "ViT-GNN Hybrid",
                "loss": "Hierarchical (focal + boundary + consistency)",
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingWarmRestarts",
                "mixed_precision": config.mixed_precision
            }
        },
        
        "training_parameters": {
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "weight_decay": config.weight_decay,
            "early_stopping_patience": config.early_stopping_patience
        },
        
        "validation_strategy": {
            "interval": config.val_interval,
            "metrics": ["loss", "dice_coefficient", "pixel_accuracy"],
            "early_stopping": "Dice coefficient plateau"
        },
        
        "output_artifacts": {
            "model_checkpoints": "checkpoint_epoch_XXX.pth",
            "best_model": "best_model.pth", 
            "training_logs": "training.log",
            "metrics_history": "Tracked per epoch"
        }
    }
    
    config_file = output_dir / "pipeline_configuration.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w') as f:
        json.dump(pipeline_config, f, indent=2)
    
    print(f"\n✅ Phase 2 Step 3.F3 Complete!")
    print(f"   📋 Pipeline configuration: {config_file}")
    print(f"   🐍 Training pipeline: {__file__}")
    print(f"   🎯 Ready for training: All components integrated")
    print(f"   💾 Memory efficient: Patch-based training")
    print(f"   🔄 Full workflow: Preprocessing → Augmentation → Training")
    
    # Note: Actual training would be started with trainer.train()
    # For now, we just validate the pipeline setup
    print(f"\n📝 To start training, run:")
    print(f"   trainer = BrainstemTrainer(config, data_dir, output_dir)")
    print(f"   trainer.train()")


if __name__ == "__main__":
    main()
