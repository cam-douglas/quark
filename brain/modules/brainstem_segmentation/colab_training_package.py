"""
Google Colab Training Package for Brainstem Segmentation
=========================================================

This script packages all necessary components for GPU training on Google Colab.
Upload this file to Colab along with the data files, then run to achieve target metrics.

Usage:
1. Upload to Google Colab
2. Upload data files (imaging_data_normalized.npy, labels.npy) 
3. Enable GPU runtime (Runtime -> Change runtime type -> GPU)
4. Run all cells
5. Download trained model weights
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import json
from pathlib import Path
import time
from datetime import datetime

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ColabTrainingConfig:
    """Optimized configuration for Colab GPU training."""
    
    # Model parameters
    input_channels: int = 4  # T2 + 3 morphogen channels
    num_classes: int = 6  # Background + 5 brainstem subdivisions
    base_filters: int = 32
    attention_gates: bool = True
    deep_supervision: bool = True
    
    # Training parameters - optimized for GPU
    batch_size: int = 8  # Larger batch size for GPU
    learning_rate: float = 1e-3
    num_epochs: int = 350  # Target for Dice ≥ 0.87
    weight_decay: float = 1e-4
    
    # Learning rate scheduling
    scheduler_T_0: int = 50
    scheduler_T_mult: int = 2
    min_lr: float = 1e-6
    
    # Loss weights
    dice_weight: float = 0.5
    focal_weight: float = 0.3
    boundary_weight: float = 0.2
    focal_gamma: float = 2.0
    focal_alpha: Optional[torch.Tensor] = None
    
    # Optimization
    gradient_clip_value: float = 1.0
    mixed_precision: bool = True  # Use AMP for faster training
    
    # Data parameters
    patch_size: Tuple[int, int, int] = (64, 64, 64)
    samples_per_epoch: int = 100
    validation_split: float = 0.2
    
    # Augmentation parameters
    augmentation_prob: float = 0.8
    rotation_angles: Tuple[int, int, int] = (20, 20, 20)
    elastic_alpha: float = 50
    elastic_sigma: float = 5
    
    # Early stopping
    patience: int = 50
    min_delta: float = 0.001


# ============================================================================
# ENHANCED U-NET ARCHITECTURE
# ============================================================================

class AttentionGate(nn.Module):
    """Attention gate for feature refinement."""
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # g comes from deeper layer (smaller spatial dimensions)
        # x comes from skip connection (larger spatial dimensions)
        
        # Upsample g to match x's spatial dimensions
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='trilinear', align_corners=False)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class EnhancedUNet3D(nn.Module):
    """Enhanced 3D U-Net with attention gates and deep supervision."""
    
    def __init__(self, config: ColabTrainingConfig):
        super().__init__()
        
        self.config = config
        f = config.base_filters
        
        # Encoder
        self.encoder1 = self._conv_block(config.input_channels, f)
        self.encoder2 = self._conv_block(f, f*2)
        self.encoder3 = self._conv_block(f*2, f*4)
        self.encoder4 = self._conv_block(f*4, f*8)
        
        # Bottleneck
        self.bottleneck = self._conv_block(f*8, f*16)
        
        # Decoder with attention gates
        self.decoder4 = self._conv_block(f*16 + f*8, f*8)
        self.decoder3 = self._conv_block(f*8 + f*4, f*4)
        self.decoder2 = self._conv_block(f*4 + f*2, f*2)
        self.decoder1 = self._conv_block(f*2 + f, f)
        
        if config.attention_gates:
            self.att4 = AttentionGate(f*16, f*8, f*8)
            self.att3 = AttentionGate(f*8, f*4, f*4)
            self.att2 = AttentionGate(f*4, f*2, f*2)
            self.att1 = AttentionGate(f*2, f, f)
        
        # Output heads
        self.final_conv = nn.Conv3d(f, config.num_classes, kernel_size=1)
        
        if config.deep_supervision:
            self.deep4 = nn.Conv3d(f*8, config.num_classes, kernel_size=1)
            self.deep3 = nn.Conv3d(f*4, config.num_classes, kernel_size=1)
            self.deep2 = nn.Conv3d(f*2, config.num_classes, kernel_size=1)
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with attention
        # Upsample and apply attention before concatenation
        d4 = F.interpolate(b, size=e4.shape[2:], mode='trilinear', align_corners=False)
        if self.config.attention_gates:
            e4 = self.att4(d4, e4)  # Apply attention with upsampled feature as gate
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.decoder4(d4)
        
        d3 = F.interpolate(d4, size=e3.shape[2:], mode='trilinear', align_corners=False)
        if self.config.attention_gates:
            e3 = self.att3(d3, e3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)
        
        d2 = F.interpolate(d3, size=e2.shape[2:], mode='trilinear', align_corners=False)
        if self.config.attention_gates:
            e2 = self.att2(d2, e2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)
        
        d1 = F.interpolate(d2, size=e1.shape[2:], mode='trilinear', align_corners=False)
        if self.config.attention_gates:
            e1 = self.att1(d1, e1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)
        
        # Output
        output = self.final_conv(d1)
        
        if self.config.deep_supervision and self.training:
            deep_outputs = [
                output,
                F.interpolate(self.deep2(d2), size=output.shape[2:], mode='trilinear', align_corners=False),
                F.interpolate(self.deep3(d3), size=output.shape[2:], mode='trilinear', align_corners=False),
                F.interpolate(self.deep4(d4), size=output.shape[2:], mode='trilinear', align_corners=False)
            ]
            return deep_outputs
        
        return output


# ============================================================================
# COMBINED LOSS FUNCTION
# ============================================================================

class CombinedSegmentationLoss(nn.Module):
    """Combined Dice + Focal + Boundary loss."""
    
    def __init__(self, config: ColabTrainingConfig):
        super().__init__()
        self.config = config
        self.focal_loss = nn.CrossEntropyLoss(reduction='none')
        
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        smooth = 1e-5
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=self.config.num_classes).permute(0, 4, 1, 2, 3).float()
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()
    
    def focal_loss_weighted(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.focal_loss(pred, target)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.config.focal_gamma) * ce_loss
        return focal_loss.mean()
    
    def boundary_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Simplified boundary loss using gradient magnitude
        pred_soft = F.softmax(pred, dim=1)
        
        # Compute gradients
        dy = torch.abs(pred_soft[:, :, 1:] - pred_soft[:, :, :-1])
        dx = torch.abs(pred_soft[:, :, :, 1:] - pred_soft[:, :, :, :-1])
        dz = torch.abs(pred_soft[:, :, :, :, 1:] - pred_soft[:, :, :, :, :-1])
        
        # Average gradient magnitude
        boundary = (dy.mean() + dx.mean() + dz.mean()) / 3.0
        return boundary
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Handle deep supervision
        if isinstance(pred, list):
            total_loss = 0
            weights = [1.0, 0.5, 0.3, 0.2]  # Decreasing weights for deeper outputs
            
            for p, w in zip(pred, weights):
                dice = self.dice_loss(p, target) * self.config.dice_weight
                focal = self.focal_loss_weighted(p, target) * self.config.focal_weight
                boundary = self.boundary_loss(p, target) * self.config.boundary_weight
                total_loss += w * (dice + focal + boundary)
            
            return total_loss / sum(weights)
        else:
            dice = self.dice_loss(pred, target) * self.config.dice_weight
            focal = self.focal_loss_weighted(pred, target) * self.config.focal_weight
            boundary = self.boundary_loss(pred, target) * self.config.boundary_weight
            return dice + focal + boundary


# ============================================================================
# MORPHOGEN GENERATOR
# ============================================================================

class MorphogenFieldGenerator:
    """Generate biological morphogen gradients."""
    
    @staticmethod
    def generate_morphogen_coarse_map(patch_size: Tuple[int, int, int]) -> Dict[str, np.ndarray]:
        H, W, D = patch_size
        
        # Anterior-Posterior gradient (front-back)
        ap_gradient = np.linspace(0, 1, H)[:, None, None]
        ap_gradient = np.broadcast_to(ap_gradient, (H, W, D))
        
        # Dorsal-Ventral gradient (top-bottom)
        dv_gradient = np.linspace(0, 1, W)[None, :, None]
        dv_gradient = np.broadcast_to(dv_gradient, (H, W, D))
        
        # Medial-Lateral gradient (left-right)
        ml_gradient = np.linspace(0, 1, D)[None, None, :]
        ml_gradient = np.broadcast_to(ml_gradient, (H, W, D))
        
        return {
            'anterior_posterior': ap_gradient,
            'dorsal_ventral': dv_gradient,
            'medial_lateral': ml_gradient
        }


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

class DataAugmentation:
    """GPU-optimized data augmentation."""
    
    def __init__(self, config: ColabTrainingConfig):
        self.config = config
    
    def random_rotation(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1) < 0.5:
            # Random 90-degree rotations only (k * 90 degrees)
            k = torch.randint(-1, 2, (1,)).item()  # -1, 0, or 1
            
            if k != 0:
                # Choose random axis to rotate around
                axis = torch.randint(0, 3, (1,)).item()
                
                # Image has shape [C, H, W, D] - dims [0, 1, 2, 3]
                # Label has shape [H, W, D] - dims [0, 1, 2]
                
                if axis == 0:  # Rotate in W-D plane
                    image = torch.rot90(image, k, [2, 3])
                    label = torch.rot90(label, k, [1, 2])
                elif axis == 1:  # Rotate in H-D plane  
                    image = torch.rot90(image, k, [1, 3])
                    label = torch.rot90(label, k, [0, 2])
                else:  # axis == 2, Rotate in H-W plane
                    image = torch.rot90(image, k, [1, 2])
                    label = torch.rot90(label, k, [0, 1])
        
        return image, label
    
    def random_flip(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Random flips along each spatial axis
        # Image shape: [C, H, W, D]
        # Label shape: [H, W, D]
        
        if torch.rand(1) < 0.3:  # Flip along H axis
            image = torch.flip(image, [1])
            label = torch.flip(label, [0])
            
        if torch.rand(1) < 0.3:  # Flip along W axis
            image = torch.flip(image, [2])
            label = torch.flip(label, [1])
            
        if torch.rand(1) < 0.3:  # Flip along D axis
            image = torch.flip(image, [3])
            label = torch.flip(label, [2])
        
        return image, label
    
    def intensity_augmentation(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < 0.5:
            # Gamma correction
            gamma = torch.rand(1) * 0.8 + 0.7  # [0.7, 1.5]
            image[:1] = torch.pow(torch.clamp(image[:1], min=0), gamma)  # Only T2 channel
            
            # Add noise
            noise = torch.randn_like(image[:1]) * 0.05
            image[:1] = image[:1] + noise
        
        return image
    
    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1) < self.config.augmentation_prob:
            image, label = self.random_rotation(image, label)
            image, label = self.random_flip(image, label)
            image = self.intensity_augmentation(image)
        
        return image, label


# ============================================================================
# DATASET
# ============================================================================

class BrainstemDataset(Dataset):
    """Dataset with morphogen integration and GPU optimization."""
    
    def __init__(self, config: ColabTrainingConfig, imaging_path: str, labels_path: str, is_train: bool = True):
        self.config = config
        self.is_train = is_train
        
        # Load data
        self.imaging_data = np.load(imaging_path)
        self.labels = np.load(labels_path)
        
        # Ensure same dimensions
        if self.imaging_data.shape != self.labels.shape:
            min_shape = np.minimum(self.imaging_data.shape, self.labels.shape)
            self.imaging_data = self.imaging_data[:min_shape[0], :min_shape[1], :min_shape[2]]
            self.labels = self.labels[:min_shape[0], :min_shape[1], :min_shape[2]]
        
        # Generate morphogens once
        self.morphogen_generator = MorphogenFieldGenerator()
        
        # Data augmentation
        self.augmentation = DataAugmentation(config) if is_train else None
        
    def __len__(self) -> int:
        return self.config.samples_per_epoch
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        H, W, D = self.imaging_data.shape
        patch_h, patch_w, patch_d = self.config.patch_size
        
        # Random patch extraction
        h = np.random.randint(0, max(1, H - patch_h + 1))
        w = np.random.randint(0, max(1, W - patch_w + 1))
        d = np.random.randint(0, max(1, D - patch_d + 1))
        
        # Extract patches
        img_patch = self.imaging_data[h:h+patch_h, w:w+patch_w, d:d+patch_d]
        label_patch = self.labels[h:h+patch_h, w:w+patch_w, d:d+patch_d]
        
        # Pad if needed
        if img_patch.shape != self.config.patch_size:
            pad_h = patch_h - img_patch.shape[0]
            pad_w = patch_w - img_patch.shape[1]
            pad_d = patch_d - img_patch.shape[2]
            
            img_patch = np.pad(img_patch, ((0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')
            label_patch = np.pad(label_patch, ((0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')
        
        # Generate morphogen
        morphogen_dict = self.morphogen_generator.generate_morphogen_coarse_map(self.config.patch_size)
        morphogen_patch = np.stack([
            morphogen_dict['anterior_posterior'],
            morphogen_dict['dorsal_ventral'],
            morphogen_dict['medial_lateral']
        ], axis=0)
        
        # Combine T2 and morphogen
        img_patch = img_patch[np.newaxis, :, :, :]  # Add channel dimension
        combined_input = np.concatenate([img_patch, morphogen_patch], axis=0)
        
        # Convert to tensors
        combined_input = torch.from_numpy(combined_input).float()
        label_patch = torch.from_numpy(label_patch).long()
        
        # Apply augmentation
        if self.augmentation is not None:
            combined_input, label_patch = self.augmentation(combined_input, label_patch)
        
        return combined_input, label_patch


# ============================================================================
# TRAINER
# ============================================================================

class ColabTrainer:
    """GPU-optimized trainer for Google Colab."""
    
    def __init__(self, config: ColabTrainingConfig):
        self.config = config
        self.device = device
        
        # Model
        self.model = EnhancedUNet3D(config).to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M")
        
        # Loss
        self.criterion = CombinedSegmentationLoss(config)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.scheduler_T_0,
            T_mult=config.scheduler_T_mult,
            eta_min=config.min_lr
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision and torch.cuda.is_available() else None
        
        # Tracking
        self.best_dice = 0.0
        self.train_losses = []
        self.val_dices = []
        self.early_stop_counter = 0
    
    def compute_dice_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute Dice score for validation."""
        pred = torch.argmax(pred, dim=1)
        dice_scores = []
        
        for cls in range(1, self.config.num_classes):  # Skip background
            pred_cls = (pred == cls).float()
            target_cls = (target == cls).float()
            
            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum()
            
            if union > 0:
                dice = (2.0 * intersection / union).item()
                dice_scores.append(dice)
        
        return np.mean(dice_scores) if dice_scores else 0.0
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)
                
                self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}: Loss = {loss.item():.4f}", end='\r')
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        dice_scores = []
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                dice_scores.append(self.compute_dice_score(outputs, targets))
        
        avg_loss = total_loss / len(dataloader)
        avg_dice = np.mean(dice_scores)
        
        return avg_loss, avg_dice
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset) -> Dict[str, Any]:
        """Full training loop."""
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"\n{'='*60}")
        print(f"Starting training on {device}")
        print(f"Target: Dice ≥ 0.87")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_dice = self.validate(val_loader)
            self.val_dices.append(val_dice)
            
            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print progress
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Dice: {val_dice:.4f} {'🎯 TARGET MET!' if val_dice >= 0.87 else ''}")
            print(f"  LR: {current_lr:.6f}")
            
            # Save best model
            if val_dice > self.best_dice:
                self.best_dice = val_dice
                self.early_stop_counter = 0
                
                # Save checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_dice': self.best_dice,
                    'config': self.config
                }, 'best_model_colab.pth')
                
                print(f"  ✅ New best model saved! Dice: {self.best_dice:.4f}")
            else:
                self.early_stop_counter += 1
            
            # Early stopping
            if val_dice >= 0.87:
                print(f"\n🎯 Target achieved! Dice: {val_dice:.4f} ≥ 0.87")
                break
            
            if self.early_stop_counter >= self.config.patience:
                print(f"\n⚠️ Early stopping triggered. Best Dice: {self.best_dice:.4f}")
                break
        
        training_time = time.time() - start_time
        
        results = {
            'best_dice': self.best_dice,
            'final_dice': val_dice,
            'total_epochs': epoch + 1,
            'training_time': training_time,
            'train_losses': self.train_losses,
            'val_dices': self.val_dices,
            'target_achieved': self.best_dice >= 0.87
        }
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"  Best Dice: {self.best_dice:.4f}")
        print(f"  Target Achieved: {'✅ YES' if results['target_achieved'] else '❌ NO'}")
        print(f"  Training Time: {training_time/60:.1f} minutes")
        print(f"  Final Model: best_model_colab.pth")
        print(f"{'='*60}\n")
        
        return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_colab_training():
    """Main training function for Google Colab."""
    
    # Check if data files exist
    if not os.path.exists('imaging_data_normalized.npy'):
        print("❌ Please upload imaging_data_normalized.npy to Colab")
        return
    
    if not os.path.exists('labels.npy'):
        print("❌ Please upload labels.npy to Colab")
        return
    
    # Configuration
    config = ColabTrainingConfig()
    
    # Create datasets
    train_dataset = BrainstemDataset(
        config,
        'imaging_data_normalized.npy',
        'labels.npy',
        is_train=True
    )
    
    val_dataset = BrainstemDataset(
        config,
        'imaging_data_normalized.npy',
        'labels.npy',
        is_train=False
    )
    
    # Create trainer
    trainer = ColabTrainer(config)
    
    # Train
    results = trainer.train(train_dataset, val_dataset)
    
    # Save results
    with open('training_results.json', 'w') as f:
        json.dump({
            'best_dice': float(results['best_dice']),
            'final_dice': float(results['final_dice']),
            'total_epochs': int(results['total_epochs']),
            'training_time': float(results['training_time']),
            'target_achieved': bool(results['target_achieved']),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print("✅ Training complete! Download these files:")
    print("   1. best_model_colab.pth (trained weights)")
    print("   2. training_results.json (metrics)")
    print("\nIntegrate back to Quark with:")
    print("   /Users/camdouglas/quark/brain/modules/brainstem_segmentation/")
    
    return results


if __name__ == "__main__":
    # Run training
    results = run_colab_training()
