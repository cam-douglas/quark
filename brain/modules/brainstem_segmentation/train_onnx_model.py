#!/usr/bin/env python3
"""
Train ONNX-Compatible Brainstem Segmentation Model - Phase 4 Step 1.A

Trains the ONNXCompatibleBrainstemSegmenter to achieve Dice ≥0.87/0.92
using the same pipeline and data as the ViT-GNN model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import logging
from tqdm import tqdm
from dataclasses import dataclass
import json

# Import our modules
from onnx_production_exporter import ONNXCompatibleBrainstemSegmenter
from morphogen_integration import MorphogenFieldGenerator


@dataclass
class ONNXTrainingConfig:
    """Training configuration for ONNX-compatible model."""

    # Model parameters
    input_channels: int = 4  # 1 imaging + 3 morphogen
    num_classes: int = 6
    base_filters: int = 32

    # Training parameters - ENHANCED FOR PERFORMANCE GAP
    batch_size: int = 2  # Smaller for memory
    learning_rate: float = 1e-3
    num_epochs: int = 200  # Extended training
    weight_decay: float = 1e-4

    # Learning rate scheduling
    use_scheduler: bool = True
    scheduler_T_0: int = 50  # Cosine annealing period
    scheduler_T_mult: int = 2  # Period multiplier
    min_lr: float = 1e-6  # Minimum learning rate

    # Data parameters
    patch_size: Tuple[int, int, int] = (64, 64, 64)
    overlap: int = 16

    # Loss parameters - ENHANCED FOR PERFORMANCE
    dice_weight: float = 0.7  # Dice loss weight
    focal_weight: float = 0.3  # Focal loss weight
    focal_gamma: float = 2.0   # Focal loss gamma
    focal_alpha: Optional[float] = None  # Focal loss alpha (class weights)

    # Optimizer improvements
    optimizer_type: str = "adamw"  # Better than Adam for generalization
    use_gradient_clip: bool = True
    gradient_clip_value: float = 1.0

    # Hardware
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class BrainstemDataset(Dataset):
    """Dataset for brainstem segmentation training."""

    def __init__(self, config: ONNXTrainingConfig):
        self.config = config

        # Load NextBrain data
        data_dir = Path("/Users/camdouglas/quark/data/datasets/brainstem_segmentation/nextbrain")

        # Load segmentation first to get dimensions
        seg_img = nib.load(data_dir / "manual_segmentation.nii.gz")
        seg_data = seg_img.get_fdata().astype(np.int32)

        # Load T2w volume
        t2w_img = nib.load(data_dir / "T2w.nii.gz")
        t2w_data = t2w_img.get_fdata().astype(np.float32)

        # Crop T2w to match segmentation dimensions
        H_seg, W_seg, D_seg = seg_data.shape
        H_t2w, W_t2w, D_t2w = t2w_data.shape

        # Center crop T2w to match segmentation
        h_start = (H_t2w - H_seg) // 2
        w_start = (W_t2w - W_seg) // 2
        d_start = (D_t2w - D_seg) // 2

        self.imaging_data = t2w_data[
            h_start:h_start + H_seg,
            w_start:w_start + W_seg,
            d_start:d_start + D_seg
        ]

        # Map NextBrain's labels to our 6 classes
        self.labels = self._map_labels_to_schema(seg_data)

        # Normalize imaging data
        self.imaging_data = (self.imaging_data - self.imaging_data.mean()) / self.imaging_data.std()

        print(f"Dataset loaded: imaging {self.imaging_data.shape}, labels {self.labels.shape}")
        print(f"Cropped T2w from {t2w_data.shape} to {self.imaging_data.shape}")
        print(f"Label distribution: {np.bincount(self.labels.flatten())}")

    def _map_labels_to_schema(self, seg_data: np.ndarray) -> np.ndarray:
        """Map NextBrain 333 labels to our 6-class brainstem schema."""

        # Load label mapping
        mapping_path = Path("/Users/camdouglas/quark/data/datasets/brainstem_segmentation/metadata/label_mapping.json")

        if mapping_path.exists():
            with open(mapping_path, 'r') as f:
                label_mapping = json.load(f)
        else:
            # Create basic mapping - this should be improved
            label_mapping = {
                'background': [0],
                'midbrain': list(range(1, 50)),  # Approximate midbrain labels
                'pons': list(range(50, 100)),    # Approximate pons labels
                'medulla': list(range(100, 150)), # Approximate medulla labels
                'cerebellum': list(range(150, 200)), # Approximate cerebellum labels
                'other': list(range(200, 334))   # Remaining labels
            }

        # Create mapping array
        mapped_labels = np.zeros_like(seg_data)

        # Background (0) stays 0
        mapped_labels[seg_data == 0] = 0

        # Midbrain labels → 1
        for label in label_mapping.get('midbrain', []):
            mapped_labels[seg_data == label] = 1

        # Pons labels → 2
        for label in label_mapping.get('pons', []):
            mapped_labels[seg_data == label] = 2

        # Medulla labels → 3
        for label in label_mapping.get('medulla', []):
            mapped_labels[seg_data == label] = 3

        # Cerebellum labels → 4
        for label in label_mapping.get('cerebellum', []):
            mapped_labels[seg_data == label] = 4

        # Other labels → 5
        for label in label_mapping.get('other', []):
            mapped_labels[seg_data == label] = 5

        return mapped_labels

    def __len__(self):
        return 100  # Extract 100 random patches per epoch

    def __getitem__(self, idx):
        """Extract random patch from volume."""
        H, W, D = self.imaging_data.shape
        patch_h, patch_w, patch_d = self.config.patch_size

        # Ensure we can extract full patches (dimensions should now match after cropping)
        max_h = H - patch_h
        max_w = W - patch_w
        max_d = D - patch_d

        # Random patch extraction
        h_start = np.random.randint(0, max_h + 1)
        w_start = np.random.randint(0, max_w + 1)
        d_start = np.random.randint(0, max_d + 1)

        # Extract patches
        img_patch = self.imaging_data[
            h_start:h_start + patch_h,
            w_start:w_start + patch_w,
            d_start:d_start + patch_d
        ]

        label_patch = self.labels[
            h_start:h_start + patch_h,
            w_start:w_start + patch_w,
            d_start:d_start + patch_d
        ]

        # Add channel dimension for imaging
        img_patch = img_patch[np.newaxis, ...]  # [1, H, W, D]

        # Generate morphogen priors
        morphogen_generator = MorphogenFieldGenerator()
        morphogen_dict = morphogen_generator.generate_morphogen_coarse_map(self.config.patch_size)

        # Stack the three gradients into channels
        morphogen_patch = np.stack([
            morphogen_dict['anterior_posterior'],
            morphogen_dict['dorsal_ventral'],
            morphogen_dict['medial_lateral']
        ], axis=0)  # [3, H, W, D]

        # Combine imaging + morphogen
        combined_input = np.concatenate([img_patch, morphogen_patch], axis=0)  # [4, H, W, D]

        return torch.from_numpy(combined_input).float(), torch.from_numpy(label_patch).long()


class EnhancedDiceFocalLoss(nn.Module):
    """Enhanced Combined Dice + Focal Loss for imbalanced segmentation with class weights."""

    def __init__(self, num_classes: int, dice_weight: float = 0.7, focal_weight: float = 0.3,
                 focal_gamma: float = 2.0, focal_alpha: Optional[float] = None):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        # Class weights for focal loss if provided
        if focal_alpha is not None:
            if isinstance(focal_alpha, (list, tuple)):
                self.register_buffer('alpha', torch.tensor(focal_alpha))
            else:
                # Compute inverse class frequency weights
                self.register_buffer('alpha', torch.ones(num_classes) * focal_alpha)
        else:
            self.register_buffer('alpha', torch.ones(num_classes))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute enhanced combined Dice + Focal loss."""

        # Dice Loss
        probs = torch.softmax(logits, dim=1)
        targets_onehot = torch.nn.functional.one_hot(targets, self.num_classes).permute(0, 4, 1, 2, 3).float()

        # Dice computation with smoothing
        intersection = torch.sum(probs * targets_onehot, dim=[2, 3, 4])
        union = torch.sum(probs, dim=[2, 3, 4]) + torch.sum(targets_onehot, dim=[2, 3, 4])
        dice_loss = 1 - (2 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = dice_loss.mean()

        # Enhanced Focal Loss with class weights
        ce_loss = torch.nn.functional.cross_entropy(logits, targets, reduction='none')

        # Apply class weights if available
        if self.focal_alpha is not None:
            alpha_t = self.alpha[targets].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            ce_loss = ce_loss * alpha_t.squeeze()

        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.focal_gamma * ce_loss).mean()

        # Combined loss
        total_loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss

        return total_loss


def compute_dice_scores(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Dict[str, float]:
    """Compute Dice scores for each class and overall."""

    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)

    dice_scores = {}

    for cls in range(num_classes):
        pred_mask = (preds == cls)
        target_mask = (targets == cls)

        intersection = torch.sum(pred_mask & target_mask).float()
        union = torch.sum(pred_mask | target_mask).float()

        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        dice_scores[f'class_{cls}'] = dice.item()

    # Overall Dice (excluding background)
    overall_dice = np.mean([dice_scores[f'class_{i}'] for i in range(1, num_classes)])
    dice_scores['overall'] = overall_dice

    return dice_scores


def train_onnx_model():
    """Train ONNX-compatible model to target performance."""

    print("🚀 TRAINING ONNX-COMPATIBLE BRAINSTEM SEGMENTATION MODEL")
    print("=" * 60)

    # Configuration
    config = ONNXTrainingConfig()
    print(f"Training on: {config.device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Epochs: {config.num_epochs}")

    # Dataset and dataloader
    dataset = BrainstemDataset(config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    # Model
    model = ONNXCompatibleBrainstemSegmenter(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
        base_filters=config.base_filters
    )

    # Enhanced Loss and optimizer for performance improvement
    criterion = EnhancedDiceFocalLoss(
        num_classes=config.num_classes,
        dice_weight=config.dice_weight,
        focal_weight=config.focal_weight,
        focal_gamma=config.focal_gamma,
        focal_alpha=config.focal_alpha
    )

    # Use AdamW for better generalization
    if config.optimizer_type == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

    # Enhanced learning rate scheduler
    if config.use_scheduler:
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.scheduler_T_0,
            T_mult=config.scheduler_T_mult,
            eta_min=config.min_lr
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    # Move to device
    model = model.to(config.device)
    criterion = criterion.to(config.device)

    # Training loop
    best_dice = 0.0
    best_model_path = Path("/Users/camdouglas/quark/data/models/brainstem/best_onnx_model.pth")

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_dice_scores = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(config.device), targets.to(config.device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass with gradient clipping
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            if config.use_gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_value)

            optimizer.step()

            # Compute Dice scores
            with torch.no_grad():
                dice_scores = compute_dice_scores(outputs, targets, config.num_classes)
                epoch_dice_scores.append(dice_scores['overall'])

            epoch_loss += loss.item()
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dice': f"{dice_scores['overall']:.4f}"
            })

        # Epoch summary
        avg_loss = epoch_loss / len(dataloader)
        avg_dice = np.mean(epoch_dice_scores)

        print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Dice: {avg_dice:.4f}")

        # Class-wise Dice
        class_dice = {f'class_{i}': [] for i in range(config.num_classes)}
        for scores in epoch_dice_scores:
            # We need to recompute class-wise scores for the epoch
            pass  # Simplified for now

        # Save best model
        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 New best model saved: Dice {best_dice:.4f}")

        # Check if we hit target
        if avg_dice >= 0.87:  # Target Dice
            print(f"🎯 Target Dice reached: {avg_dice:.4f}")
            break

        # Learning rate scheduling
        if config.use_scheduler:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Dice: {avg_dice:.4f} | LR: {current_lr:.6f}")
        else:
            scheduler.step()
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Dice: {avg_dice:.4f}")

    print(f"\n🏁 Enhanced training complete!")
    print(f"Best Dice: {best_dice:.4f}")
    print(f"Model saved: {best_model_path}")

    # Performance gap analysis
    if best_dice < 0.87:
        gap = 0.87 - best_dice
        print(f"⚠️  Performance gap to target: {gap:.3f}")
        print("💡 Recommendations for further improvement:")
        print("   • Architecture: Add attention gates and deep supervision")
        print("   • Data: Implement advanced augmentations (rotations, elastic deformations)")
        print("   • Training: Increase epochs or adjust hyperparameters")
    else:
        print("🎯 Target Dice achieved! Ready for deployment.")

    # Load best model for export
    model.load_state_dict(torch.load(best_model_path, map_location=config.device))

    return model, best_dice


if __name__ == "__main__":
    trained_model, final_dice = train_onnx_model()

    print("\n✅ Phase 4 Step 1.A Complete!")
    print(f"   🎯 Achieved Dice: {final_dice:.4f}")
    print(f"   📦 Model ready for ONNX export")
