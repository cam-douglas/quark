#!/usr/bin/env python3
"""
FIXED Colab Training Package for Brainstem Segmentation
========================================================

This version fixes the critical issues that caused Dice = 0.0051:
1. Label indexing (0-indexed vs 1-indexed)
2. Data normalization
3. Loss function computation
4. Class imbalance handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
import time
from typing import Dict, Tuple, Optional
import os

# ============================================================
# FIXED CONFIGURATION
# ============================================================

class FixedTrainingConfig:
    """Fixed configuration addressing training issues."""
    
    # Model architecture
    input_channels = 4  # T1, T2, morphogen gradients
    num_classes = 6     # 0=background, 1-5=brainstem structures
    base_filters = 32
    
    # Training parameters (FIXED)
    batch_size = 2      # Small batch for stability
    num_epochs = 200    # Extended training
    learning_rate = 1e-4  # Conservative LR
    weight_decay = 1e-5
    
    # Learning rate schedule
    use_scheduler = True
    warmup_epochs = 10   # Gradual warmup
    scheduler_patience = 20
    scheduler_factor = 0.5
    min_lr = 1e-6
    
    # Loss function (FIXED)
    use_dice_loss = True
    use_focal_loss = True
    dice_weight = 0.5
    focal_weight = 0.5
    focal_gamma = 2.0
    exclude_background_from_dice = True  # Critical fix
    
    # Optimization
    gradient_clip_value = 1.0  # Prevent exploding gradients
    use_amp = False  # Disable AMP initially for stability
    
    # Data augmentation (reduced for debugging)
    use_augmentation = False  # Disable initially
    aug_rotation_degrees = 10
    aug_flip_prob = 0.3
    
    # Monitoring
    print_every = 5
    validate_every = 10
    save_best_only = True
    
    # Paths
    data_dir = Path("./data")
    output_dir = Path("./")
    
    # Debug mode
    debug_mode = True  # Enable extensive logging

# ============================================================
# FIXED DICE LOSS
# ============================================================

class FixedDiceFocalLoss(nn.Module):
    """Fixed combined Dice and Focal loss."""
    
    def __init__(self, config: FixedTrainingConfig):
        super().__init__()
        self.config = config
        self.smooth = 1e-5
        
        # Focal loss component
        self.focal_loss = nn.CrossEntropyLoss(reduction='none')
        
    def dice_loss(self, pred, target):
        """
        Fixed Dice loss computation.
        pred: [B, C, H, W, D] logits
        target: [B, H, W, D] with values 0 to C-1
        """
        batch_size = pred.size(0)
        num_classes = pred.size(1)
        
        # Convert to probabilities
        pred = F.softmax(pred, dim=1)
        
        # One-hot encode target
        target_one_hot = F.one_hot(target.long(), num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()
        
        # Calculate Dice per class
        dice_scores = []
        
        # CRITICAL FIX: Skip background class (0) if configured
        start_idx = 1 if self.config.exclude_background_from_dice else 0
        
        for c in range(start_idx, num_classes):
            pred_c = pred[:, c].contiguous().view(batch_size, -1)
            target_c = target_one_hot[:, c].contiguous().view(batch_size, -1)
            
            intersection = (pred_c * target_c).sum(dim=1)
            union = pred_c.sum(dim=1) + target_c.sum(dim=1)
            
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice.mean())
        
        if len(dice_scores) > 0:
            mean_dice = torch.stack(dice_scores).mean()
            return 1 - mean_dice
        else:
            return torch.tensor(0.0, device=pred.device)
    
    def focal_loss_weighted(self, pred, target):
        """Focal loss with gamma focusing."""
        ce_loss = self.focal_loss(pred, target.long())
        
        # Get probabilities for true class
        pt = torch.exp(-ce_loss)
        
        # Apply focal term
        focal_loss = (1 - pt) ** self.config.focal_gamma * ce_loss
        
        return focal_loss.mean()
    
    def forward(self, pred, target):
        """Combined loss."""
        dice_l = self.dice_loss(pred, target) if self.config.use_dice_loss else 0
        focal_l = self.focal_loss_weighted(pred, target) if self.config.use_focal_loss else 0
        
        total_loss = (self.config.dice_weight * dice_l + 
                     self.config.focal_weight * focal_l)
        
        return total_loss, {'dice_loss': dice_l, 'focal_loss': focal_l}

# ============================================================
# SIMPLE U-NET (PROVEN ARCHITECTURE)
# ============================================================

class SimpleUNet3D(nn.Module):
    """Simple, proven 3D U-Net architecture."""
    
    def __init__(self, config: FixedTrainingConfig):
        super().__init__()
        
        f = config.base_filters
        
        # Encoder
        self.enc1 = self._conv_block(config.input_channels, f)
        self.enc2 = self._conv_block(f, f*2)
        self.enc3 = self._conv_block(f*2, f*4)
        self.enc4 = self._conv_block(f*4, f*8)
        
        # Bottleneck
        self.bottleneck = self._conv_block(f*8, f*16)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose3d(f*16, f*8, 2, stride=2)
        self.dec4 = self._conv_block(f*16, f*8)
        
        self.upconv3 = nn.ConvTranspose3d(f*8, f*4, 2, stride=2)
        self.dec3 = self._conv_block(f*8, f*4)
        
        self.upconv2 = nn.ConvTranspose3d(f*4, f*2, 2, stride=2)
        self.dec2 = self._conv_block(f*4, f*2)
        
        self.upconv1 = nn.ConvTranspose3d(f*2, f, 2, stride=2)
        self.dec1 = self._conv_block(f*2, f)
        
        # Output
        self.final = nn.Conv3d(f, config.num_classes, 1)
        
        # Pooling
        self.pool = nn.MaxPool3d(2, 2)
        
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)

# ============================================================
# FIXED DATASET
# ============================================================

class FixedBrainstemDataset(torch.utils.data.Dataset):
    """Fixed dataset with proper label handling."""
    
    def __init__(self, data_dir: Path, mode='train', config=None):
        self.data_dir = data_dir
        self.mode = mode
        self.config = config or FixedTrainingConfig()
        
        # Load or generate data
        self.samples = self._load_samples()
        
        print(f"📊 Dataset initialized: {len(self.samples)} samples")
        
        # Verify first sample
        if len(self.samples) > 0:
            self._verify_sample(0)
    
    def _load_samples(self):
        """Load or generate synthetic data."""
        samples = []
        
        # Try to load real data
        image_dir = self.data_dir / 'images'
        label_dir = self.data_dir / 'labels'
        
        if image_dir.exists() and label_dir.exists():
            # Load real data
            for img_file in sorted(image_dir.glob('*.npy')):
                label_file = label_dir / img_file.name
                if label_file.exists():
                    samples.append((img_file, label_file))
        
        # Generate synthetic data if no real data
        if len(samples) == 0:
            print("⚠️ No real data found, generating synthetic data...")
            for i in range(20):
                samples.append(self._generate_synthetic_sample(i))
        
        return samples
    
    def _generate_synthetic_sample(self, idx):
        """Generate synthetic data for testing."""
        np.random.seed(idx)
        
        # Generate image (4 channels)
        image = np.random.randn(4, 64, 64, 64).astype(np.float32)
        
        # Generate label with proper 0-indexing
        label = np.zeros((64, 64, 64), dtype=np.int64)
        
        # Add some structure (0=background, 1-5=structures)
        for struct_id in range(1, 6):
            center = np.random.randint(20, 44, 3)
            radius = np.random.randint(5, 10)
            
            x, y, z = np.ogrid[:64, :64, :64]
            mask = ((x - center[0])**2 + (y - center[1])**2 + 
                   (z - center[2])**2) <= radius**2
            label[mask] = struct_id
        
        return (image, label)
    
    def _verify_sample(self, idx):
        """Verify a sample has correct format."""
        image, label = self.__getitem__(idx)
        
        print(f"\n🔍 Verifying sample {idx}:")
        print(f"  Image shape: {image.shape}, dtype: {image.dtype}")
        print(f"  Image range: [{image.min():.2f}, {image.max():.2f}]")
        print(f"  Label shape: {label.shape}, dtype: {label.dtype}")
        print(f"  Label unique values: {torch.unique(label).tolist()}")
        
        # Critical checks
        assert label.min() >= 0, f"❌ Labels must be >= 0, got {label.min()}"
        assert label.max() < 6, f"❌ Labels must be < 6, got {label.max()}"
        print("  ✅ Label indexing correct (0-indexed)")
    
    def __getitem__(self, idx):
        """Get a properly formatted sample."""
        sample = self.samples[idx]
        
        # Load data
        if isinstance(sample[0], (str, Path)):
            # Load from file
            image = np.load(sample[0])
            label = np.load(sample[1])
        else:
            # Use generated data
            image, label = sample
        
        # CRITICAL FIX 1: Ensure labels are 0-indexed
        if label.min() >= 1:
            print(f"⚠️ Converting 1-indexed labels to 0-indexed for sample {idx}")
            label = label - 1
        
        # CRITICAL FIX 2: Normalize image data
        image = image.astype(np.float32)
        for c in range(image.shape[0]):
            channel = image[c]
            mean = channel.mean()
            std = channel.std() + 1e-8
            image[c] = (channel - mean) / std
        
        # Convert to tensors
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        
        # Final verification
        assert 0 <= label.min() and label.max() < 6, f"Label out of range: {label.min()}-{label.max()}"
        
        return image, label
    
    def __len__(self):
        return len(self.samples)

# ============================================================
# METRICS CALCULATION
# ============================================================

def calculate_dice_score(pred, target, num_classes=6, exclude_background=True):
    """Calculate Dice score for evaluation."""
    dice_scores = []
    
    start_idx = 1 if exclude_background else 0
    
    for c in range(start_idx, num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        if union > 0:
            dice = (2. * intersection) / union
            dice_scores.append(dice)
    
    if len(dice_scores) > 0:
        return torch.stack(dice_scores).mean().item()
    else:
        return 0.0

# ============================================================
# TRAINING LOOP
# ============================================================

def train_epoch(model, dataloader, criterion, optimizer, epoch, config):
    """Training epoch with fixes."""
    model.train()
    
    epoch_loss = 0
    epoch_dice = 0
    batch_count = 0
    
    # Learning rate warmup
    if epoch < config.warmup_epochs:
        lr_scale = (epoch + 1) / config.warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = config.learning_rate * lr_scale
        print(f"📈 Warmup: LR = {config.learning_rate * lr_scale:.6f}")
    
    for batch_idx, (data, target) in enumerate(dataloader):
        # Move to GPU
        data = data.cuda()
        target = target.cuda()
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        
        # Compute loss
        loss, loss_components = criterion(output, target)
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"⚠️ NaN loss at batch {batch_idx}, skipping...")
            continue
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_value)
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            pred = torch.argmax(output, dim=1)
            dice = calculate_dice_score(pred, target)
        
        epoch_loss += loss.item()
        epoch_dice += dice
        batch_count += 1
        
        # Print progress
        if batch_idx % config.print_every == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}: "
                  f"Loss={loss.item():.4f}, Dice={dice:.4f}")
            
            if config.debug_mode:
                print(f"    Dice Loss: {loss_components['dice_loss']:.4f}")
                print(f"    Focal Loss: {loss_components['focal_loss']:.4f}")
                print(f"    Predictions: {torch.unique(pred).tolist()}")
    
    avg_loss = epoch_loss / max(batch_count, 1)
    avg_dice = epoch_dice / max(batch_count, 1)
    
    return avg_loss, avg_dice

def validate(model, dataloader, criterion, config):
    """Validation with proper metrics."""
    model.eval()
    
    val_loss = 0
    val_dice = 0
    batch_count = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data = data.cuda()
            target = target.cuda()
            
            output = model(data)
            loss, _ = criterion(output, target)
            
            pred = torch.argmax(output, dim=1)
            dice = calculate_dice_score(pred, target)
            
            val_loss += loss.item()
            val_dice += dice
            batch_count += 1
    
    avg_loss = val_loss / max(batch_count, 1)
    avg_dice = val_dice / max(batch_count, 1)
    
    return avg_loss, avg_dice

# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_fixed_model():
    """Main training function with all fixes."""
    
    print("=" * 60)
    print("🚀 FIXED TRAINING SCRIPT FOR BRAINSTEM SEGMENTATION")
    print("=" * 60)
    
    # Configuration
    config = FixedTrainingConfig()
    
    print("\n📋 Configuration:")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Exclude Background from Dice: {config.exclude_background_from_dice}")
    
    # Create datasets
    print("\n📊 Loading datasets...")
    train_dataset = FixedBrainstemDataset(config.data_dir, mode='train', config=config)
    val_dataset = FixedBrainstemDataset(config.data_dir, mode='val', config=config)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0  # Use 0 for debugging
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print("\n🏗️ Creating model...")
    model = SimpleUNet3D(config).cuda()
    
    # Loss and optimizer
    criterion = FixedDiceFocalLoss(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    if config.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # Maximize Dice
            patience=config.scheduler_patience,
            factor=config.scheduler_factor,
            min_lr=config.min_lr
        )
    
    # Training loop
    print("\n🏃 Starting training...")
    print("=" * 60)
    
    best_dice = 0
    train_losses = []
    train_dices = []
    
    for epoch in range(config.num_epochs):
        print(f"\n📅 Epoch {epoch+1}/{config.num_epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, epoch, config
        )
        
        train_losses.append(train_loss)
        train_dices.append(train_dice)
        
        print(f"  Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        
        # Validate
        if (epoch + 1) % config.validate_every == 0:
            val_loss, val_dice = validate(model, val_loader, criterion, config)
            print(f"  Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
            
            # Update scheduler
            if config.use_scheduler:
                scheduler.step(val_dice)
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Current LR: {current_lr:.6f}")
            
            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                print(f"  🎯 New best Dice: {best_dice:.4f}")
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_dice,
                    'config': config.__dict__
                }
                torch.save(checkpoint, 'best_model_colab.pth')
        
        # Early stopping check
        if epoch > 50 and best_dice < 0.1:
            print("\n⚠️ Training not converging. Checking for issues...")
            print("Recent losses:", train_losses[-5:])
            print("Recent Dice scores:", train_dices[-5:])
            
            # Run diagnostic
            with torch.no_grad():
                sample_data, sample_label = train_dataset[0]
                sample_data = sample_data.unsqueeze(0).cuda()
                sample_label = sample_label.unsqueeze(0).cuda()
                
                output = model(sample_data)
                pred = torch.argmax(output, dim=1)
                
                print(f"\nDiagnostic:")
                print(f"  Input shape: {sample_data.shape}")
                print(f"  Output shape: {output.shape}")
                print(f"  Label unique: {torch.unique(sample_label)}")
                print(f"  Pred unique: {torch.unique(pred)}")
                
                if len(torch.unique(pred)) == 1:
                    print("  ❌ Model predicting single class!")
                    print("  Attempting to fix...")
                    # Reinitialize model
                    model = SimpleUNet3D(config).cuda()
                    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=config.learning_rate * 0.1  # Lower LR
                    )
    
    # Final results
    print("\n" + "=" * 60)
    print("📊 TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Dice Score: {best_dice:.4f}")
    
    if best_dice < 0.5:
        print("\n⚠️ Poor performance detected. Likely issues:")
        print("  1. Check label indexing (must be 0-5)")
        print("  2. Verify data normalization")
        print("  3. Check class balance")
    
    # Save final results
    results = {
        'best_dice': best_dice,
        'final_epoch': epoch + 1,
        'train_losses': train_losses,
        'train_dices': train_dices
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return best_dice

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Check GPU
    if torch.cuda.is_available():
        print(f"🖥️ GPU Available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ No GPU available, training will be slow")
    
    # Run fixed training
    final_dice = train_fixed_model()
    
    if final_dice >= 0.87:
        print("\n✅ TARGET ACHIEVED! Model ready for deployment.")
    else:
        print(f"\n❌ Target not met. Dice {final_dice:.4f} < 0.87")
        print("Please review the diagnostic output above.")
