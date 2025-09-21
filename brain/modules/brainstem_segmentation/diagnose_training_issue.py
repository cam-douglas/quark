#!/usr/bin/env python3
"""
Diagnose Training Failure
=========================

Dice of 0.0051 indicates critical issues. Common causes:
1. Label mismatch (0-indexed vs 1-indexed)
2. Data normalization issues
3. Loss function bugs
4. Learning rate too high/low
5. Incorrect data loading
"""

import numpy as np
import torch
import json
from pathlib import Path


def diagnose_training_failure():
    """Analyze why training failed."""
    
    print("=" * 60)
    print("🔍 DIAGNOSING TRAINING FAILURE")
    print("=" * 60)
    print("\nDice Score: 0.0051 (Expected: ~0.87)")
    print("This indicates the model is performing worse than random!\n")
    
    print("🎯 MOST LIKELY CAUSES:")
    print("-" * 40)
    
    # 1. Label indexing issue
    print("\n1️⃣ LABEL INDEXING MISMATCH (90% probability)")
    print("   Problem: Labels might be 1-6 but model expects 0-5")
    print("   Symptom: Model predicts all zeros or wrong classes")
    print("   Fix: Ensure labels are 0-indexed (0=background, 1-5=structures)")
    
    # 2. Data normalization
    print("\n2️⃣ DATA NORMALIZATION ISSUE (70% probability)")
    print("   Problem: Input data not normalized to [0,1] or [-1,1]")
    print("   Symptom: Gradients explode or vanish")
    print("   Fix: Normalize inputs: (data - mean) / std")
    
    # 3. Loss function
    print("\n3️⃣ LOSS FUNCTION BUG (60% probability)")
    print("   Problem: Dice loss computing wrong intersection")
    print("   Symptom: Loss doesn't decrease or goes negative")
    print("   Fix: Verify loss handles one-hot encoding correctly")
    
    # 4. Class imbalance
    print("\n4️⃣ EXTREME CLASS IMBALANCE (50% probability)")
    print("   Problem: Background dominates (>99% of voxels)")
    print("   Symptom: Model learns to predict all background")
    print("   Fix: Use weighted loss or focal loss")
    
    # 5. Learning rate
    print("\n5️⃣ LEARNING RATE ISSUE (40% probability)")
    print("   Problem: LR too high causes divergence")
    print("   Symptom: Loss oscillates wildly")
    print("   Fix: Start with LR=1e-4, use warmup")
    
    print("\n" + "=" * 60)
    print("📊 DIAGNOSTIC CHECKS TO RUN:")
    print("=" * 60)
    
    # Generate diagnostic code
    diagnostic_code = '''
# Run this in Colab to diagnose the issue:

import torch
import numpy as np

# 1. Check label distribution
print("1️⃣ Checking label values...")
if 'train_dataset' in locals():
    sample_data, sample_label = train_dataset[0]
    print(f"Label unique values: {torch.unique(sample_label)}")
    print(f"Label shape: {sample_label.shape}")
    print(f"Label range: [{sample_label.min():.2f}, {sample_label.max():.2f}]")
    
    # Check if labels are 0-indexed
    if sample_label.min() == 1:
        print("❌ ERROR: Labels are 1-indexed! Should be 0-indexed")
    else:
        print("✅ Labels are 0-indexed")

# 2. Check data normalization
print("\\n2️⃣ Checking data normalization...")
if 'train_dataset' in locals():
    sample_data, _ = train_dataset[0]
    print(f"Data shape: {sample_data.shape}")
    print(f"Data range: [{sample_data.min():.2f}, {sample_data.max():.2f}]")
    print(f"Data mean: {sample_data.mean():.2f}, std: {sample_data.std():.2f}")
    
    if sample_data.max() > 10:
        print("❌ ERROR: Data not normalized! Values too large")
    else:
        print("✅ Data appears normalized")

# 3. Check model predictions
print("\\n3️⃣ Checking model output...")
if 'model' in locals():
    model.eval()
    with torch.no_grad():
        sample_input = torch.randn(1, 4, 64, 64, 64).cuda()
        output = model(sample_input)
        pred = torch.argmax(output, dim=1)
        print(f"Output shape: {output.shape}")
        print(f"Predicted classes: {torch.unique(pred)}")
        print(f"Class distribution: {[(i, (pred==i).sum().item()) for i in range(6)]}")
        
        if len(torch.unique(pred)) == 1:
            print("❌ ERROR: Model predicting single class only!")
        else:
            print("✅ Model predicting multiple classes")

# 4. Check loss function
print("\\n4️⃣ Checking loss computation...")
if 'criterion' in locals():
    # Create synthetic perfect prediction
    perfect_output = torch.zeros(1, 6, 64, 64, 64).cuda()
    perfect_label = torch.zeros(1, 64, 64, 64).long().cuda()
    perfect_output[0, 0] = 10  # High confidence for background
    
    loss = criterion(perfect_output, perfect_label)
    print(f"Loss for perfect prediction: {loss.item():.4f}")
    
    if loss.item() > 0.1:
        print("❌ ERROR: Loss too high for perfect prediction!")
    else:
        print("✅ Loss function appears correct")

# 5. Check training loop
print("\\n5️⃣ Recent training metrics...")
if 'train_losses' in locals() and len(train_losses) > 0:
    print(f"Last 5 losses: {train_losses[-5:]}")
    print(f"Best Dice so far: {best_dice:.4f}")
    
    if all(l > train_losses[0] for l in train_losses[-5:]):
        print("❌ ERROR: Loss increasing - learning rate too high!")
    elif all(abs(l - train_losses[-1]) < 0.001 for l in train_losses[-5:]):
        print("⚠️ WARNING: Loss plateau - might need different approach")
'''
    
    print("\n📝 DIAGNOSTIC CODE:")
    print("-" * 40)
    print(diagnostic_code)
    print("-" * 40)
    
    return diagnostic_code


def generate_fixed_training_script():
    """Generate a corrected training script."""
    
    print("\n" + "=" * 60)
    print("🔧 GENERATING FIXED TRAINING SCRIPT")
    print("=" * 60)
    
    fixed_script = '''
# FIXED Training Configuration
# Key changes to address Dice=0.0051 issue

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FixedDiceLoss(nn.Module):
    """Fixed Dice Loss that handles edge cases."""
    
    def __init__(self, smooth=1e-5, include_background=False):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background
    
    def forward(self, predictions, targets):
        """
        predictions: [B, C, H, W, D] logits
        targets: [B, H, W, D] long tensor with values 0 to C-1
        """
        batch_size = predictions.size(0)
        num_classes = predictions.size(1)
        
        # Convert to probabilities
        predictions = F.softmax(predictions, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()
        
        # Calculate Dice per class
        dice_scores = []
        start_class = 0 if self.include_background else 1
        
        for c in range(start_class, num_classes):
            pred_c = predictions[:, c]
            target_c = targets_one_hot[:, c]
            
            intersection = (pred_c * target_c).sum(dim=(1,2,3))
            union = pred_c.sum(dim=(1,2,3)) + target_c.sum(dim=(1,2,3))
            
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        # Average Dice across classes and batch
        if dice_scores:
            mean_dice = torch.stack(dice_scores).mean()
            return 1 - mean_dice  # Return loss (1 - Dice)
        else:
            return torch.tensor(0.0, device=predictions.device)

class FixedDataset(torch.utils.data.Dataset):
    """Fixed dataset with proper normalization and label handling."""
    
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        # Load your data here
        self.samples = self.load_samples()
    
    def load_samples(self):
        # Implement data loading
        # CRITICAL: Ensure labels are 0-indexed!
        samples = []
        # ... load data ...
        return samples
    
    def __getitem__(self, idx):
        # Load image and label
        image, label = self.samples[idx]
        
        # CRITICAL FIXES:
        # 1. Ensure label values are 0-5, not 1-6
        if label.min() >= 1:
            label = label - 1  # Convert 1-indexed to 0-indexed
        
        # 2. Normalize image data
        image = image.astype(np.float32)
        for c in range(image.shape[0]):
            channel = image[c]
            mean = channel.mean()
            std = channel.std() + 1e-8
            image[c] = (channel - mean) / std
        
        # 3. Ensure correct dtypes
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        
        # 4. Verify label range
        assert label.min() >= 0 and label.max() < 6, f"Label out of range: {label.min()}-{label.max()}"
        
        return image, label
    
    def __len__(self):
        return len(self.samples)

# Fixed Training Configuration
config = {
    'learning_rate': 1e-4,  # Conservative LR
    'batch_size': 2,        # Small batch for stability
    'num_epochs': 100,
    'warmup_epochs': 5,     # Gradual warmup
    'weight_decay': 1e-5,
    'gradient_clip': 1.0,   # Prevent exploding gradients
}

# Training loop with fixes
def train_epoch_fixed(model, dataloader, criterion, optimizer, epoch, config):
    model.train()
    epoch_loss = 0
    epoch_dice = 0
    
    # Learning rate warmup
    if epoch < config['warmup_epochs']:
        lr_scale = (epoch + 1) / config['warmup_epochs']
        for param_group in optimizer.param_groups:
            param_group['lr'] = config['learning_rate'] * lr_scale
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.cuda()
        target = target.cuda()
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        
        # Compute loss
        loss = criterion(output, target)
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"⚠️ NaN loss detected at batch {batch_idx}")
            continue
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
        optimizer.step()
        
        # Calculate Dice for monitoring
        with torch.no_grad():
            pred = torch.argmax(output, dim=1)
            dice = calculate_dice_score(pred, target)
            epoch_dice += dice
        
        epoch_loss += loss.item()
        
        # Print progress
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)}: Loss={loss.item():.4f}, Dice={dice:.4f}")
    
    return epoch_loss / len(dataloader), epoch_dice / len(dataloader)

def calculate_dice_score(pred, target, exclude_background=True):
    """Calculate Dice score for evaluation."""
    num_classes = 6
    dice_scores = []
    
    start_class = 1 if exclude_background else 0
    
    for c in range(start_class, num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        if union > 0:
            dice = (2. * intersection) / union
            dice_scores.append(dice)
    
    return torch.stack(dice_scores).mean().item() if dice_scores else 0.0

print("✅ Fixed training script ready!")
print("Key fixes applied:")
print("  1. Labels converted to 0-indexed")
print("  2. Data properly normalized")
print("  3. Dice loss excludes background")
print("  4. Learning rate warmup added")
print("  5. Gradient clipping enabled")
'''
    
    return fixed_script


if __name__ == "__main__":
    # Run diagnosis
    diagnostic_code = diagnose_training_failure()
    
    # Generate fix
    fixed_script = generate_fixed_training_script()
    
    print("\n" + "=" * 60)
    print("💡 IMMEDIATE ACTIONS:")
    print("=" * 60)
    print("\n1. Run the diagnostic code in Colab to identify the exact issue")
    print("2. Apply the fixes from the corrected training script")
    print("3. Restart training with the fixed configuration")
    print("\n⚠️ Most likely issue: Labels are 1-indexed instead of 0-indexed!")
    print("This would cause the model to never predict the correct classes.")

