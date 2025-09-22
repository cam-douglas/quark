# -*- coding: utf-8 -*-
"""
Valid GNN-ViT Hybrid Training Script for Google Cloud VM
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import os
import sys

# Ensure the project root is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from brain.modules.morphogen_solver.gnn_vit_hybrid import GNNViTHybrid
from brain.modules.morphogen_solver.synthetic_data_generator import SyntheticEmbryoDataGenerator
from brain.modules.morphogen_solver.spatial_grid import SpatialGrid

def dice_score(pred, target, smooth=1.):
    """Calculates the Dice score for a batch."""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()

class SyntheticDataset(Dataset):
    """A PyTorch dataset to wrap the SyntheticEmbryoDataGenerator."""
    def __init__(self, generator, num_samples):
        self.generator = generator
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        data = self.generator.generate_single_sample()
        # Ensure data is in the correct format (C, D, H, W) and is a tensor
        input_tensor = torch.from_numpy(data['morphogen_volumes']).float().unsqueeze(0) # Add channel dim
        target_tensor = torch.from_numpy(data['segmentation_mask']).float()
        return input_tensor, target_tensor

def main():
    parser = argparse.ArgumentParser(description='Valid GNN-ViT Hybrid Training')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Optimizer learning rate.')
    parser.add_argument('--num_samples_train', type=int, default=200, help='Number of training samples.')
    parser.add_argument('--num_samples_val', type=int, default=50, help='Number of validation samples.')
    parser.add_argument('--grid_size', type=int, default=32, help='Grid size of the synthetic data.')
    parser.add_argument('--model_dir', type=str, default='./models', help='Directory to save the trained model.')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save TensorBoard logs.')
    parser.add_argument('--device', type=str, default='auto', help='Device to use for training (e.g., "cpu", "cuda", "auto").')
    args = parser.parse_args()

    print("🧠 Starting VALID Quark GNN-ViT Hybrid Training")
    print("=" * 50)
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🔧 Using device: {device}")

    # 1. Create the full, valid model
    model = GNNViTHybrid(
        input_channels=4,  # SHH, BMP, WNT, FGF
        num_classes=4,     # As defined in the original architecture
        grid_size=args.grid_size
    ).to(device)
    print(f"🏗️ Full GNN-ViT Hybrid model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # 2. Setup the valid data generator and dataloaders
    grid = SpatialGrid((args.grid_size, args.grid_size, args.grid_size))
    data_generator = SyntheticEmbryoDataGenerator(grid)
    
    train_dataset = SyntheticDataset(data_generator, args.num_samples_train)
    val_dataset = SyntheticDataset(data_generator, args.num_samples_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    print("📦 Dataloaders created with biologically-relevant synthetic data.")

    # 3. Setup training
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss() # More stable than separate Sigmoid + BCELoss
    writer = SummaryWriter(args.log_dir)

    print("\n🚀 Starting full training and validation loop...")
    for epoch in range(args.epochs):
        # --- Training Step ---
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs['segmentation_logits'], targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_dice += dice_score(outputs['segmentation_logits'], targets).item()
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Dice/Train', avg_train_dice, epoch)

        # --- Validation Step ---
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs['segmentation_logits'], targets)
                val_loss += loss.item()
                val_dice += dice_score(outputs['segmentation_logits'], targets).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Dice/Validation', avg_val_dice, epoch)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Train Dice: {avg_train_dice:.4f} | Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f}")

    print("\n✅ Training completed!")

    # --- Save Model ---
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, 'gnn_vit_hybrid_VALID.pth')
    torch.save(model.state_dict(), model_path)
    print(f"💾 Valid model saved to {model_path}")
    
    writer.close()
    print("🎉 VALID training pipeline completed successfully!")

if __name__ == "__main__":
    main()