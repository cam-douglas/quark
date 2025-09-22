# -*- coding: utf-8 -*-
"""
This script is the entry point for training the GNNViTHybridModel on Google Cloud AI Platform.
It handles argument parsing, model initialization, data loading, and the main training loop.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os

from brain.modules.morphogen_solver.gnn_vit_hybrid import GNNViTHybridModel
from brain.modules.morphogen_solver.synthetic_data_generator import SyntheticEmbryoDataGenerator
from brain.modules.morphogen_solver.spatial_grid import GridDimensions
from brain.modules.morphogen_solver.ml_diffusion_types import SyntheticDataConfig

def get_args():
    """Parses command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train the Brainstem Segmentation Model on GCP.")
    
    # --- Hyperparameters ---
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training. Keep small for 3D data.')

    # --- Data Generation ---
    parser.add_argument('--num_samples', type=int, default=100, help='Number of synthetic samples to generate.')
    parser.add_argument('--grid_size', type=int, default=32, help='Size of the 3D grid for synthetic data (e.g., 32 for 32x32x32).')
    
    # --- GCP & Data Paths ---
    parser.add_argument('--model_dir', type=str, required=True, help='GCS path to save model checkpoints and artifacts.')
    parser.add_argument('--log_dir', type=str, required=True, help='GCS path for TensorBoard logs.')
    
    return parser.parse_args()

def main():
    """Main training function."""
    args = get_args()

    # --- Setup ---
    print("--- Starting Training Setup ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=args.log_dir)
    print(f"TensorBoard logs will be written to: {args.log_dir}")

    # --- Data Generation ---
    print(f"Generating {args.num_samples} synthetic samples with grid size {args.grid_size}x{args.grid_size}x{args.grid_size}...")
    grid_dims = GridDimensions(x_size=args.grid_size, y_size=args.grid_size, z_size=args.grid_size, resolution=10.0)
    data_config = SyntheticDataConfig(num_samples=args.num_samples, noise_level=0.05)
    data_generator = SyntheticEmbryoDataGenerator(base_grid_dimensions=grid_dims, data_config=data_config)
    
    # Generate data and create a train/validation split
    full_dataset = data_generator.generate_synthetic_dataset()
    train_data, val_data = data_generator.create_train_val_split(full_dataset, val_fraction=0.2)

    # Create PyTorch Datasets and DataLoaders
    # Input: morphogen concentrations (N, C, D, H, W), Target: also morphogen concentrations (for auto-encoder style training)
    X_train = torch.from_numpy(train_data['morphogen_concentrations']).float()
    y_train = torch.from_numpy(train_data['morphogen_concentrations']).float()
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    X_val = torch.from_numpy(val_data['morphogen_concentrations']).float()
    y_val = torch.from_numpy(val_data['morphogen_concentrations']).float()
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print("Data generation and dataloaders created successfully.")
    
    # --- Model Initialization ---
    print("Initializing model...")
    num_morphogens = full_dataset['metadata']['morphogen_order'].__len__()
    model = GNNViTHybridModel(
        input_channels=num_morphogens,  # Channels are the morphogens
        num_classes=num_morphogens,     # Output is the reconstructed morphogen fields
        vit_embed_dim=768,              # ViT embedding dimension (divisible by 12 heads)
        gnn_hidden_dim=128,             # GNN hidden dimension
        fusion_dim=128,                 # Fusion dimension
        input_resolution=args.grid_size # Grid size for spatial resolution
    ).to(device)
    print("Model initialized successfully.")

    # --- Optimizer and Loss Function ---
    print("Initializing optimizer and loss function...")
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss() # Mean Squared Error for reconstruction loss
    print("Optimizer and loss function initialized.")

    # --- Training Loop ---
    print("\n--- Starting Training Loop ---")
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            model_output = model(data)
            output = model_output['segmentation_logits']  # Extract the segmentation logits
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # --- Logging ---
            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}')
        
        avg_train_loss = total_train_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        print(f'====> Epoch: {epoch+1} Average training loss: {avg_train_loss:.4f}')

        # --- Validation Loop ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                model_output = model(data)
                output = model_output['segmentation_logits']  # Extract the segmentation logits
                total_val_loss += criterion(output, target).item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        print(f'====> Epoch: {epoch+1} Average validation loss: {avg_val_loss:.4f}')

        # --- Checkpointing ---
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.model_dir, f"checkpoint_epoch_{epoch+1}.pth")
            # For GCS, we'd need to use a library like gcsfs, but for now, we assume a local or mounted path.
            # a more robust solution would be to use torch.save to a local path and then upload to GCS.
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    writer.close()
    print("\n--- Training Finished ---")

if __name__ == '__main__':
    main()
