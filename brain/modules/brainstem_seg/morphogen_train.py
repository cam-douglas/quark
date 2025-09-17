#!/usr/bin/env python3
"""
Morphogen-Augmented Training Script

Trains the morphogen-augmented ViT-GNN model with spatial priors.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nibabel as nib
from pathlib import Path
import json
import logging
from datetime import datetime
import sys

# Add modules to path
sys.path.append(str(Path(__file__).parent.parent / "brainstem_segmentation"))
from morphogen_integration import MorphogenAugmentedViTGNN, MorphogenDataProcessor, MorphogenConfig


class MorphogenBrainstemDataset:
    """Dataset with morphogen augmentation."""
    
    def __init__(self, data_dir: Path, patch_size=(64, 64, 64)):
        self.data_dir = data_dir
        self.patch_size = patch_size
        
        # Initialize morphogen processor
        self.morphogen_processor = MorphogenDataProcessor()
        
        # Load NextBrain data
        self.volume_path = data_dir / "nextbrain" / "T2w.nii.gz"
        self.labels_path = data_dir / "nextbrain" / "manual_segmentation.nii.gz"
        
        # Load data
        self.volume = self._load_and_normalize_volume()
        self.labels = self._load_and_map_labels()
        
        # Generate morphogen priors for full volume
        self.morphogen_priors = self._generate_morphogen_priors()
        
        # Pre-extract patches
        self.patches = self._extract_patches()
        
        print(f"Morphogen dataset loaded: {len(self.patches)} patches")
    
    def _load_and_normalize_volume(self) -> np.ndarray:
        """Load and normalize volume."""
        img = nib.load(self.volume_path)
        volume = img.get_fdata().astype(np.float32)
        
        # Z-score normalization
        volume = (volume - volume.mean()) / (volume.std() + 1e-8)
        
        return volume
    
    def _load_and_map_labels(self) -> np.ndarray:
        """Load and map labels to simplified schema."""
        img = nib.load(self.labels_path)
        labels = img.get_fdata().astype(np.int32)
        
        # Simple mapping: just use a few key structures
        mapped_labels = np.zeros_like(labels)
        
        # Map specific NextBrain labels to our classes
        label_mapping = {
            0: 0,   # Background
            4: 1,   # Red Nucleus
            9: 2,   # Brain-Stem
            29: 3,  # Pontine Nuclei
            85: 4,  # Inferior Colliculus
            99: 5,  # Medulla
        }
        
        for original, mapped in label_mapping.items():
            mapped_labels[labels == original] = mapped
        
        return mapped_labels
    
    def _generate_morphogen_priors(self) -> np.ndarray:
        """Generate morphogen priors for the full volume."""
        
        print("Generating morphogen priors for full volume...")
        morphogen_tensor = self.morphogen_processor.create_morphogen_priors_for_volume(
            self.volume.shape
        )
        
        print(f"✅ Morphogen priors generated: {morphogen_tensor.shape}")
        return morphogen_tensor.numpy()
    
    def _extract_patches(self):
        """Extract training patches with morphogen priors."""
        patches = []
        stride = 32  # 50% overlap
        
        h, w, d = self.volume.shape
        
        # Only extract patches where we can get full-size patches
        for z in range(0, d - self.patch_size[2] + 1, stride):
            for y in range(0, w - self.patch_size[1] + 1, stride):
                for x in range(0, h - self.patch_size[0] + 1, stride):
                    
                    # Ensure we don't go out of bounds
                    x_end = min(x + self.patch_size[0], h)
                    y_end = min(y + self.patch_size[1], w)
                    z_end = min(z + self.patch_size[2], d)
                    
                    # Skip if patch would be too small
                    if (x_end - x != self.patch_size[0] or 
                        y_end - y != self.patch_size[1] or 
                        z_end - z != self.patch_size[2]):
                        continue
                    
                    # Extract patches
                    vol_patch = self.volume[x:x_end, y:y_end, z:z_end]
                    label_patch = self.labels[x:x_end, y:y_end, z:z_end]
                    morphogen_patch = self.morphogen_priors[:, x:x_end, y:y_end, z:z_end]
                    
                    # Only keep patches with some foreground
                    if np.sum(label_patch > 0) > 50:
                        patches.append((vol_patch, label_patch, morphogen_patch))
        
        return patches
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        volume, labels, morphogen = self.patches[idx]
        
        # Convert to tensors
        volume_tensor = torch.from_numpy(volume).unsqueeze(0).float()  # Add channel dim
        labels_tensor = torch.from_numpy(labels).long()
        morphogen_tensor = torch.from_numpy(morphogen).float()  # Already has channel dim
        
        return volume_tensor, morphogen_tensor, labels_tensor


class MorphogenTrainer:
    """Trainer for morphogen-augmented model."""
    
    def __init__(self, data_dir: Path, output_dir: Path, device="cpu"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.device = device
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_file = output_dir / f"morphogen_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )
        
        # Create dataset
        self.dataset = MorphogenBrainstemDataset(data_dir)
        
        # Create morphogen-augmented model
        self.model = MorphogenAugmentedViTGNN(
            input_channels=1,
            morphogen_channels=3,
            embed_dim=256,  # Smaller for CPU
            vit_layers=2,   # Reduced
            gnn_layers=1,   # Reduced
            num_heads=4,    # Reduced
            num_classes=6,  # Background + 5 structures
            morphogen_weight=0.3
        ).to(device)
        
        # Simple loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
        # Morphogen processor for checkpoints
        self.morphogen_processor = MorphogenDataProcessor()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        logging.info(f"Morphogen-augmented model created with {total_params:,} parameters")
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        # Simple batch processing
        batch_size = 2
        num_batches = len(self.dataset) // batch_size
        
        for batch_idx in range(num_batches):
            # Manual batching
            batch_volumes = []
            batch_morphogens = []
            batch_labels = []
            
            for i in range(batch_size):
                idx = (batch_idx * batch_size + i) % len(self.dataset)
                vol, morph, lab = self.dataset[idx]
                batch_volumes.append(vol)
                batch_morphogens.append(morph)
                batch_labels.append(lab)
            
            # Stack into batches with error handling
            try:
                volumes = torch.stack(batch_volumes).to(self.device)
                morphogens = torch.stack(batch_morphogens).to(self.device)
                labels = torch.stack(batch_labels).to(self.device)
            except RuntimeError as e:
                if "stack expects each tensor to be equal size" in str(e):
                    print(f"Skipping batch {batch_idx} due to size mismatch")
                    continue
                else:
                    raise e
            
            # Forward pass
            self.optimizer.zero_grad()
            
            try:
                # Pass both imaging and morphogen data
                outputs = self.model(volumes, morphogens)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                
                # Calculate accuracy
                pred = outputs.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.numel()
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                          f"Loss: {loss.item():.4f}, "
                          f"Acc: {100*correct/total:.2f}%")
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        accuracy = 100 * correct / max(total, 1)
        
        logging.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
        
        return avg_loss, accuracy
    
    def train(self, num_epochs=3):
        """Main training loop."""
        
        print(f"Starting morphogen-augmented training for {num_epochs} epochs...")
        
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            try:
                loss, acc = self.train_epoch(epoch)
                
                # Save best model
                if loss < best_loss:
                    best_loss = loss
                    
                    # Save morphogen-augmented checkpoint
                    self.morphogen_processor.save_morphogen_checkpoint(
                        self.model, self.optimizer, epoch, 
                        {'loss': loss, 'accuracy': acc},
                        self.output_dir / 'best_morphogen_model.pth'
                    )
                    
                    print(f"✅ New best morphogen model saved! Loss: {loss:.4f}")
                
            except Exception as e:
                print(f"❌ Error in epoch {epoch}: {e}")
                logging.error(f"Training error in epoch {epoch}: {e}", exc_info=True)
                continue
        
        print(f"Morphogen training completed! Best loss: {best_loss:.4f}")
        return best_loss


def main():
    """Run morphogen-augmented training."""
    
    print("🧬 MORPHOGEN-AUGMENTED BRAINSTEM SEGMENTATION TRAINING")
    print("=" * 70)
    
    # Paths
    data_dir = Path("/Users/camdouglas/quark/data/datasets/brainstem_segmentation")
    output_dir = Path("/Users/camdouglas/quark/data/models/brainstem_segmentation/morphogen")
    
    # Check data
    if not (data_dir / "nextbrain" / "T2w.nii.gz").exists():
        print("❌ NextBrain data not found!")
        return False
    
    print("✅ Data found, starting morphogen-augmented training...")
    
    # Create trainer
    trainer = MorphogenTrainer(data_dir, output_dir, device="cpu")
    
    # Train
    best_loss = trainer.train(num_epochs=3)  # Short demo
    
    print(f"\n🎉 Morphogen-augmented training completed!")
    print(f"   Best loss: {best_loss:.4f}")
    print(f"   Model saved: {output_dir}/best_morphogen_model.pth")
    print(f"   Morphogen priors: 3 gradient fields integrated")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
