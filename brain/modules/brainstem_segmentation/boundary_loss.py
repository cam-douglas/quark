#!/usr/bin/env python3
"""
Boundary Loss Implementation for Brainstem Segmentation
Combines Dice, Focal, and Boundary losses for improved class separation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import Optional, Tuple


class BoundaryLoss(nn.Module):
    """Boundary loss based on distance transform maps."""
    
    def __init__(self, num_classes: int):
        """
        Args:
            num_classes: Number of segmentation classes
        """
        super(BoundaryLoss, self).__init__()
        self.num_classes = num_classes
    
    def compute_distance_map(self, mask: np.ndarray) -> np.ndarray:
        """
        Compute distance transform map for a binary mask.
        
        Args:
            mask: Binary mask (H, W, D)
            
        Returns:
            Distance map with positive values inside and negative outside
        """
        # Compute distance from boundary
        pos_dist = distance_transform_edt(mask)
        neg_dist = distance_transform_edt(1 - mask)
        
        # Signed distance: positive inside, negative outside
        distance_map = pos_dist - neg_dist
        
        return distance_map
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
               distance_maps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate boundary loss.
        
        Args:
            pred: Predicted logits [B, C, H, W, D]
            target: Ground truth labels [B, H, W, D]
            distance_maps: Precomputed distance maps [B, C, H, W, D]
            
        Returns:
            Boundary loss value
        """
        batch_size = pred.shape[0]
        
        # Convert predictions to probabilities
        pred_probs = F.softmax(pred, dim=1)
        
        # Compute distance maps if not provided
        if distance_maps is None:
            distance_maps = torch.zeros_like(pred_probs)
            
            for b in range(batch_size):
                for c in range(self.num_classes):
                    # Create binary mask for this class
                    mask = (target[b] == c).cpu().numpy().astype(np.uint8)
                    
                    # Compute distance map
                    if mask.sum() > 0:  # Only if class is present
                        dist_map = self.compute_distance_map(mask)
                        distance_maps[b, c] = torch.from_numpy(dist_map).to(pred.device)
        
        # Compute boundary loss as weighted sum
        # Loss is higher for predictions far from boundaries
        boundary_loss = torch.mean(pred_probs * distance_maps)
        
        return boundary_loss


class CombinedSegmentationLoss(nn.Module):
    """Combined loss with Dice, Focal, and Boundary components."""
    
    def __init__(self,
                 num_classes: int,
                 dice_weight: float = 0.5,
                 focal_weight: float = 0.3,
                 boundary_weight: float = 0.2,
                 focal_gamma: float = 2.0,
                 focal_alpha: Optional[torch.Tensor] = None):
        """
        Args:
            num_classes: Number of segmentation classes
            dice_weight: Weight for Dice loss
            focal_weight: Weight for Focal loss
            boundary_weight: Weight for Boundary loss
            focal_gamma: Gamma parameter for Focal loss
            focal_alpha: Class weights for Focal loss
        """
        super(CombinedSegmentationLoss, self).__init__()
        
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        
        # Initialize loss components
        self.boundary_loss = BoundaryLoss(num_classes) if boundary_weight > 0 else None
        
        # Focal loss parameters
        self.focal_gamma = focal_gamma
        if focal_alpha is not None:
            self.register_buffer('focal_alpha', focal_alpha)
        else:
            self.register_buffer('focal_alpha', torch.ones(num_classes))
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate Dice loss."""
        
        # Convert to one-hot encoding
        pred_probs = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 4, 1, 2, 3).float()
        
        # Calculate Dice for each class
        dice_losses = []
        
        for c in range(1, self.num_classes):  # Skip background
            pred_c = pred_probs[:, c]
            target_c = target_onehot[:, c]
            
            intersection = torch.sum(pred_c * target_c, dim=[1, 2, 3])
            union = torch.sum(pred_c, dim=[1, 2, 3]) + torch.sum(target_c, dim=[1, 2, 3])
            
            dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
            dice_losses.append(1.0 - dice)
        
        return torch.mean(torch.stack(dice_losses))
    
    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate Focal loss."""
        
        # Cross-entropy loss
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        # Get predicted probabilities for true class
        pred_probs = F.softmax(pred, dim=1)
        pt = torch.gather(pred_probs, 1, target.unsqueeze(1)).squeeze(1)
        
        # Apply class weights if available
        alpha_t = self.focal_alpha[target]
        
        # Focal term
        focal_term = (1 - pt) ** self.focal_gamma
        
        # Combined focal loss
        focal_loss = alpha_t * focal_term * ce_loss
        
        return focal_loss.mean()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
               distance_maps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate combined loss.
        
        Args:
            pred: Predicted logits [B, C, H, W, D]
            target: Ground truth labels [B, H, W, D]
            distance_maps: Optional precomputed distance maps
            
        Returns:
            Combined loss value
        """
        total_loss = 0.0
        loss_components = {}
        
        # Dice loss
        if self.dice_weight > 0:
            dice = self.dice_loss(pred, target)
            total_loss += self.dice_weight * dice
            loss_components['dice'] = dice.item()
        
        # Focal loss
        if self.focal_weight > 0:
            focal = self.focal_loss(pred, target)
            total_loss += self.focal_weight * focal
            loss_components['focal'] = focal.item()
        
        # Boundary loss
        if self.boundary_weight > 0 and self.boundary_loss is not None:
            boundary = self.boundary_loss(pred, target, distance_maps)
            total_loss += self.boundary_weight * boundary
            loss_components['boundary'] = boundary.item()
        
        # Store components for logging
        self.last_loss_components = loss_components
        
        return total_loss


class AdaptiveLossWeighting(nn.Module):
    """Adaptive loss weighting that adjusts weights during training."""
    
    def __init__(self, 
                 base_loss: CombinedSegmentationLoss,
                 initial_weights: Optional[dict] = None,
                 adaptation_rate: float = 0.01):
        """
        Args:
            base_loss: Base combined loss function
            initial_weights: Initial loss component weights
            adaptation_rate: Rate of weight adaptation
        """
        super(AdaptiveLossWeighting, self).__init__()
        
        self.base_loss = base_loss
        self.adaptation_rate = adaptation_rate
        
        # Initialize learnable weights
        if initial_weights:
            weights = torch.tensor([
                initial_weights.get('dice', 0.5),
                initial_weights.get('focal', 0.3),
                initial_weights.get('boundary', 0.2)
            ])
        else:
            weights = torch.tensor([0.5, 0.3, 0.2])
        
        self.log_weights = nn.Parameter(torch.log(weights))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate loss with adaptive weighting."""
        
        # Get normalized weights
        weights = F.softmax(self.log_weights, dim=0)
        
        # Update base loss weights
        self.base_loss.dice_weight = weights[0]
        self.base_loss.focal_weight = weights[1]
        self.base_loss.boundary_weight = weights[2]
        
        # Calculate loss
        loss = self.base_loss(pred, target)
        
        return loss


def test_boundary_loss():
    """Test the boundary loss implementation."""
    
    print("🎯 TESTING BOUNDARY LOSS")
    print("=" * 50)
    
    # Create synthetic data
    batch_size = 2
    num_classes = 6
    spatial_size = 32
    
    pred = torch.randn(batch_size, num_classes, spatial_size, spatial_size, spatial_size)
    target = torch.randint(0, num_classes, (batch_size, spatial_size, spatial_size, spatial_size))
    
    print("✅ Test data created")
    print(f"   Prediction shape: {pred.shape}")
    print(f"   Target shape: {target.shape}")
    
    # Test boundary loss alone
    print("\n📊 Testing Boundary Loss:")
    boundary_loss = BoundaryLoss(num_classes)
    b_loss = boundary_loss(pred, target)
    print(f"   Boundary loss: {b_loss.item():.4f}")
    
    # Test combined loss
    print("\n🔄 Testing Combined Loss:")
    combined_loss = CombinedSegmentationLoss(
        num_classes=num_classes,
        dice_weight=0.5,
        focal_weight=0.3,
        boundary_weight=0.2,
        focal_gamma=2.0
    )
    
    total_loss = combined_loss(pred, target)
    print(f"   Combined loss: {total_loss.item():.4f}")
    print(f"   Components: {combined_loss.last_loss_components}")
    
    # Test adaptive weighting
    print("\n⚖️ Testing Adaptive Loss Weighting:")
    adaptive_loss = AdaptiveLossWeighting(
        base_loss=combined_loss,
        initial_weights={'dice': 0.6, 'focal': 0.3, 'boundary': 0.1}
    )
    
    adaptive_total = adaptive_loss(pred, target)
    print(f"   Adaptive loss: {adaptive_total.item():.4f}")
    
    weights = F.softmax(adaptive_loss.log_weights, dim=0)
    print(f"   Current weights: Dice={weights[0]:.3f}, Focal={weights[1]:.3f}, Boundary={weights[2]:.3f}")
    
    # Test gradient flow
    print("\n🔙 Testing gradient flow:")
    adaptive_total.backward()
    print("   ✅ Gradients computed successfully")
    
    print("\n✅ Boundary loss implementation ready!")
    return combined_loss


if __name__ == "__main__":
    test_boundary_loss()
