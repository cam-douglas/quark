"""
Loss function comparison framework for brainstem segmentation.

Implements comprehensive ablation study comparing Dice, focal, boundary,
and hierarchical losses to optimize segmentation performance.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LossConfig:
    """Configuration for loss function experiments."""
    
    dice_weight: float = 0.7
    focal_weight: float = 0.3
    boundary_weight: float = 0.2
    hierarchy_weight: float = 0.1
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25


class ComprehensiveLoss(nn.Module):
    """Unified loss combining all loss types for ablation studies."""
    
    def __init__(self, config: LossConfig, num_classes: int = 16):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        
        # Nucleus to subdivision mapping for hierarchy
        self.nucleus_to_subdivision = {
            0: 0,  # background
            # Midbrain (1-4) → subdivision 1
            1: 1, 2: 1, 3: 1, 4: 1,
            # Pons (5-8) → subdivision 2
            5: 2, 6: 2, 7: 2, 8: 2,
            # Medulla (9-12) → subdivision 3
            9: 3, 10: 3, 11: 3, 12: 3,
            # Additional nuclei → general brainstem
            13: 4, 14: 4, 15: 4
        }
    
    def dice_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss with smoothing."""
        probs = F.softmax(inputs, dim=1)
        targets_onehot = F.one_hot(targets, self.num_classes).permute(0, 4, 1, 2, 3).float()
        
        intersection = torch.sum(probs * targets_onehot, dim=[2, 3, 4])
        union = torch.sum(probs, dim=[2, 3, 4]) + torch.sum(targets_onehot, dim=[2, 3, 4])
        
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1 - dice.mean()
        
        return dice_loss
    
    def focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss for class imbalance."""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.config.focal_alpha * (1 - pt) ** self.config.focal_gamma * ce_loss
        
        return focal_loss.mean()
    
    def boundary_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute boundary-aware loss."""
        def compute_gradients(tensor: torch.Tensor) -> torch.Tensor:
            grad_x = torch.abs(tensor[:, :, 1:, :, :] - tensor[:, :, :-1, :, :])
            grad_y = torch.abs(tensor[:, :, :, 1:, :] - tensor[:, :, :, :-1, :])
            grad_z = torch.abs(tensor[:, :, :, :, 1:] - tensor[:, :, :, :, :-1])
            
            grad_x = F.pad(grad_x, (0, 0, 0, 0, 0, 1))
            grad_y = F.pad(grad_y, (0, 0, 0, 1, 0, 0))
            grad_z = F.pad(grad_z, (0, 1, 0, 0, 0, 0))
            
            return grad_x + grad_y + grad_z
        
        target_boundaries = compute_gradients(targets.float().unsqueeze(1))
        pred_softmax = F.softmax(inputs, dim=1)
        pred_boundaries = compute_gradients(pred_softmax).sum(dim=1, keepdim=True)
        
        return F.mse_loss(pred_boundaries, target_boundaries)
    
    def hierarchy_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute hierarchical consistency loss."""
        # Convert nucleus predictions to subdivision predictions
        subdivision_targets = torch.zeros_like(targets)
        for nucleus_id, subdivision_id in self.nucleus_to_subdivision.items():
            subdivision_targets[targets == nucleus_id] = subdivision_id
        
        # Aggregate nucleus probabilities to subdivision level
        subdivision_probs = torch.zeros(inputs.shape[0], 5, *inputs.shape[2:], device=inputs.device)
        for nucleus_id, subdivision_id in self.nucleus_to_subdivision.items():
            if nucleus_id < inputs.shape[1]:  # Valid nucleus index
                subdivision_probs[:, subdivision_id] += inputs[:, nucleus_id]
        
        subdivision_loss = F.cross_entropy(subdivision_probs, subdivision_targets)
        return subdivision_loss
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, 
                loss_mask: Optional[Dict[str, bool]] = None) -> Dict[str, torch.Tensor]:
        """Compute comprehensive loss with optional component masking."""
        if loss_mask is None:
            loss_mask = {"dice": True, "focal": True, "boundary": True, "hierarchy": True}
        
        losses = {}
        total_loss = torch.tensor(0.0, device=inputs.device)
        
        if loss_mask.get("dice", True):
            dice_l = self.dice_loss(inputs, targets)
            losses["dice_loss"] = dice_l
            total_loss += self.config.dice_weight * dice_l
        
        if loss_mask.get("focal", True):
            focal_l = self.focal_loss(inputs, targets)
            losses["focal_loss"] = focal_l
            total_loss += self.config.focal_weight * focal_l
        
        if loss_mask.get("boundary", True):
            boundary_l = self.boundary_loss(inputs, targets)
            losses["boundary_loss"] = boundary_l
            total_loss += self.config.boundary_weight * boundary_l
        
        if loss_mask.get("hierarchy", True):
            hierarchy_l = self.hierarchy_loss(inputs, targets)
            losses["hierarchy_loss"] = hierarchy_l
            total_loss += self.config.hierarchy_weight * hierarchy_l
        
        losses["total_loss"] = total_loss
        return losses


class LossAblationStudy:
    """Framework for systematic loss function ablation studies."""
    
    def __init__(self, num_classes: int = 16):
        self.num_classes = num_classes
        self.results = []
    
    def run_ablation_experiment(
        self, 
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: str = "cpu",
        epochs: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """Run comprehensive ablation study across loss combinations."""
        
        # Define loss combinations to test
        loss_combinations = [
            {"name": "dice_only", "mask": {"dice": True, "focal": False, "boundary": False, "hierarchy": False}},
            {"name": "focal_only", "mask": {"dice": False, "focal": True, "boundary": False, "hierarchy": False}},
            {"name": "dice_focal", "mask": {"dice": True, "focal": True, "boundary": False, "hierarchy": False}},
            {"name": "dice_boundary", "mask": {"dice": True, "focal": False, "boundary": True, "hierarchy": False}},
            {"name": "dice_hierarchy", "mask": {"dice": True, "focal": False, "boundary": False, "hierarchy": True}},
            {"name": "full_loss", "mask": {"dice": True, "focal": True, "boundary": True, "hierarchy": True}},
        ]
        
        results = {}
        
        for combo in loss_combinations:
            logger.info(f"Testing loss combination: {combo['name']}")
            
            # Reset model weights
            model.apply(self._reset_weights)
            
            # Train with this loss combination
            metrics = self._train_with_loss_combination(
                model, train_loader, val_loader, combo["mask"], device, epochs
            )
            
            results[combo["name"]] = metrics
            logger.info(f"Results for {combo['name']}: {metrics}")
        
        return results
    
    def _reset_weights(self, module: nn.Module) -> None:
        """Reset module weights for fair comparison."""
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
        elif hasattr(module, 'weight') and module.weight is not None:
            nn.init.kaiming_normal_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)
    
    def _train_with_loss_combination(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        loss_mask: Dict[str, bool],
        device: str,
        epochs: int
    ) -> Dict[str, float]:
        """Train model with specific loss combination."""
        
        config = LossConfig()
        criterion = ComprehensiveLoss(config, self.num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        model.to(device)
        model.train()
        
        best_dice = 0.0
        
        for epoch in range(epochs):
            # Training
            train_loss = 0.0
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(data)
                
                loss_dict = criterion(outputs, targets, loss_mask)
                loss = loss_dict["total_loss"]
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx >= 10:  # Limit for quick testing
                    break
            
            # Validation
            val_metrics = self._validate_model(model, val_loader, criterion, loss_mask, device)
            
            if val_metrics["dice_score"] > best_dice:
                best_dice = val_metrics["dice_score"]
        
        return {
            "final_dice": best_dice,
            "final_loss": val_metrics["total_loss"],
            "dice_component": val_metrics.get("dice_loss", 0.0),
            "focal_component": val_metrics.get("focal_loss", 0.0),
            "boundary_component": val_metrics.get("boundary_loss", 0.0),
            "hierarchy_component": val_metrics.get("hierarchy_loss", 0.0)
        }
    
    def _validate_model(
        self,
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        criterion: ComprehensiveLoss,
        loss_mask: Dict[str, bool],
        device: str
    ) -> Dict[str, float]:
        """Validate model and compute metrics."""
        model.eval()
        
        total_loss = 0.0
        dice_scores = []
        loss_components = {"dice_loss": 0.0, "focal_loss": 0.0, "boundary_loss": 0.0, "hierarchy_loss": 0.0}
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(val_loader):
                data, targets = data.to(device), targets.to(device)
                
                outputs = model(data)
                loss_dict = criterion(outputs, targets, loss_mask)
                
                total_loss += loss_dict["total_loss"].item()
                
                # Accumulate loss components
                for component, value in loss_dict.items():
                    if component in loss_components:
                        loss_components[component] += value.item()
                
                # Compute Dice score
                preds = torch.argmax(outputs, dim=1)
                dice = self._compute_batch_dice(preds, targets)
                dice_scores.append(dice)
                
                if batch_idx >= 5:  # Limit for quick testing
                    break
        
        return {
            "total_loss": total_loss / max(batch_idx + 1, 1),
            "dice_score": np.mean(dice_scores),
            **{k: v / max(batch_idx + 1, 1) for k, v in loss_components.items()}
        }
    
    def _compute_batch_dice(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Dice score for a batch."""
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        dice_scores = []
        for class_id in range(1, self.num_classes):  # Skip background
            pred_mask = preds_np == class_id
            target_mask = targets_np == class_id
            
            intersection = np.sum(pred_mask & target_mask)
            union = np.sum(pred_mask) + np.sum(target_mask)
            
            if union > 0:
                dice = (2 * intersection) / union
                dice_scores.append(dice)
        
        return np.mean(dice_scores) if dice_scores else 0.0
    
    def save_results(self, results: Dict[str, Dict[str, float]], output_path: Path) -> None:
        """Save ablation study results."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Ablation results saved: {output_path}")


def run_loss_comparison_demo() -> Dict[str, Dict[str, float]]:
    """Run demonstration of loss comparison framework."""
    
    print("🧪 Loss Comparison Framework Demo")
    print("=" * 50)
    
    # Create synthetic data for testing
    batch_size, num_classes = 2, 16
    spatial_dims = (32, 32, 32)
    
    # Synthetic predictions and targets
    inputs = torch.randn(batch_size, num_classes, *spatial_dims)
    targets = torch.randint(0, num_classes, (batch_size, *spatial_dims))
    
    # Test different loss configurations
    configs = [
        ("Standard", LossConfig(dice_weight=0.7, focal_weight=0.3, boundary_weight=0.0, hierarchy_weight=0.0)),
        ("With Boundary", LossConfig(dice_weight=0.5, focal_weight=0.3, boundary_weight=0.2, hierarchy_weight=0.0)),
        ("Full Hierarchical", LossConfig(dice_weight=0.4, focal_weight=0.3, boundary_weight=0.2, hierarchy_weight=0.1)),
    ]
    
    results = {}
    
    for name, config in configs:
        criterion = ComprehensiveLoss(config, num_classes)
        
        # Test all components
        loss_dict = criterion(inputs, targets)
        
        metrics = {
            "total_loss": loss_dict["total_loss"].item(),
            "dice_loss": loss_dict.get("dice_loss", torch.tensor(0.0)).item(),
            "focal_loss": loss_dict.get("focal_loss", torch.tensor(0.0)).item(),
            "boundary_loss": loss_dict.get("boundary_loss", torch.tensor(0.0)).item(),
            "hierarchy_loss": loss_dict.get("hierarchy_loss", torch.tensor(0.0)).item()
        }
        
        results[name] = metrics
        
        print(f"\n📊 {name} Configuration:")
        print(f"  Total Loss: {metrics['total_loss']:.4f}")
        print(f"  Dice: {metrics['dice_loss']:.4f}")
        print(f"  Focal: {metrics['focal_loss']:.4f}")
        print(f"  Boundary: {metrics['boundary_loss']:.4f}")
        print(f"  Hierarchy: {metrics['hierarchy_loss']:.4f}")
    
    return results


if __name__ == "__main__":
    # Run demonstration
    results = run_loss_comparison_demo()
    
    # Save results
    output_dir = Path("/Users/camdouglas/quark/data/datasets/brainstem_segmentation/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "loss_comparison_demo_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved: {results_path}")
    print("✅ Loss comparison framework ready for production experiments")
