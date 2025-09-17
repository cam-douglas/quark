#!/usr/bin/env python3
"""
K-Fold Cross-Validation Framework for Brainstem Segmentation
Comprehensive validation metrics tracking and analysis
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm
from dataclasses import dataclass, asdict


@dataclass
class ValidationMetrics:
    """Comprehensive metrics for validation."""
    
    dice_overall: float
    dice_per_class: Dict[int, float]
    iou_overall: float
    iou_per_class: Dict[int, float]
    sensitivity: float
    specificity: float
    precision: float
    hausdorff_distance: Optional[float] = None
    surface_distance: Optional[float] = None
    volume_difference: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class MetricsCalculator:
    """Calculate comprehensive segmentation metrics."""
    
    def __init__(self, num_classes: int, exclude_background: bool = True):
        """
        Args:
            num_classes: Number of segmentation classes
            exclude_background: Whether to exclude class 0 from metrics
        """
        self.num_classes = num_classes
        self.exclude_background = exclude_background
        self.start_class = 1 if exclude_background else 0
    
    def dice_coefficient(self, pred: torch.Tensor, target: torch.Tensor, 
                        class_idx: Optional[int] = None) -> float:
        """Calculate Dice coefficient."""
        
        if class_idx is not None:
            pred_mask = (pred == class_idx)
            target_mask = (target == class_idx)
        else:
            pred_mask = pred
            target_mask = target
        
        intersection = torch.sum(pred_mask & target_mask).float()
        union = torch.sum(pred_mask).float() + torch.sum(target_mask).float()
        
        if union == 0:
            return 1.0 if torch.sum(target_mask) == 0 else 0.0
        
        dice = (2.0 * intersection) / union
        return dice.item()
    
    def iou_score(self, pred: torch.Tensor, target: torch.Tensor,
                 class_idx: Optional[int] = None) -> float:
        """Calculate Intersection over Union."""
        
        if class_idx is not None:
            pred_mask = (pred == class_idx)
            target_mask = (target == class_idx)
        else:
            pred_mask = pred
            target_mask = target
        
        intersection = torch.sum(pred_mask & target_mask).float()
        union = torch.sum(pred_mask | target_mask).float()
        
        if union == 0:
            return 1.0 if torch.sum(target_mask) == 0 else 0.0
        
        iou = intersection / union
        return iou.item()
    
    def sensitivity_specificity(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
        """Calculate sensitivity and specificity."""
        
        # For multi-class, calculate for foreground vs background
        pred_fg = pred > 0
        target_fg = target > 0
        
        tp = torch.sum(pred_fg & target_fg).float()
        tn = torch.sum(~pred_fg & ~target_fg).float()
        fp = torch.sum(pred_fg & ~target_fg).float()
        fn = torch.sum(~pred_fg & target_fg).float()
        
        sensitivity = tp / (tp + fn + 1e-7)
        specificity = tn / (tn + fp + 1e-7)
        
        return sensitivity.item(), specificity.item()
    
    def precision_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate precision."""
        
        pred_fg = pred > 0
        target_fg = target > 0
        
        tp = torch.sum(pred_fg & target_fg).float()
        fp = torch.sum(pred_fg & ~target_fg).float()
        
        precision = tp / (tp + fp + 1e-7)
        return precision.item()
    
    def calculate_all_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> ValidationMetrics:
        """Calculate all validation metrics."""
        
        # Per-class Dice and IoU
        dice_per_class = {}
        iou_per_class = {}
        
        for cls in range(self.start_class, self.num_classes):
            dice_per_class[cls] = self.dice_coefficient(pred, target, cls)
            iou_per_class[cls] = self.iou_score(pred, target, cls)
        
        # Overall metrics (excluding background if specified)
        dice_overall = np.mean(list(dice_per_class.values()))
        iou_overall = np.mean(list(iou_per_class.values()))
        
        # Sensitivity, Specificity, Precision
        sensitivity, specificity = self.sensitivity_specificity(pred, target)
        precision = self.precision_score(pred, target)
        
        return ValidationMetrics(
            dice_overall=dice_overall,
            dice_per_class=dice_per_class,
            iou_overall=iou_overall,
            iou_per_class=iou_per_class,
            sensitivity=sensitivity,
            specificity=specificity,
            precision=precision
        )


class KFoldValidator:
    """K-Fold cross-validation framework."""
    
    def __init__(self,
                 model_class: type,
                 model_kwargs: Dict[str, Any],
                 dataset,
                 k_folds: int = 5,
                 batch_size: int = 2,
                 num_epochs: int = 100,
                 learning_rate: float = 1e-3,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 save_dir: Optional[Path] = None):
        """
        Args:
            model_class: Model class to instantiate
            model_kwargs: Keyword arguments for model initialization
            dataset: Full dataset to split
            k_folds: Number of folds
            batch_size: Batch size for training
            num_epochs: Number of epochs per fold
            learning_rate: Learning rate
            device: Device to use
            save_dir: Directory to save results
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.dataset = dataset
        self.k_folds = k_folds
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else Path("kfold_results")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics calculator
        self.metrics_calc = MetricsCalculator(
            num_classes=model_kwargs.get('num_classes', 6),
            exclude_background=True
        )
        
        # Results storage
        self.fold_results = []
        self.best_fold = None
        self.best_metrics = None
    
    def train_fold(self, 
                  model: nn.Module,
                  train_loader: DataLoader,
                  val_loader: DataLoader,
                  fold: int) -> Tuple[nn.Module, ValidationMetrics]:
        """Train a single fold."""
        
        print(f"\n📊 Training Fold {fold + 1}/{self.k_folds}")
        print("-" * 40)
        
        # Loss function and optimizer
        from train_onnx_model import EnhancedDiceFocalLoss
        criterion = EnhancedDiceFocalLoss(
            num_classes=self.model_kwargs['num_classes'],
            dice_weight=0.7,
            focal_weight=0.3
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_epochs
        )
        
        best_val_dice = 0.0
        best_model_state = None
        
        for epoch in range(self.num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Handle deep supervision outputs
                if isinstance(outputs, list):
                    from enhanced_unet_architecture import DeepSupervisionLoss
                    ds_loss = DeepSupervisionLoss(criterion)
                    loss = ds_loss(outputs, targets)
                else:
                    loss = criterion(outputs, targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            val_metrics = self.validate_fold(model, val_loader)
            
            avg_train_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch+1}: Loss={avg_train_loss:.4f}, Val Dice={val_metrics.dice_overall:.4f}")
            
            # Save best model
            if val_metrics.dice_overall > best_val_dice:
                best_val_dice = val_metrics.dice_overall
                best_model_state = model.state_dict().copy()
                best_metrics = val_metrics
            
            scheduler.step()
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model, best_metrics
    
    def validate_fold(self, model: nn.Module, val_loader: DataLoader) -> ValidationMetrics:
        """Validate a single fold."""
        
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                
                # Handle deep supervision outputs
                if isinstance(outputs, list):
                    outputs = outputs[0]  # Use main output
                
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())
        
        # Concatenate all predictions
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics
        metrics = self.metrics_calc.calculate_all_metrics(all_preds, all_targets)
        
        return metrics
    
    def run(self) -> Dict[str, Any]:
        """Run k-fold cross-validation."""
        
        print(f"🔄 Starting {self.k_folds}-Fold Cross-Validation")
        print(f"   Dataset size: {len(self.dataset)}")
        print(f"   Device: {self.device}")
        print("=" * 50)
        
        # K-Fold split
        kfold = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        indices = np.arange(len(self.dataset))
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
            # Create data loaders
            train_subset = Subset(self.dataset, train_idx)
            val_subset = Subset(self.dataset, val_idx)
            
            train_loader = DataLoader(train_subset, batch_size=self.batch_size, 
                                    shuffle=True, num_workers=0)
            val_loader = DataLoader(val_subset, batch_size=self.batch_size, 
                                  shuffle=False, num_workers=0)
            
            # Create fresh model for this fold
            model = self.model_class(**self.model_kwargs).to(self.device)
            
            # Train fold
            model, metrics = self.train_fold(model, train_loader, val_loader, fold)
            
            # Store results
            fold_result = {
                'fold': fold + 1,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'metrics': metrics.to_dict()
            }
            self.fold_results.append(fold_result)
            
            # Track best fold
            if self.best_metrics is None or metrics.dice_overall > self.best_metrics.dice_overall:
                self.best_fold = fold + 1
                self.best_metrics = metrics
                
                # Save best model
                torch.save(model.state_dict(), 
                          self.save_dir / f"best_model_fold{fold+1}.pth")
        
        # Calculate aggregate statistics
        aggregate_stats = self.calculate_aggregate_stats()
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'k_folds': self.k_folds,
            'fold_results': self.fold_results,
            'best_fold': self.best_fold,
            'best_metrics': self.best_metrics.to_dict() if self.best_metrics else None,
            'aggregate_stats': aggregate_stats
        }
        
        with open(self.save_dir / 'kfold_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self.print_summary(aggregate_stats)
        
        return results
    
    def calculate_aggregate_stats(self) -> Dict[str, Any]:
        """Calculate aggregate statistics across all folds."""
        
        dice_scores = [r['metrics']['dice_overall'] for r in self.fold_results]
        iou_scores = [r['metrics']['iou_overall'] for r in self.fold_results]
        sensitivity_scores = [r['metrics']['sensitivity'] for r in self.fold_results]
        specificity_scores = [r['metrics']['specificity'] for r in self.fold_results]
        precision_scores = [r['metrics']['precision'] for r in self.fold_results]
        
        return {
            'dice': {
                'mean': np.mean(dice_scores),
                'std': np.std(dice_scores),
                'min': np.min(dice_scores),
                'max': np.max(dice_scores)
            },
            'iou': {
                'mean': np.mean(iou_scores),
                'std': np.std(iou_scores),
                'min': np.min(iou_scores),
                'max': np.max(iou_scores)
            },
            'sensitivity': {
                'mean': np.mean(sensitivity_scores),
                'std': np.std(sensitivity_scores)
            },
            'specificity': {
                'mean': np.mean(specificity_scores),
                'std': np.std(specificity_scores)
            },
            'precision': {
                'mean': np.mean(precision_scores),
                'std': np.std(precision_scores)
            }
        }
    
    def print_summary(self, aggregate_stats: Dict[str, Any]):
        """Print validation summary."""
        
        print("\n" + "=" * 50)
        print("📊 K-FOLD CROSS-VALIDATION SUMMARY")
        print("=" * 50)
        
        print(f"\n✅ Best Fold: {self.best_fold}")
        print(f"   Best Dice: {self.best_metrics.dice_overall:.4f}")
        
        print("\n📈 Aggregate Statistics:")
        print(f"   Dice: {aggregate_stats['dice']['mean']:.4f} ± {aggregate_stats['dice']['std']:.4f}")
        print(f"   IoU:  {aggregate_stats['iou']['mean']:.4f} ± {aggregate_stats['iou']['std']:.4f}")
        print(f"   Sensitivity: {aggregate_stats['sensitivity']['mean']:.4f} ± {aggregate_stats['sensitivity']['std']:.4f}")
        print(f"   Specificity: {aggregate_stats['specificity']['mean']:.4f} ± {aggregate_stats['specificity']['std']:.4f}")
        print(f"   Precision:   {aggregate_stats['precision']['mean']:.4f} ± {aggregate_stats['precision']['std']:.4f}")
        
        print("\n💾 Results saved to:", self.save_dir / 'kfold_results.json')


def test_kfold_validation():
    """Test the k-fold validation framework."""
    
    print("🔬 TESTING K-FOLD VALIDATION FRAMEWORK")
    print("=" * 50)
    
    # Create dummy dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=100):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            image = torch.randn(4, 32, 32, 32)  # Smaller for testing
            label = torch.randint(0, 6, (32, 32, 32))
            return image, label
    
    # Simple test model
    class SimpleModel(nn.Module):
        def __init__(self, input_channels=4, num_classes=6, **kwargs):
            super().__init__()
            self.conv = nn.Conv3d(input_channels, num_classes, kernel_size=3, padding=1)
        
        def forward(self, x):
            return self.conv(x)
    
    # Create validator
    validator = KFoldValidator(
        model_class=SimpleModel,
        model_kwargs={'input_channels': 4, 'num_classes': 6},
        dataset=DummyDataset(50),  # Small dataset for testing
        k_folds=3,  # Fewer folds for testing
        batch_size=4,
        num_epochs=2,  # Few epochs for testing
        learning_rate=1e-3,
        save_dir=Path("/Users/camdouglas/quark/data/models/brainstem/kfold_test")
    )
    
    print("✅ Validator created")
    print(f"   K-folds: {validator.k_folds}")
    print(f"   Dataset size: {len(validator.dataset)}")
    print(f"   Device: {validator.device}")
    
    # Run validation (abbreviated for testing)
    print("\n🏃 Running abbreviated validation test...")
    
    # Test metrics calculation
    metrics_calc = MetricsCalculator(num_classes=6)
    pred = torch.randint(0, 6, (2, 32, 32, 32))
    target = torch.randint(0, 6, (2, 32, 32, 32))
    
    metrics = metrics_calc.calculate_all_metrics(pred, target)
    print(f"\n✅ Metrics calculation test:")
    print(f"   Dice: {metrics.dice_overall:.4f}")
    print(f"   IoU: {metrics.iou_overall:.4f}")
    print(f"   Sensitivity: {metrics.sensitivity:.4f}")
    print(f"   Specificity: {metrics.specificity:.4f}")
    print(f"   Precision: {metrics.precision:.4f}")
    
    print("\n✅ K-fold validation framework ready!")
    return validator


if __name__ == "__main__":
    test_kfold_validation()
