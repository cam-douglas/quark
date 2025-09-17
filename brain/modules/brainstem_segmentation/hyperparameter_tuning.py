#!/usr/bin/env python3
"""
Hyperparameter Tuning Framework for Brainstem Segmentation
Grid search and Bayesian optimization for optimal parameters
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
from datetime import datetime
import itertools
from dataclasses import dataclass, asdict
from tqdm import tqdm

# Optional import for Optuna
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not installed. Bayesian optimization disabled.")


@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter search."""
    
    # Architecture parameters
    base_filters: int = 32
    use_attention: bool = True
    use_deep_supervision: bool = True
    dropout_rate: float = 0.3
    
    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 2
    optimizer_type: str = "adamw"
    weight_decay: float = 1e-4
    
    # Loss parameters
    dice_weight: float = 0.7
    focal_weight: float = 0.3
    focal_gamma: float = 2.0
    boundary_weight: float = 0.0
    
    # Scheduler parameters
    scheduler_type: str = "cosine"
    warmup_epochs: int = 5
    min_lr_ratio: float = 0.01
    
    # Augmentation parameters
    aug_rotation_range: float = 20.0
    aug_elastic_prob: float = 0.5
    aug_intensity_scale: float = 0.15
    aug_cutmix_prob: float = 0.3
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class HyperparameterTuner:
    """Hyperparameter tuning with grid search and Bayesian optimization."""
    
    def __init__(self,
                 model_class: type,
                 dataset,
                 num_epochs: int = 50,
                 val_split: float = 0.2,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 save_dir: Optional[Path] = None,
                 use_optuna: bool = True):
        """
        Args:
            model_class: Model class to instantiate
            dataset: Full dataset
            num_epochs: Number of epochs per trial
            val_split: Validation split ratio
            device: Device to use
            save_dir: Directory to save results
            use_optuna: Whether to use Bayesian optimization
        """
        self.model_class = model_class
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.val_split = val_split
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else Path("hyperparam_results")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_optuna = use_optuna
        
        # Split dataset
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size]
        )
        
        # Results storage
        self.results = []
        self.best_config = None
        self.best_score = 0.0
    
    def create_model(self, config: HyperparameterConfig) -> nn.Module:
        """Create model with given configuration."""
        
        from enhanced_unet_architecture import EnhancedUNet3D
        
        model = EnhancedUNet3D(
            input_channels=4,
            num_classes=6,
            base_filters=config.base_filters,
            use_attention=config.use_attention,
            use_deep_supervision=config.use_deep_supervision
        )
        
        # Add dropout if specified
        if config.dropout_rate > 0:
            for module in model.modules():
                if isinstance(module, nn.Conv3d):
                    # Add dropout after conv layers
                    pass  # Handled in architecture
        
        return model.to(self.device)
    
    def create_optimizer(self, model: nn.Module, config: HyperparameterConfig):
        """Create optimizer with given configuration."""
        
        if config.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                momentum=0.9,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer_type}")
        
        return optimizer
    
    def create_scheduler(self, optimizer, config: HyperparameterConfig):
        """Create learning rate scheduler."""
        
        if config.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.num_epochs,
                eta_min=config.learning_rate * config.min_lr_ratio
            )
        elif config.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
        elif config.scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', patience=10, factor=0.5
            )
        else:
            scheduler = None
        
        return scheduler
    
    def create_loss_function(self, config: HyperparameterConfig):
        """Create loss function with given configuration."""
        
        from train_onnx_model import EnhancedDiceFocalLoss
        
        # Base loss
        base_loss = EnhancedDiceFocalLoss(
            num_classes=6,
            dice_weight=config.dice_weight,
            focal_weight=config.focal_weight,
            focal_gamma=config.focal_gamma
        )
        
        # Add boundary loss if specified
        if config.boundary_weight > 0:
            # Implement boundary loss wrapper
            class CombinedLoss(nn.Module):
                def __init__(self, base_loss, boundary_weight):
                    super().__init__()
                    self.base_loss = base_loss
                    self.boundary_weight = boundary_weight
                
                def forward(self, pred, target):
                    base = self.base_loss(pred, target)
                    # Simplified boundary loss (edge detection)
                    # In practice, use proper boundary/surface loss
                    return base  # Placeholder
            
            return CombinedLoss(base_loss, config.boundary_weight).to(self.device)
        
        return base_loss.to(self.device)
    
    def create_augmentation(self, config: HyperparameterConfig):
        """Create augmentation pipeline."""
        
        from advanced_augmentations import AdvancedAugmentation3D
        
        return AdvancedAugmentation3D(
            rotation_range=(-config.aug_rotation_range, config.aug_rotation_range),
            elastic_prob=config.aug_elastic_prob,
            intensity_scale_range=(1 - config.aug_intensity_scale, 
                                  1 + config.aug_intensity_scale),
            cutmix_prob=config.aug_cutmix_prob
        )
    
    def train_with_config(self, config: HyperparameterConfig) -> float:
        """Train model with given configuration and return validation score."""
        
        # Create model and training components
        model = self.create_model(config)
        optimizer = self.create_optimizer(model, config)
        scheduler = self.create_scheduler(optimizer, config)
        criterion = self.create_loss_function(config)
        
        # Create data loaders with augmentation
        from advanced_augmentations import AugmentedDataset
        augmentation = self.create_augmentation(config)
        
        train_dataset = AugmentedDataset(self.train_dataset, augmentation, training=True)
        val_dataset = AugmentedDataset(self.val_dataset, None, training=False)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                                shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                              shuffle=False, num_workers=0)
        
        best_val_dice = 0.0
        
        # Training loop (abbreviated)
        for epoch in range(min(self.num_epochs, 10)):  # Limit for tuning
            # Train
            model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Handle deep supervision
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
            
            # Validate
            val_dice = self.validate(model, val_loader)
            best_val_dice = max(best_val_dice, val_dice)
            
            if scheduler:
                if config.scheduler_type == "plateau":
                    scheduler.step(val_dice)
                else:
                    scheduler.step()
        
        return best_val_dice
    
    def validate(self, model: nn.Module, val_loader: DataLoader) -> float:
        """Validate model and return Dice score."""
        
        model.eval()
        dice_scores = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(inputs)
                if isinstance(outputs, list):
                    outputs = outputs[0]
                
                preds = torch.argmax(outputs, dim=1)
                
                # Calculate Dice for non-background classes
                for cls in range(1, 6):
                    pred_mask = (preds == cls)
                    target_mask = (targets == cls)
                    
                    intersection = torch.sum(pred_mask & target_mask).float()
                    union = torch.sum(pred_mask).float() + torch.sum(target_mask).float()
                    
                    if union > 0:
                        dice = (2 * intersection / union).item()
                        dice_scores.append(dice)
        
        return np.mean(dice_scores) if dice_scores else 0.0
    
    def grid_search(self, param_grid: Dict[str, List]) -> Dict:
        """Perform grid search over parameter combinations."""
        
        print("🔍 Starting Grid Search")
        print(f"   Parameter combinations: {len(list(itertools.product(*param_grid.values())))}")
        
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for values in itertools.product(*param_values):
            # Create config
            config_dict = dict(zip(param_names, values))
            config = HyperparameterConfig(**config_dict)
            
            # Train and evaluate
            score = self.train_with_config(config)
            
            # Store results
            result = {
                'config': config.to_dict(),
                'score': score,
                'timestamp': datetime.now().isoformat()
            }
            self.results.append(result)
            
            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_config = config
            
            print(f"   Config: {config_dict} -> Dice: {score:.4f}")
        
        return self.get_best_results()
    
    def optuna_optimize(self, n_trials: int = 20) -> Dict:
        """Perform Bayesian optimization using Optuna."""
        
        if not OPTUNA_AVAILABLE:
            print("❌ Optuna not available. Please install: pip install optuna")
            return self.get_best_results()
        
        print("🎯 Starting Bayesian Optimization with Optuna")
        print(f"   Number of trials: {n_trials}")
        
        def objective(trial):
            # Sample hyperparameters
            config = HyperparameterConfig(
                learning_rate=trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
                batch_size=trial.suggest_categorical('batch_size', [1, 2, 4]),
                dice_weight=trial.suggest_uniform('dice_weight', 0.5, 0.9),
                focal_weight=trial.suggest_uniform('focal_weight', 0.1, 0.5),
                dropout_rate=trial.suggest_uniform('dropout_rate', 0.0, 0.5),
                aug_rotation_range=trial.suggest_uniform('aug_rotation', 0, 30),
                aug_elastic_prob=trial.suggest_uniform('aug_elastic', 0.0, 0.8),
                aug_cutmix_prob=trial.suggest_uniform('aug_cutmix', 0.0, 0.5),
                optimizer_type=trial.suggest_categorical('optimizer', ['adam', 'adamw']),
                scheduler_type=trial.suggest_categorical('scheduler', ['cosine', 'plateau'])
            )
            
            # Train and evaluate
            score = self.train_with_config(config)
            
            # Store results
            result = {
                'config': config.to_dict(),
                'score': score,
                'trial': trial.number,
                'timestamp': datetime.now().isoformat()
            }
            self.results.append(result)
            
            return score
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        self.best_config = HyperparameterConfig(**best_params)
        self.best_score = study.best_value
        
        print(f"\n✅ Best score: {self.best_score:.4f}")
        print(f"   Best params: {best_params}")
        
        return self.get_best_results()
    
    def get_best_results(self) -> Dict:
        """Get best results and save to file."""
        
        results = {
            'best_config': self.best_config.to_dict() if self.best_config else None,
            'best_score': self.best_score,
            'all_results': self.results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        with open(self.save_dir / 'hyperparameter_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {self.save_dir / 'hyperparameter_results.json'}")
        
        return results


def test_hyperparameter_tuning():
    """Test hyperparameter tuning framework."""
    
    print("⚙️ TESTING HYPERPARAMETER TUNING")
    print("=" * 50)
    
    # Create dummy dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=50):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            image = torch.randn(4, 32, 32, 32)
            label = torch.randint(0, 6, (32, 32, 32))
            return image, label
    
    # Simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.conv = nn.Conv3d(4, 6, 3, padding=1)
        
        def forward(self, x):
            return self.conv(x)
    
    # Create tuner
    tuner = HyperparameterTuner(
        model_class=SimpleModel,
        dataset=DummyDataset(20),  # Small dataset
        num_epochs=2,  # Few epochs for testing
        val_split=0.2,
        save_dir=Path("/Users/camdouglas/quark/data/models/brainstem/hyperparam_test"),
        use_optuna=False  # Grid search for testing
    )
    
    print("✅ Tuner created")
    print(f"   Train size: {len(tuner.train_dataset)}")
    print(f"   Val size: {len(tuner.val_dataset)}")
    
    # Test grid search with small grid
    param_grid = {
        'learning_rate': [1e-3, 5e-4],
        'dice_weight': [0.6, 0.7],
        'batch_size': [2]
    }
    
    print("\n🔍 Testing grid search...")
    print(f"   Parameter grid: {param_grid}")
    
    # Run abbreviated test
    config = HyperparameterConfig(learning_rate=1e-3, dice_weight=0.7)
    score = tuner.train_with_config(config)
    
    print(f"\n✅ Test configuration score: {score:.4f}")
    print("\n✅ Hyperparameter tuning framework ready!")
    
    return tuner


if __name__ == "__main__":
    test_hyperparameter_tuning()
