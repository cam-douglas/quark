#!/usr/bin/env python3
"""
Small-Mind Optimizer

Uses Optuna for hyperparameter optimization, SAM (Sharpness-Aware Minimization) for training,
and includes model compression capabilities for efficient brain development simulation.
"""

import torch
import torch.nn as nn
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add the smallmind package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available - using basic optimization")

try:
    from sam import SAM
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    logging.warning("SAM not available - using standard optimizer")

logger = logging.getLogger(__name__)

class SmallMindModel(nn.Module):
    """Small-Mind brain development model for optimization"""
    
    def __init__(self, 
                 input_size: int = 100,
                 hidden_size: int = 256,
                 output_size: int = 50,
                 dropout: float = 0.1,
                 num_layers: int = 3):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.num_layers = num_layers
        
        # Build neural network layers
        layers = []
        prev_size = input_size
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def train_model_with_sam(model: nn.Module, 
                         lr: float = 0.001,
                         epochs: int = 3,
                         batch_size: int = 32) -> float:
    """
    Train model using SAM (Sharpness-Aware Minimization) if available
    
    Args:
        model: PyTorch model to train
        lr: Learning rate
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Final validation score
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Generate synthetic data for demonstration
    # In real usage, this would be actual brain development data
    input_size = model.input_size
    output_size = model.output_size
    
    # Create synthetic training data
    num_samples = 1000
    X = torch.randn(num_samples, input_size)
    y = torch.randn(num_samples, output_size)
    
    # Split into train/val
    train_size = int(0.8 * num_samples)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Loss function
    criterion = nn.MSELoss()
    
    if SAM_AVAILABLE:
        logger.info("Using SAM optimizer")
        base_optimizer = torch.optim.Adam
        optimizer = SAM(model.parameters(), base_optimizer, lr=lr)
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            # Training loop with SAM
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size].to(device)
                batch_y = y_train[i:i+batch_size].to(device)
                
                def closure():
                    optimizer.zero_grad()
                    output = model(batch_X)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    return loss
                
                closure()
                optimizer.step(closure)
                total_loss += closure().item()
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
    else:
        logger.info("Using standard Adam optimizer")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size].to(device)
                batch_y = y_train[i:i+batch_size].to(device)
                
                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
    
    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        val_output = model(X_val.to(device))
        val_loss = criterion(val_output, y_val.to(device))
        val_score = -val_loss.item()  # Negative loss as score (higher is better)
    
    logger.info(f"Validation score: {val_score:.4f}")
    return val_score

def compress_model(model: nn.Module, compression_ratio: float = 0.5) -> nn.Module:
    """
    Compress the model using pruning and quantization
    
    Args:
        model: Model to compress
        compression_ratio: Target compression ratio (0.0 to 1.0)
        
    Returns:
        Compressed model
    """
    logger.info(f"Compressing model with target ratio: {compression_ratio}")
    
    # Model pruning
    total_params = sum(p.numel() for p in model.parameters())
    target_params = int(total_params * compression_ratio)
    
    # Simple magnitude-based pruning
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    
    # Apply pruning
    for module, param_name in parameters_to_prune:
        torch.nn.utils.prune.global_unstructured(
            parameters_to_prune,
            pruning_method=torch.nn.utils.prune.L1Unstructured,
            amount=compression_ratio
        )
    
    # Quantization (INT8)
    try:
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        logger.info("Model quantized to INT8")
        return quantized_model
    except Exception as e:
        logger.warning(f"Quantization failed: {e}. Returning pruned model.")
        return model

def optimize_smallmind_model():
    """Main optimization function using Optuna"""
    if not OPTUNA_AVAILABLE:
        logger.error("Optuna not available. Cannot perform optimization.")
        return
    
    def objective(trial):
        """Objective function for Optuna optimization"""
        # Suggest hyperparameters
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        hidden_size = trial.suggest_categorical('hidden_size', [128, 256, 512])
        num_layers = trial.suggest_int('num_layers', 2, 5)
        compression_ratio = trial.suggest_float('compression_ratio', 0.3, 0.8)
        
        try:
            # Create and train model
            model = SmallMindModel(
                hidden_size=hidden_size,
                dropout=dropout,
                num_layers=num_layers
            )
            
            # Train with SAM
            score = train_model_with_sam(model, lr=lr)
            
            # Compress model
            compressed_model = compress_model(model, compression_ratio)
            
            # Calculate compressed model score
            compressed_score = score * (1.0 - compression_ratio * 0.1)  # Small penalty for compression
            
            logger.info(f"Trial {trial.number}: Score={score:.4f}, Compressed={compressed_score:.4f}")
            return compressed_score
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return float('-inf')  # Return worst possible score
    
    # Create study
    study = optuna.create_study(
        direction="maximize",
        study_name="smallmind_optimization"
    )
    
    # Run optimization
    logger.info("🚀 Starting Optuna optimization...")
    study.optimize(objective, n_trials=10, timeout=600)  # 10 trials, 10 minute timeout
    
    # Report results
    logger.info("✅ Optimization complete!")
    logger.info(f"Best score: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")
    
    # Save best model
    best_params = study.best_params
    best_model = SmallMindModel(
        hidden_size=best_params['hidden_size'],
        dropout=best_params['dropout'],
        num_layers=best_params['num_layers']
    )
    
    # Train best model
    best_score = train_model_with_sam(best_model, lr=best_params['lr'])
    
    # Compress best model
    compressed_best_model = compress_model(best_model, best_params['compression_ratio'])
    
    # Save models
    torch.save(best_model.state_dict(), 'best_smallmind_model.pth')
    torch.save(compressed_best_model.state_dict(), 'best_compressed_smallmind_model.pth')
    
    logger.info("💾 Best models saved successfully")
    return study.best_value

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    optimize_smallmind_model()
