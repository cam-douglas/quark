#!/usr/bin/env python3
"""
Small-Mind Neural Network Optimizer

Uses SAM (Sharpness-Aware Minimization) for training, Optuna for hyperparameter optimization,
and includes model compression capabilities for efficient brain development simulation.
"""

import torch
import torch.nn as nn
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

# Add the smallmind package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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

class SmallMindNeuralModel(nn.Module):
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

class SmallMindNeuralOptimizer:
    """Neural network optimizer with SAM and advanced training techniques"""
    
    def __init__(self, 
                 model: nn.Module,
                 learning_rate: float = 0.001,
                 rho: float = 0.05,  # SAM radius
                 adaptive: bool = True):
        self.model = model
        self.learning_rate = learning_rate
        self.rho = rho
        self.adaptive = adaptive
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup optimizer
        if SAM_AVAILABLE:
            self.optimizer = SAM(
                self.model.parameters(),
                torch.optim.Adam,
                lr=learning_rate,
                rho=rho,
                adaptive=adaptive
            )
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.criterion = nn.MSELoss()
        self.training_history = []
        
    def train_with_sam(self, 
                       X: torch.Tensor, 
                       y: torch.Tensor,
                       epochs: int = 100,
                       batch_size: int = 32,
                       validation_split: float = 0.2) -> Dict[str, List[float]]:
        """
        Train model using SAM (Sharpness-Aware Minimization) if available
        
        Args:
            X: Input features
            y: Target values
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary containing training and validation loss history
        """
        # Move data to device
        X = X.to(self.device)
        y = y.to(self.device)
        
        # Split data into training and validation
        n_samples = X.shape[0]
        n_val = int(n_samples * validation_split)
        indices = torch.randperm(n_samples)
        
        X_train = X[indices[n_val:]]
        y_train = y[indices[n_val:]]
        X_val = X[indices[:n_val]]
        y_val = y[indices[:n_val]]
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                if SAM_AVAILABLE:
                    # SAM requires two forward-backward passes
                    loss.backward()
                    self.optimizer.first_step(zero_grad=True)
                    
                    # Second forward pass for SAM
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.second_step(zero_grad=True)
                else:
                    # Standard optimization
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = self.criterion(val_outputs, y_val).item()
            
            avg_train_loss = train_loss / (len(X_train) // batch_size)
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        self.training_history = {
            'train_loss': train_losses,
            'val_loss': val_losses
        }
        
        return self.training_history
    
    def optimize_hyperparameters(self, 
                                X: torch.Tensor, 
                                y: torch.Tensor,
                                n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna
        
        Args:
            X: Input features
            y: Target values
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary containing best parameters and optimization results
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available - skipping hyperparameter optimization")
            return {}
        
        def objective(trial):
            # Suggest hyperparameters
            hidden_size = trial.suggest_int('hidden_size', 64, 512)
            num_layers = trial.suggest_int('num_layers', 1, 5)
            dropout = trial.suggest_float('dropout', 0.0, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            
            # Create model with suggested parameters
            model = SmallMindNeuralModel(
                input_size=X.shape[1],
                hidden_size=hidden_size,
                output_size=y.shape[1],
                dropout=dropout,
                num_layers=num_layers
            )
            
            # Train model
            optimizer = SmallMindNeuralOptimizer(model, learning_rate=learning_rate)
            history = optimizer.train_with_sam(X, y, epochs=50, batch_size=batch_size)
            
            # Return validation loss (lower is better)
            return min(history['val_loss'])
        
        # Create study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study
        }
    
    def compress_model(self, 
                      compression_ratio: float = 0.5,
                      method: str = 'pruning') -> nn.Module:
        """
        Compress the model using various techniques
        
        Args:
            compression_ratio: Target compression ratio
            method: Compression method ('pruning', 'quantization', 'knowledge_distillation')
            
        Returns:
            Compressed model
        """
        if method == 'pruning':
            return self._prune_model(compression_ratio)
        elif method == 'quantization':
            return self._quantize_model()
        else:
            logger.warning(f"Unknown compression method: {method}")
            return self.model
    
    def _prune_model(self, compression_ratio: float) -> nn.Module:
        """Prune the model to reduce parameters"""
        logger.info(f"Pruning model with compression ratio: {compression_ratio}")
        
        # Count total parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        target_params = int(total_params * compression_ratio)
        
        # Simple magnitude-based pruning
        parameters = []
        for param in self.model.parameters():
            if param.dim() > 1:  # Only prune weight matrices
                parameters.append(param.data.view(-1))
        
        if parameters:
            all_weights = torch.cat(parameters)
            threshold = torch.quantile(torch.abs(all_weights), compression_ratio)
            
            # Apply pruning
            for param in self.model.parameters():
                if param.dim() > 1:
                    mask = torch.abs(param.data) > threshold
                    param.data *= mask.float()
        
        return self.model
    
    def _quantize_model(self) -> nn.Module:
        """Quantize the model to reduce precision"""
        logger.info("Quantizing model to INT8")
        
        try:
            # Use PyTorch's dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                self.model, {nn.Linear}, dtype=torch.qint8
            )
            return quantized_model
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            return self.model
    
    def evaluate_model(self, X_test: torch.Tensor, y_test: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate the trained model
        
        Args:
            X_test: Test input features
            y_test: Test target values
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_test)
            mse_loss = self.criterion(predictions, y_test).item()
            
            # Calculate additional metrics
            mae = torch.mean(torch.abs(predictions - y_test)).item()
            r2 = 1 - torch.sum((y_test - predictions) ** 2) / torch.sum((y_test - y_test.mean()) ** 2)
            r2 = r2.item()
        
        return {
            'mse': mse_loss,
            'mae': mae,
            'r2': r2
        }
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'model_config': {
                'input_size': self.model.input_size,
                'hidden_size': self.model.hidden_size,
                'output_size': self.model.output_size,
                'dropout': self.model.dropout,
                'num_layers': self.model.num_layers
            }
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        logger.info(f"Model loaded from {filepath}")

def create_synthetic_brain_data(input_size: int = 100, 
                               output_size: int = 50, 
                               num_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic brain development data for testing"""
    # Generate synthetic input features (e.g., gene expression levels)
    X = torch.randn(num_samples, input_size)
    
    # Generate synthetic output targets (e.g., brain region development scores)
    # Add some correlation between inputs and outputs
    weights = torch.randn(input_size, output_size) * 0.1
    y = torch.matmul(X, weights) + torch.randn(num_samples, output_size) * 0.1
    
    return X, y

def main():
    """Example usage of the neural optimizer"""
    logger.info("🧠 Small-Mind Neural Network Optimizer Demo")
    
    # Create synthetic data
    X, y = create_synthetic_brain_data(input_size=100, output_size=50, num_samples=1000)
    
    # Create model
    model = SmallMindNeuralModel(input_size=100, hidden_size=256, output_size=50)
    
    # Create optimizer
    optimizer = SmallMindNeuralOptimizer(model, learning_rate=0.001)
    
    # Train model
    logger.info("Training model with SAM...")
    history = optimizer.train_with_sam(X, y, epochs=50, batch_size=32)
    
    # Evaluate model
    logger.info("Evaluating model...")
    metrics = optimizer.evaluate_model(X, y)
    logger.info(f"Model Performance: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}, R²={metrics['r2']:.6f}")
    
    # Optimize hyperparameters
    logger.info("Optimizing hyperparameters...")
    opt_results = optimizer.optimize_hyperparameters(X, y, n_trials=20)
    if opt_results:
        logger.info(f"Best parameters: {opt_results['best_params']}")
        logger.info(f"Best validation loss: {opt_results['best_value']:.6f}")
    
    # Compress model
    logger.info("Compressing model...")
    compressed_model = optimizer.compress_model(compression_ratio=0.7, method='pruning')
    
    # Save model
    optimizer.save_model('smallmind_neural_model.pth')
    logger.info("Demo completed!")

if __name__ == "__main__":
    main()
