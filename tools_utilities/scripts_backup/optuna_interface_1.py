"""
Optuna Interface for Hyperparameter Optimization

Integrates Optuna for optimizing brain development model parameters,
neural network configurations, and simulation parameters.
"""

import os, sys
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
import json
from pathlib import Path
import time

# Try to import Optuna
try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    from optuna.visualization import plot_optimization_history, plot_param_importances
    OPTUNA_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Optuna optimization interface loaded successfully")
except ImportError:
    OPTUNA_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Optuna not available - creating mock interface")

class OptunaOptimizer:
    """Optuna-based hyperparameter optimizer for brain development models"""
    
    def __init__(self, 
                 study_name: str = "brain_development_optimization",
                 storage: Optional[str] = None,
                 sampler_type: str = "tpe"):
        self.study_name = study_name
        self.storage = storage
        self.sampler_type = sampler_type
        self.study = None
        self.best_params = None
        self.best_value = None
        
        if OPTUNA_AVAILABLE:
            self._setup_study()
        else:
            self._setup_mock()
    
    def _setup_study(self):
        """Initialize Optuna study"""
        if OPTUNA_AVAILABLE:
            # Choose sampler
            if self.sampler_type == "tpe":
                sampler = TPESampler(seed=42)
            elif self.sampler_type == "random":
                sampler = RandomSampler(seed=42)
            else:
                sampler = TPESampler(seed=42)
            
            # Choose pruner
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            
            # Create or load study
            if self.storage:
                self.study = optuna.create_study(
                    study_name=self.study_name,
                    storage=self.storage,
                    sampler=sampler,
                    pruner=pruner,
                    load_if_exists=True
                )
            else:
                self.study = optuna.create_study(
                    study_name=self.study_name,
                    sampler=sampler,
                    pruner=pruner
                )
            
            logger.info(f"Optuna study '{self.study_name}' initialized")
        else:
            logger.info("Using mock Optuna interface")
    
    def _setup_mock(self):
        """Setup mock Optuna interface for testing"""
        logger.info("Mock Optuna interface ready")
    
    def suggest_brain_development_params(self, trial) -> Dict[str, Any]:
        """
        Suggest brain development parameters for optimization
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested parameters
        """
        if OPTUNA_AVAILABLE:
            params = {
                # Neural network parameters
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
                "hidden_layers": trial.suggest_int("hidden_layers", 1, 5),
                "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256, 512]),
                "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
                
                # Brain development specific
                "growth_rate": trial.suggest_float("growth_rate", 0.01, 0.1),
                "synapse_density": trial.suggest_float("synapse_density", 0.1, 2.0),
                "plasticity_factor": trial.suggest_float("plasticity_factor", 0.1, 1.0),
                "maturation_time": trial.suggest_int("maturation_time", 10, 100),
                
                # Simulation parameters
                "time_step": trial.suggest_float("time_step", 0.01, 0.1),
                "simulation_duration": trial.suggest_int("simulation_duration", 100, 1000),
                "spatial_resolution": trial.suggest_float("spatial_resolution", 0.1, 1.0)
            }
            
            return params
        else:
            # Mock parameters
            return {
                "learning_rate": 0.001,
                "batch_size": 32,
                "hidden_layers": 3,
                "hidden_size": 256,
                "dropout_rate": 0.3,
                "growth_rate": 0.05,
                "synapse_density": 1.0,
                "plasticity_factor": 0.5,
                "maturation_time": 50,
                "time_step": 0.05,
                "simulation_duration": 500,
                "spatial_resolution": 0.5
            }
    
    def suggest_neural_network_params(self, trial) -> Dict[str, Any]:
        """
        Suggest neural network architecture parameters
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested parameters
        """
        if OPTUNA_AVAILABLE:
            params = {
                # Architecture
                "num_layers": trial.suggest_int("num_layers", 2, 8),
                "layer_sizes": [
                    trial.suggest_categorical(f"layer_{i}_size", [64, 128, 256, 512, 1024])
                    for i in range(trial.suggest_int("num_layers", 2, 8))
                ],
                
                # Training
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
                "momentum": trial.suggest_float("momentum", 0.8, 0.99),
                
                # Regularization
                "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.7),
                "batch_norm": trial.suggest_categorical("batch_norm", [True, False]),
                
                # Optimization
                "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"]),
                "scheduler": trial.suggest_categorical("scheduler", ["step", "cosine", "plateau"])
            }
            
            return params
        else:
            return {
                "num_layers": 4,
                "layer_sizes": [256, 512, 256, 128],
                "learning_rate": 0.001,
                "weight_decay": 1e-4,
                "momentum": 0.9,
                "dropout_rate": 0.3,
                "batch_norm": True,
                "optimizer": "adam",
                "scheduler": "cosine"
            }
    
    def suggest_physics_params(self, trial) -> Dict[str, Any]:
        """
        Suggest physics simulation parameters
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested parameters
        """
        if OPTUNA_AVAILABLE:
            params = {
                # MuJoCo parameters
                "timestep": trial.suggest_float("timestep", 0.001, 0.01),
                "iterations": trial.suggest_int("iterations", 10, 100),
                "solver": trial.suggest_categorical("solver", ["Newton", "PGS", "CG"]),
                
                # Brain tissue properties
                "elasticity": trial.suggest_float("elasticity", 1000, 10000),
                "viscosity": trial.suggest_float("viscosity", 0.1, 10.0),
                "density": trial.suggest_float("density", 1000, 1500),
                
                # Growth parameters
                "growth_rate": trial.suggest_float("growth_rate", 0.001, 0.1),
                "growth_direction": trial.suggest_float("growth_direction", 0.0, 1.0),
                "constraint_strength": trial.suggest_float("constraint_strength", 0.1, 10.0)
            }
            
            return params
        else:
            return {
                "timestep": 0.005,
                "iterations": 50,
                "solver": "Newton",
                "elasticity": 5000,
                "viscosity": 1.0,
                "density": 1200,
                "growth_rate": 0.01,
                "growth_direction": 0.5,
                "constraint_strength": 1.0
            }
    
    def optimize(self, 
                 objective_func: Callable,
                 n_trials: int = 100,
                 timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Run optimization
        
        Args:
            objective_func: Function to optimize (should return float)
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            
        Returns:
            Optimization results
        """
        if OPTUNA_AVAILABLE:
            logger.info(f"Starting optimization with {n_trials} trials")
            
            try:
                self.study.optimize(
                    objective_func,
                    n_trials=n_trials,
                    timeout=timeout,
                    show_progress_bar=True
                )
                
                # Get best results
                self.best_params = self.study.best_params
                self.best_value = self.study.best_value
                
                results = {
                    "best_params": self.best_params,
                    "best_value": self.best_value,
                    "n_trials": len(self.study.trials),
                    "study_name": self.study_name,
                    "optimization_history": [trial.value for trial in self.study.trials if trial.value is not None]
                }
                
                logger.info(f"Optimization completed. Best value: {self.best_value}")
                return results
                
            except Exception as e:
                logger.error(f"Optimization failed: {e}")
                return {"error": str(e)}
        else:
            logger.info(f"Mock: Simulated optimization with {n_trials} trials")
            return {
                "best_params": {"mock": True},
                "best_value": 0.0,
                "n_trials": n_trials,
                "study_name": self.study_name,
                "optimization_history": [0.0] * n_trials
            }
    
    def get_optimization_history(self) -> Dict[str, Any]:
        """Get optimization history and statistics"""
        if OPTUNA_AVAILABLE and self.study:
            trials = self.study.trials
            
            # Extract trial data
            values = [trial.value for trial in trials if trial.value is not None]
            params = [trial.params for trial in trials]
            
            history = {
                "n_trials": len(trials),
                "n_completed": len(values),
                "best_value": min(values) if values else None,
                "worst_value": max(values) if values else None,
                "mean_value": np.mean(values) if values else None,
                "std_value": np.std(values) if values else None,
                "values": values,
                "params": params
            }
            
            return history
        else:
            return {
                "n_trials": 0,
                "n_completed": 0,
                "best_value": None,
                "worst_value": None,
                "mean_value": None,
                "std_value": None,
                "values": [],
                "params": []
            }
    
    def save_study(self, filepath: str):
        """Save study to file"""
        if OPTUNA_AVAILABLE and self.study:
            try:
                with open(filepath, 'w') as f:
                    json.dump({
                        "study_name": self.study_name,
                        "best_params": self.best_params,
                        "best_value": self.best_value,
                        "n_trials": len(self.study.trials)
                    }, f, indent=2)
                logger.info(f"Study saved to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save study: {e}")
        else:
            logger.warning("Cannot save study - Optuna not available")
    
    def load_study(self, filepath: str):
        """Load study from file"""
        if OPTUNA_AVAILABLE:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                self.best_params = data.get("best_params")
                self.best_value = data.get("best_value")
                logger.info(f"Study loaded from {filepath}")
                
            except Exception as e:
                logger.error(f"Failed to load study: {e}")
        else:
            logger.warning("Cannot load study - Optuna not available")
    
    def create_visualization(self, output_dir: str = "./optimization_plots"):
        """Create optimization visualization plots"""
        if OPTUNA_AVAILABLE and self.study:
            try:
                os.makedirs(output_dir, exist_ok=True)
                
                # Optimization history
                fig1 = plot_optimization_history(self.study)
                fig1.write_html(f"{output_dir}/optimization_history.html")
                
                # Parameter importances
                fig2 = plot_param_importances(self.study)
                fig2.write_html(f"{output_dir}/parameter_importances.html")
                
                logger.info(f"Visualizations saved to {output_dir}")
                
            except Exception as e:
                logger.error(f"Failed to create visualizations: {e}")
        else:
            logger.warning("Cannot create visualizations - Optuna not available")
