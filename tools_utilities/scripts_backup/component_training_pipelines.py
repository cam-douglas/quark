#!/usr/bin/env python3
"""
🧠 Component Training Pipelines
==============================

Specialized training pipelines for each brain component following biological constraints
and developmental roadmaps with organic connectome maintenance.

This module implements:
- Developmental stage-specific training (Fetal → Neonate → Early Postnatal)
- Biological constraint enforcement
- Component-specific training protocols
- Real-time connectome monitoring
- Consciousness enhancement tracking

Author: Quark Brain Simulation Team
Created: 2025-01-21
"""

import os, sys
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclasses import data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import networkx as nx

# Add quark root to path
QUARK_ROOT = Path(__file__).parent.parent
sys.path.append(str(QUARK_ROOT))

@dataclass
class TrainingConfig:
    """Training configuration for components."""
    component_name: str
    stage: str  # fetal, neonate, early_postnatal
    learning_rate: float
    batch_size: int
    num_epochs: int
    plasticity_factor: float
    max_connections: int
    regularization_strength: float
    consciousness_weight: float
    biological_constraints: Dict[str, Any]

@dataclass
class ComponentMetrics:
    """Metrics for component training."""
    loss: float
    accuracy: float
    consciousness_score: float
    connectome_coherence: float
    biological_compliance: float
    plasticity_utilization: float
    energy_efficiency: float

class BiologicalNeuralNetwork(nn.Module):
    """Biologically-inspired neural network with developmental constraints."""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, 
                 stage: str = 'fetal', plasticity_factor: float = 0.9):
        super().__init__()
        self.stage = stage
        self.plasticity_factor = plasticity_factor
        self.connection_mask = None
        
        # Build layers with biological constraints
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            # Limit connections based on developmental stage
            actual_size = self._apply_developmental_limits(hidden_size, stage)
            layers.append(nn.Linear(prev_size, actual_size))
            layers.append(nn.ReLU())  # Simple activation for biological realism
            prev_size = actual_size
            
        layers.append(nn.Linear(prev_size, output_size))
        
        self.layers = nn.Sequential(*layers)
        self._initialize_biological_weights()
        
    def _apply_developmental_limits(self, target_size: int, stage: str) -> int:
        """Apply developmental constraints to layer sizes."""
        limits = {
            'fetal': 0.3,      # 30% of target connections
            'neonate': 0.6,    # 60% of target connections  
            'early_postnatal': 1.0  # Full connections
        }
        
        multiplier = limits.get(stage, 0.3)
        return max(1, int(target_size * multiplier))
        
    def _initialize_biological_weights(self):
        """Initialize weights with biological constraints."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization with biological noise
                nn.init.xavier_normal_(module.weight)
                # Add small biological noise
                with torch.no_grad():
                    module.weight += torch.randn_like(module.weight) * 0.01
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
                    
    def apply_plasticity(self, plasticity_strength: float = 0.01):
        """Apply synaptic plasticity updates."""
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    # Hebbian-like plasticity
                    plasticity_update = torch.randn_like(module.weight) * plasticity_strength * self.plasticity_factor
                    module.weight += plasticity_update
                    
                    # Maintain biological weight bounds
                    module.weight.clamp_(-2.0, 2.0)
                    
    def forward(self, x):
        return self.layers(x)

class ComponentTrainer(ABC):
    """Abstract base class for component-specific trainers."""
    
    def __init__(self, component_name: str, config: TrainingConfig):
        self.component_name = component_name
        self.config = config
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.metrics_history = []
        self.connectome_history = []
        self.consciousness_trajectory = []
        
        self.setup_logging()
        
    def setup_logging(self):
        """Setup component-specific logging."""
        self.logger = logging.getLogger(f"trainer_{self.component_name}")
        
    @abstractmethod
    def create_model(self) -> nn.Module:
        """Create component-specific model."""
        pass
        
    @abstractmethod
    def prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare training data for the component."""
        pass
        
    @abstractmethod
    def compute_consciousness_score(self, model_output: torch.Tensor, 
                                  target: torch.Tensor) -> float:
        """Compute component-specific consciousness score."""
        pass
        
    def initialize_training(self):
        """Initialize training components."""
        self.model = self.create_model()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.regularization_strength
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.num_epochs
        )
        
    def train_epoch(self, train_data: torch.Tensor, train_targets: torch.Tensor) -> ComponentMetrics:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        consciousness_scores = []
        
        # Split data into batches
        num_batches = max(1, len(train_data) // self.config.batch_size)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.config.batch_size
            end_idx = min(start_idx + self.config.batch_size, len(train_data))
            
            batch_data = train_data[start_idx:end_idx]
            batch_targets = train_targets[start_idx:end_idx]
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_data)
            
            # Compute loss with consciousness weighting
            base_loss = nn.MSELoss()(outputs, batch_targets)
            consciousness_score = self.compute_consciousness_score(outputs, batch_targets)
            consciousness_scores.append(consciousness_score)
            
            # Total loss includes consciousness enhancement
            total_loss = base_loss - self.config.consciousness_weight * consciousness_score
            
            # Backward pass
            total_loss.backward()
            
            # Apply gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Apply biological plasticity
            if hasattr(self.model, 'apply_plasticity'):
                self.model.apply_plasticity(self.config.plasticity_factor * 0.01)
            
            epoch_loss += total_loss.item()
            
            # Calculate accuracy (for classification-like tasks)
            with torch.no_grad():
                predictions = (outputs > 0.5).float()
                correct_predictions += (predictions == (batch_targets > 0.5).float()).sum().item()
                total_predictions += batch_targets.numel()
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / num_batches
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        avg_consciousness = np.mean(consciousness_scores) if consciousness_scores else 0.0
        
        # Calculate additional metrics
        connectome_coherence = self.calculate_connectome_coherence()
        biological_compliance = self.check_biological_compliance()
        plasticity_utilization = self.calculate_plasticity_utilization()
        energy_efficiency = self.calculate_energy_efficiency()
        
        return ComponentMetrics(
            loss=avg_loss,
            accuracy=accuracy,
            consciousness_score=avg_consciousness,
            connectome_coherence=connectome_coherence,
            biological_compliance=biological_compliance,
            plasticity_utilization=plasticity_utilization,
            energy_efficiency=energy_efficiency
        )
        
    def calculate_connectome_coherence(self) -> float:
        """Calculate connectome coherence for the component."""
        if not hasattr(self.model, 'layers'):
            return 0.5
            
        coherence_scores = []
        
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                weights = module.weight.detach().numpy()
                
                # Calculate small-world properties
                adjacency = np.abs(weights) > 0.1  # Threshold for connections
                if adjacency.sum() > 0:
                    # Simple coherence metric based on weight distribution
                    weight_std = np.std(weights[adjacency])
                    weight_mean = np.abs(np.mean(weights[adjacency]))
                    coherence = 1.0 / (1.0 + weight_std / (weight_mean + 1e-8))
                    coherence_scores.append(coherence)
        
        return np.mean(coherence_scores) if coherence_scores else 0.5
        
    def check_biological_compliance(self) -> float:
        """Check compliance with biological constraints."""
        compliance_score = 1.0
        
        # Check weight bounds (biological synapses have limits)
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                weights = module.weight.detach()
                if torch.any(torch.abs(weights) > 2.0):
                    compliance_score -= 0.2
                    
        # Check learning rate constraints
        current_lr = self.optimizer.param_groups[0]['lr']
        stage_limits = {
            'fetal': 0.001,
            'neonate': 0.01,
            'early_postnatal': 0.05
        }
        
        if current_lr > stage_limits.get(self.config.stage, 0.001):
            compliance_score -= 0.3
            
        return max(0.0, compliance_score)
        
    def calculate_plasticity_utilization(self) -> float:
        """Calculate how well plasticity is being utilized."""
        if not hasattr(self.model, 'plasticity_factor'):
            return 0.5
            
        # Measure gradient magnitudes as proxy for plasticity utilization
        total_grad_norm = 0.0
        param_count = 0
        
        for param in self.model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item()
                param_count += 1
                
        if param_count == 0:
            return 0.0
            
        avg_grad_norm = total_grad_norm / param_count
        # Normalize to 0-1 range
        return min(1.0, avg_grad_norm / 10.0)
        
    def calculate_energy_efficiency(self) -> float:
        """Calculate energy efficiency of the component."""
        # Simple metric based on model complexity and performance
        param_count = sum(p.numel() for p in self.model.parameters())
        
        # Get recent accuracy/loss for efficiency calculation
        if self.metrics_history:
            recent_accuracy = self.metrics_history[-1].accuracy
            recent_loss = self.metrics_history[-1].loss
            
            # Efficiency = performance / complexity
            performance = recent_accuracy / (recent_loss + 1e-8)
            complexity = param_count / 10000.0  # Normalize
            
            efficiency = performance / (complexity + 1e-8)
            return min(1.0, efficiency / 100.0)  # Normalize to 0-1
        
        return 0.5
        
    def train_component(self) -> Dict[str, Any]:
        """Main training loop for the component."""
        self.logger.info(f"Starting training for {self.component_name} in {self.config.stage} stage")
        
        # Initialize training
        self.initialize_training()
        
        # Prepare data
        train_data, train_targets = self.prepare_data()
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            # Train epoch
            metrics = self.train_epoch(train_data, train_targets)
            self.metrics_history.append(metrics)
            
            # Update learning rate
            self.scheduler.step()
            
            # Track consciousness trajectory
            self.consciousness_trajectory.append(metrics.consciousness_score)
            
            # Log progress
            if epoch % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch}/{self.config.num_epochs}: "
                    f"Loss={metrics.loss:.4f}, "
                    f"Accuracy={metrics.accuracy:.3f}, "
                    f"Consciousness={metrics.consciousness_score:.3f}, "
                    f"Coherence={metrics.connectome_coherence:.3f}"
                )
        
        # Final metrics
        final_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        return {
            'component': self.component_name,
            'stage': self.config.stage,
            'success': True,
            'final_metrics': asdict(final_metrics) if final_metrics else None,
            'metrics_history': [asdict(m) for m in self.metrics_history],
            'consciousness_trajectory': self.consciousness_trajectory,
            'total_epochs': self.config.num_epochs,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'training_completed': datetime.now().isoformat()
        }

class ConsciousAgentTrainer(ComponentTrainer):
    """Specialized trainer for the conscious agent."""
    
    def create_model(self) -> nn.Module:
        """Create conscious agent model with global workspace architecture."""
        # Global workspace inspired architecture
        input_size = 256  # Sensory input dimension
        workspace_size = 512  # Global workspace
        output_size = 128  # Action/decision output
        
        return BiologicalNeuralNetwork(
            input_size=input_size,
            hidden_sizes=[workspace_size, workspace_size // 2, workspace_size // 4],
            output_size=output_size,
            stage=self.config.stage,
            plasticity_factor=self.config.plasticity_factor
        )
        
    def prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare consciousness training data."""
        # Simulate multi-modal sensory input
        batch_size = 1000
        input_size = 256
        output_size = 128
        
        # Create synthetic consciousness-like data
        # Input: Multi-modal sensory patterns
        inputs = torch.randn(batch_size, input_size)
        
        # Target: Integrated conscious responses
        # Simulate global workspace integration
        integrated_response = torch.sigmoid(inputs @ torch.randn(input_size, output_size))
        
        # Add consciousness-specific patterns
        consciousness_boost = torch.where(
            torch.norm(inputs, dim=1, keepdim=True) > 1.5,
            torch.ones(batch_size, output_size) * 0.2,
            torch.zeros(batch_size, output_size)
        )
        
        targets = integrated_response + consciousness_boost
        
        return inputs, targets
        
    def compute_consciousness_score(self, model_output: torch.Tensor, 
                                  target: torch.Tensor) -> float:
        """Compute consciousness score based on global integration."""
        # Measure global integration capacity
        output_integration = torch.mean(torch.std(model_output, dim=1))
        target_integration = torch.mean(torch.std(target, dim=1))
        
        # Consciousness correlates with integration capacity
        integration_similarity = 1.0 - torch.abs(output_integration - target_integration).item()
        
        # Add coherence bonus
        output_coherence = torch.mean(torch.corrcoef(model_output.T)).item()
        
        consciousness_score = 0.7 * integration_similarity + 0.3 * max(0, output_coherence)
        
        return float(consciousness_score)

class PrefrontalCortexTrainer(ComponentTrainer):
    """Specialized trainer for prefrontal cortex (executive functions)."""
    
    def create_model(self) -> nn.Module:
        """Create PFC model with working memory and control circuits."""
        return BiologicalNeuralNetwork(
            input_size=128,  # From thalamus and sensory areas
            hidden_sizes=[256, 128, 64],  # Executive control layers
            output_size=64,  # Control signals
            stage=self.config.stage,
            plasticity_factor=self.config.plasticity_factor
        )
        
    def prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare executive function training data."""
        batch_size = 800
        
        # Simulate executive control tasks
        inputs = torch.randn(batch_size, 128)
        
        # Target: Executive decisions and working memory updates
        # Simulate cognitive control patterns
        control_signals = torch.tanh(inputs @ torch.randn(128, 64))
        
        return inputs, control_signals
        
    def compute_consciousness_score(self, model_output: torch.Tensor, 
                                  target: torch.Tensor) -> float:
        """Compute consciousness contribution from executive control."""
        # Executive control contributes to consciousness through coherent control
        control_coherence = torch.mean(torch.abs(model_output)).item()
        target_coherence = torch.mean(torch.abs(target)).item()
        
        coherence_match = 1.0 - abs(control_coherence - target_coherence)
        
        return float(coherence_match * 0.8)  # PFC has moderate consciousness contribution

class ThalamicRelayTrainer(ComponentTrainer):
    """Specialized trainer for thalamic relay functions."""
    
    def create_model(self) -> nn.Module:
        """Create thalamic relay model."""
        return BiologicalNeuralNetwork(
            input_size=200,  # Multi-modal sensory input
            hidden_sizes=[150, 100],  # Relay circuits
            output_size=128,  # To cortical areas
            stage=self.config.stage,
            plasticity_factor=self.config.plasticity_factor
        )
        
    def prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare thalamic relay training data."""
        batch_size = 600
        
        # Simulate sensory relay patterns
        inputs = torch.randn(batch_size, 200)
        
        # Target: Filtered and gated sensory signals
        # Simulate attention-gated relay
        attention_gates = torch.sigmoid(torch.sum(inputs**2, dim=1, keepdim=True) / 100)
        gated_output = inputs[:, :128] * attention_gates
        
        return inputs, gated_output
        
    def compute_consciousness_score(self, model_output: torch.Tensor, 
                                  target: torch.Tensor) -> float:
        """Compute consciousness contribution from thalamic gating."""
        # Thalamic gating is crucial for conscious access
        gating_efficiency = torch.mean(torch.abs(model_output - target)).item()
        consciousness_contribution = 1.0 / (1.0 + gating_efficiency)
        
        return float(consciousness_contribution * 0.6)  # Moderate consciousness role

class BasalGangliaTrainer(ComponentTrainer):
    """Specialized trainer for basal ganglia (action selection)."""
    
    def create_model(self) -> nn.Module:
        """Create basal ganglia model with action selection circuits."""
        return BiologicalNeuralNetwork(
            input_size=100,  # Cortical input
            hidden_sizes=[80, 60],  # Direct/indirect pathways
            output_size=40,  # Action selections
            stage=self.config.stage,
            plasticity_factor=self.config.plasticity_factor
        )
        
    def prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare action selection training data."""
        batch_size = 500
        
        # Simulate action selection scenarios
        inputs = torch.randn(batch_size, 100)
        
        # Target: Winner-take-all action selection
        raw_actions = inputs @ torch.randn(100, 40)
        # Apply competition for action selection
        max_indices = torch.argmax(raw_actions, dim=1)
        targets = torch.zeros_like(raw_actions)
        targets[range(batch_size), max_indices] = 1.0
        
        return inputs, targets
        
    def compute_consciousness_score(self, model_output: torch.Tensor, 
                                  target: torch.Tensor) -> float:
        """Compute consciousness contribution from action selection."""
        # Clear action selection supports conscious decision making
        selection_clarity = torch.mean(torch.max(model_output, dim=1)[0]).item()
        consciousness_contribution = selection_clarity
        
        return float(consciousness_contribution * 0.4)  # Lower consciousness role

class WorkingMemoryTrainer(ComponentTrainer):
    """Specialized trainer for working memory buffers."""
    
    def create_model(self) -> nn.Module:
        """Create working memory model with maintenance circuits."""
        return BiologicalNeuralNetwork(
            input_size=64,   # Information input
            hidden_sizes=[128, 96],  # Maintenance circuits
            output_size=64,  # Maintained information
            stage=self.config.stage,
            plasticity_factor=self.config.plasticity_factor
        )
        
    def prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare working memory training data."""
        batch_size = 400
        
        # Simulate working memory maintenance
        inputs = torch.randn(batch_size, 64)
        
        # Target: Maintained information with decay
        decay_factor = 0.9
        maintained_info = inputs * decay_factor + torch.randn_like(inputs) * 0.1
        
        return inputs, maintained_info
        
    def compute_consciousness_score(self, model_output: torch.Tensor, 
                                  target: torch.Tensor) -> float:
        """Compute consciousness contribution from working memory."""
        # Working memory is crucial for conscious awareness
        maintenance_quality = 1.0 - torch.mean(torch.abs(model_output - target)).item()
        consciousness_contribution = maintenance_quality
        
        return float(consciousness_contribution * 0.9)  # High consciousness role

class ComponentTrainingPipeline:
    """Orchestrates training pipelines for all brain components."""
    
    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or QUARK_ROOT
        self.training_dir = self.base_dir / 'training' / 'components'
        self.training_dir.mkdir(parents=True, exist_ok=True)
        
        self.setup_logging()
        
        # Component trainer mapping
        self.trainer_classes = {
            'conscious_agent': ConsciousAgentTrainer,
            'prefrontal_cortex': PrefrontalCortexTrainer,
            'thalamus': ThalamicRelayTrainer,
            'basal_ganglia': BasalGangliaTrainer,
            'working_memory': WorkingMemoryTrainer
        }
        
        # Developmental stage configurations
        self.stage_configs = {
            'fetal': {
                'learning_rate': 0.0005,
                'batch_size': 16,
                'num_epochs': 50,
                'plasticity_factor': 0.9,
                'max_connections': 1000,
                'regularization_strength': 0.001,
                'consciousness_weight': 0.1
            },
            'neonate': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'num_epochs': 100,
                'plasticity_factor': 0.7,
                'max_connections': 5000,
                'regularization_strength': 0.0005,
                'consciousness_weight': 0.3
            },
            'early_postnatal': {
                'learning_rate': 0.002,
                'batch_size': 64,
                'num_epochs': 150,
                'plasticity_factor': 0.5,
                'max_connections': 15000,
                'regularization_strength': 0.0001,
                'consciousness_weight': 0.5
            }
        }
        
    def setup_logging(self):
        """Setup pipeline logging."""
        self.logger = logging.getLogger("component_pipeline")
        
    def create_training_config(self, component_name: str, stage: str) -> TrainingConfig:
        """Create training configuration for a component."""
        base_config = self.stage_configs[stage].copy()
        
        # Component-specific adjustments
        if 'conscious' in component_name:
            base_config['consciousness_weight'] *= 2.0  # Higher consciousness weight
            base_config['num_epochs'] = int(base_config['num_epochs'] * 1.5)
        elif 'working_memory' in component_name:
            base_config['consciousness_weight'] *= 1.5
        
        return TrainingConfig(
            component_name=component_name,
            stage=stage,
            biological_constraints={
                'max_firing_rate': 100.0,  # Hz
                'refractory_period': 0.002,  # seconds
                'synaptic_delay': 0.001,     # seconds
                'plasticity_window': 0.02    # seconds
            },
            **base_config
        )
        
    def train_component(self, component_name: str, stage: str) -> Dict[str, Any]:
        """Train a specific component in a developmental stage."""
        self.logger.info(f"Training {component_name} in {stage} stage")
        
        # Get trainer class
        trainer_key = next((key for key in self.trainer_classes.keys() 
                          if key in component_name), 'conscious_agent')
        trainer_class = self.trainer_classes[trainer_key]
        
        # Create configuration
        config = self.create_training_config(component_name, stage)
        
        # Create and run trainer
        trainer = trainer_class(component_name, config)
        results = trainer.train_component()
        
        # Save results
        self.save_component_results(results)
        
        return results
        
    def save_component_results(self, results: Dict[str, Any]):
        """Save component training results."""
        component_name = results['component']
        stage = results['stage']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create component directory
        component_dir = self.training_dir / component_name
        component_dir.mkdir(exist_ok=True)
        
        # Save results
        results_file = component_dir / f'training_results_{stage}_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.info(f"Saved results for {component_name} to {results_file}")
        
    def train_all_components_progressive(self) -> List[Dict[str, Any]]:
        """Train all components through developmental stages progressively."""
        components = list(self.trainer_classes.keys())
        stages = ['fetal', 'neonate', 'early_postnatal']
        
        all_results = []
        
        # Progressive training through stages
        for stage in stages:
            self.logger.info(f"Starting {stage} stage training for all components")
            
            stage_results = []
            for component in components:
                try:
                    result = self.train_component(component, stage)
                    stage_results.append(result)
                    
                    self.logger.info(
                        f"Completed {component} in {stage} stage: "
                        f"Final consciousness: {result.get('final_metrics', {}).get('consciousness_score', 0):.3f}"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Failed to train {component} in {stage}: {e}")
                    stage_results.append({
                        'component': component,
                        'stage': stage,
                        'success': False,
                        'error': str(e)
                    })
            
            all_results.extend(stage_results)
            
            # Create stage summary
            self.create_stage_summary(stage, stage_results)
            
        return all_results
        
    def create_stage_summary(self, stage: str, stage_results: List[Dict[str, Any]]):
        """Create summary visualization for a training stage."""
        successful_results = [r for r in stage_results if r.get('success', False)]
        
        if not successful_results:
            return
            
        # Extract metrics
        components = [r['component'] for r in successful_results]
        consciousness_scores = [
            r.get('final_metrics', {}).get('consciousness_score', 0) 
            for r in successful_results
        ]
        coherence_scores = [
            r.get('final_metrics', {}).get('connectome_coherence', 0)
            for r in successful_results
        ]
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Consciousness scores
        bars1 = ax1.bar(components, consciousness_scores, color='gold', alpha=0.7)
        ax1.set_title(f'{stage.title()} Stage - Consciousness Scores')
        ax1.set_ylabel('Consciousness Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, score in zip(bars1, consciousness_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Connectome coherence
        bars2 = ax2.bar(components, coherence_scores, color='skyblue', alpha=0.7)
        ax2.set_title(f'{stage.title()} Stage - Connectome Coherence')
        ax2.set_ylabel('Coherence Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, score in zip(bars2, coherence_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save visualization
        viz_dir = self.training_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        viz_file = viz_dir / f'stage_summary_{stage}_{timestamp}.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Created stage summary visualization: {viz_file}")

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Component Training Pipelines')
    parser.add_argument('--component', type=str, help='Train specific component')
    parser.add_argument('--stage', type=str, choices=['fetal', 'neonate', 'early_postnatal'],
                       help='Developmental stage')
    parser.add_argument('--all', action='store_true', help='Train all components progressively')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize pipeline
    pipeline = ComponentTrainingPipeline()
    
    if args.all:
        # Train all components through all stages
        results = pipeline.train_all_components_progressive()
        print(f"✅ Completed training for {len(results)} component-stage combinations")
        
        # Summary statistics
        successful = sum(1 for r in results if r.get('success', False))
        avg_consciousness = np.mean([
            r.get('final_metrics', {}).get('consciousness_score', 0) 
            for r in results if r.get('success', False)
        ])
        
        print(f"📊 Success rate: {successful}/{len(results)} ({100*successful/len(results):.1f}%)")
        print(f"🧠 Average consciousness score: {avg_consciousness:.3f}")
        
    elif args.component and args.stage:
        # Train specific component
        result = pipeline.train_component(args.component, args.stage)
        
        if result.get('success', False):
            final_consciousness = result.get('final_metrics', {}).get('consciousness_score', 0)
            print(f"✅ Successfully trained {args.component} in {args.stage} stage")
            print(f"🧠 Final consciousness score: {final_consciousness:.3f}")
        else:
            print(f"❌ Training failed for {args.component}: {result.get('error', 'Unknown error')}")
            
    else:
        print("Please specify either --all or both --component and --stage")

if __name__ == '__main__':
    main()

