#!/usr/bin/env python3
"""
🧠 Curiosity Engine Trainer

This trainer implements curiosity-driven cognitive engines with integration to existing
cognitive engine capabilities from the updates/ folder.
"""

import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclasses import data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclass
from typing import List, Dict, Optional, Tuple
import logging
import json
from pathlib import Path

# Add existing cognitive engine capabilities to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../cognitive_engines'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

# Import existing cognitive engine capabilities
try:
    from cognitive_engines.curiosity_engine import CuriosityEngine
    from cognitive_engines.exploration_module import ExplorationModule
    from cognitive_engines.synthesis_engine import SynthesisEngine
    from cognitive_engines.interest_scorer import InterestScorer
    from cognitive_engines.question_generator import QuestionGenerator
    from cognitive_engines.uncertainty_quantifier import UncertaintyQuantifier
    from cognitive_engines.knowledge_integrator import KnowledgeIntegrator
    from cognitive_engines.pattern_recognizer import PatternRecognizer
    
    # Import existing neural architectures for enhanced cognition
    from neural_architectures.childlike_learning_system import ChildlikeLearningSystem
    from neural_architectures.continuous_training import ContinuousTraining
    
    COGNITIVE_ENGINE_CAPABILITIES_AVAILABLE = True
    print("✅ Successfully imported existing cognitive engine capabilities")
except ImportError as e:
    print(f"⚠️  Warning: Could not import some existing cognitive engine modules: {e}")
    print("   Some features may use simplified implementations")
    COGNITIVE_ENGINE_CAPABILITIES_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CuriosityEngineConfig:
    """Configuration for curiosity engine"""
    input_dim: int = 256
    hidden_dim: int = 128
    num_layers: int = 3
    learning_rate: float = 1e-4
    batch_size: int = 16
    epochs: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "models/curiosity_engine/"
    use_existing_capabilities: bool = True
    novelty_weight: float = 0.2
    uncertainty_weight: float = 0.3
    synthesis_threshold: float = 0.7

class IntegratedCuriosityEngine:
    """Integrated curiosity engine using existing capabilities"""
    
    def __init__(self, config: CuriosityEngineConfig):
        self.config = config
        self.existing_capabilities = {}
        
        if config.use_existing_capabilities and COGNITIVE_ENGINE_CAPABILITIES_AVAILABLE:
            self._initialize_existing_capabilities()
    
    def _initialize_existing_capabilities(self):
        """Initialize existing cognitive engine capabilities"""
        logger.info("Initializing existing cognitive engine capabilities...")
        
        try:
            # Initialize existing cognitive engines
            self.existing_capabilities['curiosity_engine'] = CuriosityEngine()
            self.existing_capabilities['exploration_module'] = ExplorationModule()
            self.existing_capabilities['synthesis_engine'] = SynthesisEngine()
            self.existing_capabilities['interest_scorer'] = InterestScorer()
            self.existing_capabilities['question_generator'] = QuestionGenerator()
            self.existing_capabilities['uncertainty_quantifier'] = UncertaintyQuantifier()
            self.existing_capabilities['knowledge_integrator'] = KnowledgeIntegrator()
            self.existing_capabilities['pattern_recognizer'] = PatternRecognizer()
            
            # Initialize neural architectures for enhanced cognition
            self.existing_capabilities['childlike_learner'] = ChildlikeLearningSystem()
            self.existing_capabilities['continuous_trainer'] = ContinuousTraining()
            
            logger.info("✅ All existing cognitive engine capabilities initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize some existing capabilities: {e}")
            self.existing_capabilities = {}
    
    def generate_cognitive_data(self, epoch: int) -> Dict:
        """Generate cognitive data using existing capabilities"""
        if self.existing_capabilities:
            curiosity_engine = self.existing_capabilities.get('curiosity_engine')
            exploration_module = self.existing_capabilities.get('exploration_module')
            question_generator = self.existing_capabilities.get('question_generator')
            
            if curiosity_engine and exploration_module and question_generator:
                # Use existing capabilities to generate enhanced cognitive data
                curiosity_data = curiosity_engine.generate_curious_prompts()
                exploration_data = exploration_module.generate_exploration_tasks()
                questions = question_generator.generate_questions()
                
                return {
                    "curiosity_data": curiosity_data,
                    "exploration_data": exploration_data,
                    "questions": questions,
                    "epoch": epoch,
                    "enhanced": True
                }
        
        # Fallback to simplified data generation
        return self._generate_simplified_data(epoch)
    
    def _generate_simplified_data(self, epoch: int) -> Dict:
        """Generate simplified cognitive data"""
        # Simple cognitive prompts for learning
        prompts = [
            "What happens if I mix these colors?",
            "Why does the ball bounce?",
            "How do plants grow?",
            "What makes a sound?",
            "Why is the sky blue?",
            "How do birds fly?",
            "What causes rain?",
            "Why do we dream?"
        ]
        
        # Generate random cognitive tasks
        tasks = []
        for _ in range(self.config.batch_size):
            task = {
                "prompt": np.random.choice(prompts),
                "novelty_score": np.random.random(),
                "uncertainty_level": np.random.random(),
                "interest_level": np.random.random()
            }
            tasks.append(task)
        
        return {
            "tasks": tasks,
            "epoch": epoch,
            "enhanced": False
        }
    
    def process_cognitive_task(self, data: Dict, epoch: int) -> Dict:
        """Process cognitive task using existing capabilities"""
        if self.existing_capabilities:
            interest_scorer = self.existing_capabilities.get('interest_scorer')
            uncertainty_quantifier = self.existing_capabilities.get('uncertainty_quantifier')
            pattern_recognizer = self.existing_capabilities.get('pattern_recognizer')
            knowledge_integrator = self.existing_capabilities.get('knowledge_integrator')
            
            if interest_scorer and uncertainty_quantifier and pattern_recognizer and knowledge_integrator:
                # Use existing capabilities for cognitive processing
                logger.info("Using existing cognitive engine capabilities for processing")
                
                # Process through existing systems
                interest_score = interest_scorer.score_interest(data)
                uncertainty_score = uncertainty_quantifier.quantify_uncertainty(data)
                patterns = pattern_recognizer.recognize_patterns(data)
                knowledge = knowledge_integrator.integrate_knowledge(data)
                
                return {
                    "interest_score": interest_score,
                    "uncertainty_score": uncertainty_score,
                    "patterns": patterns,
                    "knowledge": knowledge,
                    "enhanced": True
                }
        
        # Fallback to simplified processing
        return self._simulate_cognitive_processing(data, epoch)
    
    def _simulate_cognitive_processing(self, data: Dict, epoch: int) -> Dict:
        """Simulate cognitive processing"""
        # Simulate cognitive responses
        base_interest = 0.5 + np.random.random() * 0.3
        base_uncertainty = 0.3 + np.random.random() * 0.4
        base_novelty = 0.4 + np.random.random() * 0.3
        
        # Add epoch-based learning effects
        learning_factor = min(1.0, epoch / 10.0)
        interest_score = base_interest * (1 + learning_factor * 0.2)
        uncertainty_score = base_uncertainty * (1 - learning_factor * 0.1)
        novelty_score = base_novelty * (1 + learning_factor * 0.15)
        
        return {
            "interest_score": interest_score,
            "uncertainty_score": uncertainty_score,
            "novelty_score": novelty_score,
            "enhanced": False
        }

class NoveltyDetector(nn.Module):
    """Novelty detection module"""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.novelty_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.novelty_net(x)

class UncertaintyQuantifier(nn.Module):
    """Uncertainty quantification module"""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.uncertainty_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.uncertainty_net(x)

class QuestionGenerator(nn.Module):
    """Question generation module"""
    
    def __init__(self, input_dim: int, hidden_dim: int, vocab_size: int = 1000):
        super().__init__()
        self.question_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.question_net(x)

class KnowledgeSynthesizer(nn.Module):
    """Knowledge synthesis module"""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.synthesis_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.synthesis_net(x)

class CuriosityEngine(nn.Module):
    """Main curiosity engine model"""
    
    def __init__(self, config: CuriosityEngineConfig):
        super().__init__()
        self.config = config
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Cognitive modules
        self.novelty_detector = NoveltyDetector(config.hidden_dim, config.hidden_dim // 2)
        self.uncertainty_quantifier = UncertaintyQuantifier(config.hidden_dim, config.hidden_dim // 2)
        self.question_generator = QuestionGenerator(config.hidden_dim, config.hidden_dim // 2)
        self.knowledge_synthesizer = KnowledgeSynthesizer(config.hidden_dim, config.hidden_dim // 2)
        
        # Integration layer
        self.integration_layer = nn.Sequential(
            nn.Linear(4, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through curiosity engine"""
        # Extract features
        features = self.feature_extractor(x)
        
        # Apply cognitive modules
        novelty_score = self.novelty_detector(features)
        uncertainty_score = self.uncertainty_quantifier(features)
        question_logits = self.question_generator(features)
        synthesis_score = self.knowledge_synthesizer(features)
        
        # Integrate cognitive scores
        cognitive_scores = torch.cat([
            novelty_score, uncertainty_score, 
            synthesis_score, torch.sigmoid(question_logits.mean(dim=1, keepdim=True))
        ], dim=1)
        
        curiosity_score = self.integration_layer(cognitive_scores)
        
        return {
            "curiosity_score": curiosity_score,
            "novelty_score": novelty_score,
            "uncertainty_score": uncertainty_score,
            "synthesis_score": synthesis_score,
            "question_logits": question_logits,
            "features": features
        }

class CuriosityEngineDataset(Dataset):
    """Dataset for curiosity engine training with integrated capabilities"""
    
    def __init__(self, config: CuriosityEngineConfig, num_samples: int = 1000):
        self.config = config
        self.num_samples = num_samples
        self.integrated_engine = IntegratedCuriosityEngine(config)
        
        # Generate samples
        self.samples = self._generate_samples()
    
    def _generate_samples(self) -> List[Dict]:
        """Generate training samples using integrated system"""
        samples = []
        
        for i in range(self.num_samples):
            # Use integrated system to generate enhanced data
            cognitive_data = self.integrated_engine.generate_cognitive_data(i)
            
            if cognitive_data.get("enhanced", False):
                # Use enhanced data from existing capabilities
                curiosity_data = cognitive_data["curiosity_data"]
                exploration_data = cognitive_data["exploration_data"]
                questions = cognitive_data["questions"]
                
                # Combine different types of cognitive data
                combined_data = curiosity_data + exploration_data + questions
                task_data = combined_data[i % len(combined_data)]
            else:
                # Use simplified data
                task_data = cognitive_data["tasks"][i % len(cognitive_data["tasks"])]
            
            # Process through integrated system
            processed_data = self.integrated_engine.process_cognitive_task(task_data, i)
            
            # Create feature vector
            features = torch.randn(self.config.input_dim)  # Simplified features
            
            # Create target scores
            targets = {
                "novelty": torch.tensor(processed_data.get("novelty_score", 0.5), dtype=torch.float),
                "uncertainty": torch.tensor(processed_data.get("uncertainty_score", 0.5), dtype=torch.float),
                "interest": torch.tensor(processed_data.get("interest_score", 0.5), dtype=torch.float),
                "synthesis": torch.tensor(processed_data.get("synthesis_score", 0.5), dtype=torch.float)
            }
            
            samples.append({
                "features": features,
                "targets": targets,
                "task_data": task_data,
                "enhanced": cognitive_data.get("enhanced", False)
            })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]

class CuriosityEngineTrainer:
    """Trainer for curiosity engine"""
    
    def __init__(self, config: CuriosityEngineConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize integrated engine
        self.integrated_engine = IntegratedCuriosityEngine(config)
        
        # Initialize model
        self.model = CuriosityEngine(config).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        
        # Initialize dataset and dataloader
        self.dataset = CuriosityEngineDataset(config)
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=config.batch_size, 
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        # Training history
        self.train_losses = []
        self.curiosity_scores = []
    
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching"""
        features = torch.stack([item["features"] for item in batch])
        targets = {
            key: torch.stack([item["targets"][key] for item in batch])
            for key in batch[0]["targets"].keys()
        }
        
        return {
            "features": features,
            "targets": targets
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_curiosity = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.dataloader):
            # Move to device
            features = batch["features"].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch["targets"].items()}
            
            # Forward pass
            outputs = self.model(features)
            curiosity_score = outputs["curiosity_score"]
            novelty_score = outputs["novelty_score"]
            uncertainty_score = outputs["uncertainty_score"]
            synthesis_score = outputs["synthesis_score"]
            
            # Compute losses
            novelty_loss = nn.BCELoss()(novelty_score, targets["novelty"].unsqueeze(1))
            uncertainty_loss = nn.BCELoss()(uncertainty_score, targets["uncertainty"].unsqueeze(1))
            interest_loss = nn.BCELoss()(curiosity_score, targets["interest"].unsqueeze(1))
            synthesis_loss = nn.BCELoss()(synthesis_score, targets["synthesis"].unsqueeze(1))
            
            # Combined loss
            total_loss_component = (
                novelty_loss + 
                uncertainty_loss + 
                interest_loss + 
                synthesis_loss
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_component.backward()
            self.optimizer.step()
            
            # Use integrated engine for enhanced training if available
            if self.integrated_engine.existing_capabilities:
                enhanced_data = {
                    "curiosity_score": curiosity_score.detach().cpu(),
                    "novelty_score": novelty_score.detach().cpu(),
                    "uncertainty_score": uncertainty_score.detach().cpu(),
                    "synthesis_score": synthesis_score.detach().cpu(),
                    "loss": total_loss_component.item(),
                    "batch_idx": batch_idx
                }
                enhanced_loss = self.integrated_engine.process_cognitive_task(enhanced_data, epoch)
                total_loss += enhanced_loss.get("interest_score", total_loss_component.item())
            else:
                total_loss += total_loss_component.item()
            
            total_curiosity += curiosity_score.mean().item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss_component.item():.4f}, Curiosity: {curiosity_score.mean().item():.4f}")
        
        avg_loss = total_loss / num_batches
        avg_curiosity = total_curiosity / num_batches
        
        return {
            "loss": avg_loss,
            "curiosity": avg_curiosity
        }
    
    def train(self) -> Dict[str, List[float]]:
        """Complete training loop"""
        logger.info("Starting curiosity engine training...")
        logger.info(f"Using existing capabilities: {bool(self.integrated_engine.existing_capabilities)}")
        
        for epoch in range(self.config.epochs):
            metrics = self.train_epoch(epoch)
            
            self.train_losses.append(metrics["loss"])
            self.curiosity_scores.append(metrics["curiosity"])
            
            logger.info(f"Epoch {epoch}: Loss={metrics['loss']:.4f}, Curiosity={metrics['curiosity']:.4f}")
            
            # Save checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(epoch)
        
        # Save final model
        self.save_checkpoint(self.config.epochs - 1, is_final=True)
        
        # Plot training progress
        self.plot_training_progress()
        
        return {
            "train_losses": self.train_losses,
            "curiosity_scores": self.curiosity_scores
        }
    
    def save_checkpoint(self, epoch: int, is_final: bool = False):
        """Save training checkpoint"""
        os.makedirs(self.config.save_path, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "train_losses": self.train_losses,
            "curiosity_scores": self.curiosity_scores,
            "existing_capabilities_used": bool(self.integrated_engine.existing_capabilities)
        }
        
        filename = "final_model.pt" if is_final else f"checkpoint_epoch_{epoch}.pt"
        checkpoint_path = os.path.join(self.config.save_path, filename)
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def plot_training_progress(self):
        """Plot training progress"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot loss
        ax1.plot(self.train_losses)
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True)
        
        # Plot curiosity scores
        ax2.plot(self.curiosity_scores)
        ax2.set_title("Curiosity Scores")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Curiosity")
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.config.save_path, "training_progress.png")
        plt.savefig(plot_path)
        logger.info(f"Training progress plot saved: {plot_path}")
        plt.close()

def main():
    """Main training function"""
    # Configuration
    config = CuriosityEngineConfig(
        input_dim=256,
        hidden_dim=128,
        num_layers=3,
        learning_rate=1e-4,
        batch_size=16,
        epochs=50,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_path="models/curiosity_engine/",
        use_existing_capabilities=True,
        novelty_weight=0.2,
        uncertainty_weight=0.3,
        synthesis_threshold=0.7
    )
    
    # Initialize trainer
    trainer = CuriosityEngineTrainer(config)
    
    # Train model
    training_results = trainer.train()
    
    logger.info("Training completed successfully!")
    logger.info(f"Final loss: {training_results['train_losses'][-1]:.4f}")
    logger.info(f"Final curiosity: {training_results['curiosity_scores'][-1]:.4f}")
    
    # Print capabilities status
    if trainer.integrated_engine.existing_capabilities:
        logger.info("✅ Training used existing cognitive engine capabilities")
        logger.info(f"Available capabilities: {list(trainer.integrated_engine.existing_capabilities.keys())}")
    else:
        logger.info("⚠️  Training used simplified implementations (no existing capabilities available)")

if __name__ == "__main__":
    main()
