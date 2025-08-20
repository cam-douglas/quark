#!/usr/bin/env python3
"""
Domain-Specific Training Scripts
================================

Specialized training scripts for different domains of brain simulation
and AGI training, each optimized for specific use cases.
"""

import os, sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import base trainer
from integrated_brain_simulation_trainer import IntegratedBrainSimulationTrainer, BrainSimulationConfig

logger = logging.getLogger(__name__)

@dataclass
class DomainConfig:
    """Configuration for domain-specific training"""
    domain_name: str
    base_model: str
    learning_rate: float
    batch_size: int
    epochs: int
    max_length: int
    data_sources: List[str]
    specializations: List[str]

class NeuroscienceTrainer(IntegratedBrainSimulationTrainer):
    """Specialized trainer for neuroscience and brain development"""
    
    def __init__(self, config: BrainSimulationConfig):
        super().__init__(config)
        self.neuroscience_data = []
        self.brain_development_stages = []
        
    def load_neuroscience_data(self):
        """Load specialized neuroscience data"""
        logger.info("Loading neuroscience data...")
        
        # Load brain development training pack
        training_pack_path = project_root / "training_systems"
        if training_pack_path.exists():
            for md_file in training_pack_path.glob("*.md"):
                if md_file.name.startswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
                    content = md_file.read_text()
                    self.neuroscience_data.append({
                        "text": content,
                        "source": f"brain_development_{md_file.stem}",
                        "type": "neuroscience"
                    })
        
        # Load research data
        research_path = project_root / "development_tools/research"
        if research_path.exists():
            for file_path in research_path.rglob("*.md"):
                content = file_path.read_text()
                self.neuroscience_data.append({
                    "text": content,
                    "source": f"research_{file_path.stem}",
                    "type": "neuroscience"
                })
        
        logger.info(f"Loaded {len(self.neuroscience_data)} neuroscience samples")
    
    def create_neuroscience_dataset(self) -> Dataset:
        """Create specialized neuroscience dataset"""
        class NeuroscienceDataset(Dataset):
            def __init__(self, data, tokenizer, max_length):
                self.data = data
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                encoding = self.tokenizer(
                    item["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                return {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": encoding["input_ids"].squeeze(),
                    "source": item["source"],
                    "type": item["type"]
                }
        
        return NeuroscienceDataset(self.neuroscience_data, self.tokenizer, self.config.max_length)
    
    def run_neuroscience_training(self):
        """Run specialized neuroscience training"""
        logger.info("🧠 Starting neuroscience training...")
        
        # Load neuroscience data
        self.load_neuroscience_data()
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Create dataset
        dataset = self.create_neuroscience_dataset()
        
        # Setup trainer with neuroscience-specific settings
        training_args = TrainingArguments(
            output_dir="./neuroscience_output",
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model("./neuroscience_model")
        logger.info("✅ Neuroscience training complete")

class CognitiveLearningTrainer(IntegratedBrainSimulationTrainer):
    """Specialized trainer for cognitive learning and childlike development"""
    
    def __init__(self, config: BrainSimulationConfig):
        super().__init__(config)
        self.cognitive_data = []
        self.learning_progress = []
        
    def load_cognitive_data(self):
        """Load cognitive learning data"""
        logger.info("Loading cognitive learning data...")
        
        # Load childlike learning data
        childlike_path = project_root / "neural_architectures"
        if childlike_path.exists():
            for json_file in childlike_path.glob("*.json"):
                if "learning" in json_file.name or "progress" in json_file.name:
                    try:
                        data = json.loads(json_file.read_text())
                        self.cognitive_data.append({
                            "text": json.dumps(data, indent=2),
                            "source": f"childlike_{json_file.stem}",
                            "type": "cognitive_learning"
                        })
                    except:
                        continue
        
        # Load curiosity and exploration data
        cognitive_path = project_root / "cognitive_engines"
        if cognitive_path.exists():
            for py_file in cognitive_path.glob("*.py"):
                content = py_file.read_text()
                self.cognitive_data.append({
                    "text": content,
                    "source": f"cognitive_{py_file.stem}",
                    "type": "cognitive_learning"
                })
        
        logger.info(f"Loaded {len(self.cognitive_data)} cognitive learning samples")
    
    def create_cognitive_dataset(self) -> Dataset:
        """Create specialized cognitive learning dataset"""
        class CognitiveDataset(Dataset):
            def __init__(self, data, tokenizer, max_length):
                self.data = data
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                encoding = self.tokenizer(
                    item["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                return {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": encoding["input_ids"].squeeze(),
                    "source": item["source"],
                    "type": item["type"]
                }
        
        return CognitiveDataset(self.cognitive_data, self.tokenizer, self.config.max_length)
    
    def run_cognitive_training(self):
        """Run specialized cognitive learning training"""
        logger.info("🧠 Starting cognitive learning training...")
        
        # Load cognitive data
        self.load_cognitive_data()
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Create dataset
        dataset = self.create_cognitive_dataset()
        
        # Setup trainer with cognitive-specific settings
        training_args = TrainingArguments(
            output_dir="./cognitive_output",
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model("./cognitive_model")
        logger.info("✅ Cognitive learning training complete")

class PhysicsSimulationTrainer(IntegratedBrainSimulationTrainer):
    """Specialized trainer for physics simulation and brain modeling"""
    
    def __init__(self, config: BrainSimulationConfig):
        super().__init__(config)
        self.physics_data = []
        self.simulation_results = []
        
    def load_physics_data(self):
        """Load physics simulation data"""
        logger.info("Loading physics simulation data...")
        
        # Load physics simulation data
        physics_path = project_root / "agent_systems/physics_simulation"
        if physics_path.exists():
            for py_file in physics_path.glob("*.py"):
                content = py_file.read_text()
                self.physics_data.append({
                    "text": content,
                    "source": f"physics_{py_file.stem}",
                    "type": "physics_simulation"
                })
        
        # Load simulation results
        simulation_path = project_root / "development_tools"
        for sim_dir in ["demo_simulation_results", "mujoco_simulation_output", "visit_integration_outputs"]:
            sim_path = simulation_path / sim_dir
            if sim_path.exists():
                for json_file in sim_path.glob("*.json"):
                    try:
                        data = json.loads(json_file.read_text())
                        self.physics_data.append({
                            "text": json.dumps(data, indent=2),
                            "source": f"simulation_{json_file.stem}",
                            "type": "physics_simulation"
                        })
                    except:
                        continue
        
        logger.info(f"Loaded {len(self.physics_data)} physics simulation samples")
    
    def create_physics_dataset(self) -> Dataset:
        """Create specialized physics simulation dataset"""
        class PhysicsDataset(Dataset):
            def __init__(self, data, tokenizer, max_length):
                self.data = data
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                encoding = self.tokenizer(
                    item["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                return {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": encoding["input_ids"].squeeze(),
                    "source": item["source"],
                    "type": item["type"]
                }
        
        return PhysicsDataset(self.physics_data, self.tokenizer, self.config.max_length)
    
    def run_physics_training(self):
        """Run specialized physics simulation training"""
        logger.info("⚛️ Starting physics simulation training...")
        
        # Load physics data
        self.load_physics_data()
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Create dataset
        dataset = self.create_physics_dataset()
        
        # Setup trainer with physics-specific settings
        training_args = TrainingArguments(
            output_dir="./physics_output",
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model("./physics_model")
        logger.info("✅ Physics simulation training complete")

class MultiAgentTrainer(IntegratedBrainSimulationTrainer):
    """Specialized trainer for multi-agent systems and coordination"""
    
    def __init__(self, config: BrainSimulationConfig):
        super().__init__(config)
        self.agent_data = []
        self.coordination_data = []
        
    def load_agent_data(self):
        """Load multi-agent system data"""
        logger.info("Loading multi-agent system data...")
        
        # Load integration frameworks
        integration_path = project_root / "integration_frameworks"
        if integration_path.exists():
            for py_file in integration_path.glob("*.py"):
                content = py_file.read_text()
                self.agent_data.append({
                    "text": content,
                    "source": f"integration_{py_file.stem}",
                    "type": "multi_agent"
                })
        
        # Load agent systems
        agent_path = project_root / "agent_systems"
        if agent_path.exists():
            for py_file in agent_path.glob("*.py"):
                content = py_file.read_text()
                self.agent_data.append({
                    "text": content,
                    "source": f"agent_{py_file.stem}",
                    "type": "multi_agent"
                })
        
        # Load core systems
        core_path = project_root / "core_systems"
        if core_path.exists():
            for py_file in core_path.glob("*.py"):
                content = py_file.read_text()
                self.agent_data.append({
                    "text": content,
                    "source": f"core_{py_file.stem}",
                    "type": "multi_agent"
                })
        
        logger.info(f"Loaded {len(self.agent_data)} multi-agent samples")
    
    def create_agent_dataset(self) -> Dataset:
        """Create specialized multi-agent dataset"""
        class AgentDataset(Dataset):
            def __init__(self, data, tokenizer, max_length):
                self.data = data
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                encoding = self.tokenizer(
                    item["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                return {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": encoding["input_ids"].squeeze(),
                    "source": item["source"],
                    "type": item["type"]
                }
        
        return AgentDataset(self.agent_data, self.tokenizer, self.config.max_length)
    
    def run_agent_training(self):
        """Run specialized multi-agent training"""
        logger.info("🤖 Starting multi-agent training...")
        
        # Load agent data
        self.load_agent_data()
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Create dataset
        dataset = self.create_agent_dataset()
        
        # Setup trainer with agent-specific settings
        training_args = TrainingArguments(
            output_dir="./agent_output",
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model("./agent_model")
        logger.info("✅ Multi-agent training complete")

def run_domain_training(domain: str, config: BrainSimulationConfig):
    """Run training for a specific domain"""
    
    if domain == "neuroscience":
        trainer = NeuroscienceTrainer(config)
        trainer.run_neuroscience_training()
    elif domain == "cognitive":
        trainer = CognitiveLearningTrainer(config)
        trainer.run_cognitive_training()
    elif domain == "physics":
        trainer = PhysicsSimulationTrainer(config)
        trainer.run_physics_training()
    elif domain == "agent":
        trainer = MultiAgentTrainer(config)
        trainer.run_agent_training()
    else:
        raise ValueError(f"Unknown domain: {domain}")

def main():
    """Main entry point for domain-specific training"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Domain-specific brain simulation training")
    parser.add_argument("--domain", type=str, required=True, 
                       choices=["neuroscience", "cognitive", "physics", "agent"],
                       help="Domain to train on")
    parser.add_argument("--model", type=str, default="microsoft/DialoGPT-medium",
                       help="Base model to use")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate")
    
    args = parser.parse_args()
    
    # Configuration
    config = BrainSimulationConfig(
        base_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        enable_curiosity=True,
        enable_exploration=True,
        enable_synthesis=True,
        enable_physics=True,
        enable_childlike_learning=True
    )
    
    print(f"🧠 Starting {args.domain} domain training...")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    
    try:
        run_domain_training(args.domain, config)
        print(f"🎉 {args.domain} training complete!")
    except Exception as e:
        print(f"❌ {args.domain} training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
