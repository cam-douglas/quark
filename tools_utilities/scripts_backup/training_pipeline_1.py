"""
Training Pipeline for SmallMind

Integrates the dataset integrator with model training workflows, supporting
both pretraining and fine-tuning with the high-quality open LLM datasets.
"""

import os
import logging
import json
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclasses import data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclass
import time
from datetime import datetime

try:
    import torch
    from torch.utils.data import DataLoader, IterableDataset
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, TrainingArguments, 
        Trainer, DataCollatorForLanguageModeling, get_scheduler
    )
    from accelerate import Accelerator
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch/Transformers not available. Install with: pip install torch transformers accelerate")
    TORCH_AVAILABLE = False

from .....................................................dataset_integration import DatasetIntegrator, TrainingMixture

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training runs"""
    model_name_or_path: str
    output_dir: str
    mixture_name: str = "balanced"
    max_steps: int = 1000
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None
    max_samples_per_dataset: Optional[int] = None
    custom_mixture_config: Optional[Dict[str, Any]] = None

class SmallMindTrainer:
    """
    Training pipeline for SmallMind models using high-quality open datasets.
    
    Supports:
    - Multiple training mixtures (balanced, code-focused, reasoning-focused)
    - Streaming datasets for memory efficiency
    - Custom mixture configurations
    - Integration with existing model architecture
    - Progress tracking and checkpointing
    """
    
    def __init__(self, base_dir: str = "./models", cache_dir: Optional[str] = None):
        self.base_dir = Path(base_dir)
        self.cache_dir = cache_dir
        self.dataset_integrator = DatasetIntegrator(cache_dir)
        
        # Ensure directories exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        # Training state
        self.current_training = None
        self.training_history = []
        
        logger.info(f"SmallMindTrainer initialized with base_dir: {self.base_dir}")
    
    def prepare_training_mixture(self, config: TrainingConfig) -> IterableDataset:
        """Prepare the training data mixture based on configuration"""
        logger.info(f"Preparing training mixture: {config.mixture_name}")
        
        if config.custom_mixture_config:
            # Create custom mixture from config
            mixture = self.dataset_integrator.load_mixture_config(
                config.custom_mixture_config
            )
            return self.dataset_integrator.create_training_mixture(mixture.name)
        else:
            # Use predefined mixture
            return self.dataset_integrator.create_training_mixture(config.mixture_name)
    
    def setup_model_and_tokenizer(self, config: TrainingConfig):
        """Setup the model and tokenizer for training"""
        logger.info(f"Setting up model: {config.model_name_or_path}")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_name_or_path,
                trust_remote_code=True
            )
            
            # Ensure padding token is set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Resize token embeddings if needed
            model.resize_token_embeddings(len(tokenizer))
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def create_data_collator(self, tokenizer):
        """Create data collator for language modeling"""
        return DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # We're doing causal language modeling
        )
    
    def setup_training_arguments(self, config: TrainingConfig, output_dir: str) -> TrainingArguments:
        """Setup training arguments"""
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=None,  # We'll use max_steps instead
            max_steps=config.max_steps,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            weight_decay=config.weight_decay,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            eval_steps=config.eval_steps,
            save_total_limit=config.save_total_limit,
            dataloader_num_workers=config.dataloader_num_workers,
            remove_unused_columns=config.remove_unused_columns,
            push_to_hub=config.push_to_hub,
            hub_model_id=config.hub_model_id,
            hub_token=config.hub_token,
            report_to=None,  # Disable wandb/tensorboard for now
            logging_dir=f"{output_dir}/logs",
            run_name=f"smallmind-{config.mixture_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            # Optimizations
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=True,
            gradient_checkpointing=True,
            # Save strategy
            save_strategy="steps",
            evaluation_strategy="steps" if config.eval_steps > 0 else "no",
            # Learning rate scheduling
            lr_scheduler_type="cosine",
            # Mixed precision
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        )
    
    def tokenize_function(self, examples, tokenizer, max_length: int = 2048):
        """Tokenize examples for training"""
        # Handle different text field names
        text_field = None
        for field in ["text", "content", "instruction", "input"]:
            if field in examples:
                text_field = field
                break
        
        if text_field is None:
            # Try to find any field that looks like text
            for key, value in examples.items():
                if isinstance(value, str) and len(value) > 100:
                    text_field = key
                    break
        
        if text_field is None:
            logger.warning("No suitable text field found in examples")
            return {"input_ids": [], "attention_mask": []}
        
        # Tokenize the text
        tokenized = tokenizer(
            examples[text_field],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_overflowing_tokens=False,
        )
        
        return tokenized
    
    def create_dataset_loader(self, dataset: IterableDataset, tokenizer, 
                            config: TrainingConfig, max_length: int = 2048):
        """Create a dataset loader with tokenization"""
        
        def tokenize_batch(examples):
            return self.tokenize_function(examples, tokenizer, max_length)
        
        # Apply tokenization to the dataset
        tokenized_dataset = dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=dataset.column_names if hasattr(dataset, 'column_names') else None
        )
        
        # Filter out empty sequences
        tokenized_dataset = tokenized_dataset.filter(
            lambda x: len(x["input_ids"]) > 0
        )
        
        return tokenized_dataset
    
    def train(self, config: TrainingConfig) -> Dict[str, Any]:
        """Main training function"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch/Transformers not available")
        
        logger.info(f"Starting training with config: {config.mixture_name}")
        start_time = time.time()
        
        try:
            # Create output directory
            output_dir = Path(config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save training config
            config_path = output_dir / "training_config.json"
            with open(config_path, 'w') as f:
                json.dump(config.__dict__, f, indent=2, default=str)
            
            # Setup model and tokenizer
            model, tokenizer = self.setup_model_and_tokenizer(config)
            
            # Prepare training data
            training_dataset = self.prepare_training_mixture(config)
            
            # Create tokenized dataset
            tokenized_dataset = self.create_dataset_loader(
                training_dataset, tokenizer, config
            )
            
            # Setup training arguments
            training_args = self.setup_training_arguments(config, str(output_dir))
            
            # Create data collator
            data_collator = self.create_data_collator(tokenizer)
            
            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer,
            )
            
            # Start training
            logger.info("Starting training...")
            train_result = trainer.train()
            
            # Save final model
            trainer.save_model()
            tokenizer.save_pretrained(output_dir)
            
            # Save training results
            results_path = output_dir / "training_results.json"
            with open(results_path, 'w') as f:
                json.dump(train_result.metrics, f, indent=2)
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Prepare results
            results = {
                "training_successful": True,
                "output_dir": str(output_dir),
                "training_time_seconds": training_time,
                "training_time_hours": training_time / 3600,
                "final_loss": train_result.metrics.get("train_loss", "unknown"),
                "config": config.__dict__,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save results
            with open(output_dir / "training_summary.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Training completed successfully in {training_time/3600:.2f} hours")
            logger.info(f"Results saved to: {output_dir}")
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            training_time = time.time() - start_time
            
            # Save error information
            error_results = {
                "training_successful": False,
                "error": str(e),
                "training_time_seconds": training_time,
                "config": config.__dict__,
                "timestamp": datetime.now().isoformat()
            }
            
            if 'output_dir' in locals():
                error_path = Path(config.output_dir) / "training_error.json"
                with open(error_path, 'w') as f:
                    json.dump(error_results, f, indent=2)
            
            raise
    
    def create_custom_mixture(self, name: str, datasets: List[str], weights: List[float],
                            **kwargs) -> TrainingMixture:
        """Create a custom training mixture"""
        return self.dataset_integrator.create_custom_mixture(name, datasets, weights, **kwargs)
    
    def list_available_mixtures(self) -> List[str]:
        """List available training mixtures"""
        return self.dataset_integrator.list_available_mixtures()
    
    def get_mixture_info(self, mixture_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific training mixture"""
        if mixture_name not in self.dataset_integrator.training_mixtures:
            return None
        
        mixture = self.dataset_integrator.training_mixtures[mixture_name]
        return {
            "name": mixture.name,
            "datasets": [d.name for d in mixture.datasets],
            "weights": mixture.interleave_weights,
            "seed": mixture.seed,
            "max_samples": mixture.max_total_samples
        }
    
    def validate_training_config(self, config: TrainingConfig) -> List[str]:
        """Validate training configuration and return any issues"""
        issues = []
        
        # Check model path
        if not os.path.exists(config.model_name_or_path) and not config.model_name_or_path.startswith(("microsoft/", "meta-", "google/", "huggingface/")):
            issues.append(f"Model path not found: {config.model_name_or_path}")
        
        # Check mixture exists
        if config.mixture_name not in self.dataset_integrator.training_mixtures:
            issues.append(f"Unknown mixture: {config.mixture_name}")
        
        # Check batch size and gradient accumulation
        effective_batch_size = config.per_device_train_batch_size * config.gradient_accumulation_steps
        if effective_batch_size < 8:
            issues.append(f"Effective batch size ({effective_batch_size}) is very small. Consider increasing batch_size or gradient_accumulation_steps")
        
        # Check learning rate
        if config.learning_rate > 1e-3:
            issues.append(f"Learning rate ({config.learning_rate}) is quite high. Consider using 1e-5 to 1e-4 for fine-tuning")
        
        return issues

# Convenience functions for easy integration
def get_trainer(base_dir: str = "./models", cache_dir: Optional[str] = None) -> SmallMindTrainer:
    """Get a configured SmallMind trainer instance"""
    return SmallMindTrainer(base_dir, cache_dir)

def quick_train(model_path: str, mixture_name: str = "balanced", output_dir: str = "./trained_model",
                max_steps: int = 1000, **kwargs) -> Dict[str, Any]:
    """Quick training function with sensible defaults"""
    trainer = get_trainer()
    
    config = TrainingConfig(
        model_name_or_path=model_path,
        output_dir=output_dir,
        mixture_name=mixture_name,
        max_steps=max_steps,
        **kwargs
    )
    
    # Validate config
    issues = trainer.validate_training_config(config)
    if issues:
        logger.warning("Training config issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    
    return trainer.train(config)
