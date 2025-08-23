#!/usr/bin/env python3
"""
Nemotron Post-Training Dataset Integration for Quark Brain Simulation
====================================================================

Integrates NVIDIA's Nemotron Post-Training Dataset (25.6M examples) for enhanced
reasoning, coding, math, and STEM capabilities in brain simulation.

Dataset: https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1
License: CC-BY-4.0
Size: 25.6M examples across 5 categories

Author: Quark Brain Simulation Team
Date: 2025-01-20
"""

import os, sys
import json
import torch
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Transformers and training libraries
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import Dataset, load_dataset
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NemotronConfig:
    """Configuration for Nemotron dataset integration."""
    
    # Dataset information
    DATASET_INFO = {
        "name": "nvidia/Nemotron-Post-Training-Dataset-v1",
        "license": "CC-BY-4.0",
        "size": "25.6M examples",
        "categories": {
            "chat": 746622,
            "code": 1896395,
            "math": 2044407,
            "stem": 20662167,
            "tool_calling": 310051
        },
        "total": 25659642,
        "models_used": ["DeepSeek-R1-0528", "Qwen3-235B-A22B"]
    }
    
    # Training templates for different categories
    TRAINING_TEMPLATES = {
        "chat": "You are a helpful and friendly AI assistant.\n{input}",
        "code": "Write a solution for the following programming challenge. Provide a brief explanation of your approach, followed by the complete code.\n{input}",
        "math": "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}.\n{input}",
        "stem": "Read the following problem carefully and provide a detailed, step-by-step answer.\n{input}",
        "tool_calling": "{input}"  # Raw format for tool calling
    }
    
    # Model configurations for different hardware
    MODEL_CONFIGS = {
        "small": {
            "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "memory_gb": 4,
            "batch_size": 1,
            "gradient_accumulation": 8
        },
        "medium": {
            "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "memory_gb": 16,
            "batch_size": 1,
            "gradient_accumulation": 16
        },
        "large": {
            "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "memory_gb": 32,
            "batch_size": 1,
            "gradient_accumulation": 32
        }
    }

class NemotronDatasetProcessor:
    """Process and prepare Nemotron dataset for training."""
    
    def __init__(self, cache_dir: str = "./nemotron_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def load_dataset_split(self, split: str, max_samples: Optional[int] = None) -> Dataset:
        """Load a specific split of the Nemotron dataset."""
        self.logger.info(f"Loading Nemotron dataset split: {split}")
        
        try:
            # Load dataset from Hugging Face
            dataset = load_dataset(
                "nvidia/Nemotron-Post-Training-Dataset-v1",
                split=split,
                cache_dir=self.cache_dir
            )
            
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            self.logger.info(f"✅ Loaded {len(dataset)} examples from {split} split")
            return dataset
            
        except Exception as e:
            self.logger.error(f"❌ Error loading dataset split {split}: {e}")
            raise
    
    def process_messages(self, messages: List[Dict], template: str) -> str:
        """Process message format for training."""
        if not messages or len(messages) < 2:
            return ""
        
        # Extract user input and assistant response
        user_message = messages[0].get("content", "")
        assistant_message = messages[1].get("content", "")
        
        # Apply template if user message exists
        if user_message:
            formatted_input = template.format(input=user_message)
            return f"{formatted_input}\n{assistant_message}"
        else:
            return assistant_message
    
    def prepare_training_data(self, splits: List[str] = None, max_samples_per_split: int = 10000) -> Dataset:
        """Prepare training data from multiple splits."""
        if splits is None:
            splits = ["chat", "code", "math", "stem", "tool_calling"]
        
        all_data = []
        
        for split in splits:
            self.logger.info(f"Processing {split} split...")
            
            try:
                # Load dataset split
                dataset = self.load_dataset_split(split, max_samples_per_split)
                
                # Get template for this split
                template = NemotronConfig.TRAINING_TEMPLATES.get(split, "{input}")
                
                # Process each example
                for example in dataset:
                    messages = example.get("messages", [])
                    processed_text = self.process_messages(messages, template)
                    
                    if processed_text.strip():
                        all_data.append({
                            "text": processed_text,
                            "category": split,
                            "uuid": example.get("uuid", ""),
                            "license": example.get("license", ""),
                            "generator": example.get("generator", "")
                        })
                
                self.logger.info(f"✅ Processed {len(all_data)} examples from {split}")
                
            except Exception as e:
                self.logger.error(f"❌ Error processing {split} split: {e}")
                continue
        
        # Create final dataset
        final_dataset = Dataset.from_list(all_data)
        self.logger.info(f"🎉 Final dataset: {len(final_dataset)} total examples")
        
        return final_dataset

class NemotronTrainer:
    """Nemotron dataset trainer with brain simulation integration."""
    
    def __init__(self, model_config: str = "small", device: str = None):
        self.config = NemotronConfig.MODEL_CONFIGS[model_config]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.processor = NemotronDatasetProcessor()
        
        self.logger.info(f"Initialized NemotronTrainer with {model_config} config on {self.device}")
    
    def load_model(self):
        """Load the base model and tokenizer."""
        self.logger.info(f"Loading model: {self.config['model_name']}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config['model_name'],
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            self.logger.info("✅ Model and tokenizer loaded successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading model: {e}")
            raise
    
    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize the dataset for training."""
        self.logger.info("Tokenizing dataset...")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=2048,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        self.logger.info("✅ Dataset tokenization complete")
        return tokenized_dataset
    
    def setup_training(self, train_dataset: Dataset, output_dir: str = "./nemotron_fine_tuned") -> Tuple[Trainer, TrainingArguments]:
        """Setup training configuration."""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Tokenize dataset
        tokenized_dataset = self.tokenize_dataset(train_dataset)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Training arguments optimized for Nemotron
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=self.config['batch_size'],
            gradient_accumulation_steps=self.config['gradient_accumulation'],
            learning_rate=1e-5,
            weight_decay=0.01,
            warmup_steps=1000,
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="steps",
            eval_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            dataloader_drop_last=True,
            dataloader_num_workers=4,
            fp16=self.device == "cuda",
            gradient_checkpointing=True,
            report_to=None,
            push_to_hub=False,
            remove_unused_columns=False
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        )
        
        return trainer, training_args
    
    def run_training(self, splits: List[str] = None, max_samples_per_split: int = 5000) -> Dict[str, Any]:
        """Run complete Nemotron training pipeline."""
        self.logger.info("🚀 Starting Nemotron training pipeline...")
        
        try:
            # Step 1: Load model
            self.load_model()
            
            # Step 2: Prepare training data
            train_dataset = self.processor.prepare_training_data(splits, max_samples_per_split)
            
            # Step 3: Setup training
            trainer, training_args = self.setup_training(train_dataset)
            
            # Step 4: Run training
            self.logger.info("🧠 Starting fine-tuning...")
            training_result = trainer.train()
            
            # Step 5: Save model
            model_path = training_args.output_dir
            trainer.save_model()
            self.tokenizer.save_pretrained(model_path)
            
            # Step 6: Create training report
            report = {
                "success": True,
                "model_path": model_path,
                "training_loss": training_result.training_loss,
                "total_steps": training_result.global_step,
                "splits_used": splits or ["chat", "code", "math", "stem", "tool_calling"],
                "max_samples_per_split": max_samples_per_split,
                "total_samples": len(train_dataset),
                "training_time": str(training_result.metrics.get("train_runtime", "N/A")),
                "timestamp": datetime.now().isoformat()
            }
            
            # Save report
            report_path = os.path.join(model_path, "training_report.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"✅ Training completed successfully!")
            self.logger.info(f"📁 Model saved to: {model_path}")
            self.logger.info(f"📊 Training report: {report_path}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")
            return {"success": False, "error": str(e)}
    
    def generate_response(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """Generate response using the fine-tuned model."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):]  # Remove input prompt

def main():
    """Main function for Nemotron training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Nemotron Training Pipeline")
    parser.add_argument("--model-config", choices=["small", "medium", "large"], default="small",
                       help="Model configuration based on available GPU memory")
    parser.add_argument("--splits", nargs="+", 
                       default=["chat", "code", "math", "stem", "tool_calling"],
                       help="Dataset splits to use for training")
    parser.add_argument("--max-samples", type=int, default=5000,
                       help="Maximum samples per split")
    parser.add_argument("--output-dir", default="./nemotron_fine_tuned",
                       help="Output directory for fine-tuned model")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = NemotronTrainer(model_config=args.model_config)
    
    # Run training
    result = trainer.run_training(
        splits=args.splits,
        max_samples_per_split=args.max_samples
    )
    
    if result["success"]:
        print(f"🎉 Training completed successfully!")
        print(f"📁 Model saved to: {result['model_path']}")
        print(f"📊 Total samples: {result['total_samples']}")
        print(f"⏱️  Training time: {result['training_time']}")
    else:
        print(f"❌ Training failed: {result['error']}")

if __name__ == "__main__":
    main()
