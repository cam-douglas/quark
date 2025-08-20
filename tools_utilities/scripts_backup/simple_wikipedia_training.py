#!/usr/bin/env python3
"""
Simple Wikipedia Training Pipeline
=================================

A simplified version that works with available datasets and demonstrates
the complete training pipeline for brain simulation.

Author: Quark Brain Simulation Team
Date: 2025-01-20
"""

import os, sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Data processing libraries
import numpy as np
import pandas as pd
from data_knowledge.datasets_knowledge.datasets_knowledge.datasetssets import Dataset, load_dataset
import torch
from torch.utils.data import DataLoader

# Transformers for training
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleWikipediaTrainer:
    """Simple Wikipedia trainer using available datasets."""
    
    def __init__(self, model_name: str = "distilgpt2", device: str = "auto"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else device)
        self.logger = logging.getLogger(__name__)
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
    
    def create_sample_dataset(self, num_samples: int = 1000) -> Dataset:
        """Create a sample dataset for demonstration."""
        self.logger.info(f"📝 Creating sample dataset with {num_samples} examples...")
        
        # Create sample Wikipedia-style articles
        sample_articles = []
        
        topics = [
            "Artificial Intelligence", "Machine Learning", "Neural Networks",
            "Brain Simulation", "Consciousness", "Neuroscience",
            "Computer Science", "Mathematics", "Physics", "Biology",
            "Psychology", "Philosophy", "Technology", "Science", "Research"
        ]
        
        for i in range(num_samples):
            topic = topics[i % len(topics)]
            article = {
                "title": f"Introduction to {topic}",
                "text": f"""
{topic} is a fascinating field of study that has captured the attention of researchers worldwide. 
This article provides an overview of the fundamental concepts and recent developments in {topic}.

{topic} encompasses various approaches and methodologies that have evolved over time. 
Researchers in this field work on understanding the underlying principles and developing practical applications.

The study of {topic} involves both theoretical and experimental work. 
Scientists use mathematical models, computational simulations, and empirical studies to advance our understanding.

Recent advances in {topic} have led to significant breakthroughs in multiple domains. 
These developments have practical implications for technology, medicine, and society as a whole.

Future research in {topic} is expected to focus on emerging challenges and opportunities. 
The field continues to evolve rapidly, driven by new discoveries and technological innovations.
""".strip(),
                "category": topic.lower().replace(" ", "_"),
                "length": len(f"Introduction to {topic}"),
                "timestamp": datetime.now().isoformat()
            }
            sample_articles.append(article)
        
        dataset = Dataset.from_list(sample_articles)
        self.logger.info(f"✅ Created dataset with {len(dataset)} articles")
        return dataset
    
    def prepare_training_data(self, dataset: Dataset) -> Dataset:
        """Prepare dataset for training."""
        self.logger.info("🔄 Preparing training data...")
        
        def format_for_training(example):
            return {
                "text": f"Title: {example['title']}\n\n{example['text']}"
            }
        
        formatted_dataset = dataset.map(format_for_training)
        self.logger.info(f"✅ Prepared {len(formatted_dataset)} training examples")
        return formatted_dataset
    
    def setup_training(self, train_dataset: Dataset, output_dir: str = "./simple_wikipedia_trained") -> Tuple[Trainer, TrainingArguments]:
        """Setup training configuration."""
        self.logger.info(f"⚙️ Setting up training...")
        
        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=256,  # Shorter for demo
                return_tensors="pt"
            )
        
        tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=2,  # Shorter for demo
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            warmup_steps=50,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            save_total_limit=2,
            dataloader_pin_memory=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Use causal language modeling
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        return trainer, training_args
    
    def train(self, train_dataset: Dataset, output_dir: str = "./simple_wikipedia_trained") -> Dict[str, Any]:
        """Train the model on Wikipedia-style data."""
        self.logger.info(f"🚀 Starting Wikipedia training...")
        
        try:
            # Setup training
            trainer, training_args = self.setup_training(train_dataset, output_dir)
            
            # Train
            training_result = trainer.train()
            
            # Save model
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            # Create training report
            report = {
                "success": True,
                "model_path": output_dir,
                "training_loss": training_result.training_loss,
                "total_steps": training_result.global_step,
                "total_samples": len(train_dataset),
                "training_time": str(training_result.metrics.get("train_runtime", "N/A")),
                "timestamp": datetime.now().isoformat(),
                "model_name": self.model_name,
                "device": str(self.device)
            }
            
            # Save report
            report_path = os.path.join(output_dir, "training_report.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"✅ Training completed successfully!")
            self.logger.info(f"📁 Model saved to: {output_dir}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")
            return {"success": False, "error": str(e)}
    
    def test_model(self, model_path: str, test_prompts: List[str] = None) -> Dict[str, Any]:
        """Test the trained model."""
        if test_prompts is None:
            test_prompts = [
                "Title: Introduction to Brain Simulation\n\n",
                "Title: Introduction to Artificial Intelligence\n\n",
                "Title: Introduction to Neural Networks\n\n"
            ]
        
        try:
            # Load trained model
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model.to(self.device)
            
            results = []
            for prompt in test_prompts:
                # Tokenize input
                inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
                
                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=200,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Decode
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                results.append({
                    "prompt": prompt,
                    "generated": generated_text,
                    "length": len(generated_text)
                })
            
            return {"success": True, "results": results}
            
        except Exception as e:
            return {"success": False, "error": str(e)}

def main():
    """Main function for simple Wikipedia training."""
    print("🧠 Simple Wikipedia Training Pipeline")
    print("=" * 50)
    
    # Step 1: Create trainer
    trainer = SimpleWikipediaTrainer(model_name="distilgpt2", device="cpu")
    print(f"✅ Loaded model: {trainer.model_name}")
    print(f"✅ Device: {trainer.device}")
    
    # Step 2: Create sample dataset
    dataset = trainer.create_sample_dataset(num_samples=500)
    print(f"✅ Created dataset with {len(dataset)} articles")
    
    # Step 3: Prepare for training
    train_dataset = trainer.prepare_training_data(dataset)
    print(f"✅ Prepared {len(train_dataset)} training examples")
    
    # Step 4: Train model
    result = trainer.train(train_dataset, "./simple_wikipedia_trained")
    
    if result["success"]:
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Model saved to: {result['model_path']}")
        print(f"📊 Training loss: {result['training_loss']:.4f}")
        print(f"⏱️  Training time: {result['training_time']}")
        
        # Step 5: Test model
        print(f"\n🧪 Testing trained model...")
        test_result = trainer.test_model(result['model_path'])
        
        if test_result["success"]:
            print("✅ Model testing successful!")
            for i, result_item in enumerate(test_result["results"]):
                print(f"\n📝 Test {i+1}:")
                print(f"   Prompt: {result_item['prompt'][:50]}...")
                print(f"   Generated: {result_item['generated'][:100]}...")
        else:
            print(f"❌ Model testing failed: {test_result['error']}")
        
        print(f"\n🎯 Next steps:")
        print(f"1. Use with real Wikipedia data: python scripts/wikipedia_training_pipeline.py --source dumps")
        print(f"2. Scale up training: Increase num_samples and epochs")
        print(f"3. Use larger model: Change model_name to 'gpt2-medium' or 'gpt2-large'")
        
    else:
        print(f"❌ Training failed: {result['error']}")

if __name__ == "__main__":
    main()
