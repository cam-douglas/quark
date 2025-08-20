#!/usr/bin/env python3
"""
Working Cloud-Based Wikipedia Training
=====================================

Uses working cloud data sources for Wikipedia training.
Handles dataset format changes and provides reliable access.

Author: Quark Brain Simulation Team
Date: 2025-01-20
"""

import os, sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Data processing
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from tqdm import tqdm

# ML libraries
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

class WorkingCloudWikipediaLoader:
    """Loads Wikipedia data from working cloud sources."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # Working cloud-based Wikipedia datasets
        self.working_sources = {
            "wikipedia_simple": {
                "name": "wikipedia",
                "config": "20231201.simple",
                "description": "Simple English Wikipedia (easier to process)",
                "size": "~200k articles",
                "format": "Standard dataset"
            },
            "wikipedia_20220301": {
                "name": "wikipedia",
                "config": "20220301.en",
                "description": "English Wikipedia from March 2022",
                "size": "~6.8M articles",
                "format": "Standard dataset"
            },
            "wikipedia_corpus": {
                "name": "wikipedia-corpus",
                "config": None,
                "description": "Wikipedia corpus for language modeling",
                "size": "~6.8M articles",
                "format": "Standard dataset"
            }
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def list_working_sources(self):
        """List working cloud-based Wikipedia sources."""
        print("🌐 Working Cloud-Based Wikipedia Sources")
        print("=" * 60)
        
        for key, info in self.working_sources.items():
            print(f"📚 {key}")
            print(f"   Name: {info['name']}")
            print(f"   Config: {info['config']}")
            print(f"   Description: {info['description']}")
            print(f"   Size: {info['size']}")
            print(f"   Format: {info['format']}")
            print()
    
    def test_dataset_access(self, source_key: str = "wikipedia_simple") -> bool:
        """Test if a dataset source is accessible."""
        if source_key not in self.working_sources:
            print(f"❌ Unknown source: {source_key}")
            return False
        
        source_info = self.working_sources[source_key]
        print(f"🧪 Testing dataset access: {source_key}")
        
        try:
            # Try to load a small sample
            if source_info['config']:
                dataset = load_dataset(
                    source_info['name'],
                    source_info['config'],
                    split='train[:10]'  # Just 10 samples for testing
                )
            else:
                dataset = load_dataset(
                    source_info['name'],
                    split='train[:10]'
                )
            
            print(f"✅ Successfully loaded {len(dataset)} samples from {source_key}")
            print(f"   Sample columns: {dataset.column_names}")
            if len(dataset) > 0:
                print(f"   Sample data: {dataset[0]}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load {source_key}: {e}")
            return False
    
    def load_working_dataset(self, source_key: str = "wikipedia_simple", 
                           max_samples: int = 10000) -> Dataset:
        """Load Wikipedia dataset from working cloud source."""
        if source_key not in self.working_sources:
            raise ValueError(f"Unknown source: {source_key}")
        
        source_info = self.working_sources[source_key]
        self.logger.info(f"🌐 Loading Wikipedia dataset from working source: {source_key}")
        self.logger.info(f"   Dataset: {source_info['name']}")
        self.logger.info(f"   Config: {source_info['config']}")
        self.logger.info(f"   Max samples: {max_samples:,}")
        
        try:
            if source_info['config']:
                dataset = load_dataset(
                    source_info['name'],
                    source_info['config'],
                    split=f'train[:{max_samples}]'
                )
            else:
                dataset = load_dataset(
                    source_info['name'],
                    split=f'train[:{max_samples}]'
                )
            
            self.logger.info(f"✅ Dataset loaded successfully: {len(dataset):,} samples")
            return dataset
                
        except Exception as e:
            self.logger.error(f"❌ Error loading dataset: {e}")
            raise
    
    def process_dataset_for_training(self, dataset: Dataset) -> List[Dict[str, Any]]:
        """Process dataset into training-ready format."""
        self.logger.info("🔄 Processing dataset for training...")
        
        articles = []
        
        for i, article in enumerate(tqdm(dataset, desc="Processing articles")):
            # Process article based on available columns
            processed_article = self._process_article_for_training(article)
            if processed_article:
                articles.append(processed_article)
            
            if i % 1000 == 0 and i > 0:
                self.logger.info(f"Processed {i:,} articles...")
        
        self.logger.info(f"✅ Processed {len(articles):,} articles for training")
        return articles
    
    def _process_article_for_training(self, article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single article for training."""
        try:
            # Handle different dataset formats
            title = ""
            text = ""
            
            # Try different possible column names
            if 'title' in article:
                title = article['title']
            elif 'page_title' in article:
                title = article['page_title']
            
            if 'text' in article:
                text = article['text']
            elif 'content' in article:
                text = article['content']
            elif 'page_content' in article:
                text = article['page_content']
            
            # Skip if no content
            if not title or not text or len(text.strip()) < 50:
                return None
            
            # Clean text
            cleaned_text = self._clean_text(text)
            if len(cleaned_text) < 50:
                return None
            
            return {
                'title': title,
                'text': cleaned_text,
                'length': len(cleaned_text),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.warning(f"Error processing article: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean Wikipedia text content."""
        if not text:
            return ""
        
        import re
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove wiki links [[link|text]] -> text
        text = re.sub(r'\[\[([^|\]]*?)\|([^\]]*?)\]\]', r'\2', text)
        text = re.sub(r'\[\[([^\]]*?)\]\]', r'\1', text)
        
        # Remove external links [http://... text] -> text
        text = re.sub(r'\[https?://[^\s\]]+\s+([^\]]+)\]', r'\1', text)
        
        # Remove templates {{template}}
        text = re.sub(r'\{\{[^}]+\}\}', '', text)
        
        # Remove references <ref>...</ref>
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
        
        # Remove multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        return text.strip()

class WorkingCloudWikipediaTrainer:
    """Trains models on working cloud-based Wikipedia data."""
    
    def __init__(self, model_name: str = "distilgpt2", output_dir: str = "working_cloud_training"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logging()
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.logger.info(f"✅ Initialized {model_name} model")
        self.logger.info(f"   - Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def prepare_dataset(self, articles: List[Dict[str, Any]], max_length: int = 512) -> Dataset:
        """Prepare dataset for training."""
        self.logger.info("🔄 Preparing dataset for training...")
        
        def tokenize_function(examples):
            # Combine title and text
            texts = [f"Title: {title}\nText: {text}" for title, text in zip(examples['title'], examples['text'])]
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # Set labels to input_ids for language modeling
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        # Convert to HuggingFace dataset
        dataset = Dataset.from_list(articles)
        
        # Apply tokenization
        processed_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        self.logger.info("✅ Dataset prepared for training")
        return processed_dataset
    
    def train_on_working_data(self, articles: List[Dict[str, Any]], 
                             training_config: Dict[str, Any] = None):
        """Train the model on working cloud-based Wikipedia data."""
        self.logger.info("🚀 Starting training on working cloud data...")
        
        # Prepare dataset
        processed_dataset = self.prepare_dataset(articles)
        
        # Default training configuration
        if training_config is None:
            training_config = {
                "num_train_epochs": 1,
                "per_device_train_batch_size": 2,
                "per_device_eval_batch_size": 2,
                "warmup_steps": 100,
                "weight_decay": 0.01,
                "logging_steps": 50,
                "save_steps": 1000,
                "eval_steps": 1000,
                "evaluation_strategy": "steps",
                "save_total_limit": 3,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
                "dataloader_pin_memory": False,
                "gradient_accumulation_steps": 4,
                "learning_rate": 5e-5,
                "fp16": False,  # Disable for CPU training
            }
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            **training_config
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal language modeling
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=processed_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Start training
        self.logger.info("🔥 Training started...")
        trainer.train()
        
        # Save final model
        final_model_path = self.output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))
        
        self.logger.info(f"✅ Training completed! Model saved to {final_model_path}")
        
        return trainer

def main():
    """Main function to run working cloud-based Wikipedia training."""
    print("🌐 Working Cloud-Based Wikipedia Training")
    print("=" * 60)
    print("This uses reliable cloud data sources for Wikipedia training!")
    print()
    
    # Check system requirements
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Training will be slower on CPU.")
        print("   Consider using a GPU for faster training.")
    
    try:
        # Step 1: List working sources
        print("📚 Step 1: Available Working Sources")
        print("-" * 40)
        data_loader = WorkingCloudWikipediaLoader()
        data_loader.list_working_sources()
        
        # Step 2: Test dataset access
        print("🧪 Step 2: Testing Dataset Access")
        print("-" * 40)
        
        # Test simple Wikipedia first (most reliable)
        if data_loader.test_dataset_access("wikipedia_simple"):
            source_key = "wikipedia_simple"
        elif data_loader.test_dataset_access("wikipedia_20220301"):
            source_key = "wikipedia_20220301"
        elif data_loader.test_dataset_access("wikipedia_corpus"):
            source_key = "wikipedia_corpus"
        else:
            print("❌ No working datasets found. Exiting.")
            return
        
        print(f"✅ Using working source: {source_key}")
        
        # Step 3: Load working dataset
        print(f"\n🌐 Step 3: Loading Dataset from {source_key}")
        print("-" * 40)
        
        # Start with a reasonable sample size
        dataset = data_loader.load_working_dataset(
            source_key=source_key,
            max_samples=10000  # Start with 10k articles
        )
        
        # Step 4: Process dataset
        print("\n🔄 Step 4: Processing Dataset")
        print("-" * 40)
        articles = data_loader.process_dataset_for_training(dataset)
        
        print(f"✅ Ready to train on {len(articles):,} processed articles")
        
        # Step 5: Train model
        print(f"\n🚀 Step 5: Training Model")
        print("-" * 40)
        trainer = WorkingCloudWikipediaTrainer()
        
        # Training configuration
        training_config = {
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "logging_steps": 50,
            "save_steps": 1000,
            "eval_steps": 1000,
            "evaluation_strategy": "steps",
            "save_total_limit": 3,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "dataloader_pin_memory": False,
            "gradient_accumulation_steps": 4,
            "learning_rate": 5e-5,
            "fp16": False,  # Disable for CPU training
        }
        
        # Start training
        trainer.train_on_working_data(articles, training_config)
        
        print("\n🎉 Working cloud-based Wikipedia training completed!")
        print(f"📁 Model saved to: {trainer.output_dir}")
        print("\nNext steps:")
        print("1. Test the trained model")
        print("2. Scale up to more articles")
        print("3. Integrate with brain simulation")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
