#!/usr/bin/env python3
"""
Cloud-Based Wikipedia Training Pipeline
======================================

Trains on Wikipedia data from cloud sources instead of local downloads.
Uses HuggingFace datasets, cloud storage, and streaming for efficiency.

Author: Quark Brain Simulation Team
Date: 2025-01-20
License: MIT
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
from datasets import Dataset, load_dataset, IterableDataset
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

class CloudWikipediaDataLoader:
    """Loads Wikipedia data from cloud sources without local downloads."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # Available cloud-based Wikipedia datasets
        self.dataset_sources = {
            "huggingface_wikipedia": {
                "name": "wikipedia",
                "config": "20231201.en",
                "description": "Official HuggingFace Wikipedia dataset",
                "size": "~6.8M articles",
                "format": "Streaming dataset"
            },
            "huggingface_wikipedia_corpus": {
                "name": "wikipedia-corpus",
                "config": None,
                "description": "Wikipedia corpus for language modeling",
                "size": "~6.8M articles",
                "format": "Streaming dataset"
            },
            "huggingface_wikipedia_20220301": {
                "name": "wikipedia",
                "config": "20220301.en",
                "description": "Wikipedia from March 2022",
                "size": "~6.8M articles",
                "format": "Streaming dataset"
            }
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def list_available_datasets(self):
        """List all available cloud-based Wikipedia datasets."""
        print("🌐 Available Cloud-Based Wikipedia Datasets")
        print("=" * 60)
        
        for key, info in self.dataset_sources.items():
            print(f"📚 {key}")
            print(f"   Name: {info['name']}")
            print(f"   Config: {info['config']}")
            print(f"   Description: {info['description']}")
            print(f"   Size: {info['size']}")
            print(f"   Format: {info['format']}")
            print()
    
    def load_wikipedia_dataset(self, source_key: str = "huggingface_wikipedia", 
                             max_samples: int = None, streaming: bool = True) -> Dataset:
        """Load Wikipedia dataset from cloud source."""
        if source_key not in self.dataset_sources:
            raise ValueError(f"Unknown source: {source_key}")
        
        source_info = self.dataset_sources[source_key]
        self.logger.info(f"🌐 Loading Wikipedia dataset from cloud source: {source_key}")
        self.logger.info(f"   Dataset: {source_info['name']}")
        self.logger.info(f"   Config: {source_info['config']}")
        self.logger.info(f"   Streaming: {streaming}")
        
        try:
            if streaming:
                # Use streaming for large datasets
                dataset = load_dataset(
                    source_info['name'],
                    source_info['config'],
                    split='train',
                    streaming=True
                )
                self.logger.info("✅ Streaming dataset loaded successfully")
                return dataset
            else:
                # Load full dataset (use with caution for very large datasets)
                if max_samples:
                    dataset = load_dataset(
                        source_info['name'],
                        source_info['config'],
                        split=f'train[:{max_samples}]'
                    )
                else:
                    dataset = load_dataset(
                        source_info['name'],
                        source_info['config'],
                        split='train'
                    )
                self.logger.info(f"✅ Dataset loaded successfully: {len(dataset):,} samples")
                return dataset
                
        except Exception as e:
            self.logger.error(f"❌ Error loading dataset: {e}")
            raise
    
    def process_streaming_dataset(self, streaming_dataset: IterableDataset, 
                                max_samples: int = None) -> List[Dict[str, Any]]:
        """Process streaming dataset into a list for training."""
        self.logger.info("🔄 Processing streaming Wikipedia dataset...")
        
        articles = []
        sample_count = 0
        
        # Convert streaming dataset to list
        for article in tqdm(streaming_dataset, desc="Processing articles"):
            # Clean and validate article
            processed_article = self._process_article(article)
            if processed_article:
                articles.append(processed_article)
                sample_count += 1
                
                if sample_count % 1000 == 0:
                    self.logger.info(f"Processed {sample_count:,} articles...")
                
                if max_samples and sample_count >= max_samples:
                    break
        
        self.logger.info(f"✅ Processed {len(articles):,} articles from streaming dataset")
        return articles
    
    def _process_article(self, article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process and clean a single Wikipedia article."""
        try:
            # Extract title and text
            title = article.get('title', '')
            text = article.get('text', '')
            
            # Skip articles without content
            if not title or not text or len(text.strip()) < 100:
                return None
            
            # Clean text (basic cleaning)
            cleaned_text = self._clean_text(text)
            if len(cleaned_text) < 100:
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

class CloudWikipediaTrainer:
    """Trains models on cloud-based Wikipedia data."""
    
    def __init__(self, model_name: str = "distilgpt2", output_dir: str = "cloud_wikipedia_training"):
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
    
    def train_on_cloud_wikipedia(self, articles: List[Dict[str, Any]], 
                                training_config: Dict[str, Any] = None):
        """Train the model on cloud-based Wikipedia data."""
        self.logger.info("🚀 Starting cloud-based Wikipedia training...")
        
        # Prepare dataset
        processed_dataset = self.prepare_dataset(articles)
        
        # Default training configuration
        if training_config is None:
            training_config = {
                "num_train_epochs": 1,
                "per_device_train_batch_size": 4,
                "per_device_eval_batch_size": 4,
                "warmup_steps": 500,
                "weight_decay": 0.01,
                "logging_steps": 100,
                "save_steps": 5000,
                "eval_steps": 5000,
                "evaluation_strategy": "steps",
                "save_total_limit": 3,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
                "dataloader_pin_memory": False,
                "gradient_accumulation_steps": 4,
                "learning_rate": 5e-5,
                "fp16": True,  # Use mixed precision for efficiency
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
    """Main function to run cloud-based Wikipedia training."""
    print("🌐 Cloud-Based Wikipedia Training Pipeline")
    print("=" * 60)
    print("This trains on Wikipedia data from cloud sources!")
    print("No massive local downloads required.")
    print()
    
    # Check system requirements
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Training will be slower on CPU.")
        print("   Consider using a GPU for faster training.")
    
    try:
        # Step 1: List available datasets
        print("📚 Step 1: Available Cloud Datasets")
        print("-" * 40)
        data_loader = CloudWikipediaDataLoader()
        data_loader.list_available_datasets()
        
        # Step 2: Load Wikipedia dataset from cloud
        print("🌐 Step 2: Loading Wikipedia from Cloud")
        print("-" * 40)
        
        # Use streaming for large datasets
        streaming_dataset = data_loader.load_wikipedia_dataset(
            source_key="huggingface_wikipedia",
            streaming=True
        )
        
        # Step 3: Process streaming dataset
        print("🔄 Step 3: Processing Cloud Dataset")
        print("-" * 40)
        
        # Start with a reasonable sample size for testing
        print("Processing articles (starting with 50k for testing)...")
        articles = data_loader.process_streaming_dataset(
            streaming_dataset, 
            max_samples=50000  # Start with 50k articles
        )
        
        print(f"✅ Processed {len(articles):,} articles from cloud source")
        
        # Step 4: Train model
        print("\n🚀 Step 4: Training Model on Cloud Data")
        print("-" * 40)
        trainer = CloudWikipediaTrainer()
        
        # Training configuration for cloud-based data
        training_config = {
            "num_train_epochs": 1,  # Start with 1 epoch
            "per_device_train_batch_size": 2,  # Conservative batch size
            "per_device_eval_batch_size": 2,
            "warmup_steps": 500,
            "weight_decay": 0.01,
            "logging_steps": 100,
            "save_steps": 5000,
            "eval_steps": 5000,
            "evaluation_strategy": "steps",
            "save_total_limit": 5,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "dataloader_pin_memory": False,
            "gradient_accumulation_steps": 8,  # Effective batch size = 2 * 8 = 16
            "learning_rate": 3e-5,  # Conservative learning rate
            "fp16": True,
        }
        
        # Start training
        trainer.train_on_cloud_wikipedia(articles, training_config)
        
        print("\n🎉 Cloud-based Wikipedia training completed!")
        print(f"📁 Model saved to: {trainer.output_dir}")
        print("\nNext steps:")
        print("1. Test the trained model")
        print("2. Scale up to more articles from cloud")
        print("3. Integrate with brain simulation")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
