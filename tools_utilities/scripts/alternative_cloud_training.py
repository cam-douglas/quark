#!/usr/bin/env python3
"""
Alternative Cloud-Based Training for Wikipedia-Style Knowledge
============================================================

Uses working cloud datasets and alternative sources for comprehensive knowledge training.
This provides Wikipedia-style knowledge without the dataset format issues.

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

class AlternativeCloudDataLoader:
    """Loads alternative cloud datasets for Wikipedia-style knowledge training."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # Working alternative datasets with Wikipedia-style knowledge
        self.alternative_sources = {
            "bookcorpus": {
                "name": "bookcorpus",
                "description": "Large collection of free books (Wikipedia-style knowledge)",
                "size": "~11k books",
                "format": "Standard dataset",
                "testable": True
            },
            "openwebtext": {
                "name": "openwebtext",
                "description": "High-quality web content (comprehensive knowledge)",
                "size": "~8M documents",
                "format": "Standard dataset",
                "testable": True
            },
            "wikihow": {
                "name": "wikihow",
                "description": "How-to articles (structured knowledge)",
                "size": "~200k articles",
                "format": "Standard dataset",
                "testable": True
            },
            "squad": {
                "name": "squad",
                "description": "Question-answer pairs (knowledge testing)",
                "size": "~100k QA pairs",
                "format": "Standard dataset",
                "testable": True
            },
            "natural_questions": {
                "name": "natural_questions",
                "description": "Natural language questions (knowledge breadth)",
                "size": "~300k questions",
                "format": "Standard dataset",
                "testable": True
            }
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def list_alternative_sources(self):
        """List available alternative cloud datasets."""
        print("🌐 Alternative Cloud-Based Knowledge Sources")
        print("=" * 60)
        print("These provide Wikipedia-style knowledge without dataset issues!")
        print()
        
        for key, info in self.alternative_sources.items():
            print(f"📚 {key}")
            print(f"   Name: {info['name']}")
            print(f"   Description: {info['description']}")
            print(f"   Size: {info['size']}")
            print(f"   Format: {info['format']}")
            print(f"   Testable: {info['testable']}")
            print()
    
    def test_dataset_access(self, source_key: str) -> bool:
        """Test if a dataset source is accessible."""
        if source_key not in self.alternative_sources:
            print(f"❌ Unknown source: {source_key}")
            return False
        
        source_info = self.alternative_sources[source_key]
        print(f"🧪 Testing dataset access: {source_key}")
        
        try:
            # Try to load a small sample
            dataset = load_dataset(
                source_info['name'],
                split='train[:5]'  # Just 5 samples for testing
            )
            
            print(f"✅ Successfully loaded {len(dataset)} samples from {source_key}")
            print(f"   Sample columns: {dataset.column_names}")
            if len(dataset) > 0:
                print(f"   Sample data keys: {list(dataset[0].keys())}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load {source_key}: {e}")
            return False
    
    def load_alternative_dataset(self, source_key: str, max_samples: int = 10000) -> Dataset:
        """Load alternative dataset from cloud source."""
        if source_key not in self.alternative_sources:
            raise ValueError(f"Unknown source: {source_key}")
        
        source_info = self.alternative_sources[source_key]
        self.logger.info(f"🌐 Loading alternative dataset: {source_key}")
        self.logger.info(f"   Dataset: {source_info['name']}")
        self.logger.info(f"   Max samples: {max_samples:,}")
        
        try:
            dataset = load_dataset(
                source_info['name'],
                split=f'train[:{max_samples}]'
            )
            
            self.logger.info(f"✅ Dataset loaded successfully: {len(dataset):,} samples")
            return dataset
                
        except Exception as e:
            self.logger.error(f"❌ Error loading dataset: {e}")
            raise
    
    def process_alternative_dataset(self, dataset: Dataset, source_key: str) -> List[Dict[str, Any]]:
        """Process alternative dataset into training-ready format."""
        self.logger.info(f"🔄 Processing {source_key} dataset for training...")
        
        articles = []
        
        for i, item in enumerate(tqdm(dataset, desc=f"Processing {source_key}")):
            # Process based on dataset type
            processed_item = self._process_item_by_type(item, source_key)
            if processed_item:
                articles.append(processed_item)
            
            if i % 1000 == 0 and i > 0:
                self.logger.info(f"Processed {i:,} items...")
        
        self.logger.info(f"✅ Processed {len(articles):,} items for training")
        return articles
    
    def _process_item_by_type(self, item: Dict[str, Any], source_key: str) -> Optional[Dict[str, Any]]:
        """Process item based on dataset type."""
        try:
            if source_key == "bookcorpus":
                return self._process_bookcorpus_item(item)
            elif source_key == "openwebtext":
                return self._process_openwebtext_item(item)
            elif source_key == "wikihow":
                return self._process_wikihow_item(item)
            elif source_key == "squad":
                return self._process_squad_item(item)
            elif source_key == "natural_questions":
                return self._process_natural_questions_item(item)
            else:
                return self._process_generic_item(item)
                
        except Exception as e:
            self.logger.warning(f"Error processing item: {e}")
            return None
    
    def _process_bookcorpus_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process bookcorpus item."""
        text = item.get('text', '')
        if not text or len(text.strip()) < 100:
            return None
        
        cleaned_text = self._clean_text(text)
        if len(cleaned_text) < 100:
            return None
        
        return {
            'title': f"Book Excerpt {len(cleaned_text)}",
            'text': cleaned_text,
            'length': len(cleaned_text),
            'source': 'bookcorpus',
            'timestamp': datetime.now().isoformat()
        }
    
    def _process_openwebtext_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process openwebtext item."""
        text = item.get('text', '')
        if not text or len(text.strip()) < 100:
            return None
        
        cleaned_text = self._clean_text(text)
        if len(cleaned_text) < 100:
            return None
        
        return {
            'title': f"Web Content {len(cleaned_text)}",
            'text': cleaned_text,
            'length': len(cleaned_text),
            'source': 'openwebtext',
            'timestamp': datetime.now().isoformat()
        }
    
    def _process_wikihow_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process wikihow item."""
        title = item.get('title', '')
        text = item.get('text', '')
        
        if not title or not text or len(text.strip()) < 50:
            return None
        
        cleaned_text = self._clean_text(text)
        if len(cleaned_text) < 50:
            return None
        
        return {
            'title': title,
            'text': cleaned_text,
            'length': len(cleaned_text),
            'source': 'wikihow',
            'timestamp': datetime.now().isoformat()
        }
    
    def _process_squad_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process squad item."""
        context = item.get('context', '')
        question = item.get('question', '')
        answer = item.get('answers', {}).get('text', [''])[0]
        
        if not context or len(context.strip()) < 50:
            return None
        
        # Combine context, question, and answer
        combined_text = f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"
        cleaned_text = self._clean_text(combined_text)
        
        if len(cleaned_text) < 50:
            return None
        
        return {
            'title': f"QA: {question[:50]}...",
            'text': cleaned_text,
            'length': len(cleaned_text),
            'source': 'squad',
            'timestamp': datetime.now().isoformat()
        }
    
    def _process_natural_questions_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process natural questions item."""
        question = item.get('question', {}).get('text', '')
        answer = item.get('answer', {}).get('text', [''])[0] if item.get('answer') else ''
        
        if not question or len(question.strip()) < 10:
            return None
        
        # Combine question and answer
        combined_text = f"Question: {question}\nAnswer: {answer}"
        cleaned_text = self._clean_text(combined_text)
        
        if len(cleaned_text) < 20:
            return None
        
        return {
            'title': f"Q: {question[:50]}...",
            'text': cleaned_text,
            'length': len(cleaned_text),
            'source': 'natural_questions',
            'timestamp': datetime.now().isoformat()
        }
    
    def _process_generic_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process generic item."""
        # Try to find text content
        text = ""
        for key in ['text', 'content', 'body', 'article']:
            if key in item and item[key]:
                text = item[key]
                break
        
        if not text or len(text.strip()) < 50:
            return None
        
        cleaned_text = self._clean_text(text)
        if len(cleaned_text) < 50:
            return None
        
        return {
            'title': f"Content {len(cleaned_text)}",
            'text': cleaned_text,
            'length': len(cleaned_text),
            'source': 'generic',
            'timestamp': datetime.now().isoformat()
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean text content."""
        if not text:
            return ""
        
        import re
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        return text.strip()

class AlternativeCloudTrainer:
    """Trains models on alternative cloud-based knowledge data."""
    
    def __init__(self, model_name: str = "distilgpt2", output_dir: str = "alternative_cloud_training"):
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
            
            # Tokenize without padding (let the data collator handle it)
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=False,  # Don't pad here
                max_length=max_length,
                return_tensors=None  # Don't return tensors here
            )
            
            # Set labels to input_ids for language modeling
            tokenized["labels"] = tokenized["input_ids"].copy()
            
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
    
    def train_on_alternative_data(self, articles: List[Dict[str, Any]], 
                                 training_config: Dict[str, Any] = None):
        """Train the model on alternative cloud-based knowledge data."""
        self.logger.info("🚀 Starting training on alternative cloud data...")
        
        # Prepare dataset
        processed_dataset = self.prepare_dataset(articles)
        
        # Default training configuration (compatible with current transformers version)
        if training_config is None:
            training_config = {
                "num_train_epochs": 1,
                "per_device_train_batch_size": 2,
                "warmup_steps": 100,
                "weight_decay": 0.01,
                "logging_steps": 50,
                "save_steps": 1000,
                "save_total_limit": 3,
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
    """Main function to run alternative cloud-based training."""
    print("🌐 Alternative Cloud-Based Knowledge Training")
    print("=" * 60)
    print("This uses working cloud datasets for Wikipedia-style knowledge!")
    print()
    
    # Check system requirements
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Training will be slower on CPU.")
        print("   Consider using a GPU for faster training.")
    
    try:
        # Step 1: List alternative sources
        print("📚 Step 1: Available Alternative Sources")
        print("-" * 40)
        data_loader = AlternativeCloudDataLoader()
        data_loader.list_alternative_sources()
        
        # Step 2: Test dataset access
        print("🧪 Step 2: Testing Dataset Access")
        print("-" * 40)
        
        # Test datasets in order of preference
        working_sources = []
        for source_key in ['wikihow', 'bookcorpus', 'openwebtext', 'squad', 'natural_questions']:
            if data_loader.test_dataset_access(source_key):
                working_sources.append(source_key)
        
        if not working_sources:
            print("❌ No working datasets found. Exiting.")
            return
        
        # Use the first working source
        source_key = working_sources[0]
        print(f"✅ Using working source: {source_key}")
        
        # Step 3: Load alternative dataset
        print(f"\n🌐 Step 3: Loading Dataset from {source_key}")
        print("-" * 40)
        
        # Start with a reasonable sample size
        dataset = data_loader.load_alternative_dataset(
            source_key=source_key,
            max_samples=5000  # Start with 5k items
        )
        
        # Step 4: Process dataset
        print("\n🔄 Step 4: Processing Dataset")
        print("-" * 40)
        articles = data_loader.process_alternative_dataset(dataset, source_key)
        
        print(f"✅ Ready to train on {len(articles):,} processed items")
        
        # Step 5: Train model
        print(f"\n🚀 Step 5: Training Model")
        print("-" * 40)
        trainer = AlternativeCloudTrainer()
        
        # Training configuration (compatible with current transformers version)
        training_config = {
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "logging_steps": 50,
            "save_steps": 1000,
            "save_total_limit": 3,
            "dataloader_pin_memory": False,
            "gradient_accumulation_steps": 4,
            "learning_rate": 5e-5,
            "fp16": False,  # Disable for CPU training
        }
        
        # Start training
        trainer.train_on_alternative_data(articles, training_config)
        
        print("\n🎉 Alternative cloud-based training completed!")
        print(f"📁 Model saved to: {trainer.output_dir}")
        print("\nNext steps:")
        print("1. Test the trained model")
        print("2. Scale up to more data sources")
        print("3. Integrate with brain simulation")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


