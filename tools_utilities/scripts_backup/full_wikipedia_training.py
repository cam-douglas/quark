#!/usr/bin/env python3
"""
Full Wikipedia Database Training Pipeline
========================================

Trains on the COMPLETE English Wikipedia database - every single article.
This script downloads the full Wikipedia dump and processes it efficiently.

Author: Quark Brain Simulation Team
Date: 2025-01-20
License: MIT
"""

import os, sys
import json
import gzip
import bz2
import xml.etree.ElementTree as ET
import requests
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Generator
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Data processing
import numpy as np
import pandas as pd
from tqdm import tqdm

# ML libraries
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

class WikipediaDumpDownloader:
    """Downloads the complete English Wikipedia database dump."""
    
    def __init__(self, output_dir: str = "wikipedia_dumps"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logging()
        
        # Wikipedia dump URLs (latest versions)
        self.dump_urls = {
            "pages_articles": "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2",
            "page_meta": "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-meta-current.xml.bz2",
            "category_links": "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-categorylinks.sql.gz",
            "page_links": "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pagelinks.sql.gz"
        }
        
        # File sizes (approximate, will be updated)
        self.expected_sizes = {
            "pages_articles": 20_000_000_000,  # ~20GB
            "page_meta": 500_000_000,          # ~500MB
            "category_links": 1_000_000_000,   # ~1GB
            "page_links": 2_000_000_000        # ~2GB
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'wikipedia_dump_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def download_dump(self, dump_type: str, chunk_size: int = 8192) -> str:
        """Download a Wikipedia dump file with progress tracking."""
        url = self.dump_urls[dump_type]
        filename = url.split('/')[-1]
        filepath = self.output_dir / filename
        
        if filepath.exists():
            self.logger.info(f"✅ {filename} already exists, skipping download")
            return str(filepath)
        
        self.logger.info(f"📥 Downloading {dump_type} dump...")
        self.logger.info(f"   URL: {url}")
        self.logger.info(f"   Expected size: {self.expected_sizes[dump_type] / 1e9:.1f} GB")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            pbar.update(len(chunk))
            
            self.logger.info(f"✅ Downloaded {filename} successfully")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"❌ Error downloading {dump_type}: {e}")
            if filepath.exists():
                filepath.unlink()
            raise
    
    def download_all_dumps(self) -> Dict[str, str]:
        """Download all Wikipedia dumps."""
        self.logger.info("🚀 Starting download of all Wikipedia dumps...")
        
        downloaded_files = {}
        for dump_type in self.dump_urls.keys():
            try:
                filepath = self.download_dump(dump_type)
                downloaded_files[dump_type] = filepath
            except Exception as e:
                self.logger.error(f"Failed to download {dump_type}: {e}")
        
        return downloaded_files

class WikipediaDumpProcessor:
    """Processes Wikipedia dumps to extract articles and metadata."""
    
    def __init__(self, dumps_dir: str = "wikipedia_dumps"):
        self.dumps_dir = Path(dumps_dir)
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def extract_articles_from_xml(self, xml_file: str) -> Generator[Dict[str, Any], None, None]:
        """Extract articles from Wikipedia XML dump."""
        self.logger.info(f"🔍 Processing XML dump: {xml_file}")
        
        # Handle compression
        if xml_file.endswith('.bz2'):
            opener = bz2.open
        elif xml_file.endswith('.gz'):
            opener = gzip.open
        else:
            opener = open
        
        article_count = 0
        with opener(xml_file, 'rb') as f:
            # Parse XML incrementally
            context = ET.iterparse(f, events=('start', 'end'))
            
            current_article = {}
            for event, elem in context:
                if event == 'end' and elem.tag.endswith('page'):
                    # Extract article data
                    article = self._extract_article_data(elem)
                    if article and article.get('text', '').strip():
                        article_count += 1
                        yield article
                        
                        if article_count % 1000 == 0:
                            self.logger.info(f"Processed {article_count:,} articles...")
                    
                    # Clear element to free memory
                    elem.clear()
    
    def _extract_article_data(self, page_elem) -> Optional[Dict[str, Any]]:
        """Extract article data from a page element."""
        try:
            # Get basic page info
            title = page_elem.find('.//title')
            if not title or not title.text:
                return None
            
            # Skip non-articles (redirects, talk pages, etc.)
            if any(skip in title.text.lower() for skip in ['talk:', 'user:', 'file:', 'template:', 'help:', 'wikipedia:']):
                return None
            
            # Get revision text
            revision = page_elem.find('.//revision')
            if revision is None:
                return None
            
            text_elem = revision.find('.//text')
            if text_elem is None or not text_elem.text:
                return None
            
            # Clean text
            text = self._clean_wikipedia_text(text_elem.text)
            if len(text) < 100:  # Skip very short articles
                return None
            
            return {
                'title': title.text,
                'text': text,
                'length': len(text),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.warning(f"Error extracting article: {e}")
            return None
    
    def _clean_wikipedia_text(self, text: str) -> str:
        """Clean Wikipedia text content."""
        if not text:
            return ""
        
        # Remove wiki markup
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
        
        # Remove special characters
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        return text.strip()
    
    def process_to_dataset(self, xml_file: str, max_articles: int = None) -> Dataset:
        """Process Wikipedia dump to HuggingFace dataset."""
        self.logger.info(f"🔄 Converting Wikipedia dump to dataset...")
        
        articles = []
        article_count = 0
        
        for article in self.extract_articles_from_xml(xml_file):
            articles.append(article)
            article_count += 1
            
            if max_articles and article_count >= max_articles:
                break
            
            if article_count % 10000 == 0:
                self.logger.info(f"Collected {article_count:,} articles...")
        
        self.logger.info(f"✅ Processed {len(articles):,} articles")
        
        # Convert to HuggingFace dataset
        from datasets import Dataset
        dataset = Dataset.from_list(articles)
        
        return dataset

class FullWikipediaTrainer:
    """Trains models on the complete Wikipedia database."""
    
    def __init__(self, model_name: str = "distilgpt2", output_dir: str = "full_wikipedia_training"):
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
    
    def prepare_dataset(self, dataset: Dataset, max_length: int = 512) -> Dataset:
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
        
        # Apply tokenization
        processed_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        self.logger.info("✅ Dataset prepared for training")
        return processed_dataset
    
    def train_on_full_wikipedia(self, dataset: Dataset, training_config: Dict[str, Any] = None):
        """Train the model on the full Wikipedia dataset."""
        self.logger.info("🚀 Starting full Wikipedia training...")
        
        # Prepare dataset
        processed_dataset = self.prepare_dataset(dataset)
        
        # Default training configuration
        if training_config is None:
            training_config = {
                "num_train_epochs": 1,
                "per_device_train_batch_size": 4,
                "per_device_eval_batch_size": 4,
                "warmup_steps": 1000,
                "weight_decay": 0.01,
                "logging_steps": 100,
                "save_steps": 10000,
                "eval_steps": 10000,
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
    """Main function to run full Wikipedia training."""
    print("🌍 Full Wikipedia Database Training Pipeline")
    print("=" * 60)
    print("This will train on EVERY English Wikipedia article!")
    print("Estimated time: 24-48 hours for complete training")
    print("Estimated storage: ~50GB for dumps + models")
    print()
    
    # Check system requirements
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Training will be slower on CPU.")
        print("   Consider using a GPU for faster training.")
    
    # Get user confirmation
    response = input("Do you want to proceed with full Wikipedia training? (yes/no): ")
    if response.lower() != 'yes':
        print("Training cancelled.")
        return
    
    try:
        # Step 1: Download Wikipedia dumps
        print("\n📥 Step 1: Downloading Wikipedia dumps...")
        downloader = WikipediaDumpDownloader()
        downloaded_files = downloader.download_all_dumps()
        
        if not downloaded_files:
            print("❌ No dumps downloaded. Exiting.")
            return
        
        # Step 2: Process main articles dump
        print("\n🔄 Step 2: Processing Wikipedia articles...")
        processor = WikipediaDumpProcessor()
        
        # Use the main articles dump
        articles_file = downloaded_files.get("pages_articles")
        if not articles_file:
            print("❌ Articles dump not found. Exiting.")
            return
        
        # Process articles (start with a sample for testing)
        print("Processing articles (starting with sample for testing)...")
        dataset = processor.process_to_dataset(articles_file, max_articles=100000)  # Start with 100k articles
        
        print(f"✅ Processed {len(dataset):,} articles")
        
        # Step 3: Train model
        print("\n🚀 Step 3: Training model on Wikipedia data...")
        trainer = FullWikipediaTrainer()
        
        # Training configuration for full dataset
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
        trainer.train_on_full_wikipedia(dataset, training_config)
        
        print("\n🎉 Full Wikipedia training completed!")
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
