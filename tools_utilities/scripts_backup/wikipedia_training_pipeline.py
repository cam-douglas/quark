#!/usr/bin/env python3
"""
Wikipedia Training Pipeline for Quark Brain Simulation
=====================================================

Comprehensive Wikipedia data training solution with multiple data sources:
1. Official Wikipedia dumps (complete database)
2. HuggingFace Wikipedia datasets (preprocessed)
3. MediaWiki API (limited, for specific queries)
4. Wikipedia2Vec embeddings (pre-trained)

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
from typing import Dict, List, Any, Optional, Tuple, Generator
from pathlib import Path
import argparse

# Data processing libraries
import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset

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

class WikipediaDataSources:
    """Available Wikipedia data sources for training."""
    
    SOURCES = {
        "official_dumps": {
            "name": "Official Wikipedia Dumps",
            "url": "https://dumps.wikimedia.org/",
            "description": "Complete Wikipedia database dumps (XML format)",
            "size": "~20GB compressed, ~100GB uncompressed",
            "format": "XML",
            "update_frequency": "Daily",
            "license": "CC BY-SA 3.0",
            "best_for": "Complete training, custom preprocessing"
        },
        "huggingface_datasets": {
            "name": "HuggingFace Wikipedia Datasets",
            "url": "https://huggingface.co/datasets?search=wikipedia",
            "description": "Preprocessed Wikipedia datasets on HuggingFace",
            "size": "Various (1GB-50GB)",
            "format": "HuggingFace Dataset",
            "update_frequency": "Varies",
            "license": "CC BY-SA 3.0",
            "best_for": "Quick start, preprocessed data"
        },
        "mediawiki_api": {
            "name": "MediaWiki API",
            "url": "https://www.mediawiki.org/wiki/API:Main_page",
            "description": "Real-time API access to Wikipedia content",
            "size": "Rate limited",
            "format": "JSON",
            "update_frequency": "Real-time",
            "license": "CC BY-SA 3.0",
            "best_for": "Specific queries, small datasets"
        },
        "wikipedia2vec": {
            "name": "Wikipedia2Vec Embeddings",
            "url": "https://wikipedia2vec.github.io/",
            "description": "Pre-trained word embeddings from Wikipedia",
            "size": "~5GB",
            "format": "Embeddings",
            "update_frequency": "Periodic",
            "license": "CC BY-SA 3.0",
            "best_for": "Word embeddings, knowledge graph"
        }
    }

class WikipediaDumpProcessor:
    """Process official Wikipedia dumps for training."""
    
    def __init__(self, dump_dir: str = "./wikipedia_dumps"):
        self.dump_dir = Path(dump_dir)
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def download_dump(self, language: str = "en", date: str = None) -> str:
        """Download Wikipedia dump for specified language and date."""
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        
        # Wikipedia dump URLs
        base_url = "https://dumps.wikimedia.org"
        dump_url = f"{base_url}/{language}wiki/{date}/{language}wiki-{date}-pages-articles-multistream.xml.bz2"
        
        local_path = self.dump_dir / f"{language}wiki-{date}-pages-articles.xml.bz2"
        
        if local_path.exists():
            self.logger.info(f"✅ Dump already exists: {local_path}")
            return str(local_path)
        
        self.logger.info(f"📥 Downloading Wikipedia dump: {dump_url}")
        self.logger.info(f"📁 Saving to: {local_path}")
        
        try:
            response = requests.get(dump_url, stream=True)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.logger.info(f"✅ Download completed: {local_path}")
            return str(local_path)
            
        except Exception as e:
            self.logger.error(f"❌ Download failed: {e}")
            return None
    
    def parse_wikipedia_xml(self, dump_path: str, max_articles: int = None) -> Generator[Dict[str, Any], None, None]:
        """Parse Wikipedia XML dump and yield articles."""
        self.logger.info(f"🔄 Parsing Wikipedia dump: {dump_path}")
        
        article_count = 0
        
        try:
            with bz2.open(dump_path, 'rt', encoding='utf-8') as f:
                # Skip to first page
                for line in f:
                    if '<page>' in line:
                        break
                
                # Parse pages
                page_content = []
                in_page = False
                
                for line in f:
                    if '<page>' in line:
                        in_page = True
                        page_content = [line]
                    elif '</page>' in line:
                        page_content.append(line)
                        in_page = False
                        
                        # Parse the complete page
                        page_xml = ''.join(page_content)
                        article = self._parse_page_xml(page_xml)
                        
                        if article:
                            yield article
                            article_count += 1
                            
                            if max_articles and article_count >= max_articles:
                                break
                    elif in_page:
                        page_content.append(line)
                        
        except Exception as e:
            self.logger.error(f"❌ Error parsing dump: {e}")
    
    def _parse_page_xml(self, page_xml: str) -> Optional[Dict[str, Any]]:
        """Parse a single page XML and extract article data."""
        try:
            root = ET.fromstring(page_xml)
            
            # Extract page metadata
            title = root.find('.//title').text if root.find('.//title') is not None else ""
            page_id = root.find('.//id').text if root.find('.//id') is not None else ""
            revision_id = root.find('.//revision/id').text if root.find('.//revision/id') is not None else ""
            
            # Extract content
            text_elem = root.find('.//revision/text')
            if text_elem is not None and text_elem.text:
                content = text_elem.text
            else:
                return None
            
            # Skip non-articles (redirects, talk pages, etc.)
            if (title.startswith('Talk:') or 
                title.startswith('User:') or 
                title.startswith('File:') or
                title.startswith('Template:') or
                title.startswith('Category:') or
                title.startswith('Help:') or
                title.startswith('Wikipedia:') or
                title.startswith('Portal:') or
                title.startswith('Special:') or
                title.startswith('MediaWiki:')):
                return None
            
            # Clean content
            cleaned_content = self._clean_wikipedia_content(content)
            
            if len(cleaned_content.strip()) < 100:  # Skip very short articles
                return None
            
            return {
                "title": title,
                "page_id": page_id,
                "revision_id": revision_id,
                "content": cleaned_content,
                "length": len(cleaned_content),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to parse page: {e}")
            return None
    
    def _clean_wikipedia_content(self, content: str) -> str:
        """Clean Wikipedia content for training."""
        # Remove Wikipedia markup
        import re
        
        # Remove HTML tags
        content = re.sub(r'<[^>]+>', '', content)
        
        # Remove Wikipedia templates {{...}}
        content = re.sub(r'\{\{[^}]*\}\}', '', content)
        
        # Remove Wikipedia links [[...]]
        content = re.sub(r'\[\[([^|\]]*?)\]\]', r'\1', content)  # Keep link text
        content = re.sub(r'\[\[[^|\]]*\|([^\]]*?)\]\]', r'\1', content)  # Keep link text
        
        # Remove external links [...]
        content = re.sub(r'\[[^\]]*\]', '', content)
        
        # Remove references
        content = re.sub(r'<ref[^>]*>.*?</ref>', '', content, flags=re.DOTALL)
        content = re.sub(r'<ref[^>]*/>', '', content)
        
        # Remove section headers
        content = re.sub(r'^=+\s*([^=]+)\s*=+$', r'\1', content, flags=re.MULTILINE)
        
        # Remove multiple newlines
        content = re.sub(r'\n\s*\n', '\n\n', content)
        
        # Remove leading/trailing whitespace
        content = content.strip()
        
        return content
    
    def create_training_dataset(self, dump_path: str, max_articles: int = 10000) -> Dataset:
        """Create HuggingFace dataset from Wikipedia dump."""
        self.logger.info(f"📊 Creating training dataset from dump...")
        
        articles = []
        for article in self.parse_wikipedia_xml(dump_path, max_articles):
            if article:
                articles.append(article)
                
                if len(articles) % 1000 == 0:
                    self.logger.info(f"📝 Processed {len(articles)} articles...")
        
        # Create dataset
        dataset = Dataset.from_list(articles)
        self.logger.info(f"✅ Created dataset with {len(dataset)} articles")
        
        return dataset

class HuggingFaceWikipediaProcessor:
    """Process Wikipedia datasets from HuggingFace."""
    
    def __init__(self, cache_dir: str = "./wikipedia_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def get_available_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available Wikipedia datasets on HuggingFace."""
        return {
            "wikipedia": {
                "name": "wikipedia",
                "description": "Wikipedia articles in multiple languages",
                "size": "Large",
                "languages": ["en", "de", "fr", "es", "it", "pt", "nl", "pl", "ru", "ja", "zh"],
                "best_for": "Multi-language training"
            },
            "wikipedia-corpus": {
                "name": "wikipedia-corpus",
                "description": "Cleaned Wikipedia corpus",
                "size": "Medium",
                "languages": ["en"],
                "best_for": "Clean text training"
            },
            "wikipedia-sentences": {
                "name": "wikipedia-sentences",
                "description": "Wikipedia articles split into sentences",
                "size": "Large",
                "languages": ["en"],
                "best_for": "Sentence-level training"
            }
        }
    
    def load_dataset(self, dataset_name: str = "wikipedia", language: str = "en", 
                    split: str = "train", max_samples: int = None) -> Dataset:
        """Load Wikipedia dataset from HuggingFace."""
        self.logger.info(f"📥 Loading {dataset_name} dataset ({language})...")
        
        try:
            if dataset_name == "wikipedia":
                # Use the current Wikipedia dataset format
                dataset = load_dataset(
                    "wikipedia",
                    f"20231201.{language}",
                    split=split,
                    cache_dir=self.cache_dir
                )
            elif dataset_name == "wikipedia-corpus":
                dataset = load_dataset(
                    "wikipedia-corpus",
                    split=split,
                    cache_dir=self.cache_dir
                )
            elif dataset_name == "wikipedia-sentences":
                dataset = load_dataset(
                    "wikipedia-sentences",
                    split=split,
                    cache_dir=self.cache_dir
                )
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            self.logger.info(f"✅ Loaded {len(dataset)} samples from {dataset_name}")
            return dataset
            
        except Exception as e:
            self.logger.error(f"❌ Error loading dataset: {e}")
            raise
    
    def prepare_training_data(self, dataset: Dataset, format_type: str = "text") -> Dataset:
        """Prepare dataset for training."""
        self.logger.info(f"🔄 Preparing training data (format: {format_type})...")
        
        if format_type == "text":
            # For language modeling
            def format_text(example):
                if "text" in example:
                    return {"text": example["text"]}
                elif "content" in example:
                    return {"text": example["content"]}
                elif "title" in example and "text" in example:
                    return {"text": f"Title: {example['title']}\n\n{example['text']}"}
                else:
                    return {"text": str(example)}
            
            formatted_dataset = dataset.map(format_text)
            
        elif format_type == "qa":
            # For question-answering (simplified)
            def format_qa(example):
                title = example.get("title", "Unknown")
                content = example.get("text", example.get("content", ""))
                
                # Create simple QA pairs
                return {
                    "question": f"What is {title}?",
                    "answer": content[:1000]  # Truncate for simplicity
                }
            
            formatted_dataset = dataset.map(format_qa)
        
        else:
            formatted_dataset = dataset
        
        self.logger.info(f"✅ Prepared {len(formatted_dataset)} training examples")
        return formatted_dataset

class WikipediaTrainer:
    """Train models on Wikipedia data."""
    
    def __init__(self, model_name: str = "gpt2", device: str = "auto"):
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
    
    def setup_training(self, train_dataset: Dataset, output_dir: str = "./wikipedia_trained") -> Tuple[Trainer, TrainingArguments]:
        """Setup training configuration."""
        self.logger.info(f"⚙️ Setting up training...")
        
        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )
        
        tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            save_steps=1000,
            eval_steps=1000,
            evaluation_strategy="steps",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
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
    
    def train(self, train_dataset: Dataset, output_dir: str = "./wikipedia_trained") -> Dict[str, Any]:
        """Train the model on Wikipedia data."""
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
                "timestamp": datetime.now().isoformat()
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

def main():
    """Main function for Wikipedia training pipeline."""
    parser = argparse.ArgumentParser(description="Wikipedia Training Pipeline")
    parser.add_argument("--source", choices=["dumps", "huggingface", "api"], default="huggingface",
                       help="Wikipedia data source")
    parser.add_argument("--language", default="en", help="Wikipedia language")
    parser.add_argument("--max-articles", type=int, default=10000, help="Maximum articles to process")
    parser.add_argument("--model", default="gpt2", help="Base model for training")
    parser.add_argument("--output-dir", default="./wikipedia_trained", help="Output directory")
    parser.add_argument("--list-sources", action="store_true", help="List available data sources")
    
    args = parser.parse_args()
    
    if args.list_sources:
        print("📚 Available Wikipedia Data Sources:")
        print("=" * 50)
        for key, source in WikipediaDataSources.SOURCES.items():
            print(f"\n🔗 {source['name']}")
            print(f"   URL: {source['url']}")
            print(f"   Description: {source['description']}")
            print(f"   Size: {source['size']}")
            print(f"   Format: {source['format']}")
            print(f"   Best for: {source['best_for']}")
        return
    
    # Initialize components
    if args.source == "dumps":
        processor = WikipediaDumpProcessor()
        
        # Download and process dump
        dump_path = processor.download_dump(args.language)
        if dump_path:
            train_dataset = processor.create_training_dataset(dump_path, args.max_articles)
        else:
            print("❌ Failed to download Wikipedia dump")
            return
    
    elif args.source == "huggingface":
        processor = HuggingFaceWikipediaProcessor()
        
        # Load dataset
        train_dataset = processor.load_dataset("wikipedia", args.language, max_samples=args.max_articles)
        train_dataset = processor.prepare_training_data(train_dataset, "text")
    
    else:
        print("❌ API source not implemented yet")
        return
    
    # Train model
    trainer = WikipediaTrainer(args.model)
    result = trainer.train(train_dataset, args.output_dir)
    
    if result["success"]:
        print(f"🎉 Training completed successfully!")
        print(f"📁 Model saved to: {result['model_path']}")
        print(f"📊 Total samples: {result['total_samples']}")
        print(f"⏱️  Training time: {result['training_time']}")
    else:
        print(f"❌ Training failed: {result['error']}")

if __name__ == "__main__":
    main()
