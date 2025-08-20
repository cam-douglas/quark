#!/usr/bin/env python3
"""
Test Wikipedia Training Pipeline
===============================

Quick test script to demonstrate Wikipedia training with a small dataset.
This script shows how to use the Wikipedia training pipeline for brain simulation.

Usage:
    python scripts/test_wikipedia_training.py
"""

import os, sys
import json
from pathlib import Path

# Add the scripts directory to path
sys.path.append(os.path.dirname(__file__))

from wikipedia_training_pipeline import (
    WikipediaDataSources,
    HuggingFaceWikipediaProcessor,
    WikipediaTrainer
)

def test_data_sources():
    """Test listing available data sources."""
    print("🔍 Testing Wikipedia Data Sources")
    print("=" * 50)
    
    for key, source in WikipediaDataSources.SOURCES.items():
        print(f"\n📚 {source['name']}")
        print(f"   Description: {source['description']}")
        print(f"   Size: {source['size']}")
        print(f"   Best for: {source['best_for']}")
        print(f"   URL: {source['url']}")

def test_huggingface_loading():
    """Test loading Wikipedia dataset from HuggingFace."""
    print("\n🧪 Testing HuggingFace Wikipedia Loading")
    print("=" * 50)
    
    try:
        processor = HuggingFaceWikipediaProcessor()
        
        # List available datasets
        datasets = processor.get_available_datasets()
        print("Available datasets:")
        for name, info in datasets.items():
            print(f"  - {name}: {info['description']}")
        
        # Load a small sample
        print("\n📥 Loading small Wikipedia sample...")
        dataset = processor.load_dataset("wikipedia", "en", max_samples=100)
        
        print(f"✅ Loaded {len(dataset)} articles")
        
        # Show sample
        sample = dataset[0]
        print(f"\n📄 Sample article:")
        print(f"   Title: {sample.get('title', 'N/A')}")
        print(f"   Length: {len(sample.get('text', ''))} characters")
        print(f"   Preview: {sample.get('text', '')[:200]}...")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def test_training_preparation():
    """Test preparing data for training."""
    print("\n🔄 Testing Training Data Preparation")
    print("=" * 50)
    
    try:
        processor = HuggingFaceWikipediaProcessor()
        dataset = processor.load_dataset("wikipedia", "en", max_samples=50)
        
        # Prepare for training
        train_dataset = processor.prepare_training_data(dataset, "text")
        
        print(f"✅ Prepared {len(train_dataset)} training examples")
        
        # Show formatted sample
        sample = train_dataset[0]
        print(f"\n📝 Formatted sample:")
        print(f"   Text length: {len(sample.get('text', ''))} characters")
        print(f"   Preview: {sample.get('text', '')[:300]}...")
        
        return train_dataset
        
    except Exception as e:
        print(f"❌ Error preparing training data: {e}")
        return None

def test_model_loading():
    """Test loading a small model for training."""
    print("\n🤖 Testing Model Loading")
    print("=" * 50)
    
    try:
        # Use a smaller model for testing
        trainer = WikipediaTrainer(model_name="distilgpt2", device="cpu")
        
        print(f"✅ Loaded model: {trainer.model_name}")
        print(f"✅ Device: {trainer.device}")
        print(f"✅ Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
        
        return trainer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def test_mini_training():
    """Test a mini training run."""
    print("\n🚀 Testing Mini Training Run")
    print("=" * 50)
    
    try:
        # Load small dataset
        processor = HuggingFaceWikipediaProcessor()
        dataset = processor.load_dataset("wikipedia", "en", max_samples=100)
        train_dataset = processor.prepare_training_data(dataset, "text")
        
        # Load small model
        trainer = WikipediaTrainer(model_name="distilgpt2", device="cpu")
        
        # Setup mini training
        output_dir = "./test_wikipedia_training"
        
        # Create a very small training run
        def tokenize_function(examples):
            return trainer.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=256,  # Shorter for testing
                return_tensors="pt"
            )
        
        tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
        
        # Mini training arguments
        from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=1,  # Just 1 epoch for testing
            per_device_train_batch_size=2,  # Small batch size
            warmup_steps=10,  # Very few warmup steps
            logging_steps=10,
            save_steps=50,
            save_total_limit=1,
            dataloader_pin_memory=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=trainer.tokenizer,
            mlm=False,
        )
        
        # Trainer
        hf_trainer = Trainer(
            model=trainer.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        print("🧠 Starting mini training...")
        training_result = hf_trainer.train()
        
        print(f"✅ Mini training completed!")
        print(f"   Training loss: {training_result.training_loss:.4f}")
        print(f"   Total steps: {training_result.global_step}")
        
        # Save model
        hf_trainer.save_model()
        trainer.tokenizer.save_pretrained(output_dir)
        
        # Create test report
        report = {
            "success": True,
            "model_path": output_dir,
            "training_loss": training_result.training_loss,
            "total_steps": training_result.global_step,
            "total_samples": len(train_dataset),
            "test_run": True,
            "timestamp": str(datetime.now())
        }
        
        report_path = os.path.join(output_dir, "test_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📁 Test model saved to: {output_dir}")
        print(f"📊 Test report: {report_path}")
        
        return report
        
    except Exception as e:
        print(f"❌ Error in mini training: {e}")
        return None

def main():
    """Run all tests."""
    print("🧪 Wikipedia Training Pipeline Test Suite")
    print("=" * 60)
    
    # Test 1: Data sources
    test_data_sources()
    
    # Test 2: HuggingFace loading
    dataset = test_huggingface_loading()
    if dataset is None:
        print("❌ Failed to load dataset. Stopping tests.")
        return
    
    # Test 3: Training preparation
    train_dataset = test_training_preparation()
    if train_dataset is None:
        print("❌ Failed to prepare training data. Stopping tests.")
        return
    
    # Test 4: Model loading
    trainer = test_model_loading()
    if trainer is None:
        print("❌ Failed to load model. Stopping tests.")
        return
    
    # Test 5: Mini training
    result = test_mini_training()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary")
    print("=" * 60)
    
    if result and result.get("success"):
        print("✅ All tests passed!")
        print(f"📊 Final training loss: {result['training_loss']:.4f}")
        print(f"📁 Model saved to: {result['model_path']}")
        print("\n🎉 Wikipedia training pipeline is working correctly!")
        print("\nNext steps:")
        print("1. Run with larger dataset: python scripts/wikipedia_training_pipeline.py --source huggingface --max-articles 10000")
        print("2. Use official dumps: python scripts/wikipedia_training_pipeline.py --source dumps --max-articles 50000")
        print("3. Check the training guide: docs/WIKIPEDIA_TRAINING_GUIDE.md")
    else:
        print("❌ Some tests failed. Check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Install dependencies: pip install datasets transformers torch")
        print("2. Check internet connection for dataset download")
        print("3. Ensure sufficient disk space")

if __name__ == "__main__":
    from datetime import datetime
    main()
