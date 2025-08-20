#!/usr/bin/env python3
"""
Wikipedia Cloud Training Pipeline
=================================

Large-scale Wikipedia dataset training with cloud infrastructure support.
Handles the complete English Wikipedia dump (~20GB compressed, ~80GB uncompressed)
with distributed training, preprocessing, and model integration.

Purpose: Train models on complete Wikipedia dataset using cloud resources
Inputs: Wikipedia dump files, model configuration, cloud credentials
Outputs: Trained model checkpoints, training metrics, processed datasets
Seeds: Fixed random seeds for reproducible training
Dependencies: torch, transformers, datasets, accelerate, wandb, boto3
"""

import os, sys
import json
import logging
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    TrainingArguments, Trainer, TrainerCallback,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset, concatenate_datasets
from accelerate import Accelerator, DistributedDataParallelKwargs
import wandb

# Cloud providers
try:
    import boto3
    from google.cloud import storage as gcs
    from azure.storage.blob import BlobServiceClient
    HAS_CLOUD = True
except ImportError:
    HAS_CLOUD = False
    logging.warning("Cloud libraries not available. Install with: pip install boto3 google-cloud-storage azure-storage-blob")

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from expert_domains.machine_learning.auto_brain_llm import AutoBrainLLM
from expert_domains.data_engineering.knowledge_base import KnowledgeBase


@dataclass
class WikipediaTrainingConfig:
    """Configuration for Wikipedia training pipeline."""
    
    # Dataset configuration
    wikipedia_dump_date: str = "20240101"  # YYYYMMDD format
    max_articles: Optional[int] = None  # None for all articles
    min_article_length: int = 100  # Minimum article length in characters
    max_article_length: int = 50000  # Maximum article length in characters
    
    # Model configuration
    model_name: str = "microsoft/DialoGPT-medium"
    tokenizer_name: Optional[str] = None
    max_sequence_length: int = 2048
    vocab_size: Optional[int] = None
    
    # Training configuration
    batch_size: int = 8
    gradient_accumulation_steps: int = 16
    num_epochs: int = 3
    learning_rate: float = 5e-5
    warmup_steps: int = 10000
    weight_decay: float = 0.01
    fp16: bool = True
    gradient_checkpointing: bool = True
    
    # Cloud configuration
    cloud_provider: str = "aws"  # aws, gcp, azure
    cloud_region: str = "us-west-2"
    storage_bucket: str = "quark-wikipedia-training"
    instance_type: str = "p3.8xlarge"  # AWS instance type
    num_nodes: int = 4
    gpus_per_node: int = 4
    
    # Data processing
    preprocessing_workers: int = 8
    chunk_size: int = 10000  # Articles per processing chunk
    cache_dir: str = "./cache/wikipedia"
    output_dir: str = "./models/wikipedia_trained"
    
    # Monitoring and logging
    logging_steps: int = 100
    save_steps: int = 5000
    eval_steps: int = 10000
    wandb_project: str = "quark-wikipedia-training"
    
    # Safety and limits
    max_training_time: int = 48  # Hours
    memory_limit_gb: int = 200
    disk_space_limit_gb: int = 500
    
    # Random seeds for reproducibility
    seed: int = 42
    data_seed: int = 42
    model_seed: int = 42


class WikipediaDataProcessor:
    """Handles Wikipedia dataset downloading, preprocessing, and chunking."""
    
    def __init__(self, config: WikipediaTrainingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        os.makedirs(config.cache_dir, exist_ok=True)
        os.makedirs(config.output_dir, exist_ok=True)
        
    async def download_wikipedia_dump(self) -> str:
        """Download Wikipedia dump from Wikimedia."""
        dump_url = f"https://dumps.wikimedia.org/enwiki/{self.config.wikipedia_dump_date}/"
        dump_file = f"enwiki-{self.config.wikipedia_dump_date}-pages-articles.xml.bz2"
        local_path = os.path.join(self.config.cache_dir, dump_file)
        
        if os.path.exists(local_path):
            self.logger.info(f"Wikipedia dump already exists: {local_path}")
            return local_path
            
        self.logger.info(f"Downloading Wikipedia dump: {dump_url + dump_file}")
        
        # Use wget for reliable large file download
        import subprocess
        cmd = [
            "wget", 
            "-c",  # Continue partial downloads
            "-O", local_path,
            dump_url + dump_file
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"Failed to download Wikipedia dump: {stderr.decode()}")
            
        self.logger.info(f"Successfully downloaded: {local_path}")
        return local_path
        
    def preprocess_wikipedia_articles(self, dump_path: str) -> Dataset:
        """Preprocess Wikipedia XML dump into clean text dataset."""
        from wiki_extractor import WikiExtractor
        
        self.logger.info("Preprocessing Wikipedia articles...")
        
        # Extract text from XML dump
        extracted_dir = os.path.join(self.config.cache_dir, "extracted")
        os.makedirs(extracted_dir, exist_ok=True)
        
        # Configure WikiExtractor
        extractor = WikiExtractor(
            input_file=dump_path,
            output_dir=extracted_dir,
            processes=self.config.preprocessing_workers,
            min_text_length=self.config.min_article_length,
            max_text_length=self.config.max_article_length
        )
        
        articles = extractor.extract()
        
        # Convert to Hugging Face dataset
        article_texts = []
        article_titles = []
        
        for article in articles:
            if self.config.max_articles and len(article_texts) >= self.config.max_articles:
                break
                
            article_texts.append(article['text'])
            article_titles.append(article['title'])
            
            if len(article_texts) % 10000 == 0:
                self.logger.info(f"Processed {len(article_texts)} articles")
        
        dataset = Dataset.from_dict({
            'text': article_texts,
            'title': article_titles
        })
        
        self.logger.info(f"Created dataset with {len(dataset)} articles")
        return dataset
        
    def tokenize_dataset(self, dataset: Dataset, tokenizer) -> Dataset:
        """Tokenize the Wikipedia dataset for training."""
        def tokenize_function(examples):
            # Combine title and text
            texts = [f"Title: {title}\n\n{text}" for title, text in zip(examples['title'], examples['text'])]
            
            # Tokenize with truncation and padding
            tokenized = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.config.max_sequence_length,
                return_tensors="pt"
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        self.logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=self.config.chunk_size,
            num_proc=self.config.preprocessing_workers,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset


class CloudInfrastructureManager:
    """Manages cloud infrastructure for large-scale training."""
    
    def __init__(self, config: WikipediaTrainingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if not HAS_CLOUD:
            raise ImportError("Cloud libraries not installed. Run: pip install boto3 google-cloud-storage azure-storage-blob")
            
    async def setup_aws_infrastructure(self) -> Dict[str, str]:
        """Set up AWS infrastructure for training."""
        import boto3
        
        session = boto3.Session()
        ec2 = session.client('ec2', region_name=self.config.cloud_region)
        s3 = session.client('s3', region_name=self.config.cloud_region)
        
        # Create S3 bucket for data storage
        try:
            s3.create_bucket(
                Bucket=self.config.storage_bucket,
                CreateBucketConfiguration={'LocationConstraint': self.config.cloud_region}
            )
        except Exception as e:
            if "BucketAlreadyExists" not in str(e):
                self.logger.warning(f"S3 bucket creation issue: {e}")
        
        # Launch training instances
        user_data_script = f"""#!/bin/bash
        apt-get update
        apt-get install -y python3-pip docker.io
        pip3 install torch transformers datasets accelerate wandb
        
        # Setup training environment
        mkdir -p /opt/quark-training
        cd /opt/quark-training
        
        # Download training code
        # This would be replaced with actual deployment logic
        echo "Training environment ready" > /tmp/setup_complete
        """
        
        # Launch instances
        response = ec2.run_instances(
            ImageId='ami-0c55b159cbfafe1d0',  # Deep Learning AMI
            MinCount=self.config.num_nodes,
            MaxCount=self.config.num_nodes,
            InstanceType=self.config.instance_type,
            KeyName='quark-training-key',
            SecurityGroups=['quark-training-sg'],
            UserData=user_data_script,
            TagSpecifications=[
                {
                    'ResourceType': 'instance',
                    'Tags': [
                        {'Key': 'Name', 'Value': 'quark-wikipedia-training'},
                        {'Key': 'Project', 'Value': 'quark-brain-simulation'}
                    ]
                }
            ]
        )
        
        instance_ids = [instance['InstanceId'] for instance in response['Instances']]
        
        self.logger.info(f"Launched {len(instance_ids)} training instances")
        
        # Wait for instances to be running
        waiter = ec2.get_waiter('instance_running')
        waiter.wait(InstanceIds=instance_ids)
        
        # Get instance details
        instances_response = ec2.describe_instances(InstanceIds=instance_ids)
        instances = []
        for reservation in instances_response['Reservations']:
            for instance in reservation['Instances']:
                instances.append({
                    'instance_id': instance['InstanceId'],
                    'public_ip': instance.get('PublicIpAddress'),
                    'private_ip': instance.get('PrivateIpAddress')
                })
        
        return {
            'provider': 'aws',
            'bucket': self.config.storage_bucket,
            'instances': instances,
            'region': self.config.cloud_region
        }
        
    async def cleanup_aws_infrastructure(self, infrastructure_info: Dict):
        """Clean up AWS resources after training."""
        import boto3
        
        session = boto3.Session()
        ec2 = session.client('ec2', region_name=self.config.cloud_region)
        
        instance_ids = [inst['instance_id'] for inst in infrastructure_info['instances']]
        
        if instance_ids:
            ec2.terminate_instances(InstanceIds=instance_ids)
            self.logger.info(f"Terminated {len(instance_ids)} instances")


class WikipediaTrainer:
    """Main trainer for Wikipedia dataset with cloud integration."""
    
    def __init__(self, config: WikipediaTrainingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set random seeds
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
            
        # Initialize components
        self.data_processor = WikipediaDataProcessor(config)
        self.cloud_manager = CloudInfrastructureManager(config) if HAS_CLOUD else None
        self.knowledge_base = KnowledgeBase()
        
        # Initialize accelerator for distributed training
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision='fp16' if config.fp16 else 'no',
            kwargs_handlers=[ddp_kwargs]
        )
        
        # Initialize wandb
        if self.accelerator.is_main_process:
            wandb.init(
                project=config.wandb_project,
                config=config.__dict__,
                name=f"wikipedia-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
    
    async def setup_training_environment(self) -> Dict:
        """Set up the complete training environment."""
        self.logger.info("Setting up Wikipedia training environment...")
        
        # Setup cloud infrastructure if available
        infrastructure_info = None
        if self.cloud_manager:
            infrastructure_info = await self.cloud_manager.setup_aws_infrastructure()
            
        # Download and preprocess Wikipedia data
        dump_path = await self.data_processor.download_wikipedia_dump()
        
        return {
            'infrastructure': infrastructure_info,
            'dump_path': dump_path,
            'setup_time': datetime.now().isoformat()
        }
    
    def prepare_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Prepare the model and tokenizer for training."""
        # Load tokenizer
        tokenizer_name = self.config.tokenizer_name or self.config.model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load or create model
        if os.path.exists(self.config.model_name):
            # Load local model
            model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        else:
            # Load pre-trained model
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.config.fp16 else torch.float32
            )
            
        # Resize token embeddings if vocab size changed
        if self.config.vocab_size and len(tokenizer) != model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
            
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            
        return model, tokenizer
    
    async def train_on_wikipedia(self, setup_info: Dict) -> Dict:
        """Main training loop for Wikipedia dataset."""
        self.logger.info("Starting Wikipedia training...")
        
        # Prepare model and tokenizer
        model, tokenizer = self.prepare_model_and_tokenizer()
        
        # Load and preprocess dataset
        raw_dataset = self.data_processor.preprocess_wikipedia_articles(setup_info['dump_path'])
        tokenized_dataset = self.data_processor.tokenize_dataset(raw_dataset, tokenizer)
        
        # Split dataset
        train_test_split = tokenized_dataset.train_test_split(test_size=0.1, seed=self.config.data_seed)
        train_dataset = train_test_split['train']
        eval_dataset = train_test_split['test']
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.config.fp16,
            dataloader_pin_memory=True,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            report_to="wandb" if self.accelerator.is_main_process else None,
            run_name=f"wikipedia-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            max_steps=-1,  # Train for full epochs
            seed=self.config.seed,
        )
        
        # Custom callback for monitoring
        class WikipediaTrainingCallback(TrainerCallback):
            def __init__(self, logger):
                self.logger = logger
                self.start_time = datetime.now()
                
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step % 1000 == 0:
                    elapsed = datetime.now() - self.start_time
                    self.logger.info(f"Step {state.global_step}, Elapsed: {elapsed}")
                    
            def on_evaluate(self, args, state, control, **kwargs):
                self.logger.info(f"Evaluation at step {state.global_step}")
                
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[WikipediaTrainingCallback(self.logger)]
        )
        
        # Prepare for distributed training
        model, trainer.optimizer, trainer.train_dataloader, trainer.eval_dataloader = self.accelerator.prepare(
            model, trainer.optimizer, trainer.train_dataloader, trainer.eval_dataloader
        )
        
        # Start training
        train_start_time = datetime.now()
        self.logger.info(f"Starting training at {train_start_time}")
        
        try:
            train_result = trainer.train()
            
            # Save final model
            trainer.save_model()
            tokenizer.save_pretrained(self.config.output_dir)
            
            # Save training results
            training_results = {
                'train_runtime': train_result.metrics['train_runtime'],
                'train_samples_per_second': train_result.metrics['train_samples_per_second'],
                'train_loss': train_result.metrics['train_loss'],
                'total_articles_trained': len(train_dataset),
                'model_path': self.config.output_dir,
                'training_completed': datetime.now().isoformat()
            }
            
            # Save results to knowledge base
            await self.knowledge_base.store_training_results(
                experiment_name="wikipedia_complete_training",
                results=training_results,
                model_path=self.config.output_dir
            )
            
            self.logger.info(f"Training completed successfully: {training_results}")
            return training_results
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    async def run_complete_pipeline(self) -> Dict:
        """Run the complete Wikipedia training pipeline."""
        try:
            # Setup environment
            setup_info = await self.setup_training_environment()
            
            # Train model
            training_results = await self.train_on_wikipedia(setup_info)
            
            # Cleanup if using cloud
            if setup_info['infrastructure'] and self.cloud_manager:
                await self.cloud_manager.cleanup_aws_infrastructure(setup_info['infrastructure'])
            
            return {
                'status': 'completed',
                'setup_info': setup_info,
                'training_results': training_results,
                'completion_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'completion_time': datetime.now().isoformat()
            }


async def main():
    """Main entry point for Wikipedia training."""
    parser = argparse.ArgumentParser(description="Train model on complete English Wikipedia")
    
    parser.add_argument("--config", type=str, help="Path to training configuration JSON")
    parser.add_argument("--model-name", type=str, default="microsoft/DialoGPT-medium", help="Base model name")
    parser.add_argument("--cloud-provider", type=str, default="aws", choices=["aws", "gcp", "azure"], help="Cloud provider")
    parser.add_argument("--num-nodes", type=int, default=4, help="Number of training nodes")
    parser.add_argument("--max-articles", type=int, help="Maximum number of articles to train on")
    parser.add_argument("--dry-run", action="store_true", help="Run setup without actual training")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('wikipedia_training.log'),
            logging.StreamHandler()
        ]
    )
    
    # Load or create configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = WikipediaTrainingConfig(**config_dict)
    else:
        config = WikipediaTrainingConfig(
            model_name=args.model_name,
            cloud_provider=args.cloud_provider,
            num_nodes=args.num_nodes,
            max_articles=args.max_articles
        )
    
    # Initialize trainer
    trainer = WikipediaTrainer(config)
    
    if args.dry_run:
        logging.info("Dry run mode - setting up environment only")
        setup_info = await trainer.setup_training_environment()
        logging.info(f"Setup complete: {setup_info}")
    else:
        # Run complete pipeline
        results = await trainer.run_complete_pipeline()
        logging.info(f"Training pipeline results: {results}")
    
    return results


if __name__ == "__main__":
    import asyncio
    results = asyncio.run(main())
    print(f"Wikipedia training completed: {results}")
