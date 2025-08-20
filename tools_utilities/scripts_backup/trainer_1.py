#!/usr/bin/env python3
"""
Trainer for the Unified Super Mind
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import logging

from ................................................super_mind import UnifiedSuperMind, SuperMindConfig

logger = logging.getLogger(__name__)

class SuperMindTrainer:
    def __init__(self, model: UnifiedSuperMind, config: SuperMindConfig):
        self.model = model
        self.config = config
        self.device = model.device
        
        # Initialize optimizer
        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate)
        
        # Training state
        self.global_step = 0
        
        logger.info("✅ SuperMindTrainer initialized")
    
    def train(self, train_dataset, val_dataset=None, save_dir=None):
        """Train the Unified Super Mind."""
        logger.info("🚀 Starting training")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        for epoch in range(self.config.max_steps // len(train_loader) + 1):
            for batch in train_loader:
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # Training step
                    losses = self.model.train_step(batch)
                    
                    # Optimizer step
                    if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % 100 == 0:
                        loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
                        logger.info(f"Step {self.global_step}: {loss_str}")
                    
                    if self.global_step >= self.config.max_steps:
                        break
                
                except Exception as e:
                    logger.error(f"Training error: {e}")
                    continue
        
        logger.info("✅ Training completed")
    
    def _collate_fn(self, batch):
        """Collate function for batching."""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        if 'labels' in batch[0]:
            labels = torch.stack([item['labels'] for item in batch])
            result['labels'] = labels
        
        return result
