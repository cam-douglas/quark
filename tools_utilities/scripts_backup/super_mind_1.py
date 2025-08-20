#!/usr/bin/env python3
"""
Unified Super Mind - Core Implementation

This is the main class that combines all Small-Mind capabilities into one trainable model.
"""

import sys
import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [UNIFIED-SUPER-MIND] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SuperMindConfig:
    """Configuration for the Unified Super Mind."""
    
    # Model architecture
    base_model: str = "microsoft/DialoGPT-medium"
    max_length: int = 2048
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    
    # MOE configuration
    num_experts: int = 8
    expert_capacity: int = 64
    router_jitter_noise: float = 0.1
    
    # Learning parameters
    learning_rate: float = 1e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    max_steps: int = 10000
    
    # Child-like learning
    curiosity_weight: float = 0.3
    exploration_rate: float = 0.1
    emotional_learning: bool = True
    
    # Neuroscience integration
    brain_development_stages: List[str] = field(default_factory=lambda: [
        "neural_plate", "neural_tube", "primary_vesicles", "secondary_vesicles"
    ])
    
    # Training modes
    enable_continuous_learning: bool = True
    enable_meta_learning: bool = True
    enable_curiosity_driven: bool = True

class UnifiedSuperMind(nn.Module):
    """
    Unified Super Mind - A single, trainable model that integrates all Small-Mind capabilities.
    """
    
    def __init__(self, config: SuperMindConfig):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self._init_base_model()
        # Re-initialize other components after base model sets hidden_size
        self._init_moe_system()
        self._init_child_learning()
        self._init_neuroscience_experts()
        self._init_continuous_learning()
        
        # Training state
        self.training_step = 0
        self.learning_history = []
        self.curiosity_patterns = []
        self.brain_development_stage = 0
        
        logger.info(f"🚀 Unified Super Mind initialized on {self.device}")
    
    def _init_base_model(self):
        """Initialize the base language model."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
            self.base_model = AutoModelForCausalLM.from_pretrained(self.config.base_model)
            
            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Update config to match the loaded model
            if hasattr(self.base_model.config, 'hidden_size'):
                self.config.hidden_size = self.base_model.config.hidden_size
                logger.info(f"Updated hidden_size to match model: {self.config.hidden_size}")
            
            logger.info(f"✅ Base model loaded: {self.config.base_model}")
            
        except Exception as e:
            logger.warning(f"⚠️  Could not load base model: {e}")
            # Fallback to a simple transformer
            self._create_fallback_model()
    
    def _create_fallback_model(self):
        """Create a fallback model if base model loading fails."""
        logger.info("🔧 Creating fallback transformer model")
        
        # Ensure hidden_size is divisible by num_heads
        adjusted_hidden_size = max(self.config.hidden_size, self.config.num_heads * 8)
        if adjusted_hidden_size != self.config.hidden_size:
            logger.info(f"Adjusted hidden_size from {self.config.hidden_size} to {adjusted_hidden_size} for compatibility")
            self.config.hidden_size = adjusted_hidden_size
        
        self.tokenizer = None  # Will use character-level tokenization
        self.base_model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.config.hidden_size,
                nhead=self.config.num_heads,
                dim_feedforward=self.config.hidden_size * 4,
                dropout=0.1
            ),
            num_layers=self.config.num_layers
        )
        
        # Add embedding and output layers
        self.embedding = nn.Embedding(256, self.config.hidden_size)  # ASCII characters
        self.output_projection = nn.Linear(self.config.hidden_size, 256)
    
    def _init_moe_system(self):
        """Initialize the Mixture of Experts system."""
        # Create expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size * 2),
                nn.GELU(),
                nn.Linear(self.config.hidden_size * 2, self.config.hidden_size),
                nn.Dropout(0.1)
            ) for _ in range(self.config.num_experts)
        ])
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size // 2, self.config.num_experts),
            nn.Softmax(dim=-1)
        )
        
        logger.info(f"✅ MOE system initialized with {self.config.num_experts} experts")
    
    def _init_child_learning(self):
        """Initialize child-like learning mechanisms."""
        self.curiosity_module = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.exploration_module = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size // 4, 1),
            nn.Tanh()
        )
        
        if self.config.emotional_learning:
            self.emotional_module = nn.Sequential(
                nn.Linear(self.config.hidden_size, 6),  # 6 basic emotions
                nn.Softmax(dim=-1)
            )
        
        logger.info("✅ Child-like learning modules initialized")
    
    def _init_neuroscience_experts(self):
        """Initialize neuroscience-inspired processing."""
        self.brain_development_modules = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.LayerNorm(self.config.hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in self.config.brain_development_stages
        ])
        
        # Synaptic plasticity simulation
        self.synaptic_strength = nn.Parameter(torch.ones(self.config.hidden_size))
        self.plasticity_rate = nn.Parameter(torch.ones(1) * 0.01)
        
        logger.info("✅ Neuroscience modules initialized")
    
    def _init_continuous_learning(self):
        """Initialize continuous learning mechanisms."""
        # Meta-learning components
        self.meta_learner = nn.Sequential(
            nn.Linear(self.config.hidden_size * 2, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.Tanh()
        )
        
        # Memory consolidation
        self.memory_bank = nn.Parameter(torch.randn(1000, self.config.hidden_size) * 0.1)
        self.memory_importance = nn.Parameter(torch.ones(1000))
        
        logger.info("✅ Continuous learning modules initialized")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the Unified Super Mind."""
        batch_size, seq_len = input_ids.shape
        
        # Get base model representations
        if hasattr(self, 'embedding'):
            # Fallback model
            embeddings = self.embedding(input_ids)
            # TransformerEncoder expects (seq_len, batch, features) format
            embeddings = embeddings.transpose(0, 1)
            hidden_states = self.base_model(embeddings)
            # Convert back to (batch, seq_len, features) format
            hidden_states = hidden_states.transpose(0, 1)
            logits = self.output_projection(hidden_states)
        else:
            # HuggingFace model
            if attention_mask is not None:
                outputs = self.base_model(input_ids, attention_mask=attention_mask)
            else:
                outputs = self.base_model(input_ids)
            # Handle different output formats
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                hidden_states = outputs.hidden_states[-1]
            elif hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            else:
                # Use logits as hidden states if nothing else available
                hidden_states = outputs.logits
            logits = outputs.logits
        
        # Apply MOE processing
        hidden_states = self._apply_moe(hidden_states)
        
        # Apply child-like learning
        curiosity_scores = self.curiosity_module(hidden_states)
        exploration_scores = self.exploration_module(hidden_states)
        
        # Apply neuroscience processing
        brain_processed = self._apply_brain_development(hidden_states)
        
        # Apply continuous learning
        meta_learned = self._apply_meta_learning(hidden_states, brain_processed)
        
        # Update learning state
        self._update_learning_state(curiosity_scores, exploration_scores)
        
        return {
            'logits': logits,
            'hidden_states': hidden_states,
            'curiosity_scores': curiosity_scores,
            'exploration_scores': exploration_scores,
            'brain_processed': brain_processed,
            'meta_learned': meta_learned
        }
    
    def _apply_moe(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply Mixture of Experts processing."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Get routing probabilities
        router_logits = self.router(hidden_states)  # [batch, seq, num_experts]
        
        # Add jitter for exploration
        if self.training:
            jitter = torch.randn_like(router_logits) * self.config.router_jitter_noise
            router_logits = router_logits + jitter
        
        # Route to experts
        expert_outputs = []
        for i in range(self.config.num_experts):
            expert_output = self.experts[i](hidden_states)
            expert_outputs.append(expert_output)
        
        expert_outputs = torch.stack(expert_outputs, dim=2)  # [batch, seq, num_experts, hidden]
        
        # Weighted combination
        router_probs = F.softmax(router_logits, dim=-1)
        router_probs = router_probs.unsqueeze(-1)  # [batch, seq, num_experts, 1]
        
        moe_output = torch.sum(expert_outputs * router_probs, dim=2)
        return moe_output
    
    def _apply_brain_development(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply neuroscience-inspired brain development processing."""
        processed = hidden_states
        
        # Apply current brain development stage
        if self.brain_development_stage < len(self.brain_development_modules):
            module = self.brain_development_modules[self.brain_development_stage]
            processed = module(processed)
        
        # Apply synaptic plasticity
        processed = processed * self.synaptic_strength.unsqueeze(0).unsqueeze(0)
        
        return processed
    
    def _apply_meta_learning(self, current_states: torch.Tensor, brain_states: torch.Tensor) -> torch.Tensor:
        """Apply meta-learning for continuous adaptation."""
        # Combine current and brain states
        combined = torch.cat([current_states, brain_states], dim=-1)
        
        # Apply meta-learner
        meta_output = self.meta_learner(combined)
        
        # Update memory bank
        if self.training:
            self._update_memory_bank(meta_output)
        
        return meta_output
    
    def _update_memory_bank(self, new_memory: torch.Tensor):
        """Update the memory bank with new information."""
        # Simple FIFO update
        batch_size = new_memory.shape[0]
        if batch_size > 0:
            # Get the first memory from batch and flatten if needed
            memory_to_store = new_memory[0].detach()
            if memory_to_store.dim() > 1:
                memory_to_store = memory_to_store.mean(dim=0)  # Average across sequence dimension
            
            # Ensure memory_to_store has the right shape
            if memory_to_store.shape[0] != self.memory_bank.shape[1]:
                # Pad or truncate to match memory bank dimensions
                if memory_to_store.shape[0] > self.memory_bank.shape[1]:
                    memory_to_store = memory_to_store[:self.memory_bank.shape[1]]
                else:
                    memory_to_store = F.pad(memory_to_store, (0, self.memory_bank.shape[1] - memory_to_store.shape[0]))
            
            # Find least important memory to replace
            min_importance_idx = torch.argmin(self.memory_importance)
            
            # Update memory and importance
            with torch.no_grad():
                self.memory_bank[min_importance_idx] = memory_to_store.clone()
                self.memory_importance[min_importance_idx] = 1.0
    
    def _update_learning_state(self, curiosity_scores: torch.Tensor, exploration_scores: torch.Tensor):
        """Update the learning state based on curiosity and exploration."""
        if self.training:
            # Record curiosity patterns
            avg_curiosity = curiosity_scores.mean().item()
            self.curiosity_patterns.append(avg_curiosity)
            
            # Update brain development stage based on learning progress
            if len(self.curiosity_patterns) > 100:
                recent_curiosity = np.mean(self.curiosity_patterns[-100:])
                if recent_curiosity > 0.7 and self.brain_development_stage < len(self.config.brain_development_stages) - 1:
                    self.brain_development_stage += 1
                    logger.info(f"🧠 Brain development advanced to stage: {self.config.brain_development_stages[self.brain_development_stage]}")
    
    def generate_response(self, prompt: str, max_length: int = 100) -> str:
        """Generate a response using the unified model."""
        # Tokenize input
        if self.tokenizer:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
        else:
            # Character-level tokenization for fallback
            input_ids = torch.tensor([[ord(c) for c in prompt]], dtype=torch.long).to(self.device)
            attention_mask = torch.ones_like(input_ids)
        
        # Generate response
        with torch.no_grad():
            outputs = self(input_ids, attention_mask)
            logits = outputs['logits']
            
            # Simple greedy decoding
            next_token = torch.argmax(logits[0, -1, :])
            generated_tokens = [next_token.item()]
            
            for _ in range(max_length - 1):
                # Add new token to sequence
                new_input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                new_attention_mask = torch.ones_like(new_input_ids)
                
                # Get next prediction
                outputs = self(new_input_ids, new_attention_mask)
                logits = outputs['logits']
                next_token = torch.argmax(logits[0, -1, :])
                generated_tokens.append(next_token.item())
                
                # Stop if EOS token
                if self.tokenizer and next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode response
        if self.tokenizer:
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        else:
            response = ''.join([chr(t) if 32 <= t <= 126 else ' ' for t in generated_tokens])
        
        return response
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step."""
        self.train()
        
        # Forward pass
        outputs = self(batch['input_ids'], batch['attention_mask'])
        
        # Calculate losses
        losses = {}
        
        # Language modeling loss
        if 'labels' in batch:
            lm_loss = F.cross_entropy(
                outputs['logits'].view(-1, outputs['logits'].size(-1)),
                batch['labels'].view(-1)
            )
            losses['lm_loss'] = lm_loss.item()
        
        # Curiosity-driven learning loss
        curiosity_loss = F.mse_loss(
            outputs['curiosity_scores'],
            torch.ones_like(outputs['curiosity_scores']) * 0.5  # Target curiosity
        )
        losses['curiosity_loss'] = curiosity_loss.item()
        
        # Exploration loss
        exploration_loss = F.mse_loss(
            outputs['exploration_scores'],
            torch.zeros_like(outputs['exploration_scores'])  # Target exploration
        )
        losses['exploration_loss'] = exploration_loss.item()
        
        # Total loss
        total_loss = sum(losses.values())
        
        # Backward pass
        total_loss.backward()
        
        # Update step counter
        self.training_step += 1
        
        return losses
