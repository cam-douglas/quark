#!/usr/bin/env python3
"""
Dataset for the Unified Super Mind

This provides training data that combines various types of learning:
- Language modeling
- Curiosity-driven exploration
- Neuroscience-inspired patterns
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional
import random

class SuperMindDataset(Dataset):
    """Dataset for training the Unified Super Mind."""
    
    def __init__(self, texts: List[str], max_length: int = 512, tokenizer=None):
        self.texts = texts
        self.max_length = max_length
        self.tokenizer = tokenizer
        
        # Generate training examples
        self.examples = self._generate_examples()
    
    def _generate_examples(self) -> List[Dict[str, Any]]:
        """Generate training examples from texts."""
        examples = []
        
        for text in self.texts:
            # Create input-output pairs
            if self.tokenizer:
                # Use proper tokenizer
                tokens = self.tokenizer.encode(text, max_length=self.max_length, truncation=True)
                
                # Create sliding window examples
                for i in range(len(tokens) - 1):
                    input_ids = tokens[:i+1]
                    labels = tokens[1:i+2]
                    
                    # Pad to max_length
                    input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
                    labels = labels + [-100] * (self.max_length - len(labels))  # -100 for padding
                    
                    examples.append({
                        'input_ids': input_ids,
                        'labels': labels
                    })
            else:
                # Character-level tokenization
                chars = [ord(c) for c in text[:self.max_length]]
                
                # Create sliding window examples
                for i in range(len(chars) - 1):
                    input_ids = chars[:i+1]
                    labels = chars[1:i+2]
                    
                    # Pad to max_length
                    input_ids = input_ids + [0] * (self.max_length - len(input_ids))
                    labels = labels + [-100] * (self.max_length - len(labels))
                    
                    examples.append({
                        'input_ids': input_ids,
                        'labels': labels
                    })
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Convert to tensors
        input_ids = torch.tensor(example['input_ids'], dtype=torch.long)
        labels = torch.tensor(example['labels'], dtype=torch.long)
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        attention_mask[input_ids == 0] = 0  # Mask padding tokens
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def create_sample_dataset() -> SuperMindDataset:
    """Create a sample dataset for testing."""
    sample_texts = [
        "The Unified Super Mind combines multiple AI capabilities into one trainable system.",
        "Curiosity-driven learning allows the model to explore and discover new patterns.",
        "Neuroscience-inspired processing simulates brain development and synaptic plasticity.",
        "Continuous learning enables the model to adapt and evolve over time.",
        "Meta-learning allows the system to learn how to learn more effectively.",
        "The Mixture of Experts architecture routes information to specialized neural networks.",
        "Child-like learning mechanisms include exploration, curiosity, and emotional responses.",
        "Brain development stages progress from neural plate to complex cortical structures.",
        "Synaptic plasticity enables the model to strengthen important connections.",
        "Memory consolidation helps retain and integrate new information."
    ]
    
    return SuperMindDataset(sample_texts, max_length=128)
