#!/usr/bin/env python3
"""
🤖 Multi-Agent Coordinator Trainer

Specialized training script for multi-agent coordination systems.
This trainer implements:
- Agent coordination and communication
- Distributed learning and decision making
- Collective intelligence optimization
- Emergent behavior simulation
"""

import os, sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MultiAgentConfig:
    """Configuration for multi-agent training"""
    model_type: str = "agent_coordinator"
    hidden_dim: int = 512
    num_layers: int = 6
    vocab_size: int = 10000
    batch_size: int = 8
    learning_rate: float = 1e-4
    epochs: int = 300
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    agent_count: int = 5
    communication_channels: int = 3
    coordination_weight: float = 0.2
    consensus_threshold: float = 0.7
    dataset_path: str = "data/multi_agent/"
    save_path: str = "models/multi_agent/"
    checkpoint_interval: int = 25
    validation_interval: int = 12

class IndividualAgent(nn.Module):
    """Individual agent in the multi-agent system"""
    
    def __init__(self, agent_id: int, config: MultiAgentConfig):
        super().__init__()
        self.agent_id = agent_id
        self.config = config
        
        # Agent's local processing
        self.local_encoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU()
        )
        
        # Agent's decision making
        self.decision_head = nn.Sequential(
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Agent's communication encoder
        self.communication_encoder = nn.Sequential(
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.communication_channels)
        )
    
    def forward(self, input_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Local processing
        local_features = self.local_encoder(input_features)
        
        # Decision making
        decision = self.decision_head(local_features)
        
        # Communication encoding
        communication = self.communication_encoder(local_features)
        
        return {
            "local_features": local_features,
            "decision": decision,
            "communication": communication,
            "agent_id": self.agent_id
        }

class CommunicationNetwork(nn.Module):
    """Communication network for multi-agent coordination"""
    
    def __init__(self, config: MultiAgentConfig):
        super().__init__()
        self.config = config
        
        # Communication channels
        self.channel_attention = nn.MultiheadAttention(
            embed_dim=config.communication_channels,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Message aggregation
        self.message_aggregator = nn.Sequential(
            nn.Linear(config.communication_channels * config.agent_count, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU()
        )
        
        # Coordination scoring
        self.coordination_scorer = nn.Sequential(
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, agent_communications: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Stack communications from all agents
        communications = torch.stack(agent_communications, dim=1)  # [batch, num_agents, channels]
        
        # Apply attention across agents
        attended_communications, attention_weights = self.channel_attention(
            communications, communications, communications
        )
        
        # Aggregate messages
        flattened = attended_communications.view(attended_communications.size(0), -1)
        aggregated = self.message_aggregator(flattened)
        
        # Score coordination
        coordination_score = self.coordination_scorer(aggregated)
        
        return {
            "aggregated_messages": aggregated,
            "coordination_score": coordination_score,
            "attention_weights": attention_weights
        }

class ConsensusModule(nn.Module):
    """Module for achieving consensus among agents"""
    
    def __init__(self, config: MultiAgentConfig):
        super().__init__()
        self.config = config
        
        # Consensus voting mechanism
        self.vote_aggregator = nn.Sequential(
            nn.Linear(config.agent_count, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Consensus validation
        self.consensus_validator = nn.Sequential(
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, agent_decisions: List[torch.Tensor], aggregated_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Stack decisions from all agents
        decisions = torch.cat(agent_decisions, dim=1)  # [batch, num_agents]
        
        # Aggregate votes
        vote_aggregated = self.vote_aggregator(decisions)
        
        # Validate consensus
        consensus_input = torch.cat([vote_aggregated, aggregated_features], dim=1)
        consensus_score = self.consensus_validator(consensus_input)
        
        # Check if consensus threshold is met
        consensus_achieved = consensus_score > self.config.consensus_threshold
        
        return {
            "vote_aggregated": vote_aggregated,
            "consensus_score": consensus_score,
            "consensus_achieved": consensus_achieved
        }

class AgentCoordinator(nn.Module):
    """Main multi-agent coordination system"""
    
    def __init__(self, config: MultiAgentConfig):
        super().__init__()
        self.config = config
        
        # Word embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # Transformer encoder
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=8,
                dim_feedforward=config.hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(config.num_layers)
        ])
        
        # Individual agents
        self.agents = nn.ModuleList([
            IndividualAgent(i, config) for i in range(config.agent_count)
        ])
        
        # Communication network
        self.communication_network = CommunicationNetwork(config)
        
        # Consensus module
        self.consensus_module = ConsensusModule(config)
        
        # Global coordination head
        self.coordination_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Get embeddings
        embeddings = self.word_embeddings(input_ids)
        
        # Apply transformer layers
        hidden_states = embeddings
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, src_key_padding_mask=attention_mask == 0 if attention_mask is not None else None)
        
        # Extract features
        features = hidden_states.mean(dim=1)
        
        # Process through individual agents
        agent_outputs = []
        agent_decisions = []
        agent_communications = []
        
        for agent in self.agents:
            agent_output = agent(features)
            agent_outputs.append(agent_output)
            agent_decisions.append(agent_output["decision"])
            agent_communications.append(agent_output["communication"])
        
        # Communication and coordination
        communication_output = self.communication_network(agent_communications)
        
        # Consensus building
        consensus_output = self.consensus_module(agent_decisions, communication_output["aggregated_messages"])
        
        # Global coordination
        coordination_input = torch.cat([features, communication_output["aggregated_messages"]], dim=1)
        global_coordination = self.coordination_head(coordination_input)
        
        return {
            "features": features,
            "agent_outputs": agent_outputs,
            "communication_output": communication_output,
            "consensus_output": consensus_output,
            "global_coordination": global_coordination
        }

class MultiAgentDataset(Dataset):
    """Dataset for multi-agent training"""
    
    def __init__(self, data_path: str, config: MultiAgentConfig):
        self.config = config
        self.samples = self._generate_samples()
        
    def _generate_samples(self) -> List[Dict]:
        """Generate training samples"""
        coordination_tasks = [
            "Coordinate team members to solve complex problems",
            "Distribute tasks among agents efficiently",
            "Achieve consensus on important decisions",
            "Collaborate on research and development",
            "Coordinate emergency response operations",
            "Manage distributed computing resources",
            "Coordinate autonomous vehicle fleets",
            "Collaborate on scientific discovery",
            "Coordinate supply chain operations",
            "Manage smart grid energy distribution",
            "Coordinate disaster relief efforts",
            "Collaborate on creative projects",
            "Coordinate financial trading strategies",
            "Manage distributed sensor networks",
            "Coordinate healthcare delivery systems",
            "Collaborate on environmental monitoring",
            "Coordinate educational content delivery",
            "Manage distributed manufacturing",
            "Coordinate security and surveillance",
            "Collaborate on space exploration missions"
        ]
        
        samples = []
        for task in coordination_tasks:
            # Simple tokenization
            words = task.split()
            token_ids = [hash(word) % self.config.vocab_size for word in words]
            
            sample = {
                "tokens": token_ids,
                "text": task,
                "length": len(words),
                "coordination_difficulty": np.random.uniform(0.3, 1.0),
                "consensus_required": np.random.choice([True, False], p=[0.7, 0.3])
            }
            samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Pad sequence
        max_len = 30
        tokens = sample["tokens"][:max_len]
        tokens = tokens + [0] * (max_len - len(tokens))
        
        attention_mask = [1] * len(sample["tokens"][:max_len]) + [0] * (max_len - len(sample["tokens"][:max_len]))
        
        return {
            "input_ids": torch.LongTensor(tokens),
            "attention_mask": torch.LongTensor(attention_mask),
            "text": sample["text"],
            "coordination_difficulty": sample["coordination_difficulty"],
            "consensus_required": sample["consensus_required"]
        }

class MultiAgentTrainer:
    """Trainer for multi-agent coordination models"""
    
    def __init__(self, config: MultiAgentConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize model
        self.model = AgentCoordinator(config).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        
        # Loss functions
        self.coordination_loss = nn.BCELoss()
        self.consensus_loss = nn.BCELoss()
        self.decision_loss = nn.BCELoss()
        
        # Training history
        self.train_losses = []
        self.coordination_scores = []
        self.consensus_scores = []
        self.agent_decisions = []
        
        # Create save directory
        os.makedirs(config.save_path, exist_ok=True)
    
    def train(self, train_loader: DataLoader):
        """Train the multi-agent coordination model"""
        logger.info("Starting multi-agent coordination training...")
        
        for epoch in range(self.config.epochs):
            # Training phase
            train_loss, coordination_score, consensus_score, avg_agent_decision = self._train_epoch(train_loader, epoch)
            
            self.train_losses.append(train_loss)
            self.coordination_scores.append(coordination_score)
            self.consensus_scores.append(consensus_score)
            self.agent_decisions.append(avg_agent_decision)
            
            logger.info(f"Epoch {epoch}: Loss: {train_loss:.4f}, Coordination: {coordination_score:.4f}, "
                       f"Consensus: {consensus_score:.4f}, Avg Agent Decision: {avg_agent_decision:.4f}")
            
            # Checkpointing
            if epoch % self.config.checkpoint_interval == 0:
                self._save_checkpoint(epoch)
        
        # Save final model
        self._save_final_model()
        
        # Plot training curves
        self._plot_training_curves()
        
        logger.info("Multi-agent coordination training completed!")
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float, float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_coordination = 0.0
        total_consensus = 0.0
        total_agent_decisions = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            coordination_difficulty = batch["coordination_difficulty"].to(self.device).unsqueeze(1)
            consensus_required = batch["consensus_required"].to(self.device).unsqueeze(1)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            
            # Calculate losses
            coordination_loss = self.coordination_loss(outputs["global_coordination"], coordination_difficulty)
            consensus_loss = self.consensus_loss(outputs["consensus_output"]["consensus_score"], consensus_required.float())
            
            # Agent decision consistency loss
            agent_decisions = [output["decision"] for output in outputs["agent_outputs"]]
            decision_variance = torch.var(torch.cat(agent_decisions, dim=1), dim=1, keepdim=True)
            decision_loss = torch.mean(decision_variance)
            
            # Combined loss
            total_batch_loss = (coordination_loss + 
                              self.config.coordination_weight * consensus_loss +
                              0.1 * decision_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += total_batch_loss.item()
            total_coordination += outputs["global_coordination"].mean().item()
            total_consensus += outputs["consensus_output"]["consensus_score"].mean().item()
            total_agent_decisions += torch.mean(torch.cat(agent_decisions, dim=1)).item()
            num_batches += 1
            
            if batch_idx % 5 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {total_batch_loss.item():.4f}")
        
        return (total_loss / num_batches, total_coordination / num_batches, 
                total_consensus / num_batches, total_agent_decisions / num_batches)
    
    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "train_losses": self.train_losses,
            "coordination_scores": self.coordination_scores,
            "consensus_scores": self.consensus_scores,
            "agent_decisions": self.agent_decisions
        }
        
        checkpoint_path = os.path.join(self.config.save_path, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_final_model(self):
        """Save the final trained model"""
        model_path = os.path.join(self.config.save_path, "agent_coordinator_final.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "training_history": {
                "train_losses": self.train_losses,
                "coordination_scores": self.coordination_scores,
                "consensus_scores": self.consensus_scores,
                "agent_decisions": self.agent_decisions
            }
        }, model_path)
        logger.info(f"Final model saved: {model_path}")
    
    def _plot_training_curves(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Training loss
        axes[0, 0].plot(self.train_losses)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Coordination scores
        axes[0, 1].plot(self.coordination_scores, color='blue')
        axes[0, 1].set_title('Coordination Scores')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Coordination Score')
        axes[0, 1].grid(True)
        
        # Consensus scores
        axes[1, 0].plot(self.consensus_scores, color='green')
        axes[1, 0].set_title('Consensus Scores')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Consensus Score')
        axes[1, 0].grid(True)
        
        # Agent decisions
        axes[1, 1].plot(self.agent_decisions, color='red')
        axes[1, 1].set_title('Average Agent Decisions')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Decision Score')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.config.save_path, "training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Training curves saved: {plot_path}")

def main():
    """Main entry point for multi-agent training"""
    parser = argparse.ArgumentParser(description="Multi-Agent Coordinator Trainer")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--agent-count", type=int, default=5, help="Number of agents")
    parser.add_argument("--coordination-weight", type=float, default=0.2, help="Coordination weight")
    
    args = parser.parse_args()
    
    # Create configuration
    config = MultiAgentConfig()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.agent_count = args.agent_count
    config.coordination_weight = args.coordination_weight
    
    # Initialize trainer
    trainer = MultiAgentTrainer(config)
    
    # Create datasets and data loaders
    train_dataset = MultiAgentDataset(config.dataset_path, config)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    # Start training
    trainer.train(train_loader)
    
    logger.info("Multi-agent coordination training completed successfully!")

if __name__ == "__main__":
    main()
