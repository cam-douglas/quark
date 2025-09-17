"""
Brainstem Segmentation Training Pipeline

Complete implementation of preprocessing → augmentation → training pipeline
for brainstem subdivision segmentation using ViT-GNN hybrid architecture.
"""

from .pipeline import BrainstemTrainer, BrainstemDataset, TrainingConfig, MetricsTracker

__version__ = "1.0.0"
__all__ = ["BrainstemTrainer", "BrainstemDataset", "TrainingConfig", "MetricsTracker"]
