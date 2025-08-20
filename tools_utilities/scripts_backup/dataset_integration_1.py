"""
Dataset Integration Module for SmallMind

Integrates high-quality open LLM datasets for enhanced natural language understanding
while maintaining code capabilities. Supports streaming, interleaving, and curriculum learning.
"""

import os
import logging
from typing import Dict, List, Optional, Iterator, Any, Union
from dataclasses import dataclass
from pathlib import Path
import json

try:
    from datasets import load_dataset, interleave_datasets, Dataset, IterableDataset
    from datasets.utils.logging import set_verbosity_error
    # Suppress verbose logging from datasets library
    set_verbosity_error()
except ImportError:
    print("Warning: datasets library not available. Install with: pip install datasets")
    Dataset = None
    IterableDataset = None

logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    """Configuration for a dataset in the training pipeline"""
    name: str
    dataset_id: str
    split: str
    subset: Optional[str] = None
    weight: float = 1.0
    max_samples: Optional[int] = None
    streaming: bool = True
    filters: Optional[Dict[str, Any]] = None

@dataclass
class TrainingMixture:
    """Configuration for a training data mixture"""
    name: str
    datasets: List[DatasetConfig]
    interleave_weights: List[float]
    seed: int = 42
    max_total_samples: Optional[int] = None

class DatasetIntegrator:
    """
    Integrates multiple high-quality open LLM datasets for comprehensive training.
    
    Supports:
    - Web-scale pretraining data (FineWeb, Dolma, RedPajama, RefinedWeb)
    - Code-aware corpora (The Stack v2, OpenCodeReasoning)
    - Post-training datasets (Tülu, UltraFeedback, OpenMathInstruct)
    - Streaming and interleaving for memory efficiency
    - Curriculum learning with configurable weights
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/smallmind/datasets")
        self._ensure_cache_dir()
        
        # Predefined dataset configurations
        self.dataset_configs = self._init_dataset_configs()
        
        # Training mixtures
        self.training_mixtures = self._init_training_mixtures()
        
        logger.info(f"DatasetIntegrator initialized with cache_dir: {self.cache_dir}")
    
    def _ensure_cache_dir(self):
        """Ensure the cache directory exists"""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _init_dataset_configs(self) -> Dict[str, DatasetConfig]:
        """Initialize predefined dataset configurations"""
        return {
            # Web-scale pretraining cores (general NLU)
            "fineweb": DatasetConfig(
                name="FineWeb",
                dataset_id="HuggingFaceFW/fineweb",
                split="train",
                subset="CC-MAIN-2024-10",  # Recent, high-quality subset
                weight=0.18,
                streaming=True
            ),
            "fineweb_edu": DatasetConfig(
                name="FineWeb-Edu",
                dataset_id="HuggingFaceFW/fineweb-edu",
                split="train",
                weight=0.15,
                streaming=True
            ),
            "dolma": DatasetConfig(
                name="Dolma",
                dataset_id="allenai/dolma",
                split="train",
                weight=0.18,
                streaming=True
            ),
            "redpajama_v2": DatasetConfig(
                name="RedPajama-Data-V2",
                dataset_id="togethercomputer/RedPajama-Data-V2",
                split="train",
                weight=0.10,
                streaming=True
            ),
            "refinedweb": DatasetConfig(
                name="RefinedWeb",
                dataset_id="tiiuae/falcon-refinedweb",
                split="train",
                weight=0.07,
                streaming=True
            ),
            
            # Code-aware corpora
            "stack_v2": DatasetConfig(
                name="The Stack v2",
                dataset_id="bigcode/the-stack-v2",
                split="train",
                weight=0.10,
                streaming=True,
                filters={"license": ["mit", "apache-2.0", "bsd-3-clause"]}  # Filter by permissive licenses
            ),
            "opencode_reasoning": DatasetConfig(
                name="OpenCodeReasoning",
                dataset_id="nvidia/OpenCodeReasoning",
                split="split_0",
                weight=0.06,
                streaming=True
            ),
            
            # Post-training datasets
            "tulu_sft": DatasetConfig(
                name="Tülu 3 SFT",
                dataset_id="allenai/tulu-3-sft-mixture",
                split="train",
                weight=0.12,
                streaming=True
            ),
            "ultrafeedback": DatasetConfig(
                name="UltraFeedback",
                dataset_id="openbmb/UltraFeedback",
                split="train",
                weight=0.06,
                streaming=True
            ),
            "openmath": DatasetConfig(
                name="OpenMathInstruct-1",
                dataset_id="openmathinstruct/OpenMathInstruct-1",
                split="train",
                weight=0.06,
                streaming=True
            ),
            
            # Multilingual support
            "madlad": DatasetConfig(
                name="MADLAD-400",
                dataset_id="allenai/MADLAD-400",
                split="train",
                weight=0.07,
                streaming=True,
                max_samples=1000000  # Limit for disk space
            )
        }
    
    def _init_training_mixtures(self) -> Dict[str, TrainingMixture]:
        """Initialize predefined training mixtures"""
        return {
            "balanced": TrainingMixture(
                name="Balanced Training Mixture",
                datasets=[
                    self.dataset_configs["fineweb"],
                    self.dataset_configs["dolma"],
                    self.dataset_configs["redpajama_v2"],
                    self.dataset_configs["refinedweb"],
                    self.dataset_configs["stack_v2"],
                    self.dataset_configs["tulu_sft"],
                    self.dataset_configs["opencode_reasoning"],
                    self.dataset_configs["openmath"],
                    self.dataset_configs["ultrafeedback"],
                    self.dataset_configs["madlad"]
                ],
                interleave_weights=[0.18, 0.18, 0.10, 0.07, 0.10, 0.12, 0.06, 0.06, 0.06, 0.07],
                seed=42
            ),
            "code_focused": TrainingMixture(
                name="Code-Focused Mixture",
                datasets=[
                    self.dataset_configs["stack_v2"],
                    self.dataset_configs["opencode_reasoning"],
                    self.dataset_configs["fineweb"],
                    self.dataset_configs["dolma"],
                    self.dataset_configs["tulu_sft"]
                ],
                interleave_weights=[0.30, 0.20, 0.20, 0.20, 0.10],
                seed=42
            ),
            "reasoning_focused": TrainingMixture(
                name="Reasoning-Focused Mixture",
                datasets=[
                    self.dataset_configs["openmath"],
                    self.dataset_configs["opencode_reasoning"],
                    self.dataset_configs["fineweb_edu"],
                    self.dataset_configs["tulu_sft"],
                    self.dataset_configs["ultrafeedback"]
                ],
                interleave_weights=[0.25, 0.20, 0.25, 0.20, 0.10],
                seed=42
            )
        }
    
    def load_dataset(self, config: DatasetConfig) -> Union[Dataset, IterableDataset]:
        """Load a single dataset with error handling and fallbacks"""
        try:
            kwargs = {
                "path": config.dataset_id,
                "split": config.split,
                "streaming": config.streaming
            }
            
            if config.subset:
                kwargs["name"] = config.subset
            
            dataset = load_dataset(**kwargs)
            
            # Apply filters if specified
            if config.filters and hasattr(dataset, 'filter'):
                for filter_key, filter_value in config.filters.items():
                    if isinstance(filter_value, list):
                        dataset = dataset.filter(lambda x: x.get(filter_key) in filter_value)
                    else:
                        dataset = dataset.filter(lambda x: x.get(filter_key) == filter_value)
            
            # Limit samples if specified
            if config.max_samples and hasattr(dataset, 'take'):
                dataset = dataset.take(config.max_samples)
            
            logger.info(f"Successfully loaded dataset: {config.name}")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset {config.name}: {e}")
            # Return a minimal fallback dataset
            return self._create_fallback_dataset(config)
    
    def _create_fallback_dataset(self, config: DatasetConfig) -> IterableDataset:
        """Create a minimal fallback dataset when loading fails"""
        fallback_data = [{"text": f"Fallback text for {config.name}", "source": "fallback"}]
        return IterableDataset.from_list(fallback_data)
    
    def create_training_mixture(self, mixture_name: str = "balanced") -> IterableDataset:
        """Create an interleaved training mixture"""
        if mixture_name not in self.training_mixtures:
            raise ValueError(f"Unknown mixture: {mixture_name}. Available: {list(self.training_mixtures.keys())}")
        
        mixture = self.training_mixtures[mixture_name]
        logger.info(f"Creating training mixture: {mixture.name}")
        
        # Load all datasets
        datasets = []
        for config in mixture.datasets:
            dataset = self.load_dataset(config)
            datasets.append(dataset)
        
        # Create interleaved mixture
        interleaved = interleave_datasets(
            datasets,
            probabilities=mixture.interleave_weights,
            seed=mixture.seed
        )
        
        # Apply total sample limit if specified
        if mixture.max_total_samples:
            interleaved = interleaved.take(mixture.max_total_samples)
        
        logger.info(f"Created interleaved mixture with {len(datasets)} datasets")
        return interleaved
    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific dataset"""
        if dataset_name not in self.dataset_configs:
            return None
        
        config = self.dataset_configs[dataset_name]
        try:
            # Try to get dataset info without loading the full dataset
            from datasets import get_dataset_config_names, get_dataset_split_names
            
            info = {
                "name": config.name,
                "dataset_id": config.dataset_id,
                "split": config.split,
                "subset": config.subset,
                "weight": config.weight,
                "streaming": config.streaming
            }
            
            # Get available configs and splits
            try:
                configs = get_dataset_config_names(config.dataset_id)
                info["available_configs"] = configs
            except:
                info["available_configs"] = []
            
            try:
                splits = get_dataset_split_names(config.dataset_id)
                info["available_splits"] = splits
            except:
                info["available_splits"] = []
            
            return info
            
        except Exception as e:
            logger.warning(f"Could not get detailed info for {dataset_name}: {e}")
            return {
                "name": config.name,
                "dataset_id": config.dataset_id,
                "split": config.split,
                "subset": config.subset,
                "weight": config.weight,
                "streaming": config.streaming
            }
    
    def list_available_mixtures(self) -> List[str]:
        """List available training mixtures"""
        return list(self.training_mixtures.keys())
    
    def create_custom_mixture(self, name: str, datasets: List[str], weights: List[float], 
                             seed: int = 42, max_samples: Optional[int] = None) -> TrainingMixture:
        """Create a custom training mixture"""
        if len(datasets) != len(weights):
            raise ValueError("Number of datasets must match number of weights")
        
        if abs(sum(weights) - 1.0) > 1e-6:
            logger.warning(f"Weights sum to {sum(weights)}, normalizing to 1.0")
            weights = [w / sum(weights) for w in weights]
        
        dataset_configs = []
        for dataset_name in datasets:
            if dataset_name not in self.dataset_configs:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            dataset_configs.append(self.dataset_configs[dataset_name])
        
        mixture = TrainingMixture(
            name=name,
            datasets=dataset_configs,
            interleave_weights=weights,
            seed=seed,
            max_total_samples=max_samples
        )
        
        self.training_mixtures[name] = mixture
        logger.info(f"Created custom mixture: {name}")
        return mixture
    
    def save_mixture_config(self, mixture_name: str, filepath: str):
        """Save a training mixture configuration to file"""
        if mixture_name not in self.training_mixtures:
            raise ValueError(f"Unknown mixture: {mixture_name}")
        
        mixture = self.training_mixtures[mixture_name]
        
        config_data = {
            "name": mixture.name,
            "datasets": [{"name": d.name, "dataset_id": d.dataset_id, "split": d.split, 
                         "subset": d.subset, "weight": d.weight} for d in mixture.datasets],
            "interleave_weights": mixture.interleave_weights,
            "seed": mixture.seed,
            "max_total_samples": mixture.max_total_samples
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Saved mixture config to: {filepath}")
    
    def load_mixture_config(self, filepath: str) -> TrainingMixture:
        """Load a training mixture configuration from file"""
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        
        # Reconstruct dataset configs
        dataset_configs = []
        for d in config_data["datasets"]:
            # Find matching dataset config
            for name, config in self.dataset_configs.items():
                if (config.dataset_id == d["dataset_id"] and 
                    config.split == d["split"] and 
                    config.subset == d.get("subset")):
                    dataset_configs.append(config)
                    break
        
        mixture = TrainingMixture(
            name=config_data["name"],
            datasets=dataset_configs,
            interleave_weights=config_data["interleave_weights"],
            seed=config_data.get("seed", 42),
            max_total_samples=config_data.get("max_total_samples")
        )
        
        self.training_mixtures[mixture.name] = mixture
        logger.info(f"Loaded mixture config: {mixture.name}")
        return mixture

# Convenience functions for easy integration
def get_dataset_integrator(cache_dir: Optional[str] = None) -> DatasetIntegrator:
    """Get a configured dataset integrator instance"""
    return DatasetIntegrator(cache_dir=cache_dir)

def create_balanced_mixture(cache_dir: Optional[str] = None) -> IterableDataset:
    """Quick access to balanced training mixture"""
    integrator = get_dataset_integrator(cache_dir)
    return integrator.create_training_mixture("balanced")

def create_code_focused_mixture(cache_dir: Optional[str] = None) -> IterableDataset:
    """Quick access to code-focused training mixture"""
    integrator = get_dataset_integrator(cache_dir)
    return integrator.create_training_mixture("code_focused")
