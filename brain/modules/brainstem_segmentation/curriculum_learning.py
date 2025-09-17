"""
Curriculum learning framework for brainstem segmentation.

Implements easy-to-hard sample ordering and multi-stage refinement
to improve learning efficiency and final performance.
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning pipeline."""
    
    # Difficulty assessment
    complexity_metric: str = "boundary_complexity"  # or "class_imbalance", "morphogen_variance"
    
    # Curriculum stages
    num_stages: int = 3
    stage_durations: List[int] = None  # epochs per stage, auto-computed if None
    
    # Sample selection
    easy_percentile: float = 0.3  # Bottom 30% = easy samples
    hard_percentile: float = 0.7  # Top 30% = hard samples
    
    # Multi-stage refinement
    enable_refinement: bool = True
    refinement_overlap: float = 0.2  # 20% overlap between stages


class SampleDifficultyAssessor:
    """Assesses training sample difficulty for curriculum ordering."""
    
    def __init__(self, metric_type: str = "boundary_complexity"):
        self.metric_type = metric_type
    
    def assess_volume_difficulty(self, volume: np.ndarray, labels: np.ndarray) -> float:
        """Assess difficulty of a volume-label pair.
        
        Args:
            volume: Input volume (H, W, D)
            labels: Label volume (H, W, D)
            
        Returns:
            Difficulty score (higher = more difficult)
        """
        if self.metric_type == "boundary_complexity":
            return self._boundary_complexity(labels)
        elif self.metric_type == "class_imbalance":
            return self._class_imbalance(labels)
        elif self.metric_type == "morphogen_variance":
            return self._morphogen_variance(volume)
        else:
            raise ValueError(f"Unknown metric: {self.metric_type}")
    
    def _boundary_complexity(self, labels: np.ndarray) -> float:
        """Compute boundary complexity as difficulty metric."""
        # Count boundary voxels
        boundary_count = 0
        
        for axis in range(3):
            # Compute gradients along each axis
            diff = np.diff(labels, axis=axis)
            boundary_count += np.sum(diff != 0)
        
        # Normalize by volume size
        complexity = boundary_count / labels.size
        return float(complexity)
    
    def _class_imbalance(self, labels: np.ndarray) -> float:
        """Compute class imbalance as difficulty metric."""
        unique, counts = np.unique(labels, return_counts=True)
        
        if len(counts) <= 1:
            return 0.0  # No imbalance if only one class
        
        # Shannon entropy as imbalance measure
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        
        # Normalize by max possible entropy
        max_entropy = np.log(len(counts))
        imbalance = 1.0 - (entropy / max_entropy)
        
        return float(imbalance)
    
    def _morphogen_variance(self, volume: np.ndarray) -> float:
        """Compute morphogen concentration variance as difficulty metric."""
        # Higher variance = more complex morphogen patterns = harder
        return float(np.var(volume))


class CurriculumScheduler:
    """Manages curriculum learning schedule and sample ordering."""
    
    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.assessor = SampleDifficultyAssessor(config.complexity_metric)
        self.sample_difficulties = {}
        self.stage_boundaries = []
    
    def assess_dataset_difficulty(
        self, 
        volumes: List[np.ndarray], 
        labels: List[np.ndarray],
        sample_ids: List[str]
    ) -> Dict[str, float]:
        """Assess difficulty for entire dataset."""
        
        difficulties = {}
        
        for i, (volume, label, sample_id) in enumerate(zip(volumes, labels, sample_ids)):
            difficulty = self.assessor.assess_volume_difficulty(volume, label)
            difficulties[sample_id] = difficulty
            
            if i % 10 == 0:
                logger.info(f"Assessed {i+1}/{len(volumes)} samples")
        
        self.sample_difficulties = difficulties
        self._compute_stage_boundaries()
        
        return difficulties
    
    def _compute_stage_boundaries(self) -> None:
        """Compute difficulty boundaries for curriculum stages."""
        if not self.sample_difficulties:
            return
        
        difficulties = list(self.sample_difficulties.values())
        difficulties.sort()
        
        # Compute percentile boundaries for stages
        self.stage_boundaries = []
        for stage in range(self.config.num_stages):
            percentile = (stage + 1) / self.config.num_stages * 100
            boundary = np.percentile(difficulties, percentile)
            self.stage_boundaries.append(boundary)
        
        logger.info(f"Stage boundaries: {self.stage_boundaries}")
    
    def get_samples_for_stage(self, stage: int) -> List[str]:
        """Get sample IDs for a specific curriculum stage.
        
        Args:
            stage: Stage number (0-indexed)
            
        Returns:
            List of sample IDs for this stage
        """
        if stage >= self.config.num_stages:
            # Final stage includes all samples
            return list(self.sample_difficulties.keys())
        
        # Get difficulty range for this stage
        if stage == 0:
            min_difficulty = 0.0
        else:
            min_difficulty = self.stage_boundaries[stage - 1]
        
        max_difficulty = self.stage_boundaries[stage]
        
        # Add overlap with previous stage if enabled
        if self.config.enable_refinement and stage > 0:
            prev_boundary = self.stage_boundaries[stage - 1]
            overlap_range = (max_difficulty - prev_boundary) * self.config.refinement_overlap
            min_difficulty = max(0.0, prev_boundary - overlap_range)
        
        # Select samples in difficulty range
        selected_samples = []
        for sample_id, difficulty in self.sample_difficulties.items():
            if min_difficulty <= difficulty <= max_difficulty:
                selected_samples.append(sample_id)
        
        logger.info(f"Stage {stage}: {len(selected_samples)} samples (difficulty {min_difficulty:.3f}-{max_difficulty:.3f})")
        return selected_samples
    
    def get_curriculum_schedule(self, total_epochs: int) -> List[Dict[str, any]]:
        """Generate complete curriculum schedule.
        
        Args:
            total_epochs: Total number of training epochs
            
        Returns:
            List of stage configurations
        """
        if self.config.stage_durations is None:
            # Auto-compute stage durations
            base_duration = total_epochs // self.config.num_stages
            stage_durations = [base_duration] * self.config.num_stages
            # Add remaining epochs to final stage
            stage_durations[-1] += total_epochs % self.config.num_stages
        else:
            stage_durations = self.config.stage_durations
        
        schedule = []
        current_epoch = 0
        
        for stage in range(self.config.num_stages):
            stage_config = {
                "stage": stage,
                "start_epoch": current_epoch,
                "end_epoch": current_epoch + stage_durations[stage],
                "duration": stage_durations[stage],
                "samples": self.get_samples_for_stage(stage),
                "difficulty_range": (
                    0.0 if stage == 0 else self.stage_boundaries[stage - 1],
                    self.stage_boundaries[stage] if stage < len(self.stage_boundaries) else 1.0
                )
            }
            
            schedule.append(stage_config)
            current_epoch += stage_durations[stage]
        
        return schedule


class MultiStageRefinementPipeline:
    """Pipeline for multi-stage curriculum learning with refinement."""
    
    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.scheduler = CurriculumScheduler(config)
        self.stage_history = []
    
    def prepare_curriculum(
        self,
        volumes: List[np.ndarray],
        labels: List[np.ndarray], 
        sample_ids: List[str],
        total_epochs: int
    ) -> List[Dict[str, any]]:
        """Prepare complete curriculum learning schedule."""
        
        logger.info("🎓 Preparing curriculum learning schedule...")
        
        # Assess sample difficulties
        difficulties = self.scheduler.assess_dataset_difficulty(volumes, labels, sample_ids)
        
        # Generate schedule
        schedule = self.scheduler.get_curriculum_schedule(total_epochs)
        
        logger.info(f"📚 Curriculum prepared: {len(schedule)} stages")
        for stage_config in schedule:
            logger.info(f"  Stage {stage_config['stage']}: epochs {stage_config['start_epoch']}-{stage_config['end_epoch']}, {len(stage_config['samples'])} samples")
        
        return schedule
    
    def execute_stage(
        self,
        stage_config: Dict[str, any],
        model: torch.nn.Module,
        train_function: Callable,
        validation_function: Callable
    ) -> Dict[str, float]:
        """Execute a single curriculum stage.
        
        Args:
            stage_config: Stage configuration from curriculum schedule
            model: Model to train
            train_function: Training function (model, samples, epochs) -> metrics
            validation_function: Validation function (model) -> metrics
            
        Returns:
            Stage training metrics
        """
        stage_num = stage_config["stage"]
        logger.info(f"🎯 Executing curriculum stage {stage_num}")
        
        # Train on stage samples
        stage_samples = stage_config["samples"]
        stage_epochs = stage_config["duration"]
        
        train_metrics = train_function(model, stage_samples, stage_epochs)
        val_metrics = validation_function(model)
        
        # Record stage results
        stage_results = {
            "stage": stage_num,
            "samples_count": len(stage_samples),
            "epochs": stage_epochs,
            "difficulty_range": stage_config["difficulty_range"],
            "train_metrics": train_metrics,
            "val_metrics": val_metrics
        }
        
        self.stage_history.append(stage_results)
        
        logger.info(f"✅ Stage {stage_num} complete: val_dice={val_metrics.get('dice', 0.0):.3f}")
        return stage_results
    
    def save_curriculum_results(self, output_path: Path) -> None:
        """Save curriculum learning results."""
        results = {
            "config": {
                "complexity_metric": self.config.complexity_metric,
                "num_stages": self.config.num_stages,
                "enable_refinement": self.config.enable_refinement
            },
            "stage_history": self.stage_history,
            "sample_difficulties": self.scheduler.sample_difficulties
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Curriculum results saved: {output_path}")


def create_curriculum_demo() -> Dict[str, any]:
    """Create demonstration of curriculum learning framework."""
    
    print("🎓 Curriculum Learning Framework Demo")
    print("=" * 50)
    
    # Create synthetic dataset with varying difficulty
    num_samples = 20
    volumes = []
    labels = []
    sample_ids = []
    
    for i in range(num_samples):
        # Create volumes with increasing complexity
        complexity_factor = i / num_samples
        
        volume = np.random.rand(32, 32, 32).astype(np.float32)
        volume += complexity_factor * np.random.rand(32, 32, 32)  # Add noise
        
        # Create labels with varying boundary complexity
        labels_vol = np.zeros((32, 32, 32), dtype=np.int32)
        
        # Simple case: single region
        if complexity_factor < 0.3:
            labels_vol[10:22, 10:22, 10:22] = 1
        # Medium case: two regions
        elif complexity_factor < 0.7:
            labels_vol[8:16, 8:16, 8:16] = 1
            labels_vol[18:26, 18:26, 18:26] = 2
        # Hard case: multiple small regions
        else:
            labels_vol[6:12, 6:12, 6:12] = 1
            labels_vol[14:18, 14:18, 14:18] = 2
            labels_vol[20:24, 20:24, 20:24] = 3
            labels_vol[26:30, 26:30, 26:30] = 4
        
        volumes.append(volume)
        labels.append(labels_vol)
        sample_ids.append(f"sample_{i:03d}")
    
    # Create curriculum pipeline
    config = CurriculumConfig(
        complexity_metric="boundary_complexity",
        num_stages=3,
        enable_refinement=True
    )
    
    pipeline = MultiStageRefinementPipeline(config)
    
    # Prepare curriculum
    schedule = pipeline.prepare_curriculum(volumes, labels, sample_ids, total_epochs=30)
    
    print("\n📋 Curriculum Schedule:")
    for stage_config in schedule:
        print(f"  Stage {stage_config['stage']}: {stage_config['duration']} epochs, {len(stage_config['samples'])} samples")
        print(f"    Difficulty range: {stage_config['difficulty_range'][0]:.3f} - {stage_config['difficulty_range'][1]:.3f}")
    
    return {
        "schedule": schedule,
        "sample_difficulties": pipeline.scheduler.sample_difficulties,
        "config": config
    }


if __name__ == "__main__":
    # Run demonstration
    demo_results = create_curriculum_demo()
    
    # Save results
    output_dir = Path("/Users/camdouglas/quark/data/datasets/brainstem_segmentation/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "curriculum_learning_demo.json"
    with open(results_path, 'w') as f:
        json.dump(demo_results, f, indent=2, default=str)
    
    print(f"\n📄 Demo results saved: {results_path}")
    print("✅ Curriculum learning framework ready for training integration")
