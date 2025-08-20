"""
Continuous Training System for SmallMind

Runs training in an infinite loop, continuously improving your model
with high-quality open datasets until manually stopped.
"""

import os, sys
import time
import signal
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ................................................dataset_integration import get_dataset_integrator
from ................................................training_pipeline import get_trainer

logger = logging.getLogger(__name__)

@dataclass
class ContinuousTrainingConfig:
    """Configuration for continuous training"""
    model_name_or_path: str
    output_base_dir: str
    checkpoint_dir: str
    
    # Training parameters
    steps_per_epoch: int = 1000
    learning_rate: float = 5e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    
    # Dataset rotation
    mixture_rotation: List[str] = None
    rotation_interval: int = 3
    
    # Checkpointing
    save_every_epoch: bool = True
    max_checkpoints: int = 5
    
    # Recovery
    resume_from_checkpoint: bool = True
    auto_restart: bool = True
    max_failures: int = 3

class ContinuousTrainer:
    """Continuous training system that runs indefinitely until manually stopped."""
    
    def __init__(self, config: ContinuousTrainingConfig):
        self.config = config
        self.should_stop = False
        self.current_epoch = 0
        self.total_steps = 0
        self.best_loss = float('inf')
        self.failure_count = 0
        
        # Setup directories
        self.output_base_dir = Path(config.output_base_dir)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.dataset_integrator = get_dataset_integrator()
        self.trainer = get_trainer()
        
        # Training state
        self.training_history = []
        self.current_mixture = None
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        logger.info(f"ContinuousTrainer initialized with output_dir: {self.output_base_dir}")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.should_stop = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _get_available_mixtures(self) -> List[str]:
        """Get list of available training mixtures"""
        if self.config.mixture_rotation:
            return self.config.mixture_rotation
        else:
            return self.trainer.list_available_mixtures()
    
    def _rotate_mixture(self) -> str:
        """Rotate to the next training mixture"""
        available_mixtures = self._get_available_mixtures()
        if not available_mixtures:
            raise ValueError("No training mixtures available")
        
        mixture_index = (self.current_epoch // self.config.rotation_interval) % len(available_mixtures)
        mixture_name = available_mixtures[mixture_index]
        
        logger.info(f"Rotating to mixture: {mixture_name} (epoch {self.current_epoch})")
        return mixture_name
    
    def _create_epoch_config(self, mixture_name: str, epoch: int):
        """Create training configuration for a specific epoch"""
        from ................................................training_pipeline import TrainingConfig
        
        epoch_output_dir = self.output_base_dir / f"epoch_{epoch:04d}_{mixture_name}"
        
        return TrainingConfig(
            model_name_or_path=self.config.model_name_or_path,
            output_dir=str(epoch_output_dir),
            mixture_name=mixture_name,
            max_steps=self.config.steps_per_epoch,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=min(100, self.config.steps_per_epoch // 10),
            weight_decay=0.01,
            logging_steps=100,
            save_steps=500,
            eval_steps=500,
            save_total_limit=2,
            dataloader_num_workers=4,
            remove_unused_columns=False
        )
    
    def _save_training_state(self, epoch: int, mixture_name: str, loss: float):
        """Save current training state"""
        state = {
            "epoch": epoch,
            "total_steps": self.total_steps,
            "mixture_name": mixture_name,
            "loss": loss,
            "best_loss": self.best_loss,
            "timestamp": datetime.now().isoformat(),
            "config": self.config.__dict__
        }
        
        state_file = self.checkpoint_dir / "training_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Save training history
        self.training_history.append(state)
        history_file = self.checkpoint_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def _load_training_state(self) -> Optional[Dict[str, Any]]:
        """Load previous training state if available"""
        state_file = self.checkpoint_dir / "training_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                logger.info(f"Resumed from epoch {state['epoch']}, total steps: {state['total_steps']}")
                return state
            except Exception as e:
                logger.warning(f"Failed to load training state: {e}")
        return None
    
    def _run_epoch(self, epoch: int) -> Dict[str, Any]:
        """Run a single training epoch"""
        start_time = time.time()
        
        # Rotate to next mixture
        mixture_name = self._rotate_mixture()
        self.current_mixture = mixture_name
        
        # Create epoch configuration
        epoch_config = self._create_epoch_config(mixture_name, epoch)
        
        logger.info(f"Starting epoch {epoch} with mixture: {mixture_name}")
        logger.info(f"Output directory: {epoch_config.output_dir}")
        
        try:
            # Run training
            results = self.trainer.train(epoch_config)
            
            # Update training state
            epoch_loss = results.get("final_loss", float('inf'))
            if isinstance(epoch_loss, str):
                epoch_loss = float('inf')
            
            self.total_steps += self.config.steps_per_epoch
            
            # Update best loss
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                logger.info(f"New best loss: {self.best_loss:.6f}")
            
            # Save training state
            self._save_training_state(epoch, mixture_name, epoch_loss)
            
            # Calculate epoch statistics
            epoch_time = time.time() - start_time
            steps_per_second = self.config.steps_per_epoch / epoch_time
            
            epoch_results = {
                "epoch": epoch,
                "mixture_name": mixture_name,
                "loss": epoch_loss,
                "best_loss": self.best_loss,
                "epoch_time_seconds": epoch_time,
                "steps_per_second": steps_per_second,
                "total_steps": self.total_steps,
                "success": True
            }
            
            logger.info(f"Epoch {epoch} completed successfully")
            logger.info(f"  Loss: {epoch_loss:.6f}")
            logger.info(f"  Best Loss: {self.best_loss:.6f}")
            logger.info(f"  Time: {epoch_time/60:.2f} minutes")
            logger.info(f"  Speed: {steps_per_second:.2f} steps/sec")
            
            return epoch_results
            
        except Exception as e:
            logger.error(f"Epoch {epoch} failed: {e}")
            self.failure_count += 1
            
            # Save error information
            error_results = {
                "epoch": epoch,
                "mixture_name": mixture_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
            
            error_file = self.checkpoint_dir / f"epoch_{epoch:04d}_error.json"
            with open(error_file, 'w') as f:
                json.dump(error_results, f, indent=2)
            
            return error_results
    
    def _should_continue_training(self) -> bool:
        """Determine if training should continue"""
        if self.should_stop:
            logger.info("Training stopped by user")
            return False
        
        if self.failure_count >= self.config.max_failures:
            logger.error(f"Too many failures ({self.failure_count}), stopping training")
            return False
        
        return True
    
    def train_continuously(self):
        """Main continuous training loop"""
        logger.info("🚀 Starting continuous training loop...")
        logger.info("Press Ctrl+C to stop training gracefully")
        
        # Load previous state if resuming
        if self.config.resume_from_checkpoint:
            previous_state = self._load_training_state()
            if previous_state:
                self.current_epoch = previous_state["epoch"] + 1
                self.total_steps = previous_state["total_steps"]
                self.best_loss = previous_state.get("best_loss", float('inf'))
                logger.info(f"Resuming from epoch {self.current_epoch}")
        
        try:
            while self._should_continue_training():
                # Run epoch
                epoch_results = self._run_epoch(self.current_epoch)
                
                if epoch_results.get("success", False):
                    # Successful epoch
                    self.failure_count = 0  # Reset failure count
                    self.current_epoch += 1
                    time.sleep(1)  # Small delay between epochs
                else:
                    # Failed epoch
                    logger.error(f"Epoch {self.current_epoch} failed, failure count: {self.failure_count}")
                    
                    if self.config.auto_restart:
                        logger.info("Waiting 30 seconds before retrying...")
                        time.sleep(30)
                    else:
                        logger.error("Auto-restart disabled, stopping training")
                        break
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, stopping training...")
        except Exception as e:
            logger.error(f"Unexpected error in training loop: {e}")
        finally:
            self._cleanup_and_save()
    
    def _cleanup_and_save(self):
        """Cleanup and save final state"""
        logger.info("Cleaning up and saving final state...")
        
        # Save final training state
        if hasattr(self, 'current_epoch') and hasattr(self, 'current_mixture'):
            self._save_training_state(self.current_epoch, self.current_mixture, self.best_loss)
        
        # Save training summary
        summary = {
            "total_epochs": self.current_epoch,
            "total_steps": self.total_steps,
            "best_loss": self.best_loss,
            "final_timestamp": datetime.now().isoformat(),
            "training_history": self.training_history
        }
        
        summary_file = self.output_base_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("🎉 Training completed!")
        logger.info(f"Total epochs: {self.current_epoch}")
        logger.info(f"Total steps: {self.total_steps}")
        logger.info(f"Best loss: {self.best_loss:.6f}")
        logger.info(f"Results saved to: {self.output_base_dir}")
    
    def stop_training(self):
        """Manually stop training"""
        logger.info("Manually stopping training...")
        self.should_stop = True

# Convenience functions
def start_continuous_training(
    model_path: str,
    output_dir: str = "./continuous_training",
    checkpoint_dir: str = "./continuous_checkpoints",
    **kwargs
) -> ContinuousTrainer:
    """Start continuous training with sensible defaults"""
    
    config = ContinuousTrainingConfig(
        model_name_or_path=model_path,
        output_base_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        **kwargs
    )
    
    trainer = ContinuousTrainer(config)
    
    # Start training in a separate thread so you can control it
    training_thread = threading.Thread(target=trainer.train_continuously)
    training_thread.daemon = True
    training_thread.start()
    
    return trainer

def run_continuous_training(
    model_path: str,
    output_dir: str = "./continuous_training",
    checkpoint_dir: str = "./continuous_checkpoints",
    **kwargs
):
    """Run continuous training in the current thread (blocking)"""
    
    config = ContinuousTrainingConfig(
        model_name_or_path=model_path,
        output_base_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        **kwargs
    )
    
    trainer = ContinuousTrainer(config)
    trainer.train_continuously()
    
    return trainer
