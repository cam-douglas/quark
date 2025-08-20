"""
Simple Continuous Training for SmallMind

Runs training in an infinite loop until manually stopped.
"""

import time
import signal
import logging
from pathlib import Path
from ...........................................................training_pipeline import get_trainer, TrainingConfig

logger = logging.getLogger(__name__)

class ContinuousTrainer:
    """Simple continuous training that runs forever until stopped."""
    
    def __init__(self, model_path: str, output_dir: str = "./continuous_training"):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.trainer = get_trainer()
        self.should_stop = False
        self.epoch = 0
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Fix tokenizer issues if needed
        self._fix_tokenizer_issues()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Received stop signal, finishing current epoch...")
        self.should_stop = True
    
    def _fix_tokenizer_issues(self):
        """Fix common tokenizer compatibility issues"""
        try:
            # Check if tokenizer files exist
            tokenizer_path = Path(self.model_path)
            tokenizer_files = list(tokenizer_path.glob("*.json")) + list(tokenizer_path.glob("*.model"))
            
            if not tokenizer_files:
                logger.warning(f"No tokenizer files found in {self.model_path}")
                return False
            
            # Try to load tokenizer with different approaches
            from transformers import AutoTokenizer
            
            try:
                # First try: standard loading
                tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                logger.info(f"✅ Tokenizer loaded successfully: {type(tokenizer).__name__}")
                return True
            except Exception as e1:
                logger.warning(f"Standard tokenizer loading failed: {e1}")
                
                try:
                    # Second try: force legacy mode for LLaMA models
                    if "llama" in self.model_path.lower() or "deepseek" in self.model_path.lower():
                        tokenizer = AutoTokenizer.from_pretrained(
                            self.model_path, 
                            trust_remote_code=True,
                            use_fast=False  # Force slow tokenizer
                        )
                        logger.info(f"✅ Tokenizer loaded with legacy mode: {type(tokenizer).__name__}")
                        return True
                except Exception as e2:
                    logger.warning(f"Legacy tokenizer loading failed: {e2}")
            
            return False
            
        except Exception as e:
            logger.error(f"Error fixing tokenizer issues: {e}")
            return False
    
    def train_forever(self):
        """Train forever until manually stopped"""
        logger.info("🚀 Starting infinite training loop...")
        logger.info("Press Ctrl+C to stop")
        
        mixtures = ["balanced", "code_focused", "reasoning_focused"]
        
        while not self.should_stop:
            # Rotate through mixtures
            mixture = mixtures[self.epoch % len(mixtures)]
            
            logger.info(f"\n=== EPOCH {self.epoch + 1} with {mixture} ===")
            
            try:
                # Create config for this epoch
                config = TrainingConfig(
                    model_name_or_path=self.model_path,
                    output_dir=str(self.output_dir / f"epoch_{self.epoch + 1:04d}"),
                    mixture_name=mixture,
                    max_steps=1000,  # Adjust as needed
                    per_device_train_batch_size=2,
                    gradient_accumulation_steps=4,
                    learning_rate=5e-5
                )
                
                # Train this epoch
                results = self.trainer.train(config)
                
                logger.info(f"✅ Epoch {self.epoch + 1} completed!")
                logger.info(f"   Loss: {results.get('final_loss', 'unknown')}")
                logger.info(f"   Time: {results.get('training_time_hours', 0):.2f} hours")
                
                self.epoch += 1
                
                # Small delay between epochs
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"❌ Epoch {self.epoch + 1} failed: {e}")
                logger.info("Waiting 30 seconds before retrying...")
                time.sleep(30)
        
        logger.info("🎉 Training stopped by user")
        logger.info(f"Completed {self.epoch} epochs")

def train_forever(model_path: str, output_dir: str = "./continuous_training"):
    """Simple function to train forever"""
    trainer = ContinuousTrainer(model_path, output_dir)
    trainer.train_forever()
    return trainer
