#!/usr/bin/env python3
"""
Small-Mind Simulation Launcher

Main script that runs continuous optimization cycles for the small-mind brain development simulation.
Includes safety checks and configurable cycle intervals.
"""

import time
import logging
import sys
from pathlib import Path

# Add the smallmind package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_optimization.optuna_interface import OptunaOptimizer
from scripts.safety_check import safety_passed
from scripts.smallmind_optimizer import optimize_smallmind_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('smallmind_sim.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main_loop():
    """Main simulation loop with continuous optimization cycles"""
    logger.info("🚀 Starting Small-Mind Simulation System")
    
    cycle_count = 0
    while True:
        cycle_count += 1
        logger.info(f"🔁 Starting small-mind simulation cycle #{cycle_count}")
        
        try:
            if safety_passed():
                logger.info("✅ Safety check passed. Proceeding with optimization...")
                optimize_smallmind_model()
                logger.info(f"✅ Cycle #{cycle_count} completed successfully")
            else:
                logger.warning("⚠️ Safety check failed. Skipping optimization cycle.")
                
        except Exception as e:
            logger.error(f"❌ Error in cycle #{cycle_count}: {str(e)}")
            logger.exception("Full traceback:")
        
        # Wait before next cycle (5 minutes = 300 seconds)
        logger.info("⏳ Waiting 5 minutes before next cycle...")
        time.sleep(300)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        logger.info("🛑 Small-mind simulation stopped by user")
    except Exception as e:
        logger.error(f"💥 Fatal error in main loop: {str(e)}")
        logger.exception("Fatal error traceback:")
        sys.exit(1)
