#!/usr/bin/env python3
"""
Simple Cloud Integration for Conscious Agent
Purpose: Provide cloud offload functionality without complex integration
Inputs: Agent instance, task parameters
Outputs: Cloud processing results integrated into agent state
Seeds: Deterministic offload behavior
Deps: cloud_offload, time
"""

import time
import logging
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleCloudIntegration:
    """Simple cloud integration for conscious agents"""
    
    def __init__(self, agent_instance):
        self.agent = agent_instance
        self.cloud_offloader = None
        self.last_offload_time = 0
        self.offload_cooldown = 30  # seconds
        
        # Initialize cloud offloader
        self._init_cloud_offloader()
    
    def _init_cloud_offloader(self):
        """Initialize the cloud offloader"""
        try:
            from cloud_offload import SkyOffloader
            self.cloud_offloader = SkyOffloader()
            logger.info("✅ Cloud offloader initialized")
        except ImportError as e:
            logger.warning(f"⚠️ Could not import cloud offloader: {e}")
            self.cloud_offloader = None
        except Exception as e:
            logger.error(f"❌ Cloud offloader initialization failed: {e}")
            self.cloud_offloader = None
    
    def check_and_offload(self, consciousness_level: float, cognitive_load: float) -> bool:
        """
        Check if offload is needed and execute it
        
        Args:
            consciousness_level: Current consciousness level (0-1)
            cognitive_load: Current cognitive load (0-1)
            
        Returns:
            True if offload was executed, False otherwise
        """
        if not self.cloud_offloader:
            return False
        
        # Check if offload is needed
        if consciousness_level > 0.7 or cognitive_load > 0.8:
            current_time = time.time()
            
            # Check cooldown
            if current_time - self.last_offload_time < self.offload_cooldown:
                return False
            
            # Choose task type based on load
            if consciousness_level > 0.8:
                task_type = "neural_simulation"
                parameters = {
                    'duration': 5000,
                    'num_neurons': 200,
                    'scale': consciousness_level
                }
            elif cognitive_load > 0.8:
                task_type = "memory_consolidation"
                parameters = {
                    'duration': 3000,
                    'scale': cognitive_load
                }
            else:
                task_type = "attention_modeling"
                parameters = {
                    'duration': 2000,
                    'scale': 0.8
                }
            
            # Execute offload
            return self._execute_offload(task_type, parameters)
        
        return False
    
    def _execute_offload(self, task_type: str, parameters: Dict[str, Any]) -> bool:
        """Execute a cloud offload task"""
        try:
            logger.info(f"🚀 Offloading {task_type} to cloud...")
            
            job_id, result = self.cloud_offloader.submit(task_type, parameters)
            
            # Update agent state with results
            self._integrate_results(task_type, result)
            
            # Update timing
            self.last_offload_time = time.time()
            
            logger.info(f"✅ Cloud offload completed: {task_type}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Cloud offload failed: {e}")
            return False
    
    def _integrate_results(self, task_type: str, result: Dict[str, Any]):
        """Integrate cloud results into agent state"""
        if not hasattr(self.agent, 'unified_state'):
            return
        
        if task_type == "neural_simulation":
            self.agent.unified_state['neural_activity'] = result.get('activity_level', 0.0)
        elif task_type == "memory_consolidation":
            self.agent.unified_state['memory_consolidation'] = result.get('consolidation_level', 0.0)
        elif task_type == "attention_modeling":
            self.agent.unified_state['attention_focus'] = result.get('focus_level', 0.0)
        
        # Update cloud metrics if they exist
        if 'cloud_offload_metrics' in self.agent.unified_state:
            metrics = self.agent.unified_state['cloud_offload_metrics']
            metrics['tasks_offloaded'] = metrics.get('tasks_offloaded', 0) + 1
            metrics['last_offload_time'] = time.time()
    
    def manual_offload(self, task_type: str, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Manually trigger a cloud offload"""
        if not self.cloud_offloader:
            logger.warning("⚠️ Cloud offloader not available")
            return None
        
        try:
            job_id, result = self.cloud_offloader.submit(task_type, parameters)
            self._integrate_results(task_type, result)
            return result
        except Exception as e:
            logger.error(f"❌ Manual offload failed: {e}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get cloud integration status"""
        return {
            'cloud_offloader_available': self.cloud_offloader is not None,
            'last_offload_time': self.last_offload_time,
            'offload_cooldown': self.offload_cooldown,
            'time_since_last_offload': time.time() - self.last_offload_time
        }

def add_cloud_integration_to_agent(agent_instance) -> SimpleCloudIntegration:
    """Add cloud integration to an agent instance"""
    cloud_integration = SimpleCloudIntegration(agent_instance)
    
    # Add cloud integration methods to agent
    agent_instance.cloud_integration = cloud_integration
    agent_instance.check_cloud_offload = cloud_integration.check_and_offload
    agent_instance.manual_cloud_offload = cloud_integration.manual_offload
    agent_instance.get_cloud_status = cloud_integration.get_status
    
    logger.info("✅ Cloud integration added to agent")
    return cloud_integration
