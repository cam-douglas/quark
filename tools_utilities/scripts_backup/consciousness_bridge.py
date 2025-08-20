"""
Consciousness Bridge Utility
===========================

This module provides utilities to connect Jupyter notebooks to the main consciousness agent
as a tool for interactive training, simulation, and analysis.

Author: Quark Brain Simulation Framework
Purpose: Enable notebook-based interaction with consciousness components
Dependencies: Core brain modules, consciousness agent
"""

import sys
import os
import json
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

try:
    from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsbase.unified_consciousness_agent import UnifiedConsciousnessAgent
    from development.src.core.brain_launcher_v3 import BrainLauncherV3
    from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsbase.agent_connector import AgentConnector
except ImportError as e:
    print(f"Warning: Could not import consciousness components: {e}")
    print("Make sure you're running from the project root with venv activated")

class ConsciousnessBridge:
    """
    Bridge class to connect notebooks to the main consciousness agent
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """Initialize the consciousness bridge"""
        self.session_id = session_id or f"notebook_session_{int(datetime.now().timestamp())}"
        self.consciousness_agent = None
        self.brain_launcher = None
        self.agent_connector = None
        self.is_connected = False
        
    def connect(self, config_path: Optional[str] = None) -> bool:
        """
        Connect to the main consciousness agent
        
        Args:
            config_path: Optional path to brain configuration
            
        Returns:
            bool: True if connection successful
        """
        try:
            print(f"🧠 Connecting to consciousness agent (Session: {self.session_id})...")
            
            # Initialize agent connector
            self.agent_connector = AgentConnector()
            
            # Initialize consciousness agent
            self.consciousness_agent = UnifiedConsciousnessAgent(
                session_id=self.session_id,
                enable_speech=False,  # Disable speech in notebooks
                enable_learning=True
            )
            
            # Initialize brain launcher if config provided
            if config_path:
                self.brain_launcher = BrainLauncherV3()
                print(f"📡 Brain launcher initialized with config: {config_path}")
            
            self.is_connected = True
            print("✅ Successfully connected to consciousness agent!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to consciousness agent: {e}")
            return False
    
    def get_brain_state(self) -> Optional[Dict[str, Any]]:
        """Get current brain state from consciousness agent"""
        if not self.is_connected:
            print("❌ Not connected to consciousness agent")
            return None
            
        try:
            state = self.consciousness_agent.get_brain_state()
            return state
        except Exception as e:
            print(f"❌ Error getting brain state: {e}")
            return None
    
    def run_simulation_step(self, input_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Run a single simulation step
        
        Args:
            input_data: Optional input data for the simulation
            
        Returns:
            Dict: Simulation results
        """
        if not self.is_connected:
            print("❌ Not connected to consciousness agent")
            return None
            
        try:
            if input_data:
                result = self.consciousness_agent.process_input(input_data)
            else:
                result = self.consciousness_agent.think()
            
            return result
        except Exception as e:
            print(f"❌ Error running simulation step: {e}")
            return None
    
    def train_component(self, component_name: str, training_data: List[Dict]) -> bool:
        """
        Train a specific brain component
        
        Args:
            component_name: Name of the component to train
            training_data: List of training examples
            
        Returns:
            bool: True if training successful
        """
        if not self.is_connected:
            print("❌ Not connected to consciousness agent")
            return False
            
        try:
            print(f"🎯 Training component: {component_name}")
            success = self.consciousness_agent.train_component(component_name, training_data)
            if success:
                print(f"✅ Training completed for {component_name}")
            else:
                print(f"❌ Training failed for {component_name}")
            return success
        except Exception as e:
            print(f"❌ Error training component {component_name}: {e}")
            return False
    
    def get_metrics(self) -> Optional[Dict]:
        """Get current performance metrics"""
        if not self.is_connected:
            print("❌ Not connected to consciousness agent")
            return None
            
        try:
            metrics = self.consciousness_agent.get_performance_metrics()
            return metrics
        except Exception as e:
            print(f"❌ Error getting metrics: {e}")
            return None
    
    def visualize_brain_activity(self):
        """Create visualization of current brain activity"""
        if not self.is_connected:
            print("❌ Not connected to consciousness agent")
            return
            
        try:
            import matplotlib.pyplot as plt
            
            state = self.get_brain_state()
            if not state:
                print("❌ Could not get brain state for visualization")
                return
                
            # Create a simple activity visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f'Brain Activity - Session: {self.session_id}', fontsize=16)
            
            # Plot 1: Overall activity levels
            if 'activity_levels' in state:
                activities = state['activity_levels']
                axes[0, 0].bar(range(len(activities)), activities)
                axes[0, 0].set_title('Component Activity Levels')
                axes[0, 0].set_xlabel('Component')
                axes[0, 0].set_ylabel('Activity')
            
            # Plot 2: Memory usage
            if 'memory_usage' in state:
                memory = state['memory_usage']
                axes[0, 1].pie(memory.values(), labels=memory.keys(), autopct='%1.1f%%')
                axes[0, 1].set_title('Memory Usage Distribution')
            
            # Plot 3: Learning progress
            if 'learning_metrics' in state:
                metrics = state['learning_metrics']
                axes[1, 0].plot(metrics.get('timestamps', []), metrics.get('accuracy', []))
                axes[1, 0].set_title('Learning Progress')
                axes[1, 0].set_xlabel('Time')
                axes[1, 0].set_ylabel('Accuracy')
            
            # Plot 4: Network connectivity
            if 'connectivity_matrix' in state:
                matrix = np.array(state['connectivity_matrix'])
                im = axes[1, 1].imshow(matrix, cmap='viridis')
                axes[1, 1].set_title('Network Connectivity')
                plt.colorbar(im, ax=axes[1, 1])
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("❌ matplotlib not available for visualization")
        except Exception as e:
            print(f"❌ Error creating visualization: {e}")
    
    def save_session(self, filepath: str):
        """Save current session data"""
        if not self.is_connected:
            print("❌ Not connected to consciousness agent")
            return
            
        try:
            session_data = {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'brain_state': self.get_brain_state(),
                'metrics': self.get_metrics()
            }
            
            with open(filepath, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            print(f"💾 Session saved to: {filepath}")
            
        except Exception as e:
            print(f"❌ Error saving session: {e}")
    
    def disconnect(self):
        """Disconnect from consciousness agent"""
        if self.is_connected:
            try:
                if self.consciousness_agent:
                    self.consciousness_agent.shutdown()
                print("🔌 Disconnected from consciousness agent")
                self.is_connected = False
            except Exception as e:
                print(f"❌ Error during disconnect: {e}")

# Convenience function for quick setup
def quick_connect(session_id: Optional[str] = None) -> ConsciousnessBridge:
    """
    Quick setup function to connect to consciousness agent
    
    Args:
        session_id: Optional session identifier
        
    Returns:
        ConsciousnessBridge: Connected bridge instance
    """
    bridge = ConsciousnessBridge(session_id)
    bridge.connect()
    return bridge

# Example usage for notebooks
if __name__ == "__main__":
    print("🧠 Consciousness Bridge Utility")
    print("Usage:")
    print("  from consciousness_bridge import quick_connect")
    print("  bridge = quick_connect('my_session')")
    print("  state = bridge.get_brain_state()")
    print("  bridge.visualize_brain_activity()")
