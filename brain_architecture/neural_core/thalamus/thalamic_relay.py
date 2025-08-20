# brain_modules/thalamus/thalamic_relay.py

"""
Purpose: Implements a thalamic relay for context-sensitive information routing.
Inputs: Information streams, attentional signals
Outputs: Routed information
Dependencies: numpy
"""

import numpy as np
from typing import Dict, Any, List

class ThalamicRelay:
    """A thalamic relay for attentional gating and information routing"""
    def __init__(self, num_sources: int):
        self.num_sources = num_sources
        self.attention_weights = np.ones(num_sources) / num_sources  # Equal attention initially
        self.information_sources: Dict[int, Any] = {}
    
    def update_attention(self, new_weights: np.ndarray):
        """Update attentional weights for different information sources"""
        if len(new_weights) != self.num_sources:
            raise ValueError(f"Expected {self.num_sources} weights, but got {len(new_weights)}")
        
        # Normalize weights to ensure they sum to 1
        self.attention_weights = new_weights / np.sum(new_weights)
    
    def add_information_source(self, source_id: int, data: Any):
        """Add or update an information source"""
        if source_id >= self.num_sources:
            raise ValueError(f"Source ID {source_id} is out of bounds")
        
        self.information_sources[source_id] = data
    
    def route_information(self) -> Dict[str, Any]:
        """Route information based on attentional weights"""
        routed_info = {}
        
        for source_id, weight in enumerate(self.attention_weights):
            if source_id in self.information_sources:
                # The amount of information routed is proportional to the attention weight
                # (Simplified - in practice this would be more complex)
                routed_info[f"source_{source_id}"] = {
                    'data': self.information_sources[source_id],
                    'attention_weight': weight
                }
        
        return routed_info
    
    def get_attention_state(self) -> np.ndarray:
        """Get current attentional weights"""
        return self.attention_weights

if __name__ == '__main__':
    # Test the thalamic relay
    print("Testing Thalamic Relay")
    
    # Create relay
    relay = ThalamicRelay(num_sources=4)
    
    # Add information sources
    relay.add_information_source(0, "visual_data")
    relay.add_information_source(1, "auditory_data")
    relay.add_information_source(2, "somatosensory_data")
    relay.add_information_source(3, "proprioceptive_data")
    
    # Route with initial attention
    routed_info = relay.route_information()
    print(f"Routed info with initial attention: {routed_info}")
    
    # Update attention
    new_weights = np.array([0.7, 0.1, 0.1, 0.1])  # Focus on visual data
    relay.update_attention(new_weights)
    
    # Route with new attention
    routed_info = relay.route_information()
    print(f"Routed info with updated attention: {routed_info}")
    
    print("Thalamic Relay test completed.")
