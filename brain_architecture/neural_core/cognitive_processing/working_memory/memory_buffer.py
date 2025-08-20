# brain_modules/working_memory/memory_buffer.py

"""
Purpose: Implements a short-term memory buffer for working memory.
Inputs: Information to be stored
Outputs: Stored information with temporal dynamics
Dependencies: numpy
"""

import numpy as np
from typing import List, Any, Dict, Optional
import time

class MemoryItem:
    """Represents an item in the working memory buffer"""
    def __init__(self, content: Any, decay_rate: float = 0.98, timestamp: Optional[float] = None):
        self.content = content
        self.decay_rate = decay_rate
        self.activation = 1.0  # Initial activation
        self.timestamp = timestamp if timestamp is not None else time.time()
    
    def update_activation(self):
        """Update activation level based on decay"""
        self.activation *= self.decay_rate
    
    def refresh(self, reactivation_strength: float = 0.5):
        """Refresh activation of the memory item"""
        self.activation = min(1.0, self.activation + reactivation_strength)
    
    def is_active(self, threshold: float = 0.1) -> bool:
        """Check if memory item is still active"""
        return self.activation > threshold
    
    def __repr__(self) -> str:
        return f"MemoryItem(content={self.content}, activation={self.activation:.3f})"

class MemoryBuffer:
    """A short-term memory buffer with capacity and decay"""
    def __init__(self, capacity: int = 7, decay_rate: float = 0.98):
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.buffer: List[MemoryItem] = []
    
    def add_item(self, content: Any):
        """Add a new item to the buffer, removing the oldest if at capacity"""
        if len(self.buffer) >= self.capacity:
            # Remove the least active item
            least_active_idx = np.argmin([item.activation for item in self.buffer])
            self.buffer.pop(least_active_idx)
        
        item = MemoryItem(content, self.decay_rate)
        self.buffer.append(item)
    
    def retrieve_items(self, threshold: float = 0.1) -> List[Any]:
        """Retrieve all active items from the buffer"""
        return [item.content for item in self.buffer if item.is_active(threshold)]
    
    def update_buffer(self):
        """Update activations of all items in the buffer"""
        for item in self.buffer:
            item.update_activation()
        
        # Remove inactive items
        self.buffer = [item for item in self.buffer if item.is_active()]
    
    def refresh_item(self, content: Any, reactivation_strength: float = 0.5):
        """Refresh a specific item in the buffer"""
        for item in self.buffer:
            if item.content == content:
                item.refresh(reactivation_strength)
                break
    
    def get_buffer_state(self) -> List[Dict]:
        """Get current state of the buffer"""
        return [{'content': item.content, 'activation': item.activation} for item in self.buffer]
    
    def get_size(self) -> int:
        """Get current number of items in the buffer"""
        return len(self.buffer)

if __name__ == '__main__':
    # Test the memory buffer
    print("Testing Working Memory Buffer")
    
    # Create buffer
    memory = MemoryBuffer(capacity=5)
    
    # Add items
    for i in range(7):
        memory.add_item(f"item_{i}")
        print(f"Added item_{i}, Buffer size: {memory.get_size()}")
    
    # Update buffer and show decay
    for _ in range(5):
        memory.update_buffer()
        print(f"Buffer state after update: {memory.get_buffer_state()}")
    
    # Refresh an item
    memory.refresh_item("item_5")
    print(f"Buffer state after refreshing item_5: {memory.get_buffer_state()}")
    
    # Retrieve active items
    active_items = memory.retrieve_items()
    print(f"Active items: {active_items}")
    
    print("Working Memory Buffer test completed.")
