#!/usr/bin/env python3
"""🧠 Working Memory Module
Handles short-term information storage, retrieval, and cognitive load management

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from collections import deque
import time

@dataclass
class MemoryItem:
    """Individual memory item in working memory"""
    content: Any
    priority: float
    created_at: float
    last_accessed: float
    access_count: int = 0
    decay_rate: float = 0.1

class WorkingMemory:
    """Working memory system for short-term information storage"""

    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.memory_slots: deque[MemoryItem] = deque(maxlen=capacity)
        self.cognitive_load = 0.0
        self.attention_focus = "general"

        # Neural representation
        self.memory_neurons = np.random.rand(capacity, 50)  # 50 features per slot

    def store(self, content: Any, priority: float = 0.5) -> bool:
        """Store new information in working memory"""
        if len(self.memory_slots) >= self.capacity:
            # Remove lowest priority item
            if not self._remove_lowest_priority():
                return False  # Couldn't make space

        item = MemoryItem(
            content=content,
            priority=priority,
            created_at=time.time(),
            last_accessed=time.time()
        )

        self.memory_slots.append(item)
        self._update_cognitive_load()
        return True

    def retrieve(self, query: str, search_key: str = None) -> Optional[MemoryItem]:
        """
        Retrieve memory item based on query.
        Can perform a simple string search or a more precise key-value search
        if the content is a dictionary.
        """
        best_match = None
        best_score = 0.0

        for item in self.memory_slots:
            match = False
            if search_key and isinstance(item.content, dict) and item.content.get('key') == query:
                # Precise key-based search
                match = True
            elif not search_key and query.lower() in str(item.content).lower():
                # Fallback to simple content string search
                match = True

            if match:
                score = item.priority * (1.0 / (time.time() - item.last_accessed + 1))
                if score > best_score:
                    best_score = score
                    best_match = item

        if best_match:
            best_match.last_accessed = time.time()
            best_match.access_count += 1
            self._update_cognitive_load()

        return best_match

    def _remove_lowest_priority(self) -> bool:
        """Remove lowest priority memory item"""
        if not self.memory_slots:
            return False

        # Find item with lowest priority and oldest access
        lowest_item = min(self.memory_slots,
                         key=lambda x: (x.priority, x.last_accessed))

        self.memory_slots.remove(lowest_item)
        return True

    def _update_cognitive_load(self):
        """Update cognitive load based on memory usage"""
        usage_ratio = len(self.memory_slots) / self.capacity
        complexity_factor = sum(len(str(item.content)) for item in self.memory_slots) / 1000

        self.cognitive_load = min(1.0, usage_ratio + complexity_factor * 0.3)

    def get_status(self) -> Dict[str, Any]:
        """Get working memory status"""
        return {
            "used_slots": len(self.memory_slots),
            "available_slots": self.capacity - len(self.memory_slots),
            "cognitive_load": self.cognitive_load,
            "attention_focus": self.attention_focus,
            "memory_contents": [str(item.content)[:50] + "..." for item in self.memory_slots]
        }

    def step(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Step working memory forward"""
        # Process new information
        if "new_info" in inputs:
            for info in inputs["new_info"]:
                priority = info.get("priority", 0.5)
                self.store(info["content"], priority)

        # Process retrieval requests
        if "retrieval_requests" in inputs:
            for request in inputs["retrieval_requests"]:
                result = self.retrieve(request["query"])
                if result:
                    request["result"] = result.content

        # Update neural representations
        self._update_neural_representations()

        return self.get_status()

    def _update_neural_representations(self):
        """Update neural representations of memory items"""
        for i, item in enumerate(self.memory_slots):
            if i < len(self.memory_neurons):
                # Update based on item properties
                priority_factor = item.priority
                recency_factor = 1.0 / (time.time() - item.last_accessed + 1)
                access_factor = min(1.0, item.access_count / 10)

                # Combine factors
                activation = (priority_factor + recency_factor + access_factor) / 3

                # Update neural representation
                self.memory_neurons[i] += np.random.normal(0, 0.01, 50) * activation
                self.memory_neurons[i] = np.clip(self.memory_neurons[i], 0.0, 1.0)

    def get_contents(self):
        """Returns the current contents of the working memory."""
        return self.memory_slots

    def reset(self):
        """Clears all items from the working memory."""
        self.memory_slots.clear()
        print("Working Memory has been reset.")
