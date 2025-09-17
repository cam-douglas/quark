

"""
Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
# brain_modules/thalamus/thalamic_relay.py

"""
Purpose: Implements a thalamic relay for context-sensitive information routing.
Inputs: Information streams, attentional signals
Outputs: Routed information
Dependencies: numpy
"""

from typing import Dict, Any

class ThalamicRelay:
    """A fully dynamic thalamic relay for attentional gating and information routing."""
    def __init__(self):
        self.attention_weights: Dict[str, float] = {}
        self.information_sources: Dict[str, Any] = {}

    def _update_attention(self, new_weights: Dict[str, float]):
        """Update attentional weights for different information sources."""
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            # Normalize weights to ensure they sum to 1
            self.attention_weights = {source: weight / total_weight for source, weight in new_weights.items()}
        else:
            self.attention_weights = new_weights

    def _add_information_source(self, source_id: str, data: Any):
        """Adds or updates a sensory information source."""
        self.information_sources[source_id] = {"data": data}

    def _route_information(self) -> Dict[str, Any]:
        """Route information based on attentional weights."""
        routed_info = {}
        for source_id, source_data in self.information_sources.items():
            weight = self.attention_weights.get(source_id, 0)
            if weight > 0:
                routed_info[source_id] = {
                    'data': source_data['data'],
                    'attention_weight': weight
                }
        return routed_info

    def step(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """The main step function, called by the BrainSimulator."""
        if "attention_weights" in inputs:
            self._update_attention(inputs["attention_weights"])

        if "sensory_data" in inputs:
            for source_id, data in inputs["sensory_data"].items():
                self._add_information_source(source_id, data)

        routed_info = self._route_information()
        return {"routed_information": routed_info}
