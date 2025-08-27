import random
from typing import Dict, Any, List

# Integration: Import the actual WorkingMemory module from Quark's architecture
from brain_architecture.neural_core.working_memory.working_memory import WorkingMemory
from testing.cognitive_benchmarks.base_benchmark import BaseBenchmark

class WorkingMemoryBenchmark(BaseBenchmark):
    """
    Benchmark for evaluating working memory using a simplified n-back task.

    In the n-back task, a sequence of stimuli is presented, and the agent must
    indicate whether the current stimulus matches the one from 'n' steps earlier.
    """
    def __init__(self, n_back: int = 2, sequence_length: int = 20):
        super().__init__(
            name=f"Working Memory (n-back, n={n_back})",
            description=f"Tests the ability to monitor and update working memory content over a sequence of {sequence_length} items."
        )
        self.n_back = n_back
        self.sequence_length = sequence_length
        self.stimuli_sequence: List[int] = []
        self.correct_responses: List[bool] = []
        
        # Integration: Add a placeholder for the working memory instance
        self.memory_module: WorkingMemory

    def setup(self):
        """
        Generate the stimulus sequence, correct responses, and initialize the memory module.
        """
        self.stimuli_sequence = [random.randint(0, 9) for _ in range(self.sequence_length)]
        self.correct_responses = []
        for i in range(self.sequence_length):
            if i < self.n_back:
                self.correct_responses.append(False)  # No target possible yet
            else:
                is_target = (self.stimuli_sequence[i] == self.stimuli_sequence[i - self.n_back])
                self.correct_responses.append(is_target)

        # Integration: Initialize the working memory module for the test
        # We give it a capacity large enough to handle the n-back requirement
        self.memory_module = WorkingMemory(capacity=self.n_back + 5)

    def run(self) -> List[bool]:
        """
        Present the sequence to the actual WorkingMemory module and get its responses.
        This now uses the real cognitive module instead of a simulation.
        """
        agent_responses = []
        for i, stimulus in enumerate(self.stimuli_sequence):
            # The agent's task is to determine if the current stimulus
            # matches the one from n steps ago.
            
            is_match_response = False
            if i >= self.n_back:
                # We need a way to query the memory for the item n-steps back.
                # The current `retrieve` method is string-based. We'll adapt by
                # storing items with a unique identifier we can query.
                
                # Query for the stimulus that occurred n-steps ago.
                # We'll use the new precise key-based search.
                query_key = f"stimulus_at_step_{i - self.n_back}"
                retrieved_item = self.memory_module.retrieve(query_key, search_key='key')
                
                if retrieved_item and retrieved_item.content.get('value') == stimulus:
                    is_match_response = True
            
            agent_responses.append(is_match_response)
            
            # Store the current stimulus in the working memory module.
            # We give it a unique key and the value.
            current_key = f"stimulus_at_step_{i}"
            self.memory_module.store(
                content={'key': current_key, 'value': stimulus},
                priority=1.0 # High priority for task-relevant items
            )
        
        return agent_responses

    def evaluate(self, agent_responses: List[bool]) -> Dict[str, Any]:
        """
        Compare the agent's responses to the correct responses and calculate metrics.
        """
        if len(agent_responses) != len(self.correct_responses):
            raise ValueError("Agent responses and correct responses have different lengths.")

        hits = 0
        misses = 0
        false_alarms = 0
        correct_rejections = 0

        for i in range(self.sequence_length):
            agent_response = agent_responses[i]
            correct_response = self.correct_responses[i]

            if agent_response and correct_response:
                hits += 1
            elif not agent_response and correct_response:
                misses += 1
            elif agent_response and not correct_response:
                false_alarms += 1
            elif not agent_response and not correct_response:
                correct_rejections += 1

        total_targets = hits + misses
        total_non_targets = false_alarms + correct_rejections
        
        accuracy = (hits + correct_rejections) / self.sequence_length if self.sequence_length > 0 else 0
        hit_rate = hits / total_targets if total_targets > 0 else 0
        false_alarm_rate = false_alarms / total_non_targets if total_non_targets > 0 else 0

        return {
            "accuracy": round(accuracy, 4),
            "hit_rate": round(hit_rate, 4),
            "false_alarm_rate": round(false_alarm_rate, 4),
            "hits": hits,
            "misses": misses,
            "false_alarms": false_alarms,
            "correct_rejections": correct_rejections,
            "sequence_length": self.sequence_length,
            "n_back": self.n_back
        }
