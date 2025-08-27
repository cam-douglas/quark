import random
from typing import Dict, Any, List
from brain_architecture.neural_core.motor_control.basal_ganglia.rl_agent import QLearningAgent
from testing.cognitive_benchmarks.base_benchmark import BaseBenchmark

# Integration: Import the new executive control function
from brain_architecture.neural_core.prefrontal_cortex.executive_control import ExecutiveControl

class ProbabilisticReversalBenchmark(BaseBenchmark):
    """
    Benchmark for adaptive learning, based on a probabilistic reversal learning task.

    The agent must learn which of two choices has a higher probability of being
    rewarded. After a certain number of trials, these probabilities are secretly
    reversed, and the agent must adapt its strategy to continue maximizing rewards.
    """
    def __init__(self, num_trials: int = 200, reversal_point: int = 100, high_prob: float = 0.8, low_prob: float = 0.2):
        super().__init__(
            name="Probabilistic Reversal Learning",
            description=f"Tests adaptability by reversing reward probabilities after {reversal_point} trials."
        )
        self.num_trials = num_trials
        self.reversal_point = reversal_point
        self.probabilities = [high_prob, low_prob]

        # Integration: A single executive control instance will manage the agent
        self.executive = ExecutiveControl()
        self.rl_agent = None

    def setup(self):
        """Initialize the reinforcement learning agent for the task."""
        # Integration: Use the executive function to select the agent
        task_properties = {
            'name': 'Probabilistic Reversal Learning',
            'dynamic': True
        }
        self.rl_agent = self.executive.select_strategy(
            task_properties,
            num_states=1,
            num_actions=2
        )

    def _get_reward(self, choice: int) -> int:
        """Get a reward based on the current probability structure."""
        if random.random() < self.probabilities[choice]:
            return 1
        return 0

    def run(self) -> List[int]:
        """
        Run the reversal learning task, having the agent learn and adapt.
        """
        agent_choices: List[int] = []
        current_state = 0

        for trial in range(self.num_trials):
            # Reverse the probabilities at the specified point
            if trial == self.reversal_point:
                print("\n--- PROBABILITIES REVERSED ---\n")
                self.probabilities.reverse()

            action = self.rl_agent.choose_action(current_state)
            agent_choices.append(action)

            reward = self._get_reward(action)
            
            next_state = 0
            self.rl_agent.learn(current_state, action, reward, next_state)
            
            self.rl_agent.decay_exploration()

        return agent_choices

    def evaluate(self, agent_choices: List[int]) -> Dict[str, Any]:
        """
        Analyze the agent's performance and provide feedback to the executive.
        """
        pre_reversal_choices = agent_choices[:self.reversal_point]
        post_reversal_choices = agent_choices[self.reversal_point:]

        # Initially, choice 0 is optimal
        pre_reversal_optimal_choices = sum(1 for c in pre_reversal_choices if c == 0)
        pre_reversal_accuracy = pre_reversal_optimal_choices / self.reversal_point if self.reversal_point > 0 else 0
        
        # After reversal, choice 1 is optimal
        post_reversal_optimal_choices = sum(1 for c in post_reversal_choices if c == 1)
        post_reversal_accuracy = post_reversal_optimal_choices / len(post_reversal_choices) if post_reversal_choices else 0

        # Integration: Provide feedback to the meta-learner.
        # The reward will be the post-reversal accuracy, as that's the key metric for this task.
        self.executive.provide_feedback(post_reversal_accuracy)

        return {
            "pre_reversal_accuracy": round(pre_reversal_accuracy, 4),
            "post_reversal_accuracy": round(post_reversal_accuracy, 4),
            "num_trials": self.num_trials,
            "reversal_point": self.reversal_point
        }
