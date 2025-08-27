import random
from typing import Dict, Any, List

# Integration: Import the QLearningAgent from Quark's architecture
from brain_architecture.neural_core.motor_control.basal_ganglia.rl_agent import QLearningAgent
from testing.cognitive_benchmarks.base_benchmark import BaseBenchmark

# Integration: Import the new executive control function
from brain_architecture.neural_core.prefrontal_cortex.executive_control import ExecutiveControl

class DecisionMakingBenchmark(BaseBenchmark):
    """
    Benchmark for decision-making under uncertainty, inspired by the Iowa Gambling Task.

    The agent must choose from four decks of cards to maximize its score.
    - Decks A & B: High immediate reward, but larger, unpredictable penalties (bad decks).
    - Decks C & D: Lower immediate reward, but smaller penalties (good decks).
    A successful agent learns to avoid the bad decks and favor the good ones.
    """
    def __init__(self, num_trials: int = 100, dynamic_rewards: bool = False):
        super().__init__(
            name=f"Decision Making (Iowa Gambling Task, {'Dynamic' if dynamic_rewards else 'Static'})",
            description=f"Tests strategic decision-making over {num_trials} trials to maximize long-term gain."
        )
        self.num_trials = num_trials
        self.dynamic_rewards = dynamic_rewards
        self.decks: Dict[str, Dict[str, Any]] = {}
        self.deck_choices: List[str] = []

        # Integration: A single executive control instance will manage the agent
        self.executive = ExecutiveControl()
        self.rl_agent = None
        self.deck_map = ['A', 'B', 'C', 'D']

    def setup(self):
        """Define the properties of the four decks and have the executive select an agent."""
        self.decks = {
            'A': {'reward': 100, 'penalty_chance': 0.5, 'penalty_amount': -250, 'id': 0},
            'B': {'reward': 100, 'penalty_chance': 0.1, 'penalty_amount': -1250, 'id': 1},
            'C': {'reward': 50, 'penalty_chance': 0.5, 'penalty_amount': -50, 'id': 2},
            'D': {'reward': 50, 'penalty_chance': 0.1, 'penalty_amount': -250, 'id': 3}
        }
        self.deck_choices = []

        # Integration: Use the executive function to select the agent
        task_properties = {
            'name': 'Iowa Gambling Task',
            'dynamic': self.dynamic_rewards
        }
        self.rl_agent = self.executive.select_strategy(
            task_properties,
            num_states=1,
            num_actions=4
        )

    def _swap_deck_properties(self):
        """Swaps the properties of 'good' and 'bad' decks."""
        print("\n--- REWARD STRUCTURE SHIFTED ---\n")
        self.decks['A'], self.decks['C'] = self.decks['C'], self.decks['A']
        self.decks['B'], self.decks['D'] = self.decks['D'], self.decks['B']

    def run(self) -> List[str]:
        """
        Have the QLearningAgent make a series of choices and learn from the outcomes.
        """
        agent_choices_numeric: List[int] = []
        current_state = 0 # Single state for this task

        for trial in range(self.num_trials):
            # If dynamic rewards are enabled, swap the decks halfway through
            if self.dynamic_rewards and trial == self.num_trials // 2:
                self._swap_deck_properties()
                # We might want to reset parts of the agent's learning here
                # to simulate a cognitive 'reset' or 're-evaluation'.
                # For now, we'll let it learn through the change.

            # 1. Agent chooses an action (a deck)
            action = self.rl_agent.choose_action(current_state)
            agent_choices_numeric.append(action)
            
            # 2. Calculate the reward based on the chosen deck
            chosen_deck_char = self.deck_map[action]
            deck_info = self.decks[chosen_deck_char]
            
            reward = deck_info['reward']
            if random.random() < deck_info['penalty_chance']:
                reward += deck_info['penalty_amount']
                
            # 3. Agent learns from the reward
            # The next state is still the same, as the task is stateless.
            next_state = 0
            self.rl_agent.learn(current_state, action, reward, next_state)

            # 4. Decay exploration rate so the agent exploits more over time
            if self.rl_agent.exploration_rate > self.rl_agent.min_exploration_rate:
                self.rl_agent.exploration_rate *= self.rl_agent.exploration_decay_rate

        # Convert numeric choices back to deck letters for evaluation
        agent_choices_char = [self.deck_map[choice] for choice in agent_choices_numeric]
        return agent_choices_char

    def evaluate(self, agent_choices: List[str]) -> Dict[str, Any]:
        """
        Analyze the agent's choices and provide feedback to the executive.
        """
        if not agent_choices:
            return {"error": "No choices were made."}

        # Reset decks to original state for consistent evaluation scoring
        if self.dynamic_rewards:
            self._swap_deck_properties()

        score = 0
        choices_over_time = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        
        for choice in agent_choices:
            deck = self.decks[choice]
            score += deck['reward']
            if random.random() < deck['penalty_chance']:
                score += deck['penalty_amount']
            choices_over_time[choice] += 1
            
        first_half_choices = agent_choices[:self.num_trials // 2]
        second_half_choices = agent_choices[self.num_trials // 2:]

        good_deck_choices_first_half = sum(1 for c in first_half_choices if c in ['C', 'D'])
        good_deck_choices_second_half = sum(1 for c in second_half_choices if c in ['C', 'D'])
        
        # This metric is less meaningful for dynamic rewards, but we'll keep it.
        # A better metric would be adaptability.
        shift_towards_good_decks = (good_deck_choices_second_half - good_deck_choices_first_half) / (self.num_trials / 2) if self.num_trials > 0 else 0

        # New metric for dynamic rewards: choices after the swap
        adaptability_score = 0
        if self.dynamic_rewards:
            # After the swap, A & B are the good decks
            good_deck_choices_after_swap = sum(1 for c in second_half_choices if c in ['A', 'B'])
            adaptability_score = good_deck_choices_after_swap / (self.num_trials / 2) if self.num_trials > 0 else 0

        # Integration: Provide feedback to the meta-learner
        # We can use the net score per choice as the reward signal.
        net_score = score / self.num_trials if self.num_trials > 0 else 0
        self.executive.provide_feedback(net_score)

        return {
            "total_score": score,
            "net_score_per_choice": net_score,
            "percentage_good_decks": (choices_over_time['C'] + choices_over_time['D']) / self.num_trials * 100 if self.num_trials > 0 else 0,
            "shift_towards_good_decks": round(shift_towards_good_decks * 100, 2),
            "adaptability_score (post-swap good choices)": round(adaptability_score, 4) if self.dynamic_rewards else "N/A",
            "deck_distribution": choices_over_time,
            "num_trials": self.num_trials,
            "dynamic_rewards": self.dynamic_rewards
        }
