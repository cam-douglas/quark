# brain_architecture/neural_core/prefrontal_cortex/executive_control.py
"""
Purpose: Implements a high-level executive control function to select the
         appropriate cognitive strategy for a given task.
Inputs: Task properties (dictionary)
Outputs: An instantiated learning agent (Strategic or Tactical)
Dependencies: StrategicAgent, TacticalAgent
"""

from brain_architecture.neural_core.motor_control.basal_ganglia.strategic_agent import StrategicAgent
from brain_architecture.neural_core.motor_control.basal_ganglia.tactical_agent import TacticalAgent
from brain_architecture.neural_core.prefrontal_cortex.meta_learning_agent import MetaLearningAgent
from typing import Union, List
from brain_architecture.neural_core.prefrontal_cortex.plan import Plan
from brain_architecture.neural_core.prefrontal_cortex.decision import Decision
from brain_architecture.neural_core.prefrontal_cortex.scientific_validator import ScientificValidator
from brain_architecture.neural_core.prefrontal_cortex.dopamine_system import DopamineSystem
import os

class ExecutiveControl:
    """Core executive control for planning and decision-making"""
    
    def __init__(self):
        self.plans: List[Plan] = []
        self.decisions: List[Decision] = []
        self.cognitive_resources = {"attention": 1.0, "memory": 1.0}
        self.validator = ScientificValidator()
        self.dopamine_system = DopamineSystem()

        # Meta-Learning Integration
        self.strategies = [StrategicAgent, TacticalAgent]
        self.meta_learner = MetaLearningAgent(num_states=1, num_strategies=len(self.strategies))
        self.last_decision = {'state': 0, 'strategy_index': -1}
        
        self._last_goal_push_s: float = 0.0
        self._viewer_url: str = os.environ.get("QUARK_VIEWER_URL", "http://127.0.0.1:8011")
        
    def select_strategy(self, task_properties: dict, num_states: int, num_actions: int) -> Union[StrategicAgent, TacticalAgent]:
        """
        Uses the meta-learning agent to select the best strategy for the task.
        """
        # For now, we use a single state for the meta-learner.
        # This could be expanded to a state vector based on task_properties.
        current_meta_state = 0
        
        strategy_index = self.meta_learner.choose_strategy(current_meta_state)
        SelectedAgentClass = self.strategies[strategy_index]
        
        print(f"Executive Control (Meta-Learner): Selected {SelectedAgentClass.__name__}")

        # Store the decision for the feedback loop
        self.last_decision = {'state': current_meta_state, 'strategy_index': strategy_index}
        
        return SelectedAgentClass(num_states, num_actions)

    def provide_feedback(self, reward: float):
        """
        Provides performance feedback to the meta-learning agent to help it learn.
        """
        if self.last_decision['strategy_index'] == -1:
            print("Meta-Learner: No decision to provide feedback on.")
            return

        state = self.last_decision['state']
        strategy = self.last_decision['strategy_index']
        # The next state is the same in our simple meta-learning model
        next_state = 0

        self.meta_learner.learn(state, strategy, reward, next_state)
        self.meta_learner.decay_exploration()
        print(f"Meta-Learner: Feedback provided. Reward: {reward}. Q-Table updated.")
        # Reset last decision
        self.last_decision = {'state': 0, 'strategy_index': -1}
    
    def make_decision(self, options: List[str]) -> Decision:
        # This method can be repurposed or used for other types of decisions.
        # For agent selection, we now use `select_strategy`.
        selected = options[0] if options else ""
        confidence = 0.7
        reasoning = f"Selected {selected} based on current context (non-meta-learning decision)"
        
        decision = Decision(
            options=options,
            selected=selected,
            confidence=confidence,
            reasoning=reasoning
        )
        self.decisions.append(decision)
        return decision
