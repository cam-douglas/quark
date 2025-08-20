# tests/test_pillar2_basal_ganglia.py

"""
Purpose: Comprehensive validation test for Pillar 2 (Gating & Reinforcement) Basal Ganglia system.
Inputs: Test scenarios with known reward patterns
Outputs: Validation results and learning metrics
Dependencies: brain_modules.basal_ganglia.*, numpy, pytest
"""

import sys
import os
import numpy as np
import pytest

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from brain_modules.basal_ganglia.gating_system import GatingSystem
from brain_modules.basal_ganglia.dopamine_system import DopamineSystem
from brain_modules.basal_ganglia.rl_agent import QLearningAgent

class TestPillar2BasalGanglia:
    """Test suite for Pillar 2: Gating & Reinforcement Learning System"""
    
    def setup_method(self):
        """Set up test environment before each test"""
        self.num_states = 5
        self.num_actions = 3
        self.gating_system = GatingSystem(self.num_states, self.num_actions)
        
    def test_architecture_initialization(self):
        """Test that all Basal Ganglia components initialize correctly"""
        assert self.gating_system.architecture is not None
        assert self.gating_system.agent is not None
        assert self.gating_system.dopamine_system is not None
        
        # Test Q-learning agent initialization
        assert self.gating_system.agent.num_states == self.num_states
        assert self.gating_system.agent.num_actions == self.num_actions
        assert self.gating_system.agent.q_table.shape == (self.num_states, self.num_actions)
        
        # Test dopamine system initialization
        assert self.gating_system.dopamine_system.baseline_dopamine == 0.5
        assert self.gating_system.dopamine_system.current_dopamine_level == 0.5
        
    def test_dopamine_response_to_rewards(self):
        """Test dopamine system response to different reward scenarios"""
        dopamine_system = DopamineSystem()
        
        # Test positive surprise (unexpected high reward)
        predicted_reward = 0.2
        actual_reward = 1.0
        dopamine_level, rpe = dopamine_system.process_reward(predicted_reward, actual_reward)
        
        assert rpe > 0  # Positive reward prediction error
        assert dopamine_level > dopamine_system.baseline_dopamine  # Dopamine surge
        
        # Test negative surprise (reward less than expected)
        predicted_reward = 0.8
        actual_reward = 0.1
        dopamine_level, rpe = dopamine_system.process_reward(predicted_reward, actual_reward)
        
        assert rpe < 0  # Negative reward prediction error
        assert dopamine_level < dopamine_system.baseline_dopamine  # Dopamine dip
        
        # Test no surprise (reward as expected)
        predicted_reward = 0.5
        actual_reward = 0.5
        dopamine_level, rpe = dopamine_system.process_reward(predicted_reward, actual_reward)
        
        assert rpe == 0  # No reward prediction error
        assert abs(dopamine_level - dopamine_system.baseline_dopamine) < 0.01  # Baseline dopamine
        
    def test_action_selection_learning(self):
        """Test that the agent learns to prefer high-reward actions"""
        # Create a simple environment where action 1 in state 2 gives high reward
        optimal_state = 2
        optimal_action = 1
        high_reward = 1.0
        low_reward = 0.1
        
        # Train the agent for multiple episodes
        num_episodes = 1000
        state = 0
        
        for episode in range(num_episodes):
            action = self.gating_system.choose_action(state)
            
            # Determine reward based on state-action pair
            if state == optimal_state and action == optimal_action:
                reward = high_reward
            else:
                reward = low_reward
                
            next_state = (state + 1) % self.num_states
            self.gating_system.learn_from_feedback(state, action, reward, next_state)
            state = next_state
            
        # Test that the agent learned the optimal action
        optimal_q_value = self.gating_system.agent.q_table[optimal_state, optimal_action]
        other_q_values = [
            self.gating_system.agent.q_table[optimal_state, a] 
            for a in range(self.num_actions) if a != optimal_action
        ]
        
        # The optimal action should have the highest Q-value
        assert optimal_q_value > max(other_q_values), f"Optimal action not learned. Optimal Q-value: {optimal_q_value}, Other Q-values: {other_q_values}"
        
    def test_exploration_exploitation_balance(self):
        """Test that the agent balances exploration and exploitation correctly"""
        initial_exploration_rate = self.gating_system.agent.exploration_rate
        
        # Run some episodes to trigger exploration decay
        for episode in range(100):
            state = 0
            action = self.gating_system.choose_action(state)
            reward = np.random.random()
            next_state = (state + 1) % self.num_states
            self.gating_system.learn_from_feedback(state, action, reward, next_state)
            
        # Exploration rate should have decreased
        final_exploration_rate = self.gating_system.agent.exploration_rate
        assert final_exploration_rate < initial_exploration_rate, "Exploration rate should decrease over time"
        
        # But should not go below minimum
        assert final_exploration_rate >= self.gating_system.agent.min_exploration_rate, "Exploration rate should not go below minimum"
        
    def test_learning_rate_modulation(self):
        """Test that learning rate is modulated by dopamine levels"""
        initial_learning_rate = self.gating_system.agent.learning_rate
        
        # Simulate high dopamine (positive surprise)
        high_dopamine = 0.8
        self.gating_system.modulate_learning_rate(high_dopamine)
        high_dopamine_learning_rate = self.gating_system.agent.learning_rate
        
        # Simulate low dopamine (negative surprise)
        low_dopamine = 0.2
        self.gating_system.modulate_learning_rate(low_dopamine)
        low_dopamine_learning_rate = self.gating_system.agent.learning_rate
        
        # Learning rate should be modulated by dopamine deviation from baseline
        baseline_dopamine = self.gating_system.dopamine_system.baseline_dopamine
        high_deviation = abs(high_dopamine - baseline_dopamine)
        low_deviation = abs(low_dopamine - baseline_dopamine)
        
        # Higher deviation should lead to higher learning rate
        assert high_dopamine_learning_rate > low_dopamine_learning_rate, "Higher dopamine deviation should increase learning rate"
        
    def test_convergence_to_optimal_policy(self):
        """Test that the agent converges to an optimal policy in a deterministic environment"""
        # Create a deterministic environment with clear optimal actions
        optimal_actions = {0: 1, 1: 2, 2: 0, 3: 1, 4: 2}  # Optimal action for each state
        
        # Train the agent extensively
        num_episodes = 2000
        state = 0
        
        for episode in range(num_episodes):
            action = self.gating_system.choose_action(state)
            
            # Give high reward for optimal actions, low reward for others
            if action == optimal_actions[state]:
                reward = 1.0
            else:
                reward = 0.0
                
            next_state = (state + 1) % self.num_states
            self.gating_system.learn_from_feedback(state, action, reward, next_state)
            state = next_state
            
        # Test convergence: optimal actions should have highest Q-values
        for state in range(self.num_states):
            optimal_action = optimal_actions[state]
            optimal_q_value = self.gating_system.agent.q_table[state, optimal_action]
            
            # Check that optimal action has highest Q-value
            for action in range(self.num_actions):
                if action != optimal_action:
                    other_q_value = self.gating_system.agent.q_table[state, action]
                    assert optimal_q_value >= other_q_value, f"State {state}: Optimal action {optimal_action} should have highest Q-value"
                    
    def test_dopamine_integration_with_learning(self):
        """Test that dopamine signals properly integrate with the learning process"""
        # Test that RPE is used as learning signal
        state = 0
        action = 0
        predicted_reward = 0.3
        actual_reward = 0.8  # Positive surprise
        
        # Get initial Q-value
        initial_q_value = self.gating_system.agent.q_table[state, action]
        
        # Learn from feedback
        next_state = 1
        self.gating_system.learn_from_feedback(state, action, actual_reward, next_state)
        
        # Get final Q-value
        final_q_value = self.gating_system.agent.q_table[state, action]
        
        # Q-value should have increased due to positive RPE
        assert final_q_value > initial_q_value, "Q-value should increase with positive RPE"
        
        # Test negative RPE
        state = 1
        action = 1
        predicted_reward = 0.7
        actual_reward = 0.2  # Negative surprise
        
        initial_q_value = self.gating_system.agent.q_table[state, action]
        self.gating_system.learn_from_feedback(state, action, actual_reward, next_state)
        final_q_value = self.gating_system.agent.q_table[state, action]
        
        # Q-value should have decreased due to negative RPE
        assert final_q_value < initial_q_value, "Q-value should decrease with negative RPE"

def run_pillar2_validation():
    """Run comprehensive validation of Pillar 2 implementation"""
    print("🧠 Pillar 2 Validation: Gating & Reinforcement Learning System")
    print("=" * 60)
    
    # Create test instance
    test_suite = TestPillar2BasalGanglia()
    test_suite.setup_method()
    
    # Run all tests
    test_methods = [
        test_suite.test_architecture_initialization,
        test_suite.test_dopamine_response_to_rewards,
        test_suite.test_action_selection_learning,
        test_suite.test_exploration_exploitation_balance,
        test_suite.test_learning_rate_modulation,
        test_suite.test_convergence_to_optimal_policy,
        test_suite.test_dopamine_integration_with_learning
    ]
    
    passed_tests = 0
    total_tests = len(test_methods)
    
    for test_method in test_methods:
        try:
            test_method()
            print(f"✅ {test_method.__name__}: PASSED")
            passed_tests += 1
        except Exception as e:
            print(f"❌ {test_method.__name__}: FAILED - {str(e)}")
    
    print("=" * 60)
    print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 Pillar 2 Validation: ALL TESTS PASSED")
        print("✅ Basal Ganglia Gating & Reinforcement System is working correctly")
        return True
    else:
        print("⚠️  Pillar 2 Validation: SOME TESTS FAILED")
        print("🔧 Review implementation and fix issues")
        return False

if __name__ == "__main__":
    success = run_pillar2_validation()
    exit(0 if success else 1)
