# tests/test_enhanced_pillar2_comprehensive.py

"""
Purpose: Comprehensive validation test for all enhanced Pillar 2 components.
Inputs: Test scenarios for biological models, actor-critic, experience replay, and integration
Outputs: Validation results and performance metrics
Dependencies: brain_modules.basal_ganglia.*, numpy, pytest
"""

import sys
import os
import numpy as np
import pytest

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from brain_modules.basal_ganglia.enhanced_architecture import (
    SpikingNeuron,
    EnhancedStriatum,
    EnhancedDopamineSystem,
    EnhancedBasalGanglia
)
from brain_modules.basal_ganglia.actor_critic import (
    Actor,
    Critic,
    ActorCriticAgent,
    ExperienceReplay,
    PrioritizedExperienceReplay
)
from brain_modules.basal_ganglia.enhanced_gating_system import EnhancedGatingSystem

class TestEnhancedPillar2Comprehensive:
    """Comprehensive test suite for all enhanced Pillar 2 components"""
    
    def test_actor_critic_basic_functionality(self):
        """Test basic actor-critic functionality"""
        agent = ActorCriticAgent(num_states=5, num_actions=3)
        
        # Test action selection
        state = 0
        action = agent.choose_action(state)
        assert 0 <= action < 3, "Action should be valid"
        
        # Test learning
        next_state = 1
        reward = 1.0
        agent.learn(next_state, reward, done=False)
        
        # Test policy entropy
        entropy = agent.get_policy_entropy(state)
        assert entropy >= 0, "Policy entropy should be non-negative"
        
        # Test state value
        value = agent.get_state_value(state)
        assert isinstance(value, float), "State value should be float"
    
    def test_actor_critic_policy_convergence(self):
        """Test that actor-critic learns optimal policy"""
        agent = ActorCriticAgent(num_states=3, num_actions=2)
        
        # Train on simple environment where action 1 in state 1 is optimal
        for episode in range(100):
            state = 0
            for step in range(10):
                action = agent.choose_action(state)
                reward = 1.0 if action == 1 and state == 1 else 0.1
                next_state = (state + 1) % 3
                agent.learn(next_state, reward, done=(step == 9))
                state = next_state
        
        # Check that optimal action has higher probability
        probabilities = agent.get_action_probabilities(1)
        assert probabilities[1] > probabilities[0], "Optimal action should have higher probability"
    
    def test_experience_replay_basic(self):
        """Test basic experience replay functionality"""
        replay = ExperienceReplay(capacity=100)
        
        # Add experiences
        for i in range(50):
            replay.add_experience(i % 5, i % 3, i * 0.1, (i + 1) % 5, i % 10 == 0)
        
        # Test sampling
        batch = replay.sample_batch(10)
        assert len(batch) == 10, "Should sample requested batch size"
        
        # Test buffer size
        assert replay.get_size() == 50, "Buffer should contain 50 experiences"
    
    def test_prioritized_experience_replay(self):
        """Test prioritized experience replay with TD errors"""
        replay = PrioritizedExperienceReplay(capacity=100)
        
        # Add experiences with different TD errors
        for i in range(20):
            td_error = i * 0.1  # Varying TD errors
            replay.add_experience(i % 5, i % 3, i * 0.1, (i + 1) % 5, False, td_error)
        
        # Test sampling with priorities
        batch, weights, indices = replay.sample_batch(5)
        assert len(batch) == 5, "Should sample requested batch size"
        assert len(weights) == 5, "Should return weights for each sample"
        assert len(indices) == 5, "Should return indices for each sample"
        
        # Test priority update
        new_td_errors = np.array([0.5, 0.3, 0.8, 0.2, 0.6])
        replay.update_priorities(indices, new_td_errors)
        
        # Check that max priority was updated
        assert replay.max_priority >= 0.8, "Max priority should be updated"
    
    def test_enhanced_gating_system_integration(self):
        """Test integration of all enhanced components"""
        system = EnhancedGatingSystem(
            num_states=4,
            num_actions=2,
            num_neurons=20,
            use_biological_models=True,
            use_experience_replay=True
        )
        
        # Test action selection
        state = 0
        action = system.choose_action(state)
        assert 0 <= action < 2, "Action should be valid"
        
        # Test learning
        reward = 1.0
        next_state = 1
        system.learn_from_experience(state, action, reward, next_state, done=False)
        
        # Test system state
        state_info = system.get_system_state()
        assert 'actor_critic_state' in state_info, "Should include actor-critic state"
        assert 'learning_stats' in state_info, "Should include learning stats"
        assert 'replay_stats' in state_info, "Should include replay stats"
        assert 'biological_state' in state_info, "Should include biological state"
    
    def test_enhanced_gating_system_learning(self):
        """Test learning performance of enhanced system"""
        system = EnhancedGatingSystem(
            num_states=3,
            num_actions=2,
            num_neurons=15,
            use_biological_models=True,
            use_experience_replay=True
        )
        
        # Train on simple environment
        for episode in range(30):
            state = 0
            episode_reward = 0
            
            for step in range(5):
                action = system.choose_action(state)
                reward = 1.0 if action == 1 and state == 1 else 0.1
                next_state = (state + 1) % 3
                
                system.learn_from_experience(state, action, reward, next_state, done=(step == 4))
                state = next_state
                episode_reward += reward
            
            system.end_episode()
        
        # Check performance
        performance = system.get_performance_summary()
        assert 'avg_reward' in performance, "Should include average reward"
        assert 'learning_progress' in performance, "Should include learning progress"
    
    def test_robustness_features(self):
        """Test robustness features of enhanced system"""
        system = EnhancedGatingSystem(
            num_states=3,
            num_actions=2,
            num_neurons=10,
            use_biological_models=True,
            use_experience_replay=True
        )
        
        # Test noise injection
        clean_input = np.random.rand(10)
        noisy_input = system.add_noise_to_input(clean_input)
        assert np.any(clean_input != noisy_input), "Noise should be added to input"
        
        # Test robustness
        robustness = system.test_robustness(num_trials=20)
        assert 'noise_tolerance' in robustness, "Should include noise tolerance"
        assert 'action_consistency' in robustness, "Should include action consistency"
        assert 0 <= robustness['noise_tolerance'] <= 1, "Noise tolerance should be in [0,1]"
        assert 0 <= robustness['action_consistency'] <= 1, "Action consistency should be in [0,1]"
    
    def test_biological_plausibility_enhanced(self):
        """Test enhanced biological plausibility features"""
        # Test spiking neuron refractory period
        neuron = SpikingNeuron(threshold=1.0, decay=0.9, refractory_period=3)
        
        # Generate spike
        neuron.update(2.0, 0)
        assert neuron.spike_history[0] == 1, "Should spike with supra-threshold input"
        
        # Test refractory period
        for i in range(1, 4):
            spiked = neuron.update(2.0, i)
            assert not spiked, f"Should not spike during refractory period at time {i}"
        
        # Test enhanced dopamine system
        dopamine_system = EnhancedDopamineSystem()
        
        # Test phasic/tonic separation
        dopamine_system.process_reward(0.2, 1.0, 0)  # Positive surprise
        tonic, phasic, total = dopamine_system.response_history[-1]
        assert phasic > 0, "Should have positive phasic response"
        assert tonic > dopamine_system.baseline_dopamine, "Should have elevated tonic level"
        
        # Test temporal dynamics
        for i in range(1, 10):
            dopamine_system.process_reward(0.3, 0.8 if i % 3 == 0 else 0.2, i)
        
        assert len(dopamine_system.response_history) == 10, "Should track all responses"
        assert len(dopamine_system.time_history) == 10, "Should track all times"
    
    def test_performance_monitoring(self):
        """Test performance monitoring and metrics"""
        system = EnhancedGatingSystem(
            num_states=4,
            num_actions=2,
            num_neurons=20,
            use_biological_models=True,
            use_experience_replay=True
        )
        
        # Run some episodes
        for episode in range(10):
            state = 0
            for step in range(3):
                action = system.choose_action(state)
                reward = np.random.random()
                next_state = (state + 1) % 4
                system.learn_from_experience(state, action, reward, next_state, done=(step == 2))
                state = next_state
            system.end_episode()
        
        # Check performance metrics
        performance = system.get_performance_summary()
        assert 'avg_reward' in performance, "Should track average reward"
        assert 'std_reward' in performance, "Should track reward standard deviation"
        assert 'max_reward' in performance, "Should track maximum reward"
        assert 'avg_entropy' in performance, "Should track average entropy"
        assert 'learning_progress' in performance, "Should track learning progress"
        
        # Check learning history
        assert len(system.learning_history) == 10, "Should track all episodes"
        assert len(system.performance_metrics['episode_rewards']) == 10, "Should track episode rewards"
    
    def test_system_scalability(self):
        """Test system scalability with larger state/action spaces"""
        # Test with larger system
        large_system = EnhancedGatingSystem(
            num_states=20,
            num_actions=5,
            num_neurons=100,
            use_biological_models=True,
            use_experience_replay=True
        )
        
        # Test action selection in larger space
        for state in range(5):  # Test subset of states
            action = large_system.choose_action(state)
            assert 0 <= action < 5, "Action should be valid in larger space"
        
        # Test learning in larger space
        for i in range(20):
            state = i % 20
            action = large_system.choose_action(state)
            reward = np.random.random()
            next_state = (state + 1) % 20
            large_system.learn_from_experience(state, action, reward, next_state, done=False)
        
        # Check system state
        state_info = large_system.get_system_state()
        assert len(state_info['actor_critic_state']['state_values']) == 5, "Should track subset of state values"
    
    def test_integration_consistency(self):
        """Test consistency between different components"""
        system = EnhancedGatingSystem(
            num_states=3,
            num_actions=2,
            num_neurons=15,
            use_biological_models=True,
            use_experience_replay=True
        )
        
        # Test that biological and learning systems are consistent
        state = 0
        
        # Get action from actor-critic
        ac_action = system.actor_critic.choose_action(state)
        
        # Get action from biological system
        neural_input = system._state_to_neural_input(state)
        bio_action, _ = system.biological_system.process_action_selection(neural_input, 0.0, 0.0)
        
        # Both should produce valid actions
        assert 0 <= ac_action < 2, "Actor-critic action should be valid"
        assert bio_action in [0, 1], "Biological action should be valid"
        
        # Test that experience replay stores experiences correctly
        reward = 1.0
        next_state = 1
        system.learn_from_experience(state, ac_action, reward, next_state, done=False)
        
        # Check replay buffer
        if system.use_experience_replay:
            assert system.experience_replay.get_size() > 0, "Should store experiences in replay buffer"

def run_comprehensive_enhanced_validation():
    """Run comprehensive validation of all enhanced Pillar 2 components"""
    print("🧠 Comprehensive Enhanced Pillar 2 Validation")
    print("=" * 70)
    
    # Create test instance
    test_suite = TestEnhancedPillar2Comprehensive()
    
    # Run all tests
    test_methods = [
        test_suite.test_actor_critic_basic_functionality,
        test_suite.test_actor_critic_policy_convergence,
        test_suite.test_experience_replay_basic,
        test_suite.test_prioritized_experience_replay,
        test_suite.test_enhanced_gating_system_integration,
        test_suite.test_enhanced_gating_system_learning,
        test_suite.test_robustness_features,
        test_suite.test_biological_plausibility_enhanced,
        test_suite.test_performance_monitoring,
        test_suite.test_system_scalability,
        test_suite.test_integration_consistency
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
    
    print("=" * 70)
    print(f"📊 Comprehensive Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 Comprehensive Enhanced Pillar 2 Validation: ALL TESTS PASSED")
        print("✅ All enhanced components are working correctly")
        print("🧬 Enhanced features:")
        print("   - Spiking neural networks with refractory periods")
        print("   - Direct/indirect pathway separation")
        print("   - Phasic/tonic dopamine responses")
        print("   - Actor-critic learning with policy gradients")
        print("   - Prioritized experience replay")
        print("   - Noise injection for robustness")
        print("   - Comprehensive performance monitoring")
        return True
    else:
        print("⚠️  Comprehensive Enhanced Pillar 2 Validation: SOME TESTS FAILED")
        print("🔧 Review enhanced implementation and fix issues")
        return False

if __name__ == "__main__":
    success = run_comprehensive_enhanced_validation()
    exit(0 if success else 1)
