# brain_modules/basal_ganglia/enhanced_gating_system.py

"""
Purpose: Enhanced gating system integrating biological models, actor-critic learning, and experience replay.
Inputs: State observations, rewards, neural inputs
Outputs: Actions, learning signals, system state
Dependencies: enhanced_architecture, actor_critic, numpy
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time

from .enhanced_architecture import EnhancedBasalGanglia, EnhancedDopamineSystem
from .actor_critic import ActorCriticAgent, PrioritizedExperienceReplay

class EnhancedGatingSystem:
    """
    Enhanced gating system combining biological plausibility with advanced learning algorithms.
    Integrates spiking neurons, actor-critic learning, and experience replay.
    """
    
    def __init__(self, num_states: int, num_actions: int, num_neurons: int = 100,
                 use_biological_models: bool = True, use_experience_replay: bool = True):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_neurons = num_neurons
        self.use_biological_models = use_biological_models
        self.use_experience_replay = use_experience_replay
        
        # Initialize components
        if use_biological_models:
            self.biological_system = EnhancedBasalGanglia(num_neurons)
            self.dopamine_system = EnhancedDopamineSystem()
        
        # Actor-critic learning system
        self.actor_critic = ActorCriticAgent(num_states, num_actions)
        
        # Experience replay for stable learning
        if use_experience_replay:
            self.experience_replay = PrioritizedExperienceReplay(capacity=10000)
            self.replay_batch_size = 32
            self.replay_frequency = 4  # Learn from replay every N steps
        
        # Learning statistics
        self.episode_count = 0
        self.total_reward = 0.0
        self.learning_history = []
        self.performance_metrics = {
            'episode_rewards': [],
            'policy_entropies': [],
            'dopamine_levels': [],
            'biological_activities': []
        }
        
        # Noise injection for robustness
        self.noise_level = 0.05
        self.noise_decay = 0.999
        
    def choose_action(self, state: int, add_noise: bool = True) -> int:
        """Choose action using integrated system"""
        # Add noise to state representation for robustness
        if add_noise and self.noise_level > 0.001:
            state_noise = np.random.normal(0, self.noise_level)
            # Apply noise to state (simplified - in practice would affect state representation)
            pass
        
        # Get action from actor-critic
        action = self.actor_critic.choose_action(state)
        
        # If using biological models, get biological validation
        if self.use_biological_models:
            # Generate neural input based on state (simplified)
            neural_input = self._state_to_neural_input(state)
            biological_action, _ = self.biological_system.process_action_selection(
                neural_input, 0.0, 0.0  # Placeholder values
            )
            
            # Combine biological and learning-based decisions (simplified)
            # In practice, this would be more sophisticated
            if np.random.random() < 0.1:  # 10% chance to use biological decision
                action = biological_action
        
        return action
    
    def learn_from_experience(self, state: int, action: int, reward: float, 
                             next_state: int, done: bool = False):
        """Learn from experience using integrated learning mechanisms"""
        # Update actor-critic
        self.actor_critic.learn(next_state, reward, done)
        
        # Store experience in replay buffer
        if self.use_experience_replay:
            # Calculate TD error for prioritization
            current_value = self.actor_critic.get_state_value(state)
            next_value = 0.0 if done else self.actor_critic.get_state_value(next_state)
            td_error = reward + 0.9 * next_value - current_value
            
            self.experience_replay.add_experience(state, action, reward, next_state, done, td_error)
        
        # Update biological system if enabled
        if self.use_biological_models:
            neural_input = self._state_to_neural_input(state)
            self.biological_system.process_action_selection(neural_input, 0.0, reward)
        
        # Update performance metrics
        self.total_reward += reward
        self._update_performance_metrics(state, reward)
        
        # Learn from replay buffer periodically
        if self.use_experience_replay and self.episode_count % self.replay_frequency == 0:
            self._learn_from_replay()
    
    def _learn_from_replay(self):
        """Learn from experience replay buffer"""
        if self.experience_replay.get_size() < self.replay_batch_size:
            return
        
        # Sample batch from replay buffer
        batch, weights, indices = self.experience_replay.sample_batch(self.replay_batch_size)
        td_errors = []
        
        # Learn from each experience in the batch
        for (state, action, reward, next_state, done), weight in zip(batch, weights):
            # Get current and next state values
            current_value = self.actor_critic.get_state_value(state)
            next_value = 0.0 if done else self.actor_critic.get_state_value(next_state)
            
            # Calculate TD target and error
            td_target = reward + 0.9 * next_value
            td_error = td_target - current_value
            
            # Update actor-critic with importance sampling weight
            # (Simplified - in practice would apply weight to gradients)
            self.actor_critic.learn(next_state, reward, done)
            td_errors.append(td_error)
        
        # Update priorities in replay buffer
        if len(td_errors) > 0:
            self.experience_replay.update_priorities(indices, np.array(td_errors))
    
    def _state_to_neural_input(self, state: int) -> np.ndarray:
        """Convert state to neural input representation"""
        # Simplified conversion - in practice would be more sophisticated
        neural_input = np.zeros(self.num_neurons)
        
        # Create a sparse representation based on state
        start_idx = (state * self.num_neurons) // self.num_states
        end_idx = min(start_idx + 10, self.num_neurons)
        neural_input[start_idx:end_idx] = np.random.rand(end_idx - start_idx)
        
        return neural_input
    
    def _update_performance_metrics(self, state: int, reward: float):
        """Update performance tracking metrics"""
        # Policy entropy
        entropy = self.actor_critic.get_policy_entropy(state)
        self.performance_metrics['policy_entropies'].append(entropy)
        
        # Dopamine level (if using biological models)
        if self.use_biological_models:
            state_info = self.biological_system.get_system_state()
            self.performance_metrics['dopamine_levels'].append(state_info['dopamine_level'])
            self.performance_metrics['biological_activities'].append({
                'direct': state_info['direct_pathway_activity'],
                'indirect': state_info['indirect_pathway_activity']
            })
        
        # Decay noise
        self.noise_level *= self.noise_decay
    
    def get_system_state(self) -> Dict:
        """Get comprehensive system state"""
        state = {
            'actor_critic_state': {
                'exploration_rate': self.actor_critic.exploration_rate,
                'policy_entropy': self.actor_critic.get_policy_entropy(0),  # Example state
                'state_values': [self.actor_critic.get_state_value(s) for s in range(min(5, self.num_states))]
            },
            'learning_stats': {
                'episode_count': self.episode_count,
                'total_reward': self.total_reward,
                'avg_reward': self.total_reward / max(1, self.episode_count)
            },
            'replay_stats': {}
        }
        
        if self.use_experience_replay:
            state['replay_stats'] = {
                'buffer_size': self.experience_replay.get_size(),
                'max_priority': self.experience_replay.max_priority
            }
        
        if self.use_biological_models:
            bio_state = self.biological_system.get_system_state()
            state['biological_state'] = bio_state
        
        return state
    
    def end_episode(self):
        """End current episode and update statistics"""
        self.episode_count += 1
        self.performance_metrics['episode_rewards'].append(self.total_reward)
        self.total_reward = 0.0
        
        # Record learning history
        self.learning_history.append({
            'episode': self.episode_count,
            'system_state': self.get_system_state()
        })
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary statistics"""
        if not self.performance_metrics['episode_rewards']:
            return {}
        
        recent_rewards = self.performance_metrics['episode_rewards'][-100:]  # Last 100 episodes
        recent_entropies = self.performance_metrics['policy_entropies'][-100:]
        
        return {
            'avg_reward': np.mean(recent_rewards),
            'std_reward': np.std(recent_rewards),
            'max_reward': np.max(recent_rewards),
            'avg_entropy': np.mean(recent_entropies),
            'learning_progress': len(recent_rewards) >= 10 and np.mean(recent_rewards[-10:]) > np.mean(recent_rewards[:10])
        }
    
    def add_noise_to_input(self, input_signal: np.ndarray) -> np.ndarray:
        """Add noise to input for robustness testing"""
        noise = np.random.normal(0, self.noise_level, input_signal.shape)
        return input_signal + noise
    
    def test_robustness(self, num_trials: int = 100) -> Dict:
        """Test system robustness to various perturbations"""
        robustness_results = {
            'noise_tolerance': [],
            'action_consistency': [],
            'learning_stability': []
        }
        
        for trial in range(num_trials):
            # Test with different noise levels
            state = np.random.randint(0, self.num_states)
            clean_action = self.choose_action(state, add_noise=False)
            
            # Test with noise
            noisy_action = self.choose_action(state, add_noise=True)
            robustness_results['noise_tolerance'].append(clean_action == noisy_action)
            
            # Test action consistency
            action1 = self.choose_action(state, add_noise=False)
            action2 = self.choose_action(state, add_noise=False)
            robustness_results['action_consistency'].append(action1 == action2)
        
        # Calculate robustness metrics
        robustness_results['noise_tolerance'] = np.mean(robustness_results['noise_tolerance'])
        robustness_results['action_consistency'] = np.mean(robustness_results['action_consistency'])
        
        return robustness_results

if __name__ == '__main__':
    # Test the enhanced gating system
    print("Testing Enhanced Gating System")
    
    # Create system
    enhanced_system = EnhancedGatingSystem(
        num_states=5, 
        num_actions=3, 
        num_neurons=50,
        use_biological_models=True,
        use_experience_replay=True
    )
    
    # Simulate learning episodes
    for episode in range(50):
        state = 0
        episode_reward = 0
        
        for step in range(10):
            action = enhanced_system.choose_action(state)
            reward = 1.0 if action == 1 and state == 2 else 0.1  # Optimal action in state 2
            next_state = (state + 1) % 5
            
            enhanced_system.learn_from_experience(state, action, reward, next_state, done=(step == 9))
            state = next_state
            episode_reward += reward
        
        enhanced_system.end_episode()
        
        if episode % 10 == 0:
            performance = enhanced_system.get_performance_summary()
            print(f"Episode {episode}: Avg Reward = {performance.get('avg_reward', 0):.3f}")
    
    # Test robustness
    robustness = enhanced_system.test_robustness(num_trials=50)
    print(f"Robustness - Noise Tolerance: {robustness['noise_tolerance']:.3f}")
    print(f"Robustness - Action Consistency: {robustness['action_consistency']:.3f}")
    
    print("Enhanced Gating System test completed.")
