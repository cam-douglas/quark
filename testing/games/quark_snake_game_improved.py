#!/usr/bin/env python3
"""
Quark Snake Game - Improved Learning Demonstration

An enhanced Snake game where Quark learns to play using improved reinforcement learning.
Better state representation and learning parameters for more visible progress.
"""

import pygame
import numpy as np
import random
import time
from collections import deque
import threading
import json

class ImprovedSnakeGame:
    """Improved Snake game environment for Quark to learn."""
    
    def __init__(self, width=800, height=600, grid_size=20):
        pygame.init()
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.grid_width = width // grid_size
        self.grid_height = height // grid_size
        
        # Pygame setup
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Quark Snake Game - Advanced Learning Live!")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Game state
        self.reset_game()
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.PURPLE = (128, 0, 128)
        self.ORANGE = (255, 165, 0)
        
        # Improved learning parameters
        self.epsilon = 0.3  # Higher initial exploration
        self.learning_rate = 0.1  # Higher learning rate
        self.gamma = 0.9  # Slightly lower discount
        self.epsilon_decay = 0.999  # Slower decay
        
        # Neural network (Q-learning with better state representation)
        self.q_table = {}
        self.state_size = 16  # Enhanced state representation
        self.action_size = 4  # Up, Down, Left, Right
        
        # Statistics
        self.episode = 0
        self.total_score = 0
        self.best_score = 0
        self.learning_progress = []
        self.episode_scores = []
        self.episode_lengths = []
        
        # Learning thread
        self.learning_active = False
        self.learning_thread = None
        
        # Display info
        self.show_learning = True
        self.show_q_values = False
        self.show_state_info = False
        
        # Performance tracking
        self.consecutive_zeros = 0
        self.learning_stalled = False
        
    def reset_game(self):
        """Reset the game to initial state."""
        self.snake = [(self.grid_width // 2, self.grid_height // 2)]
        self.direction = [1, 0]  # Start moving right
        self.food = self.generate_food()
        self.score = 0
        self.steps = 0
        self.max_steps = self.grid_width * self.grid_height  # Reduced max steps
        self.game_over = False
        
    def generate_food(self):
        """Generate food at random position."""
        while True:
            food = (random.randint(0, self.grid_width - 1), 
                   random.randint(0, self.grid_height - 1))
            if food not in self.snake:
                return food
    
    def get_state(self):
        """Get enhanced game state for learning."""
        head = self.snake[0]
        
        # Enhanced snake vision (8 directions with distance)
        vision = []
        for dx, dy in [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]:
            distance = 0
            x, y = head[0], head[1]
            
            while True:
                x += dx
                y += dy
                distance += 1
                
                if (x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height or 
                    (x, y) in self.snake):
                    break
                
                if distance > 5:  # Cap distance at 5
                    break
            
            # Normalize distance to [0, 1]
            vision.append(min(distance / 5.0, 1.0))
        
        # Current direction (one-hot encoded)
        direction = [0, 0, 0, 0]  # Up, Right, Down, Left
        if self.direction == [0, -1]: direction[0] = 1
        elif self.direction == [1, 0]: direction[1] = 1
        elif self.direction == [0, 1]: direction[2] = 1
        elif self.direction == [-1, 0]: direction[3] = 1
        
        # Food direction and distance
        dx = self.food[0] - head[0]
        dy = self.food[1] - head[1]
        food_distance = abs(dx) + abs(dy)
        
        # Normalize food distance
        max_distance = self.grid_width + self.grid_height
        normalized_food_distance = food_distance / max_distance
        
        # Food direction (normalized)
        food_dir_x = dx / max(self.grid_width, 1)
        food_dir_y = dy / max(self.grid_height, 1)
        
        # Combine all state information
        state = vision + direction + [normalized_food_distance, food_dir_x, food_dir_y]
        return tuple(state)
    
    def get_action(self, state):
        """Get action using epsilon-greedy policy with improvement."""
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        
        # Get Q-values for current state
        if state not in self.q_table:
            self.q_table[state] = np.zeros(4)
        
        # Add small random noise to break ties
        q_values = self.q_table[state] + np.random.random(4) * 0.01
        return np.argmax(q_values)
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using improved Q-learning."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(4)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(4)
        
        # Q-learning update rule
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def get_reward(self, old_head, new_head, ate_food, steps_taken):
        """Calculate improved reward for the action."""
        if self.game_over:
            return -50  # Reduced death penalty
        
        if ate_food:
            return 100  # Increased food reward
        
        # Distance-based reward with step penalty
        old_dist = abs(old_head[0] - self.food[0]) + abs(old_head[1] - self.food[1])
        new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
        
        distance_reward = 0
        if new_dist < old_dist:
            distance_reward = 2  # Moving towards food
        elif new_dist > old_dist:
            distance_reward = -1  # Moving away from food
        
        # Step penalty to encourage efficiency
        step_penalty = -0.1
        
        return distance_reward + step_penalty
    
    def step(self, action):
        """Take one step in the game."""
        if self.game_over:
            return False
        
        # Convert action to direction
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left
        new_direction = list(directions[action])
        
        # Don't allow reverse direction
        if (new_direction[0] != -self.direction[0] or 
            new_direction[1] != -self.direction[1]):
            self.direction = new_direction
        
        # Move snake
        old_head = self.snake[0]
        new_head = (old_head[0] + self.direction[0], old_head[1] + self.direction[1])
        
        # Check collision
        if (new_head[0] < 0 or new_head[0] >= self.grid_width or
            new_head[1] < 0 or new_head[1] >= self.grid_height or
            new_head in self.snake):
            self.game_over = True
            return False
        
        # Check food
        ate_food = False
        if new_head == self.food:
            ate_food = True
            self.food = self.generate_food()
            self.score += 1
        else:
            self.snake.pop()
        
        self.snake.insert(0, new_head)
        self.steps += 1
        
        # Check if stuck
        if self.steps > self.max_steps:
            self.game_over = True
        
        return ate_food
    
    def run_episode(self):
        """Run one complete episode with improved learning."""
        self.reset_game()
        total_reward = 0
        episode_states = []
        
        while not self.game_over:
            state = self.get_state()
            action = self.get_action(state)
            
            # Take action
            ate_food = self.step(action)
            new_state = self.get_state()
            
            # Calculate reward
            reward = self.get_reward(self.snake[1] if len(self.snake) > 1 else self.snake[0], 
                                   self.snake[0], ate_food, self.steps)
            total_reward += reward
            
            # Update Q-values
            self.update_q_value(state, action, reward, new_state)
            
            # Store state for analysis
            episode_states.append({
                'state': state,
                'action': action,
                'reward': reward,
                'score': self.score,
                'steps': self.steps
            })
            
            # Small delay for visualization
            time.sleep(0.03)  # Faster episodes
        
        return total_reward, episode_states
    
    def start_learning(self):
        """Start the learning process in a separate thread."""
        if not self.learning_active:
            self.learning_active = True
            self.learning_thread = threading.Thread(target=self.learning_loop)
            self.learning_thread.daemon = True
            self.learning_thread.start()
    
    def stop_learning(self):
        """Stop the learning process."""
        self.learning_active = False
    
    def learning_loop(self):
        """Main learning loop with improvements."""
        while self.learning_active:
            reward, episode_states = self.run_episode()
            self.episode += 1
            self.total_score += self.score
            self.episode_scores.append(self.score)
            self.episode_lengths.append(self.steps)
            
            if self.score > self.best_score:
                self.best_score = self.score
                self.consecutive_zeros = 0
            elif self.score == 0:
                self.consecutive_zeros += 1
            
            # Check if learning is stalled
            if self.consecutive_zeros > 20:
                self.learning_stalled = True
                # Increase exploration temporarily
                self.epsilon = min(0.8, self.epsilon * 1.1)
            else:
                self.learning_stalled = False
            
            # Update learning progress
            avg_score = np.mean(self.episode_scores[-50:]) if self.episode_scores else 0
            avg_length = np.mean(self.episode_lengths[-50:]) if self.episode_lengths else 0
            
            self.learning_progress.append({
                'episode': self.episode,
                'score': self.score,
                'avg_score': avg_score,
                'best_score': self.best_score,
                'epsilon': self.epsilon,
                'avg_length': avg_length,
                'total_reward': reward,
                'learning_stalled': self.learning_stalled
            })
            
            # Adaptive epsilon decay
            if self.score > 0:
                self.epsilon = max(0.05, self.epsilon * self.epsilon_decay)
            else:
                # Slower decay when not improving
                self.epsilon = max(0.05, self.epsilon * 0.9999)
            
            # Small delay between episodes
            time.sleep(0.05)
    
    def draw(self):
        """Draw the game with enhanced information."""
        self.screen.fill(self.BLACK)
        
        # Draw grid
        for x in range(0, self.width, self.grid_size):
            pygame.draw.line(self.screen, (50, 50, 50), (x, 0), (x, self.height))
        for y in range(0, self.height, self.grid_size):
            pygame.draw.line(self.screen, (50, 50, 50), (0, y), (self.width, y))
        
        # Draw snake with gradient colors
        for i, segment in enumerate(self.snake):
            # Create gradient from head to tail
            if i == 0:
                color = self.GREEN  # Head
            else:
                # Gradient from blue to purple
                ratio = i / max(len(self.snake), 1)
                color = (
                    int(self.BLUE[0] * (1 - ratio) + self.PURPLE[0] * ratio),
                    int(self.BLUE[1] * (1 - ratio) + self.PURPLE[1] * ratio),
                    int(self.BLUE[2] * (1 - ratio) + self.PURPLE[2] * ratio)
                )
            
            rect = pygame.Rect(segment[0] * self.grid_size, segment[1] * self.grid_size,
                             self.grid_size, self.grid_size)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.WHITE, rect, 2)
        
        # Draw food with pulsing effect
        pulse = abs(np.sin(time.time() * 5)) * 0.3 + 0.7
        food_color = (
            int(self.RED[0] * pulse),
            int(self.RED[1] * pulse),
            int(self.RED[2] * pulse)
        )
        food_rect = pygame.Rect(self.food[0] * self.grid_size, self.food[1] * self.grid_size,
                               self.grid_size, self.grid_size)
        pygame.draw.rect(self.screen, food_color, food_rect)
        pygame.draw.rect(self.screen, self.WHITE, food_rect, 2)
        
        # Draw learning information
        if self.show_learning:
            self.draw_learning_info()
        
        # Draw Q-values if requested
        if self.show_q_values:
            self.draw_q_values()
        
        # Draw state information if requested
        if self.show_state_info:
            self.draw_state_info()
        
        pygame.display.flip()
    
    def draw_learning_info(self):
        """Draw enhanced learning progress information."""
        # Episode info
        episode_text = self.font.render(f"Episode: {self.episode}", True, self.WHITE)
        self.screen.blit(episode_text, (10, 10))
        
        score_text = self.font.render(f"Score: {self.score}", True, self.WHITE)
        self.screen.blit(score_text, (10, 50))
        
        best_text = self.font.render(f"Best: {self.best_score}", True, self.WHITE)
        self.screen.blit(best_text, (10, 90))
        
        # Learning stats
        if self.learning_progress:
            recent = self.learning_progress[-1]
            avg_text = self.font.render(f"Avg (50): {recent['avg_score']:.1f}", True, self.YELLOW)
            self.screen.blit(avg_text, (10, 130))
            
            epsilon_text = self.font.render(f"ε: {recent['epsilon']:.3f}", True, self.PURPLE)
            self.screen.blit(epsilon_text, (10, 170))
            
            # Show if learning is stalled
            if recent.get('learning_stalled', False):
                stall_text = self.font.render("LEARNING STALLED!", True, self.ORANGE)
                self.screen.blit(stall_text, (10, 210))
            else:
                stall_text = self.font.render("Learning Active", True, self.GREEN)
                self.screen.blit(stall_text, (10, 210))
        
        # Learning status
        status_color = self.GREEN if self.learning_active else self.RED
        status_text = self.font.render("LEARNING" if self.learning_active else "PAUSED", 
                                     True, status_color)
        self.screen.blit(status_text, (10, 250))
        
        # Instructions
        instructions = [
            "SPACE: Start/Stop Learning",
            "R: Reset Game",
            "Q: Toggle Q-values",
            "S: Toggle State Info",
            "ESC: Quit"
        ]
        
        for i, instruction in enumerate(instructions):
            inst_text = self.small_font.render(instruction, True, self.WHITE)
            self.screen.blit(inst_text, (self.width - 250, 10 + i * 25))
    
    def draw_q_values(self):
        """Draw Q-values for current state."""
        if self.game_over:
            return
        
        state = self.get_state()
        if state in self.q_table:
            q_values = self.q_table[state]
            actions = ["UP", "RIGHT", "DOWN", "LEFT"]
            
            # Draw Q-values in top-right corner
            for i, (action, q_val) in enumerate(zip(actions, q_values)):
                color = self.YELLOW if q_val == max(q_values) else self.WHITE
                q_text = self.small_font.render(f"{action}: {q_val:.3f}", True, color)
                self.screen.blit(q_text, (self.width - 200, 200 + i * 25))
    
    def draw_state_info(self):
        """Draw current state information."""
        if self.game_over:
            return
        
        state = self.get_state()
        
        # Draw state breakdown
        info_texts = [
            f"State Size: {len(state)}",
            f"Vision: {state[:8]}",
            f"Direction: {state[8:12]}",
            f"Food Info: {state[12:]}"
        ]
        
        for i, text in enumerate(info_texts):
            info_text = self.small_font.render(text, True, self.ORANGE)
            self.screen.blit(info_text, (10, 300 + i * 20))
    
    def run(self):
        """Main game loop."""
        running = True
        self.start_learning()
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        if self.learning_active:
                            self.stop_learning()
                        else:
                            self.start_learning()
                    elif event.key == pygame.K_r:
                        self.reset_game()
                    elif event.key == pygame.K_q:
                        self.show_q_values = not self.show_q_values
                    elif event.key == pygame.K_s:
                        self.show_state_info = not self.show_state_info
            
            self.draw()
            self.clock.tick(60)
        
        self.stop_learning()
        pygame.quit()
        
        # Save learning progress
        self.save_learning_data()
    
    def save_learning_data(self):
        """Save learning progress to file."""
        data = {
            'episode': self.episode,
            'best_score': self.best_score,
            'total_score': self.total_score,
            'learning_progress': self.learning_progress,
            'episode_scores': self.episode_scores,
            'episode_lengths': self.episode_lengths,
            'q_table_size': len(self.q_table),
            'final_epsilon': self.epsilon
        }
        
        try:
            with open('testing/games/quark_snake_improved_learning_data.json', 'w') as f:
                json.dump(data, f, indent=2)
            print(f"💾 Learning data saved! Quark learned for {self.episode} episodes")
            print(f"🏆 Best score achieved: {self.best_score}")
            print(f"🧠 Q-table size: {len(self.q_table)} states")
            print(f"📊 Final exploration rate: {self.epsilon:.3f}")
            
            if self.learning_progress:
                recent = self.learning_progress[-1]
                print(f"📈 Average score (last 50): {recent['avg_score']:.2f}")
                print(f"⏱️  Average episode length: {recent['avg_length']:.1f} steps")
                
        except Exception as e:
            print(f"❌ Failed to save learning data: {e}")

def main():
    """Main function to run the game."""
    print("🐍 Starting Quark Snake Game - Advanced Learning Demo!")
    print("🧠 Quark will learn to play Snake using improved reinforcement learning")
    print("🎮 Watch as Quark improves its strategy in real-time!")
    print("📊 Enhanced learning progress and state information displayed live")
    print("\n🎯 Controls:")
    print("   SPACE: Start/Stop Learning")
    print("   R: Reset Game")
    print("   Q: Toggle Q-values display")
    print("   S: Toggle State Information")
    print("   ESC: Quit")
    print("\n🚀 Starting game...")
    
    game = ImprovedSnakeGame()
    game.run()

if __name__ == "__main__":
    main()
