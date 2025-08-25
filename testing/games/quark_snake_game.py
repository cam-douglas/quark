#!/usr/bin/env python3
"""
Quark Snake Game - Live Learning Demonstration

A Snake game where Quark learns to play using reinforcement learning.
Shows real-time learning progress and neural network adaptation.
"""

import pygame
import numpy as np
import random
import time
from collections import deque
import threading
import json

class SnakeGame:
    """Snake game environment for Quark to learn."""
    
    def __init__(self, width=800, height=600, grid_size=20):
        pygame.init()
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.grid_width = width // grid_size
        self.grid_height = height // grid_size
        
        # Pygame setup
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Quark Snake Game - Learning Live!")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
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
        
        # Learning parameters
        self.epsilon = 0.1  # Exploration rate
        self.learning_rate = 0.001
        self.gamma = 0.95  # Discount factor
        
        # Neural network (simple Q-learning table for now)
        self.q_table = {}
        self.state_size = 11  # Snake vision + direction + food direction
        self.action_size = 4  # Up, Down, Left, Right
        
        # Statistics
        self.episode = 0
        self.total_score = 0
        self.best_score = 0
        self.learning_progress = []
        self.episode_scores = []
        
        # Learning thread
        self.learning_active = False
        self.learning_thread = None
        
        # Display info
        self.show_learning = True
        self.show_q_values = False
        
    def reset_game(self):
        """Reset the game to initial state."""
        self.snake = [(self.grid_width // 2, self.grid_height // 2)]
        self.direction = [1, 0]  # Start moving right
        self.food = self.generate_food()
        self.score = 0
        self.steps = 0
        self.max_steps = self.grid_width * self.grid_height * 2
        self.game_over = False
        
    def generate_food(self):
        """Generate food at random position."""
        while True:
            food = (random.randint(0, self.grid_width - 1), 
                   random.randint(0, self.grid_height - 1))
            if food not in self.snake:
                return food
    
    def get_state(self):
        """Get current game state for learning."""
        head = self.snake[0]
        
        # Snake vision (8 directions)
        vision = []
        for dx, dy in [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]:
            x, y = head[0] + dx, head[1] + dy
            if (x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height or 
                (x, y) in self.snake):
                vision.append(1)  # Wall or snake
            else:
                vision.append(0)  # Empty space
        
        # Current direction
        direction = [0, 0, 0, 0]  # Up, Right, Down, Left
        if self.direction == [0, -1]: direction[0] = 1
        elif self.direction == [1, 0]: direction[1] = 1
        elif self.direction == [0, 1]: direction[2] = 1
        elif self.direction == [-1, 0]: direction[3] = 1
        
        # Food direction
        food_dir = [0, 0, 0, 0]  # Up, Right, Down, Left
        dx = self.food[0] - head[0]
        dy = self.food[1] - head[1]
        if abs(dx) > abs(dy):
            if dx > 0: food_dir[1] = 1  # Right
            else: food_dir[3] = 1        # Left
        else:
            if dy > 0: food_dir[2] = 1  # Down
            else: food_dir[0] = 1        # Up
        
        return tuple(vision + direction + food_dir)
    
    def get_action(self, state):
        """Get action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        
        # Get Q-values for current state
        if state not in self.q_table:
            self.q_table[state] = np.zeros(4)
        
        return np.argmax(self.q_table[state])
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using Q-learning."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(4)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(4)
        
        # Q-learning update rule
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def get_reward(self, old_head, new_head, ate_food):
        """Calculate reward for the action."""
        if self.game_over:
            return -100  # Death penalty
        
        if ate_food:
            return 50  # Food reward
        
        # Distance-based reward
        old_dist = abs(old_head[0] - self.food[0]) + abs(old_head[1] - self.food[1])
        new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
        
        if new_dist < old_dist:
            return 1  # Moving towards food
        elif new_dist > old_dist:
            return -1  # Moving away from food
        else:
            return 0  # No change in distance
    
    def step(self, action):
        """Take one step in the game."""
        if self.game_over:
            return
        
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
            return
        
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
        """Run one complete episode."""
        self.reset_game()
        total_reward = 0
        
        while not self.game_over:
            state = self.get_state()
            action = self.get_action(state)
            
            # Take action
            ate_food = self.step(action)
            new_state = self.get_state()
            
            # Calculate reward
            reward = self.get_reward(self.snake[1] if len(self.snake) > 1 else self.snake[0], 
                                   self.snake[0], ate_food)
            total_reward += reward
            
            # Update Q-values
            self.update_q_value(state, action, reward, new_state)
            
            # Small delay for visualization
            time.sleep(0.05)
        
        return total_reward
    
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
        """Main learning loop."""
        while self.learning_active:
            reward = self.run_episode()
            self.episode += 1
            self.total_score += self.score
            self.episode_scores.append(self.score)
            
            if self.score > self.best_score:
                self.best_score = self.score
            
            # Update learning progress
            avg_score = np.mean(self.episode_scores[-100:]) if self.episode_scores else 0
            self.learning_progress.append({
                'episode': self.episode,
                'score': self.score,
                'avg_score': avg_score,
                'best_score': self.best_score,
                'epsilon': self.epsilon
            })
            
            # Decay epsilon
            self.epsilon = max(0.01, self.epsilon * 0.9995)
            
            # Small delay between episodes
            time.sleep(0.1)
    
    def draw(self):
        """Draw the game."""
        self.screen.fill(self.BLACK)
        
        # Draw grid
        for x in range(0, self.width, self.grid_size):
            pygame.draw.line(self.screen, (50, 50, 50), (x, 0), (x, self.height))
        for y in range(0, self.height, self.grid_size):
            pygame.draw.line(self.screen, (50, 50, 50), (0, y), (self.width, y))
        
        # Draw snake
        for i, segment in enumerate(self.snake):
            color = self.GREEN if i == 0 else self.BLUE
            rect = pygame.Rect(segment[0] * self.grid_size, segment[1] * self.grid_size,
                             self.grid_size, self.grid_size)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.WHITE, rect, 2)
        
        # Draw food
        food_rect = pygame.Rect(self.food[0] * self.grid_size, self.food[1] * self.grid_size,
                               self.grid_size, self.grid_size)
        pygame.draw.rect(self.screen, self.RED, food_rect)
        pygame.draw.rect(self.screen, self.WHITE, food_rect, 2)
        
        # Draw learning information
        if self.show_learning:
            self.draw_learning_info()
        
        # Draw Q-values if requested
        if self.show_q_values:
            self.draw_q_values()
        
        pygame.display.flip()
    
    def draw_learning_info(self):
        """Draw learning progress information."""
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
            avg_text = self.font.render(f"Avg (100): {recent['avg_score']:.1f}", True, self.YELLOW)
            self.screen.blit(avg_text, (10, 130))
            
            epsilon_text = self.font.render(f"ε: {recent['epsilon']:.3f}", True, self.PURPLE)
            self.screen.blit(epsilon_text, (10, 170))
        
        # Learning status
        status_color = self.GREEN if self.learning_active else self.RED
        status_text = self.font.render("LEARNING" if self.learning_active else "PAUSED", 
                                     True, status_color)
        self.screen.blit(status_text, (10, 210))
        
        # Instructions
        instructions = [
            "SPACE: Start/Stop Learning",
            "R: Reset Game",
            "Q: Toggle Q-values",
            "ESC: Quit"
        ]
        
        for i, instruction in enumerate(instructions):
            inst_text = pygame.font.Font(None, 24).render(instruction, True, self.WHITE)
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
                q_text = pygame.font.Font(None, 24).render(f"{action}: {q_val:.3f}", True, color)
                self.screen.blit(q_text, (self.width - 200, 200 + i * 25))
    
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
            'q_table_size': len(self.q_table)
        }
        
        try:
            with open('testing/games/quark_snake_learning_data.json', 'w') as f:
                json.dump(data, f, indent=2)
            print(f"💾 Learning data saved! Quark learned for {self.episode} episodes")
            print(f"🏆 Best score achieved: {self.best_score}")
            print(f"🧠 Q-table size: {len(self.q_table)} states")
        except Exception as e:
            print(f"❌ Failed to save learning data: {e}")

def main():
    """Main function to run the game."""
    print("🐍 Starting Quark Snake Game - Live Learning Demo!")
    print("🧠 Quark will learn to play Snake using reinforcement learning")
    print("🎮 Watch as Quark improves its strategy in real-time!")
    print("📊 Learning progress and Q-values are displayed live")
    print("\n🎯 Controls:")
    print("   SPACE: Start/Stop Learning")
    print("   R: Reset Game")
    print("   Q: Toggle Q-values display")
    print("   ESC: Quit")
    print("\n🚀 Starting game...")
    
    game = SnakeGame()
    game.run()

if __name__ == "__main__":
    main()
