#!/usr/bin/env python3
"""
Test file for autonomous code editing demonstration
Simple neural network implementation for testing
"""

import numpy as np
from typing import List, Tuple

class SimpleNeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights randomly
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        
        # Initialize biases
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))
    
    def forward(self, X):
        # Forward pass
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def backward(self, X, y, learning_rate=0.1):
        # Backward pass
        m = X.shape[0]
        
        # Calculate gradients
        dz2 = self.a2 - y
        dw2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        dz1 = np.dot(dz2, self.weights2.T) * self.sigmoid_derivative(self.a1)
        dw1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights and biases
        self.weights2 -= learning_rate * dw2
        self.bias2 -= learning_rate * db2
        self.weights1 -= learning_rate * dw1
        self.bias1 -= learning_rate * db1
    
    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Backward pass
            self.backward(X, y)
            
            # Print loss every 100 epochs
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss}")

def main():
    # Create sample data
    X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Create and train neural network
    nn = SimpleNeuralNetwork(3, 4, 1)
    nn.train(X, y, epochs=1000)
    
    # Test the network
    test_input = np.array([[0, 0, 1]])
    prediction = nn.forward(test_input)
    print(f"Prediction for [0, 0, 1]: {prediction[0][0]}")

if __name__ == "__main__":
    main()

