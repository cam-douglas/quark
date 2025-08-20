#!/usr/bin/env python3
"""
Test file for autonomous editing - this file is NOT protected
Simple neural network implementation for testing
"""

def create_neural_network(layers):
    """Create a simple neural network"""
    network = []
    for i in range(len(layers) - 1):
        layer = {
            'weights': [[0.1 for _ in range(layers[i+1])] for _ in range(layers[i])],
            'biases': [0.1 for _ in range(layers[i+1])]
        }
        network.append(layer)
    return network

def forward_propagate(network, inputs):
    """Forward propagation through the network"""
    current_inputs = inputs
    for layer in network:
        new_inputs = []
        for i in range(len(layer['weights'][0])):
            sum_val = 0
            for j in range(len(current_inputs)):
                sum_val += current_inputs[j] * layer['weights'][j][i]
            new_inputs.append(sum_val + layer['biases'][i])
        current_inputs = new_inputs
    return current_inputs

def train_network(network, training_data, learning_rate=0.1):
    """Train the network using backpropagation"""
    for data in training_data:
        inputs = data[0]
        targets = data[1]
        
        # Forward pass
        outputs = forward_propagate(network, inputs)
        
        # Calculate error
        errors = [targets[i] - outputs[i] for i in range(len(targets))]
        
        # Backpropagate and update weights
        for i in range(len(network)):
            layer = network[i]
            for j in range(len(layer['weights'])):
                for k in range(len(layer['weights'][j])):
                    layer['weights'][j][k] += learning_rate * errors[k] * inputs[j]
                    layer['biases'][k] += learning_rate * errors[k]

def main():
    """Main function to test the neural network"""
    print("Simple Neural Network Test")
    print("=" * 30)
    
    # Create a simple network
    network = create_neural_network([2, 3, 1])
    
    # Training data (XOR problem)
    training_data = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0])
    ]
    
    # Train the network
    print("Training network...")
    for epoch in range(1000):
        train_network(network, training_data)
    
    # Test the network
    print("Testing network...")
    for inputs, expected in training_data:
        output = forward_propagate(network, inputs)
        print(f"Input: {inputs}, Expected: {expected}, Output: {[round(x, 3) for x in output]}")

if __name__ == "__main__":
    main()
