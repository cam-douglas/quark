import torch
import torch.nn as nn
import random

class MyBrainModel(nn.Module):
    def __init__(self, dropout=0.3):
        super(MyBrainModel, self).__init__()
        self.layer1 = nn.Linear(100, 50)
        self.layer2 = nn.Linear(50, 25)
        self.layer3 = nn.Linear(25, 10)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

def train_model_with_sam(model, lr=0.001):
    """Simple training simulation for optimization"""
    # Simulate training process
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Generate dummy data
    x = torch.randn(32, 100)
    y = torch.randn(32, 10)
    
    # Training loop simulation
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    
    # Return a simulated score (higher is better)
    score = 1.0 / (1.0 + loss.item()) + random.random() * 0.1
    return score
