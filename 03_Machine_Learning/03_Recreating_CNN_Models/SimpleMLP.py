"""
Title: SimpleMLP
Description: Implementation of a simple multi-layer perceptron for image classification on handwritten digits.
"""


import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        """
        Constructor for the SimpleMLP class.
        @param input_size: The input size of the MLP.
        @param hidden_size: The hidden size of the MLP.
        @param num_classes: The number of classes in the dataset.
        """
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out