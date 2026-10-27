"""Multilayer perceptrons, carried over unchanged from geocomp-2025/src/models.py.

Each takes a batch of pixel predictor vectors (batch x variables) and returns
one raw logit per pixel (batch x 1).
"""

import torch
from torch import nn


class FeedforwardSingle(nn.Module):
    """Single hidden-layer MLP: input → hidden → output (1 node, binary logit)."""

    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        # Fully-connected layer from inputs to hidden neurons
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Output layer: hidden neurons → single logit (no sigmoid here;
        # BCEWithLogitsLoss applies sigmoid internally for numerical stability)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        # Dropout randomly zeroes a fraction of neurons each forward pass
        # to reduce overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden = self.relu(self.fc1(x))   # activate hidden layer
        hidden = self.dropout(hidden)      # apply dropout
        return self.fc3(hidden)            # produce raw logit


class FeedforwardTwo(nn.Module):
    """Two hidden-layer MLP with equal-width layers."""

    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)   # first hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # second hidden layer (same width)
        self.fc3 = nn.Linear(hidden_size, 1)            # output logit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden = self.relu(self.fc1(x))
        hidden = self.dropout(hidden)
        hidden = self.relu(self.fc2(hidden))
        hidden = self.dropout(hidden)
        return self.fc3(hidden)


class FeedforwardTapered(nn.Module):
    """Two hidden-layer MLP that narrows: hidden_size → hidden_size//2 → 1.
    Forces the network to compress its representation progressively."""

    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)           # wide first layer
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)     # narrower second layer
        self.fc3 = nn.Linear(hidden_size // 2, 1)               # output logit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden = self.relu(self.fc1(x))
        hidden = self.dropout(hidden)
        hidden = self.relu(self.fc2(hidden))
        hidden = self.dropout(hidden)
        return self.fc3(hidden)
