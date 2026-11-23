"""Multilayer perceptrons (MLPs), carried over from geocomp-2025/src/models.py.

Each takes a batch of pixels' predictor values, shape (batch, variables), and
returns one raw logit per pixel, shape (batch, 1). No sigmoid at the end:
BCEWithLogitsLoss applies it inside the loss during training (numerically
safer), and predict_probabilities applies it when probabilities are needed.

All three share one constructor, (input_size, hidden_size, dropout), so
run.py can build whichever the experiment config names.

Layers were renamed on 2026-11-19 (fc1/fc2/fc3 -> hidden/output layer names);
the architectures are unchanged.
"""

from torch import nn


class FeedforwardSingle(nn.Module):
    """One hidden layer: input -> hidden -> output (one logit)."""

    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()   # sets up the PyTorch machinery (weight tracking etc.)
        # Fully connected layer: every predictor feeds every hidden neuron,
        # each a weighted sum plus a bias (like X %*% W + b in R)
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        # One neuron that combines the hidden neurons into a single logit
        self.output_layer = nn.Linear(hidden_size, 1)
        # ReLU = max(0, x): the bend that lets the network learn non-linear
        # responses; without it, stacked layers collapse into one linear model
        self.relu = nn.ReLU()
        # Dropout switches off a random fraction of hidden neurons on each
        # training pass, so the network can't lean on any single one (less
        # overfitting). Off automatically in evaluation mode (model.eval()).
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        # forward() is what model(features) runs: the path from inputs to logit
        hidden = self.relu(self.hidden_layer(features))   # weighted sums, then the bend
        hidden = self.dropout(hidden)                      # thin out during training
        return self.output_layer(hidden)                   # raw logit


class FeedforwardTwo(nn.Module):
    """Two hidden layers of equal width: input -> hidden -> hidden -> output.

    The second layer combines the first layer's features into new ones, so the
    network can learn interactions between predictors (e.g. dry fuel AND high
    fire weather). In geocomp-2025 this gave the biggest gain: validation AUC
    about 0.80 with one hidden layer, 0.86 with two.
    """

    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.first_hidden_layer = nn.Linear(input_size, hidden_size)
        self.second_hidden_layer = nn.Linear(hidden_size, hidden_size)   # same width
        self.output_layer = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)   # one Dropout object, reused after each layer

    def forward(self, features):
        hidden = self.relu(self.first_hidden_layer(features))
        hidden = self.dropout(hidden)
        hidden = self.relu(self.second_hidden_layer(hidden))   # features of features
        hidden = self.dropout(hidden)
        return self.output_layer(hidden)


class FeedforwardTapered(nn.Module):
    """Two hidden layers that narrow: input -> hidden -> hidden/2 -> output.

    Same idea as FeedforwardTwo, but the second layer has half the neurons,
    forcing the network to squeeze what it learned into fewer features. Fewer
    weights than FeedforwardTwo, which can reduce overfitting. Never trained
    in geocomp-2025; first tried here.
    """

    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.first_hidden_layer = nn.Linear(input_size, hidden_size)
        # // is whole-number division (256 // 2 = 128), like %/% in R
        self.second_hidden_layer = nn.Linear(hidden_size, hidden_size // 2)
        self.output_layer = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        hidden = self.relu(self.first_hidden_layer(features))
        hidden = self.dropout(hidden)
        hidden = self.relu(self.second_hidden_layer(hidden))   # half as many neurons
        hidden = self.dropout(hidden)
        return self.output_layer(hidden)
