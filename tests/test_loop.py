"""Tests for edf_dnn.train.loop, on 200 made-up pixels with an easy rule to learn.

Run with:  pixi run test
"""

import torch
from torch.utils.data import TensorDataset

from edf_dnn.data.datasets import batch_loader
from edf_dnn.models.mlp import FeedforwardSingle
from edf_dnn.train.loop import choose_device, fit, predict_probabilities, train_one_epoch
from edf_dnn.train.metrics import compute_metrics

CPU = torch.device("cpu")


def make_pixels():
    """200 pixels with two random predictors; a pixel 'burned' if the first is above 0.5."""
    torch.manual_seed(1984)
    features = torch.rand(200, 2)
    burned = (features[:, 0] > 0.5).float()
    cell_ids = torch.arange(1, 201)
    return features, burned, cell_ids


def make_model(dropout=0.0):
    torch.manual_seed(1984)
    model = FeedforwardSingle(input_size=2, hidden_size=8, dropout=dropout)
    return model, torch.optim.Adam(model.parameters(), lr=0.01)


def test_choose_device():
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert choose_device("auto").type == expected
    assert choose_device("cpu").type == "cpu"


def test_training_lowers_the_loss():
    features, burned, cell_ids = make_pixels()
    loader = batch_loader(TensorDataset(features, burned, cell_ids), batch_size=50, shuffle=True)
    model, optimizer = make_model()

    losses = [train_one_epoch(model, loader, torch.nn.BCEWithLogitsLoss(), optimizer, CPU)
              for _ in range(30)]

    assert losses[-1] < losses[0] * 0.5


def test_predict_probabilities_gives_one_row_per_pixel():
    features, burned, cell_ids = make_pixels()
    loader = batch_loader(TensorDataset(features, burned, cell_ids), batch_size=50, shuffle=False)
    # Heavy dropout: predictions are only repeatable if dropout is switched off
    model, _ = make_model(dropout=0.5)

    predictions = predict_probabilities(model, loader, CPU)

    assert list(predictions.columns) == ["cell_id", "label", "probability"]
    assert predictions["cell_id"].tolist() == cell_ids.tolist()
    assert predictions["label"].tolist() == burned.tolist()
    assert predictions["probability"].between(0, 1).all()
    assert predictions.equals(predict_probabilities(model, loader, CPU))


def test_fit_stops_after_patience_and_keeps_the_best_weights():
    features, burned, cell_ids = make_pixels()
    training = batch_loader(TensorDataset(features, burned, cell_ids), batch_size=50, shuffle=True)
    # Validation pixels follow the OPPOSITE rule: the better the model learns
    # the training rule, the worse its validation AUC. So the best epoch is
    # the first, and the last is much worse (AUC 0.89 at epoch 1, 0.004 at 6).
    validation = batch_loader(TensorDataset(features, 1 - burned, cell_ids), batch_size=50,
                              shuffle=False)
    model, optimizer = make_model()

    history, best_epoch = fit(model, training, validation, torch.nn.BCEWithLogitsLoss(),
                              optimizer, CPU, max_epochs=50, patience=5)

    # Stopped 5 epochs after the best one, long before max_epochs
    assert best_epoch == 1
    assert history["epoch"].tolist() == [1, 2, 3, 4, 5, 6]
    # The model now holds epoch 1's weights, not epoch 6's
    predictions = predict_probabilities(model, validation, CPU)
    roc_auc_now = compute_metrics(predictions["label"], predictions["probability"])["roc_auc"]
    assert roc_auc_now == history.loc[0, "validation_roc_auc"]
    assert roc_auc_now > history.loc[5, "validation_roc_auc"]


def test_fit_runs_to_max_epochs_while_improving():
    features, burned, cell_ids = make_pixels()
    training = batch_loader(TensorDataset(features, burned, cell_ids), batch_size=50, shuffle=True)
    validation = batch_loader(TensorDataset(features, burned, cell_ids), batch_size=50,
                              shuffle=False)
    model, optimizer = make_model()

    history, _ = fit(model, training, validation, torch.nn.BCEWithLogitsLoss(),
                     optimizer, CPU, max_epochs=4, patience=100)

    assert len(history) == 4
    # Every validation metric is logged, with a validation_ prefix
    assert {"training_loss", "validation_roc_auc", "validation_log_loss"} <= set(history.columns)
