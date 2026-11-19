"""Tests for edf_dnn.data.datasets, on ten made-up pixels.

Run with:  pixi run test
"""

import pandas as pd
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler

from edf_dnn.data.datasets import batch_loader, pixel_dataset

# Ten pixels built so every row can be checked: elevation = 10 x cell_id,
# and the even cell_ids burned
TABLE = pd.DataFrame({"cell_id": range(101, 111)})
TABLE["elevation"] = TABLE["cell_id"] * 10.0
TABLE["burned01"] = (TABLE["cell_id"] % 2 == 0).astype(float)
SCALER = MinMaxScaler().fit(TABLE[["elevation"]])


def make_dataset():
    return pixel_dataset(TABLE, ["elevation"], SCALER)


def test_dataset_holds_every_pixel_scaled():
    features, labels, cell_ids = make_dataset().tensors

    # Scaled exactly as the scaler does it, as float32
    assert features.dtype == torch.float32
    assert torch.equal(features, torch.tensor(SCALER.transform(TABLE[["elevation"]]),
                                              dtype=torch.float32))
    # All ten labels, burned and unburned: nothing filtered
    assert labels.tolist() == TABLE["burned01"].tolist()
    assert cell_ids.tolist() == TABLE["cell_id"].tolist()


def test_dataset_uses_the_given_scaler_not_its_own_table():
    # Validation pixels outside the training range (1010-1100 m). Scaled with
    # the TRAINING scaler they fall outside 0-1; a dataset that refitted the
    # scaler on its own table (data leakage) would give exactly 0 and 1.
    validation = pd.DataFrame({"cell_id": [201, 202], "elevation": [1000.0, 1190.0],
                               "burned01": [0.0, 1.0]})

    features, _, _ = pixel_dataset(validation, ["elevation"], SCALER).tensors

    # (1000 - 1010) / 90 and (1190 - 1010) / 90
    assert features.ravel().tolist() == pytest.approx([-10 / 90, 2.0])


def test_shuffled_batches_keep_each_pixel_together():
    torch.manual_seed(1984)
    for features, labels, cell_ids in batch_loader(make_dataset(), batch_size=4, shuffle=True):
        # Undo the scaling: each elevation must be 10 x its own cell_id,
        # and each label must say whether its own cell_id is even
        elevations = SCALER.inverse_transform(features.numpy()).ravel()
        assert elevations.round().tolist() == (cell_ids * 10).tolist()
        assert labels.tolist() == (cell_ids % 2 == 0).float().tolist()


def test_shuffled_epoch_covers_every_pixel_once():
    torch.manual_seed(1984)
    loader = batch_loader(make_dataset(), batch_size=4, shuffle=True)

    epoch_orders = []
    for _ in range(2):
        batches = [cell_ids.tolist() for _, _, cell_ids in loader]
        # 4 + 4 + 2: the last, shorter batch is kept
        assert [len(batch) for batch in batches] == [4, 4, 2]
        order = [cell_id for batch in batches for cell_id in batch]
        # every pixel exactly once per epoch, in a shuffled order
        assert sorted(order) == list(range(101, 111))
        assert order != list(range(101, 111))
        epoch_orders.append(order)

    # a new order every epoch
    assert epoch_orders[0] != epoch_orders[1]


def test_same_seed_gives_the_same_shuffle():
    orders = []
    for _ in range(2):
        torch.manual_seed(1984)
        loader = batch_loader(make_dataset(), batch_size=4, shuffle=True)
        orders.append([cell_ids.tolist() for _, _, cell_ids in loader])

    assert orders[0] == orders[1]


def test_evaluation_batches_keep_table_order():
    loader = batch_loader(make_dataset(), batch_size=4, shuffle=False)

    batches = [cell_ids.tolist() for _, _, cell_ids in loader]

    assert batches == [[101, 102, 103, 104], [105, 106, 107, 108], [109, 110]]
