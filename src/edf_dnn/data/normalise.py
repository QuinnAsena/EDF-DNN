"""Save and load the fitted scaler, so every use of a model scales its inputs the same way.

Contents
--------
- save_scaler : write a fitted scaler into a run folder.
- load_scaler : read it back, e.g. in predict.py or a report.

The scaling itself is sklearn's MinMaxScaler, as in geocomp-2025. Neural
networks train poorly when inputs have very different ranges (elevation in
metres next to forest fractions of 0-1), so every predictor is mapped onto
0-1: (value - min) / (max - min). Used like this:

    scaler = MinMaxScaler().fit(training[variables])     # training pixels only
    training_values = scaler.transform(training[variables])
    validation_values = scaler.transform(validation[variables])

- Fit on the training pixels only. Fitted on all pixels, the min and max
  would leak information about the validation and test pixels into training.
- Fit on the table's columns (a DataFrame), not a bare numpy array. The
  scaler then records the variable names, and transform() refuses a table
  whose columns differ or come in a different order. Transform DataFrames
  too: given a bare numpy array, it only warns and scales by position.
- Values outside the training range scale to below 0 or above 1 (a hotter
  future, say). The network still runs, but its predictions there are
  extrapolation, so it is worth knowing how much of the future grid lies
  outside the training range.
"""

import pickle
from pathlib import Path

# One file name for the whole project: run.py writes it, predict.py reads it
SCALER_FILE = "scaler.pkl"


def save_scaler(scaler, run_dir):
    """Write a fitted scaler to run_dir/scaler.pkl. Like saveRDS() in R."""
    # "wb" = write binary (a pickle is not text). The with-block closes the
    # file when done, even after an error, like on.exit(close(con)) in R.
    with open(Path(run_dir) / SCALER_FILE, "wb") as file:
        pickle.dump(scaler, file)


def load_scaler(run_dir):
    """Read the scaler saved by save_scaler. Like readRDS() in R.

    Only load pickles this project wrote: unpickling can run code.
    """
    with open(Path(run_dir) / SCALER_FILE, "rb") as file:   # "rb" = read binary
        return pickle.load(file)
