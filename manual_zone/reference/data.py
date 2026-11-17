import os
import torch
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# The 25 predictor columns used across all model variants (order matters for
# consistency with the saved MinMaxScaler and future-climate CSV)
PREDICTORS = [
    "evergreen", "decid", "mixed", "forest", "wetland",
    "unburnable", "tcc", "frac_Critical", "frac_Full",
    "frac_Modified", "ysb_mean", "ysb_min", "ysb_max",
    "elevation_1km", "slope_1km", "TPI_1km", "aspect_1km",
    "flowacc_1km", "twi_1km", "fwi_summer_mean", "fwi_summer_max",
    "fwi_summer_sum", "fwi_annual_mean", "fwi_annual_max", "fwi_annual_sum"
]


def load_historical(data_dir):
    """Load tabular training data; return predictors X and binary target y."""
    df = pd.read_csv(os.path.join(data_dir, "data_historical_allvars.csv"))
    return df[PREDICTORS], df["burned01"]


def load_future(data_dir):
    """Load CMIP6 NorESM2-MM SSP3.70 future-scenario predictor data."""
    df = pd.read_csv(os.path.join(data_dir, "data_cmip6_fwivars_NorESM2-MMssp370.csv"))
    return df[PREDICTORS]


def split(X, y, test_size=0.2, random_state=1984):
    """80/20 train/validation split with fixed random seed for reproducibility."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def split_holdout(X, y, test_size=0.1, random_state=1984):
    """Split off a held-out test set before cross-validation begins.
    Stratified to preserve the burned/unburned class ratio in the test set.
    The returned train_pool is the only data seen during CV and model selection;
    X_test/y_test are used once at the very end for unbiased evaluation."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def downsample(X_train, y_train, random_state=1984):
    """Randomly discard majority-class (unburned) rows until classes are balanced.
    Only applied to the training set — validation keeps the real-world ratio."""
    burned_idx = y_train[y_train == 1].index
    # Sample the same number of unburned rows as there are burned rows
    unburned_downsampled_idx = (
        y_train[y_train == 0].index.to_series()
        .sample(n=len(burned_idx), random_state=random_state)
    )
    balanced_idx = burned_idx.union(unburned_downsampled_idx)
    # Shuffle so burned/unburned rows aren't ordered by class
    X_balanced = X_train.loc[balanced_idx].sample(frac=1, random_state=random_state)
    y_balanced = y_train.loc[X_balanced.index]
    return X_balanced, y_balanced


def pos_weight(y_train):
    """Compute the positive-class weight for BCEWithLogitsLoss.
    Returns (num_unburned / num_burned) as a tensor — tells the loss function
    to penalise missed burns more heavily than missed non-burns."""
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    return torch.tensor(num_neg / num_pos, dtype=torch.float32)


def fit_scaler(X_train, out_dir):
    """Fit a MinMaxScaler on the training data, scale it, and save the fitted
    scaler to out_dir/scaler.pkl (needed later to scale future-climate data
    with the same parameters)."""
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    return scaler, X_train_scaled


def to_tensors(X_train_scaled, y_train, X_val_scaled, y_val, device):
    """Convert numpy arrays / pandas Series to PyTorch float tensors on `device`."""
    return (
        torch.FloatTensor(X_train_scaled).to(device),
        torch.FloatTensor(y_train.values).to(device),
        torch.FloatTensor(X_val_scaled).to(device),
        torch.FloatTensor(y_val.values).to(device),
    )
