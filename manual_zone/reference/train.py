import os
import pickle
from itertools import product

import numpy as np
import torch
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

# Default hyperparameter grid — same as the original experiments
HID_DIMS = [128, 256]
LRS      = [1e-3, 5e-4, 1e-4]
DROPOUTS = [0.1, 0.2]
EPOCHS   = 1000
PATIENCE = 50   # epochs without val_loss improvement before stopping early

# Full-scale grid for future runs (swap in when ready)
# HID_DIMS = [64, 128, 256, 512]
# LRS      = [1e-3, 5e-4, 1e-4, 1e-5]
# DROPOUTS = [0.1, 0.2, 0.3]
# EPOCHS   = 10000
# PATIENCE = 100


def run_grid_search(model_cls, X_train_t, y_train_t, X_val_t, y_val_t,
                    criterion_fn, out_dir, ckpt_dir, device,
                    hid_dims=HID_DIMS, lrs=LRS, dropouts=DROPOUTS, epochs=EPOCHS,
                    patience=PATIENCE, random_seed=1984):
    """Train model_cls for every combination of hidden_size x lr x dropout.

    model_cls    : an nn.Module subclass (e.g. FeedforwardSingle) — instantiated
                   fresh for each hyperparameter combination
    criterion_fn : a PyTorch loss function (BCEWithLogitsLoss, optionally with
                   pos_weight for the weighted variant)
    Results are written to out_dir/gridsearch_results.csv and per-run history
    pickles; checkpoints are saved to ckpt_dir/.
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    results = []   # will accumulate one dict per hyperparameter combination

    for hidden_size, lr, dropout in product(hid_dims, lrs, dropouts):
        print(f"\nhidden_size={hidden_size}, lr={lr}, dropout={dropout}")

        # Fixed seed so each config starts from the same weight initialisation,
        # making results reproducible across runs
        torch.manual_seed(random_seed)

        # Fresh model and optimizer for each combination
        model     = model_cls(X_train_t.shape[1], hidden_size, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Lists to record per-epoch metrics
        loss_train, loss_val, acc_train, acc_val = [], [], [], []

        # Early-stopping state: track the best validation loss seen so far and
        # save a copy of the weights at that point
        best_val_loss    = float('inf')
        patience_counter = 0
        best_state_dict  = None

        for epoch in range(epochs):
            # --- Training step ---
            model.train()
            optimizer.zero_grad()                                   # clear previous gradients
            logits_train = model(X_train_t)                         # forward pass
            loss = criterion_fn(logits_train.squeeze(), y_train_t)  # compute loss
            loss.backward()                                         # backpropagate
            optimizer.step()                                        # update weights

            loss_train.append(loss.item())
            train_preds = (torch.sigmoid(logits_train) > 0.5).float()
            acc_train.append((train_preds.squeeze() == y_train_t).float().mean().item())

            # --- Validation step (no gradient tracking needed) ---
            model.eval()
            with torch.no_grad():
                logits_val = model(X_val_t)
                loss_val.append(criterion_fn(logits_val.squeeze(), y_val_t).item())
                probs_val  = torch.sigmoid(logits_val)
                preds_val  = (probs_val > 0.5).float()
                acc_val.append((preds_val.squeeze() == y_val_t).float().mean().item())

            if epoch % 500 == 0:
                print(f"  epoch {epoch}: "
                      f"train_loss={loss_train[-1]:.4f}  "
                      f"val_loss={loss_val[-1]:.4f}  "
                      f"val_acc={acc_val[-1]:.4f}")

            # --- Early stopping check ---
            if loss_val[-1] < best_val_loss:
                best_val_loss    = loss_val[-1]
                patience_counter = 0
                # Clone weights so we can restore the best model after stopping
                best_state_dict  = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(no improvement for {patience} epochs)")
                    break

        # --- Post-training evaluation using the best (not final) weights ---
        # This ensures metrics and the saved checkpoint reflect peak performance,
        # not wherever the model happened to be when training ended
        model.load_state_dict(best_state_dict)
        model.eval()
        with torch.no_grad():
            probs_val = torch.sigmoid(model(X_val_t))

        probs_val_np = probs_val.cpu().numpy().squeeze()
        y_val_np     = y_val_t.cpu().numpy().squeeze()
        cm           = confusion_matrix(y_val_np, (probs_val_np > 0.5).astype(float))

        # AUC-ROC: overall discrimination ability across all thresholds
        fpr, tpr, _ = roc_curve(y_val_np, probs_val_np)
        val_auc     = auc(fpr, tpr)

        # AUC-PR: precision-recall tradeoff, more informative than AUC-ROC
        # for imbalanced data because it focuses entirely on the positive class
        pr_precision, pr_recall, _ = precision_recall_curve(y_val_np, probs_val_np)
        val_pr_auc = float(auc(pr_recall, pr_precision))

        # Brier score: mean squared error of predicted probabilities
        # Measures calibration — whether p=0.7 actually corresponds to ~70% burn rate
        # https://doi.org/10.1016/j.ijforecast.2018.11.010: brier score: BS = (p(Y ) − Y )^2
        val_brier = float(np.mean((probs_val_np - y_val_np) ** 2))

        # Recall (sensitivity) and precision at the 0.5 decision threshold,
        # derived from the confusion matrix. Recall is the ecologically important
        # number: what fraction of actual burns did the model detect?
        # https://en.wikipedia.org/wiki/Precision_and_recall
        tp = cm[1, 1]
        fp = cm[0, 1]
        fn = cm[1, 0]
        val_recall    = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        val_precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

        # Save the best weights (not necessarily the final epoch)
        checkpoint_name = f"model_h{hidden_size}_lr{lr}_do{dropout}.pt"
        torch.save(best_state_dict, os.path.join(ckpt_dir, checkpoint_name))

        # Save full training history (loss curves, confusion matrix, ROC data)
        # as a pickle so the .qmd can load and plot without re-training
        run_history = dict(
            hid_dim=hidden_size, lr=lr, dropout=dropout,
            loss_train=loss_train, loss_val=loss_val,
            acc_train=acc_train,   acc_val=acc_val,
            cm=cm,
            fpr=fpr,          tpr=tpr,          roc_auc=val_auc,
            pr_precision=pr_precision, pr_recall=pr_recall, pr_auc=val_pr_auc,
            brier=val_brier,
            recall=val_recall, precision=val_precision,
        )
        history_path = os.path.join(out_dir, f"history_h{hidden_size}_lr{lr}_do{dropout}.pkl")
        with open(history_path, "wb") as f:
            pickle.dump(run_history, f)

        # Append summary row for the CSV
        results.append(dict(
            hid_dim=hidden_size, lr=lr, dropout=dropout,
            val_loss=loss_val[-1],     val_acc=acc_val[-1],
            train_loss=loss_train[-1], train_acc=acc_train[-1],
            val_auc=val_auc, val_pr_auc=val_pr_auc,
            val_brier=val_brier,
            val_recall=val_recall, val_precision=val_precision,
            checkpoint=checkpoint_name,
        ))

    # Write all results to a single CSV for easy comparison across hyperparameters
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(out_dir, "gridsearch_results.csv"), index=False)
    return results_df


# =============================================================================
# Helpers used internally by run_grid_search_cv
# =============================================================================

def _downsample_np(X, y, random_state=1984):
    """Downsample majority class in numpy arrays.
    Mirrors src/data.downsample but works on the numpy arrays produced by
    StratifiedKFold indexing rather than pandas DataFrames."""
    rng = np.random.default_rng(random_state)
    burned_idx         = np.where(y == 1)[0]
    unburned_idx       = np.where(y == 0)[0]
    sampled_unburned   = rng.choice(unburned_idx, size=len(burned_idx), replace=False)
    balanced_idx       = np.concatenate([burned_idx, sampled_unburned])
    rng.shuffle(balanced_idx)
    return X[balanced_idx], y[balanced_idx]


def _pos_weight_np(y, device):
    """Compute BCEWithLogitsLoss pos_weight tensor from a numpy label array."""
    num_neg = (y == 0).sum()
    num_pos = (y == 1).sum()
    return torch.tensor(num_neg / num_pos, dtype=torch.float32).to(device)


# =============================================================================
# K-fold cross-validated grid search
# =============================================================================

def run_grid_search_cv(model_cls, X_train_pool, y_train_pool, X_test, y_test,
                       imbalance, out_dir, ckpt_dir, device,
                       hid_dims=HID_DIMS, lrs=LRS, dropouts=DROPOUTS,
                       epochs=EPOCHS, patience=PATIENCE, k=5, random_seed=1984):
    """K-fold cross-validated grid search with a held-out test set.

    imbalance : "downsample", "pos_weight", or None — applied inside each fold
                so the scaler and class balance are computed from fold training
                data only (no information leakage into the validation fold)
    X_train_pool, y_train_pool : the training pool (pandas DataFrame/Series);
                                 all CV folds are drawn from this set
    X_test, y_test : held-out test set, never seen during CV or model selection

    Workflow:
      1. For each hyperparameter combination, run k stratified folds; each fold
         fits its own scaler and applies the imbalance strategy independently.
         Record mean/std of val metrics across folds.
      2. Select the config with the highest mean val_auc.
      3. Retrain that config on the full pool for exactly mean_stop_epoch epochs
         (the average epoch at which early stopping fired across folds), so no
         second validation split is needed for the final model.
      4. Evaluate the final model on the held-out test set.

    Outputs (all written to out_dir / ckpt_dir):
      gridsearch_results_cv.csv      — one row per config, mean/std columns
      history_cv_h*_lr*_do*.pkl     — per-config, per-fold training histories
      final_test_results.pkl         — test-set metrics + checkpoint filename
      scaler.pkl                     — scaler fitted on the full training pool
      checkpoints/model_h*_lr*_do*.pt — final model weights
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Convert pandas inputs to numpy for StratifiedKFold indexing
    X_pool_np = X_train_pool.values if hasattr(X_train_pool, 'values') else np.array(X_train_pool)
    y_pool_np = y_train_pool.values if hasattr(y_train_pool, 'values') else np.array(y_train_pool)
    X_test_np = X_test.values if hasattr(X_test, 'values') else np.array(X_test)
    y_test_np = y_test.values if hasattr(y_test, 'values') else np.array(y_test)

    # Stratified so each fold preserves the burned/unburned class ratio
    kf      = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_seed)
    results = []

    for hidden_size, lr, dropout in product(hid_dims, lrs, dropouts):
        print(f"\nhidden_size={hidden_size}, lr={lr}, dropout={dropout}")

        # Accumulators for metrics and histories across folds
        fold_val_aucs       = []
        fold_val_pr_aucs    = []
        fold_val_briers     = []
        fold_val_recalls    = []
        fold_val_precisions = []
        fold_stop_epochs    = []
        fold_histories      = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_pool_np, y_pool_np)):
            print(f"  fold {fold_idx + 1}/{k}")

            X_fold_train = X_pool_np[train_idx]
            y_fold_train = y_pool_np[train_idx]
            X_fold_val   = X_pool_np[val_idx]
            y_fold_val   = y_pool_np[val_idx]

            # Apply imbalance strategy to fold training data only (not val)
            if imbalance == "downsample":
                X_fold_train, y_fold_train = _downsample_np(X_fold_train, y_fold_train, random_seed)
                criterion_fn = torch.nn.BCEWithLogitsLoss()
            elif imbalance == "pos_weight":
                criterion_fn = torch.nn.BCEWithLogitsLoss(
                    pos_weight=_pos_weight_np(y_fold_train, device)
                )
            else:
                criterion_fn = torch.nn.BCEWithLogitsLoss()

            # Fit scaler on fold training data; transform val with the same params
            # so no val-set statistics leak into the model's view of training data
            fold_scaler         = MinMaxScaler()
            X_fold_train_scaled = fold_scaler.fit_transform(X_fold_train)
            X_fold_val_scaled   = fold_scaler.transform(X_fold_val)

            # Convert to tensors
            X_train_t = torch.FloatTensor(X_fold_train_scaled).to(device)
            y_train_t = torch.FloatTensor(y_fold_train).to(device)
            X_val_t   = torch.FloatTensor(X_fold_val_scaled).to(device)
            y_val_t   = torch.FloatTensor(y_fold_val).to(device)

            # Fresh model and optimizer; fixed seed for reproducibility
            torch.manual_seed(random_seed)
            model     = model_cls(X_train_t.shape[1], hidden_size, dropout).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            loss_train_fold, loss_val_fold = [], []
            best_val_loss    = float('inf')
            patience_counter = 0
            best_state_dict  = None
            stop_epoch       = epochs - 1   # default: ran to the end

            for epoch in range(epochs):
                # --- Training step ---
                model.train()
                optimizer.zero_grad()
                logits = model(X_train_t)
                loss   = criterion_fn(logits.squeeze(), y_train_t)
                loss.backward()
                optimizer.step()
                loss_train_fold.append(loss.item())

                # --- Validation step ---
                model.eval()
                with torch.no_grad():
                    val_loss = criterion_fn(model(X_val_t).squeeze(), y_val_t).item()
                loss_val_fold.append(val_loss)

                # --- Early stopping ---
                if val_loss < best_val_loss:
                    best_val_loss    = val_loss
                    patience_counter = 0
                    best_state_dict  = {kk: vv.clone() for kk, vv in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        stop_epoch = epoch
                        break

            fold_stop_epochs.append(stop_epoch)

            # Evaluate fold with the best (not final) weights
            model.load_state_dict(best_state_dict)
            model.eval()
            with torch.no_grad():
                probs_val_fold = torch.sigmoid(model(X_val_t)).cpu().numpy().squeeze()

            cm_fold            = confusion_matrix(y_fold_val, (probs_val_fold > 0.5).astype(float))
            fpr_f, tpr_f, _   = roc_curve(y_fold_val, probs_val_fold)
            fold_auc           = auc(fpr_f, tpr_f)
            pr_prec, pr_rec, _ = precision_recall_curve(y_fold_val, probs_val_fold)
            fold_pr_auc        = float(auc(pr_rec, pr_prec))
            fold_brier         = float(np.mean((probs_val_fold - y_fold_val) ** 2))
            tp = cm_fold[1, 1]; fp = cm_fold[0, 1]; fn = cm_fold[1, 0]
            fold_recall    = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            fold_precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

            fold_val_aucs.append(fold_auc)
            fold_val_pr_aucs.append(fold_pr_auc)
            fold_val_briers.append(fold_brier)
            fold_val_recalls.append(fold_recall)
            fold_val_precisions.append(fold_precision)
            fold_histories.append(dict(
                fold=fold_idx,
                loss_train=loss_train_fold, loss_val=loss_val_fold,
                val_auc=fold_auc, val_pr_auc=fold_pr_auc,
                brier=fold_brier, recall=fold_recall, precision=fold_precision,
                stop_epoch=stop_epoch,
            ))

        mean_val_auc       = float(np.mean(fold_val_aucs))
        std_val_auc        = float(np.std(fold_val_aucs))
        mean_val_pr_auc    = float(np.mean(fold_val_pr_aucs))
        mean_val_brier     = float(np.mean(fold_val_briers))
        mean_val_recall    = float(np.mean(fold_val_recalls))
        mean_val_precision = float(np.mean(fold_val_precisions))
        # Mean stop epoch becomes the fixed training duration for the final model
        mean_stop_epoch    = int(round(np.mean(fold_stop_epochs)))

        print(f"  → mean AUC={mean_val_auc:.4f} ± {std_val_auc:.4f}  "
              f"(mean stop epoch={mean_stop_epoch})")

        # Save per-fold histories for this config (loss curves, metrics per fold)
        history_path = os.path.join(out_dir, f"history_cv_h{hidden_size}_lr{lr}_do{dropout}.pkl")
        with open(history_path, "wb") as fh:
            pickle.dump(dict(
                hid_dim=hidden_size, lr=lr, dropout=dropout,
                fold_histories=fold_histories,
                mean_val_auc=mean_val_auc,    std_val_auc=std_val_auc,
                mean_val_pr_auc=mean_val_pr_auc,
                mean_val_brier=mean_val_brier,
                mean_val_recall=mean_val_recall, mean_val_precision=mean_val_precision,
                mean_stop_epoch=mean_stop_epoch,
            ), fh)

        results.append(dict(
            hid_dim=hidden_size, lr=lr, dropout=dropout,
            mean_val_auc=mean_val_auc,    std_val_auc=std_val_auc,
            mean_val_pr_auc=mean_val_pr_auc,
            mean_val_brier=mean_val_brier,
            mean_val_recall=mean_val_recall, mean_val_precision=mean_val_precision,
            mean_stop_epoch=mean_stop_epoch,
        ))

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(out_dir, "gridsearch_results_cv.csv"), index=False)

    # -------------------------------------------------------------------------
    # Final training: retrain best config on the full pool
    # -------------------------------------------------------------------------
    best_row     = results_df.loc[results_df["mean_val_auc"].idxmax()]
    best_hidden  = int(best_row["hid_dim"])
    best_lr      = float(best_row["lr"])
    best_dropout = float(best_row["dropout"])
    final_epochs = int(best_row["mean_stop_epoch"])

    print(f"\nBest config: hidden={best_hidden}, lr={best_lr}, dropout={best_dropout}")
    print(f"Retraining on full pool for {final_epochs} epochs (mean CV stop epoch)")

    X_final_train = X_pool_np.copy()
    y_final_train = y_pool_np.copy()

    if imbalance == "downsample":
        X_final_train, y_final_train = _downsample_np(X_final_train, y_final_train, random_seed)
        final_criterion = torch.nn.BCEWithLogitsLoss()
    elif imbalance == "pos_weight":
        final_criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=_pos_weight_np(y_final_train, device)
        )
    else:
        final_criterion = torch.nn.BCEWithLogitsLoss()

    # Fit the final scaler on the full training pool and save it so QMDs can
    # apply the same transformation to future-climate data at prediction time
    final_scaler   = MinMaxScaler()
    X_final_scaled = final_scaler.fit_transform(X_final_train)
    X_test_scaled  = final_scaler.transform(X_test_np)
    joblib.dump(final_scaler, os.path.join(out_dir, "scaler.pkl"))

    X_final_t = torch.FloatTensor(X_final_scaled).to(device)
    y_final_t = torch.FloatTensor(y_final_train).to(device)
    X_test_t  = torch.FloatTensor(X_test_scaled).to(device)

    torch.manual_seed(random_seed)
    final_model     = model_cls(X_final_t.shape[1], best_hidden, best_dropout).to(device)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_lr)

    for epoch in range(final_epochs):
        final_model.train()
        final_optimizer.zero_grad()
        logits = final_model(X_final_t)
        loss   = final_criterion(logits.squeeze(), y_final_t)
        loss.backward()
        final_optimizer.step()
        if epoch % 500 == 0:
            print(f"  final epoch {epoch}: train_loss={loss.item():.4f}")

    # Evaluate on the held-out test set (only time this set is used)
    final_model.eval()
    with torch.no_grad():
        probs_test = torch.sigmoid(final_model(X_test_t)).cpu().numpy().squeeze()

    test_cm              = confusion_matrix(y_test_np, (probs_test > 0.5).astype(float))
    fpr_t, tpr_t, _     = roc_curve(y_test_np, probs_test)
    test_auc             = float(auc(fpr_t, tpr_t))
    pr_prec_t, pr_rec_t, _ = precision_recall_curve(y_test_np, probs_test)
    test_pr_auc          = float(auc(pr_rec_t, pr_prec_t))
    test_brier           = float(np.mean((probs_test - y_test_np) ** 2))
    tp = test_cm[1, 1]; fp = test_cm[0, 1]; fn = test_cm[1, 0]
    test_recall    = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    test_precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    print(f"\nFinal test set: AUC={test_auc:.4f}  PR-AUC={test_pr_auc:.4f}  "
          f"Brier={test_brier:.4f}  Recall={test_recall:.4f}")

    checkpoint_name = f"model_h{best_hidden}_lr{best_lr}_do{best_dropout}.pt"
    torch.save(final_model.state_dict(), os.path.join(ckpt_dir, checkpoint_name))

    # Save all final-model artefacts so QMDs can load without re-running training
    final_results = dict(
        hid_dim=best_hidden, lr=best_lr, dropout=best_dropout,
        final_epochs=final_epochs,
        test_cm=test_cm,
        test_fpr=fpr_t,   test_tpr=tpr_t,            test_auc=test_auc,
        test_pr_precision=pr_prec_t, test_pr_recall=pr_rec_t, test_pr_auc=test_pr_auc,
        test_brier=test_brier,
        test_recall=test_recall, test_precision=test_precision,
        checkpoint=checkpoint_name,
    )
    with open(os.path.join(out_dir, "final_test_results.pkl"), "wb") as fh:
        pickle.dump(final_results, fh)

    return results_df
