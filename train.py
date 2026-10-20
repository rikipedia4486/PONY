"""
Unified PONY training / test script.

Usage:
    python train.py configs/<experiment>.json [--set key=value ...]

Every setting lives in the JSON config (see DEFAULTS below for all keys and their
defaults). --set overrides individual keys from the command line, e.g.
    python train.py configs/mofs.json --set model_size=small --set output_dir=runs/mofs_small
Values are parsed as JSON where possible (numbers, true/false, lists), else as strings.
Keys starting with "_" in the JSON (e.g. "_comment") are ignored.

Set "test_only": true to skip training and only run test-set inference (with and
without TTA) from an existing checkpoint.
"""
print("script initialized successfully, beginning preprocessing...")

import argparse
import json
import math
import os
import time
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from codecarbon import EmissionsTracker

from training_util.training_util import StructureDataset, train_one_epoch, evaluate, evaluate_tta
from data_util.pre_processing import fit_feature_scalers, fit_film_scalers, prepare_data, upsample_collate, check_cell_widths
from data_util.calculate_stats import compute_mae, compute_rmse
from data_util.plotting import plot_learning_curve, parity_plot
from models.pointnet2 import get_model, get_loss

DEFAULTS = {
    # ---- Paths (CSV paths are relative to data_dir unless absolute) ----
    "h5_path":                 "TODO",   # HDF5 with point clouds for train, val, and test structures
    "data_dir":                "/scratch/sk10275/PONY_ML/PONY_refactored_4/datasets",
    "train_csv":               "TODO",
    "val_csv":                 "TODO",
    "test_csv":                "TODO",
    "output_dir":              "TODO",

    # ---- Data ----
    "target":                  "TODO",   # CSV column to regress
    "structure_id_col":        "ID",     # CSV column matching the HDF5 group names
    "task_mode":               "molecule",  # "crystal" (loads lattice, applies PBC) | "molecule" (no PBC)
    "num_atomic_features":     9,        # feature columns after xyz in the H5 point cloud
    "feature_scaler":          "maxabs", # per atomic-feature channel: "maxabs" | "standard" | "none"
    "film_feature_cols":       [],       # per-structure FiLM inputs, e.g. ["density", "volume"]
    "film_scaler":             "standard",
    "center_coordinates":      True,     # subtract each structure's centroid (translational invariance)
    "min_points":              1024,     # minimum point cloud size after upsampling
    "debug_subset":            None,     # int: use only the first N rows of each split (smoke tests)

    # ---- Model ----
    "model_size":              "medium", # SA radii preset: xsmall | small | medium | large | xlarge
    "sa1_radius":              None,     # optional override of the preset (Å)
    "sa2_radius":              None,
    "dropout":                 [0.2, 0.2, 0.3],  # for final FC layers after the SA blocks
    "enable_rotation_aug":     True,     # random SO(3) rotation of every training sample
    "compile":                 True,     # torch.compile the model

    # ---- Training ----
    "batch_size":              128,
    "learning_rate":           3e-4,
    "weight_decay":            5e-4,
    "epochs":                  500,
    "t_max_epochs":            300,      # epoch at which cosine decay reaches eta_min; LR stays there afterwards
    "warmup_epochs":           10,       # linear LR warmup before cosine decay begins
    "eta_min":                 1e-6,
    "early_stopping_patience": 50,       # stop if the val checkpoint metric does not improve for this many epochs
    "checkpoint_metric":       "mae",    # "mae" | "rmse" — metric used to select the best checkpoint
    "n_train_augments":        1,        # rotational augmentation passes per epoch (1 = standard)
    "standardize_target":      False,    # train in z-score space; metrics/CSVs/plots stay in raw units
    "num_workers":             12,

    # ---- Test ----
    "n_tta_rotations":         16,       # rotations averaged at test time (TTA)
    "test_only":               False,    # skip training; evaluate an existing checkpoint
    "checkpoint_path":         None,     # default: <output_dir>/best_model.pth
    "overwrite":               False,    # allow training into an output_dir that already has best_model.pth
}

TASK_MODES = {"crystal": 0, "molecule": 1}
SCALERS = {"maxabs": MaxAbsScaler, "standard": StandardScaler, "none": lambda: None}


# =============================================================================
# Config
# =============================================================================

def _parse_value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def load_config():
    parser = argparse.ArgumentParser(description="Train / test PONY (PointNet++) models.")
    parser.add_argument("config", help="path to a JSON config file")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config key (repeatable)")
    args = parser.parse_args()

    with open(args.config) as fh:
        user_cfg = {k: v for k, v in json.load(fh).items() if not k.startswith("_")}
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects KEY=VALUE, got '{item}'")
        user_cfg[key] = _parse_value(value)

    unknown = sorted(set(user_cfg) - set(DEFAULTS))
    if unknown:
        raise KeyError(f"Unknown config keys: {unknown}. Valid keys: {sorted(DEFAULTS)}")
    cfg = {**DEFAULTS, **user_cfg}

    todo = [k for k, v in cfg.items() if v == "TODO"]
    if todo:
        raise ValueError(f"Config values still set to \"TODO\": {todo}")
    if cfg["task_mode"] not in TASK_MODES:
        raise ValueError(f"task_mode must be one of {list(TASK_MODES)}, got {cfg['task_mode']}")
    for key in ("feature_scaler", "film_scaler"):
        if cfg[key] not in SCALERS:
            raise ValueError(f"{key} must be one of {list(SCALERS)}, got {cfg[key]}")
    if cfg["checkpoint_metric"] not in ("mae", "rmse"):
        raise ValueError(f"checkpoint_metric must be 'mae' or 'rmse', got {cfg['checkpoint_metric']}")
    return cfg


def output_paths(cfg):
    out = cfg["output_dir"]
    return {
        "model":            cfg["checkpoint_path"] or os.path.join(out, "best_model.pth"),
        "train_csv":        os.path.join(out, "train_data.csv"),
        "val_csv":          os.path.join(out, "validation_data.csv"),
        "test_csv_tta":     os.path.join(out, "test_data_tta.csv"),
        "test_csv_no_tta":  os.path.join(out, "test_data_no_tta.csv"),
        "learning_curve_csv": os.path.join(out, "learning_curve_data.csv"),
        "learning_curve":   os.path.join(out, "learning_curve.png"),
        "val_parity":       os.path.join(out, "validation_parity_plot.png"),
        "test_parity_tta":  os.path.join(out, "test_parity_plot.png"),
        "test_parity_no_tta": os.path.join(out, "test_parity_plot_no_tta.png"),
    }


# =============================================================================
# Helpers
# =============================================================================

def get_device():
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name())
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")   # Apple Silicon GPU
    else:
        device = torch.device("cpu")
    print("Using device:", device)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    return device


def warn_split_overlap(dfs, id_col):
    """Warn if the same structure ID appears in more than one split (possible leakage)."""
    ids = {name: set(df[id_col].astype(str)) for name, df in dfs.items()}
    names = list(ids)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            n_shared = len(ids[a] & ids[b])
            if n_shared:
                print(f"WARNING: {n_shared} structure IDs appear in both '{a}' and '{b}' splits "
                      f"(possible data leakage, e.g. isotherm rows of one structure in different splits).")


def make_loader(dataset, cfg, shuffle, drop_last=False):
    workers = cfg["num_workers"]
    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=shuffle,
        drop_last=drop_last,   # train: avoids a final batch of size 1, which BatchNorm cannot handle
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        collate_fn=partial(upsample_collate, min_points=cfg["min_points"]),
    )


def build_scheduler(optimizer, cfg):
    """
    Linear warmup (learning_rate/warmup_epochs → learning_rate over warmup_epochs), then
    cosine decay to eta_min at t_max_epochs, then constant eta_min. Identical to the old
    LinearLR + CosineAnnealingLR schedule up to t_max_epochs; the old schedule then rose
    back up because CosineAnnealingLR is periodic.
    """
    lr, eta_min = cfg["learning_rate"], cfg["eta_min"]
    warmup, t_max = cfg["warmup_epochs"], cfg["t_max_epochs"]

    def lr_factor(epoch):
        if warmup > 0 and epoch < warmup:
            return 1.0 / warmup + (1.0 - 1.0 / warmup) * epoch / warmup
        if epoch < t_max:
            progress = (epoch - warmup) / (t_max - warmup)
            return (eta_min + (lr - eta_min) * 0.5 * (1.0 + math.cos(math.pi * progress))) / lr
        return eta_min / lr

    return LambdaLR(optimizer, lr_lambda=lr_factor)


def save_checkpoint(model, path):
    # Save the uncompiled module's weights so keys carry no "_orig_mod." prefix.
    torch.save(getattr(model, "_orig_mod", model).state_dict(), path)


def load_checkpoint(model, path, device):
    # Strips the "_orig_mod." prefix of checkpoints saved from a torch.compile'd model.
    state = torch.load(path, map_location=device)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    getattr(model, "_orig_mod", model).load_state_dict(state)


def save_predictions(df, row_indices, columns, path):
    """Join prediction columns to the CSV rows they belong to (by DataFrame index) and save."""
    metadata_df = df.loc[row_indices].reset_index(drop=True)
    combined = pd.concat([metadata_df, pd.DataFrame(columns)], axis=1)
    combined.to_csv(path, index=False)
    return combined


# =============================================================================
# Training / test
# =============================================================================

def train(model, cfg, paths, device, train_loader, val_loader, val_df, unscale):
    loss_fn = get_loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = build_scheduler(optimizer, cfg)
    amp_scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    metric_name = cfg["checkpoint_metric"]

    best_metric = float("inf")
    epochs_without_improvement = 0
    training_loss, validation_loss = [], []

    tracker = EmissionsTracker(output_dir=cfg["output_dir"], log_level="error")
    tracker.start()

    for epoch in range(cfg["epochs"]):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, amp_scaler,
                                     n_augments=cfg["n_train_augments"])
        val_loss, val_preds, val_targets, val_row_indices = evaluate(model, val_loader, loss_fn, device)
        # Back-transform to raw units for all metrics, checkpointing, and plotting.
        val_preds_raw, val_targets_raw = unscale(val_preds), unscale(val_targets)
        val_mae  = compute_mae(val_targets_raw, val_preds_raw)
        val_rmse = compute_rmse(val_targets_raw, val_preds_raw)
        val_metric = val_mae if metric_name == "mae" else val_rmse

        # Save model/predictions with lowest checkpoint metric on validation set
        if val_metric < best_metric and not np.isnan(val_metric):
            best_metric = val_metric
            save_checkpoint(model, paths["model"])
            combined_val_df = save_predictions(val_df, val_row_indices,
                                               {"val_true": val_targets_raw, "val_pred": val_preds_raw},
                                               paths["val_csv"])
            print(f"New best model saved ({metric_name}={val_metric:.5f}) at epoch {epoch+1}")
            parity_plot(combined_val_df, paths["val_parity"], f"Validation Set: Epoch {epoch+1}",
                        "val_true", "val_pred", quantity_label=cfg["target"])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"No improvement (best {metric_name}={best_metric:.5f}, "
                  f"patience {epochs_without_improvement}/{cfg['early_stopping_patience']})")

        epoch_seconds = time.time() - start_time
        scheduler.step()

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"Val MAE: {val_mae:.5f} | "
            f"Val RMSE: {val_rmse:.5f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"Time: {epoch_seconds:.5f} sec"
        )

        # For Plotting Learning Curve
        training_loss.append(train_loss)
        validation_loss.append(val_loss)
        plot_learning_curve(paths["learning_curve"], training_loss, validation_loss)
        pd.DataFrame({"training_loss": training_loss, "validation_loss": validation_loss}).to_csv(
            paths["learning_curve_csv"], index=False)

        if epochs_without_improvement >= cfg["early_stopping_patience"]:
            print(f"Early stopping triggered at epoch {epoch+1} "
                  f"(no improvement for {cfg['early_stopping_patience']} epochs).")
            break

    tracker.stop()


def evaluate_train_set(model, cfg, paths, device, train_dataset, train_df, unscale):
    """Eval-mode predictions of the best checkpoint on the training set."""
    print("\nEvaluating best model on the training set (eval mode)...")
    loader = make_loader(train_dataset, cfg, shuffle=False)
    _, preds, targets, row_indices = evaluate(model, loader, get_loss(), device, label="Train")
    save_predictions(train_df, row_indices,
                     {"train_true": unscale(targets), "train_pred": unscale(preds)}, paths["train_csv"])
    print(f"Train-set predictions saved to {paths['train_csv']}")


def run_test(model, cfg, paths, device, test_loader, test_df, unscale):
    loss_fn = get_loss()
    n_rot = cfg["n_tta_rotations"]

    ### Prediction without TTA
    print("Evaluating on test set without TTA...")
    t0 = time.perf_counter()
    _, preds, targets, row_indices = evaluate(model, test_loader, loss_fn, device, label="Test")
    t1 = time.perf_counter()
    combined = save_predictions(test_df, row_indices,
                                {"test_true": unscale(targets), "test_pred": unscale(preds)},
                                paths["test_csv_no_tta"])
    parity_plot(combined, paths["test_parity_no_tta"], "Pointnet++ Regression — No TTA (Test Prediction)",
                true_column="test_true", pred_column="test_pred", quantity_label=cfg["target"])
    print(f"No-TTA inference time : {t1 - t0:.2f} s  ({len(test_df)} structures)")
    print(f"No-TTA predictions saved to {paths['test_csv_no_tta']}")

    ### Prediction with TTA (individual rotation predictions saved alongside the mean)
    print(f"\nEvaluating on test set with {n_rot}-rotation TTA...")
    t0 = time.perf_counter()
    _, preds, targets, row_indices, rotation_preds = evaluate_tta(model, test_loader, loss_fn, device, n_rotations=n_rot)
    t1 = time.perf_counter()
    rotation_preds = unscale(rotation_preds)                       # [N, n_rot]
    columns = {"test_true": unscale(targets)}
    columns.update({f"test_TTA{i+1}": rotation_preds[:, i] for i in range(n_rot)})
    columns["test_pred"] = unscale(preds)                          # mean over rotations
    columns["test_tta_std"] = rotation_preds.std(axis=1)           # spread over rotations
    combined = save_predictions(test_df, row_indices, columns, paths["test_csv_tta"])
    parity_plot(combined, paths["test_parity_tta"], f"Pointnet++ Regression — {n_rot}-Rotation TTA (Test Prediction)",
                true_column="test_true", pred_column="test_pred", quantity_label=cfg["target"])
    print(f"TTA inference time    : {t1 - t0:.2f} s  ({len(test_df)} structures, {n_rot} rotations)")
    print(f"TTA predictions (mean + every rotation) saved to {paths['test_csv_tta']}")


# =============================================================================
# Main
# =============================================================================

def main():
    cfg = load_config()
    paths = output_paths(cfg)
    test_only = cfg["test_only"]
    task_mode = TASK_MODES[cfg["task_mode"]]

    os.makedirs(cfg["output_dir"], exist_ok=True)
    if not test_only and os.path.exists(paths["model"]) and not cfg["overwrite"]:
        raise FileExistsError(f"{paths['model']} already exists. Set \"overwrite\": true to retrain "
                              f"into this directory, or \"test_only\": true to only evaluate it.")
    config_out = "config_used_test_only.json" if test_only else "config_used.json"
    with open(os.path.join(cfg["output_dir"], config_out), "w") as fh:
        json.dump(cfg, fh, indent=4)
    print(json.dumps(cfg, indent=4))

    device = get_device()

    ####### Load CSVs
    csv = lambda name: pd.read_csv(os.path.join(cfg["data_dir"], cfg[name]))
    train_df, val_df, test_df = csv("train_csv"), csv("val_csv"), csv("test_csv")
    if cfg["debug_subset"]:
        n = cfg["debug_subset"]
        train_df, val_df, test_df = train_df.head(n), val_df.head(n), test_df.head(n)
        print(f"DEBUG: using only the first {n} rows of each split")
    warn_split_overlap({"train": train_df, "val": val_df, "test": test_df}, cfg["structure_id_col"])

    ####### Target Standardization
    # Fit on train set only. mean/std are printed so the transform can be
    # reproduced at inference time without re-loading the training CSV.
    target = cfg["target"]
    if cfg["standardize_target"]:
        target_mean = float(train_df[target].mean())
        target_std  = float(train_df[target].std())
        print(f"Target standardization ENABLED  (target={target})")
        print(f"  mean = {target_mean:.8f}")
        print(f"  std  = {target_std:.8f}")
    else:
        target_mean, target_std = 0.0, 1.0
        print("Target standardization DISABLED")

    def standardized(df):
        # Copy so the original keeps raw values for the metadata CSVs.
        df_fit = df.copy()
        df_fit[target] = (df_fit[target] - target_mean) / target_std
        return df_fit

    def unscale(values):
        return values * target_std + target_mean

    ####### Fit scalers on the training set only
    feature_scalers = [SCALERS[cfg["feature_scaler"]]() for _ in range(cfg["num_atomic_features"])]
    film_scalers    = [SCALERS[cfg["film_scaler"]]() for _ in cfg["film_feature_cols"]]
    fit_feature_scalers(train_df[cfg["structure_id_col"]].tolist(), cfg["h5_path"], feature_scalers)
    fit_film_scalers(train_df, cfg["film_feature_cols"], film_scalers)

    print("---- Feature Scalers for Point Cloud -------")
    for i, s in enumerate(feature_scalers):
        print(f"  Feature {i}: {s}")

    prepare = partial(prepare_data, hdf5_path=cfg["h5_path"], feature_scalers=feature_scalers,
                      target_col=target, structure_id_col=cfg["structure_id_col"], task_mode=task_mode,
                      film_feature_cols=cfg["film_feature_cols"], film_scalers=film_scalers,
                      center_coordinates=cfg["center_coordinates"])

    ####### Load Model (PointNet++ Single Scale Grouping + FiLM Modulation)
    model = get_model(extra_feat_dim=len(cfg["film_feature_cols"]),
                      atomic_feature_channel_num=cfg["num_atomic_features"],
                      dropout=cfg["dropout"],
                      enable_rotation_aug=cfg["enable_rotation_aug"],
                      model_size=cfg["model_size"],
                      sa1_radius=cfg["sa1_radius"],
                      sa2_radius=cfg["sa2_radius"]).to(device)
    print(model)
    print(f"SA radii: sa1 = {model.sa1_radius} Å, sa2 = {model.sa2_radius} Å")
    max_radius = max(model.sa1_radius, model.sa2_radius)
    if cfg["compile"]:
        # torch.compile fuses SA-layer Conv2d/BN/SiLU chains and FC layers into
        # optimised Triton kernels. dynamic=True handles any variation in point-cloud
        # size across batches. Note: saved_features/fps_indices dict assignments in
        # forward() create minor graph breaks (compiled in sub-graphs).
        model = torch.compile(model, dynamic=True)

    ####### Training
    if not test_only:
        train_samples = prepare(standardized(train_df))
        val_samples   = prepare(standardized(val_df))
        if len(train_samples) < cfg["batch_size"]:
            raise ValueError(f"Only {len(train_samples)} training samples for batch_size={cfg['batch_size']} "
                             f"(the last incomplete batch is dropped). Lower batch_size or raise debug_subset.")
        if task_mode == 0:
            check_cell_widths(train_samples, max_radius, "train")
            check_cell_widths(val_samples, max_radius, "val")
        train_dataset = StructureDataset(train_samples)
        train_loader = make_loader(train_dataset, cfg, shuffle=True, drop_last=True)
        val_loader   = make_loader(StructureDataset(val_samples), cfg, shuffle=False)

        train(model, cfg, paths, device, train_loader, val_loader, val_df, unscale)
    else:
        print("test_only = true: skipping training, evaluating existing checkpoint.")

    ####### Reload best checkpoint
    load_checkpoint(model, paths["model"], device)
    model.eval()
    print(f"Loaded checkpoint: {paths['model']}")

    if not test_only:
        evaluate_train_set(model, cfg, paths, device, train_dataset, train_df, unscale)

    ####### Prediction on Test Set
    test_samples = prepare(standardized(test_df))
    if task_mode == 0:
        check_cell_widths(test_samples, max_radius, "test")
    test_loader = make_loader(StructureDataset(test_samples), cfg, shuffle=False)
    run_test(model, cfg, paths, device, test_loader, test_df, unscale)


if __name__ == "__main__":
    main()
