print("script initialized successfully, beginning preprocessing...")

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
from models.pointnet2_rad_large import get_model, get_loss
from training_util.training_util_publication import StructureDataset, train_one_epoch, evaluate, evaluate_tta
from data_util.pre_processing_v2 import fit_feature_scalers, fit_film_scalers, prepare_data, upsample_collate
from data_util.calculate_stats import compute_r2, compute_mae, compute_rmse
from data_util.plotting import plot_learning_curve, parity_plot
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from codecarbon import EmissionsTracker
 
### CONFIGURE TRAINING:
### Model checkpointing metric is controlled by CHECKPOINT_METRIC below.

film_feature_cols = [
    "T",
    "log10(P)",
    "molecular_mass",
    "critical_temperature",
    "critical_pressure",
    "acentric_factor",
    "triple_point_temperature",
    "triple_point_pressure",
    "radius_of_gyration",
    "site_1_connectivity",
    "site_1_radii",
    "site_1_partial_charge",
    "site_1_epsilon",
    "site_1_sigma",
    "site_2_connectivity",
    "site_2_radii",
    "site_2_partial_charge",
    "site_2_epsilon",
    "site_2_sigma",
    "site_3_connectivity",
    "site_3_radii",
    "site_3_partial_charge",
    "site_3_epsilon",
    "site_3_sigma"
]   # list of CSV column names to use as per-structure FiLM inputs; e.g. ["density", "volume"]

film_scalers                    =  [MaxAbsScaler() for _ in film_feature_cols]
extra_feat_dim                  =  len(film_feature_cols)
task_mode                       =  0   # 0 = crystal (loads lattice, applies PBC), 1 = molecule (no lattice, no PBC)
num_cpus                        =  12
learning_rate                   =  3e-4  # linear-scaled from 3e-4 @ bs=64 → bs=128
weight_decay                    =  5e-4  
dropout_strength                =  [0.2, 0.2, 0.3] # for final FC layers after the SA blocks.
EPOCHS                          =  100 # Refer to SI
T_MAX_EPOCHS                    =  75 # this defines when lr decay will stop. 
WARMUP_EPOCHS                   =  5   # linear LR warmup before cosine decay begins
N_TRAIN_AUGMENTS                =  1   # rotational augmentation passes per epoch (1 = standard)
N_TTA_ROTATIONS                 =  16  # rotations averaged at test time (TTA)
EARLY_STOPPING_PATIENCE         =  25  # stop if val MAE does not improve for this many epochs
CHECKPOINT_METRIC               =  "mae"   # "mae" or "rmse" — metric used to select the best checkpoint
num_atomic_features             =  8
# feature_scalers                 = [StandardScaler() for _ in range(num_atomic_features)]
feature_scalers                 = [MaxAbsScaler() for _ in range(num_atomic_features)]


# ---- Collation / augmentation hyperparameters ----
XYZ_NOISE_STD                   =  0.00   # Gaussian noise std added to xyz coordinates (set 0.0 for electronic structure tasks)
FEAT_NOISE_STD                  =  0.00  # Gaussian noise std added to atomic feature channels
MIN_POINTS                      =  1024   # minimum point cloud size after upsampling
DROP_FRACTION                   =  0.0    # fraction of points randomly dropped before upsampling
COLOR_DROP_FRACTION             =  0.0    # fraction zeroed; meaning depends on COLOR_DROP_MODE (see below)
COLOR_DROP_MODE                 =  "channel"  # "vector": zero all features for COLOR_DROP_FRACTION of atoms | "channel": zero COLOR_DROP_FRACTION of all (atom, channel) values independently

# ---- Regularization ----
target_name                     =  "Uptake(mol/kg)"
structure_id_col                =  "ID"

h5_path                         = "/scratch/sk10275/PONY_ML/PONY_refactored_4/datasets/mofxdb_point_clouds.h5" # HDF5 file containing atomic point clouds across training, validation, and test structures. 
train_df_path                   = "/scratch/sk10275/PONY_ML/PONY_refactored_4/datasets/hmof_isotherm_train.csv"
val_df_path                     = "/scratch/sk10275/PONY_ML/PONY_refactored_4/datasets/hmof_isotherm_val.csv"
test_data_path                  = "/scratch/sk10275/PONY_ML/PONY_refactored_4/datasets/hmof_isotherm_test.csv"

model_file_name                 = "/scratch/sk10275/PONY_ML/PONY_refactored_4/models_trained_0628/model_hmof_isotherm_large_1fold_maxabs/best_model.pth"
train_csv_name                  = "/scratch/sk10275/PONY_ML/PONY_refactored_4/models_trained_0628/model_hmof_isotherm_large_1fold_maxabs/train_data.csv"
val_csv_name                    = "/scratch/sk10275/PONY_ML/PONY_refactored_4/models_trained_0628/model_hmof_isotherm_large_1fold_maxabs/validation_data.csv"
test_csv_name                   = "/scratch/sk10275/PONY_ML/PONY_refactored_4/models_trained_0628/model_hmof_isotherm_large_1fold_maxabs/test_data_tta.csv"
test_csv_name_no_tta            = "/scratch/sk10275/PONY_ML/PONY_refactored_4/models_trained_0628/model_hmof_isotherm_large_1fold_maxabs/test_data_no_tta.csv"

learning_curve_df_path          = "/scratch/sk10275/PONY_ML/PONY_refactored_4/models_trained_0628/model_hmof_isotherm_large_1fold_maxabs/learning_curve_data.csv"
learning_curve_path             = "/scratch/sk10275/PONY_ML/PONY_refactored_4/models_trained_0628/model_hmof_isotherm_large_1fold_maxabs/learning_curve.png"
parity_plot_path                = "/scratch/sk10275/PONY_ML/PONY_refactored_4/models_trained_0628/model_hmof_isotherm_large_1fold_maxabs/validation_parity_plot.png"
test_parity_plot_path           = "/scratch/sk10275/PONY_ML/PONY_refactored_4/models_trained_0628/model_hmof_isotherm_large_1fold_maxabs/test_parity_plot.png"
test_parity_plot_path_no_tta    = "/scratch/sk10275/PONY_ML/PONY_refactored_4/models_trained_0628/model_hmof_isotherm_large_1fold_maxabs/test_parity_plot_no_tta.png"

### FOR LEARNING CURVE:
training_loss = []
validation_loss = []

if __name__ == "__main__":

    ####### Load Device 
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name())
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")   # Apple Silicon GPU
    else:
        device = torch.device("cpu")
    print("Using device:", device)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    # ####### Prepare Training/Validation Data 
    train_df = pd.read_csv(train_df_path)
    val_df = pd.read_csv(val_df_path)

    train_ids = train_df[structure_id_col].tolist()
    val_ids   = val_df[structure_id_col].tolist()

    atomic_feature_scaling_coef = fit_feature_scalers(
        train_ids, h5_path, feature_scalers
    )

    fit_film_scalers(train_df, film_feature_cols, film_scalers)

    train_samples = prepare_data(train_df, h5_path,
                                 feature_scalers,
                                 target_col=target_name, task_mode=task_mode,
                                 film_feature_cols=film_feature_cols, film_scalers=film_scalers
                                 )
    
    val_samples = prepare_data(val_df, h5_path,
                               feature_scalers,
                               target_col=target_name, task_mode=task_mode,
                               film_feature_cols=film_feature_cols, film_scalers=film_scalers
                               )

    # Print shape for train_sample list:
    print("Outer length:", len(train_samples))
    print("Inner length:", len(train_samples[0]) if len(train_samples) > 0 else "Empty")

    train_dataset = StructureDataset(train_samples)
    val_dataset = StructureDataset(val_samples)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=128, 
        shuffle=True, 
        num_workers=num_cpus,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=lambda batch: upsample_collate(
            batch, enable_train_augmentations=True,
            xyz_noise_std=XYZ_NOISE_STD, feat_noise_std=FEAT_NOISE_STD,
            min_points=MIN_POINTS, drop_fraction=DROP_FRACTION,
            color_drop_fraction=COLOR_DROP_FRACTION, color_drop_mode=COLOR_DROP_MODE
        )
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=128, 
        shuffle=False, 
        num_workers=num_cpus,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=lambda batch: upsample_collate(
            batch, enable_train_augmentations=False,
            xyz_noise_std=XYZ_NOISE_STD, feat_noise_std=FEAT_NOISE_STD,
            min_points=MIN_POINTS, drop_fraction=DROP_FRACTION,
            color_drop_fraction=COLOR_DROP_FRACTION, color_drop_mode=COLOR_DROP_MODE
        )
    )
    
    ####### Print Scaling Coefficients for Manual Setup
    print("---- Feature Scalers for Point Cloud -------")
    for i, s in enumerate(feature_scalers):
        print(f"  Feature {i}: {s}")


    # ####### Load Model (Pointnet++ Single Scale Grouping + FiLM Modulation)
    model = get_model(extra_feat_dim=extra_feat_dim, atomic_feature_channel_num=num_atomic_features, dropout=dropout_strength).to(device)
    print(model)
    # torch.compile fuses SA-layer Conv2d/BN/SiLU chains and FC layers into
    # optimised Triton kernels. dynamic=True handles any variation in point-cloud
    # size across batches. Note: saved_features/fps_indices dict assignments in
    # forward() create minor graph breaks (compiled in sub-graphs).
    model = torch.compile(model, dynamic=True)

    loss_fn = get_loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Linear warmup: lr ramps from learning_rate/WARMUP_EPOCHS → learning_rate over WARMUP_EPOCHS epochs.
    # Cosine decay: lr then decays smoothly from learning_rate → eta_min over the remaining epochs.
    warmup_scheduler = LinearLR(optimizer, start_factor=1.0/WARMUP_EPOCHS, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=T_MAX_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS])

    ###### Training Loop
    best_val_mae = 9999
    epochs_without_improvement = 0
    amp_scaler = torch.amp.GradScaler('cuda')

    tracker = EmissionsTracker(output_dir=os.path.dirname(model_file_name), log_level="error")
    tracker.start()

    for epoch in range(EPOCHS):
        start_time = time.time()

        train_loss, train_preds, train_targets, train_row_indicies = train_one_epoch(model, train_loader, optimizer, loss_fn, device, amp_scaler, n_augments=N_TRAIN_AUGMENTS)
        val_loss, val_preds, val_targets, val_row_indicies = evaluate(model, val_loader, loss_fn, device)
        val_mae  = compute_mae(val_targets, val_preds)
        val_rmse = compute_rmse(val_targets, val_preds)
        val_metric = val_mae if CHECKPOINT_METRIC == "mae" else val_rmse

        # Save model/predictions with lowest checkpoint metric on validation set
        if val_metric < best_val_mae and not np.isnan(val_metric):
            best_val_mae = val_metric
            best_epoch = epoch
            torch.save(model.state_dict(), model_file_name)
            
            # Save Validation Target Predictions + Metadata
            valid_indices = val_df.index.values
            valid_mask = np.isin(val_row_indicies, valid_indices)
            safe_indices = val_row_indicies[valid_mask]
            metadata_df = val_df.loc[safe_indices].reset_index(drop=True)

            val_target_df = pd.DataFrame({
                'val_true': val_targets,
                'val_pred': val_preds
            })

            combined_val_df = pd.concat([metadata_df, val_target_df], axis=1)

            combined_val_df.to_csv(val_csv_name, index=False)

            # Save Training Target Predictions + Metadata
            valid_indices_train = train_df.index.values
            valid_mask_train = np.isin(train_row_indicies, valid_indices_train)
            safe_indices_train = train_row_indicies[valid_mask_train]
            metadata_df_train = train_df.loc[safe_indices_train].reset_index(drop=True)

            train_target_df = pd.DataFrame({
                'train_true': train_targets,
                'train_pred': train_preds
            })

            combined_train_df = pd.concat([metadata_df_train, train_target_df], axis=1)

            combined_train_df.to_csv(train_csv_name, index=False)

            print(f"New best model saved ({CHECKPOINT_METRIC}={val_metric:.5f}) at epoch {epoch+1}")
            parity_plot(combined_val_df, parity_plot_path, "Validation Set: Epoch " + str(epoch+1),
            'val_true', 'val_pred')
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"No improvement (best {CHECKPOINT_METRIC}={best_val_mae:.5f}, patience {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE})")

        end_time = time.time()
        epoch_minutes = (end_time - start_time) / 60
        scheduler.step()

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"Val MAE: {val_mae:.5f} | "
            f"Val RMSE: {val_rmse:.5f} | "
            f"Time: {epoch_minutes:.2f} min"
        )
        
        # For Plotting Learning Curve
        training_loss.append(train_loss)
        validation_loss.append(val_loss)
        plot_learning_curve(learning_curve_path, training_loss, validation_loss)

        learning_curve_df = pd.DataFrame({
            'training_loss': training_loss,
            'validation_loss': validation_loss
        })
        
        learning_curve_df.to_csv(learning_curve_df_path, index=False)

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1} (no improvement for {EARLY_STOPPING_PATIENCE} epochs).")
            break

    tracker.stop()

    ###### Prediction on Test Set:
    test_df = pd.read_csv(test_data_path)
    test_ids = test_df[structure_id_col].tolist()
    test_samples = prepare_data(
        test_df, h5_path,
        feature_scalers,
        target_col=target_name, task_mode=task_mode,
        film_feature_cols=film_feature_cols, film_scalers=film_scalers
    )
    
    test_dataset = StructureDataset(test_samples)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=num_cpus,
        pin_memory=True,
        collate_fn=lambda batch: upsample_collate(
            batch, enable_train_augmentations=False,
            xyz_noise_std=XYZ_NOISE_STD, feat_noise_std=FEAT_NOISE_STD,
            min_points=MIN_POINTS, drop_fraction=DROP_FRACTION,
            color_drop_fraction=COLOR_DROP_FRACTION, color_drop_mode=COLOR_DROP_MODE
        )
    )
    
    model.load_state_dict(torch.load(model_file_name, map_location=device))
    model.eval()

    dummy_loss_fn = torch.nn.MSELoss()

    ### Prediction without TTA
    print("Evaluating on test set without TTA...")
    t0_no_tta = time.perf_counter()
    test_loss_no_tta, test_preds_no_tta, test_targets_no_tta, test_row_indicies_no_tta = evaluate(model, test_loader, dummy_loss_fn, device)
    t1_no_tta = time.perf_counter()

    valid_indicies_no_tta = test_df.index.values
    valid_test_mask_no_tta = np.isin(test_row_indicies_no_tta, valid_indicies_no_tta)
    test_safe_indices_no_tta = test_row_indicies_no_tta[valid_test_mask_no_tta]
    test_metadata_df_no_tta = test_df.loc[test_safe_indices_no_tta].reset_index(drop=True)

    test_target_df_no_tta = pd.DataFrame({
        'test_true': test_targets_no_tta,
        'test_pred': test_preds_no_tta
    })

    combined_test_df_no_tta = pd.concat([test_metadata_df_no_tta, test_target_df_no_tta], axis=1)
    combined_test_df_no_tta.to_csv(test_csv_name_no_tta, index=False)
    parity_plot(combined_test_df_no_tta, test_parity_plot_path_no_tta, "Pointnet++ Regression — No TTA (Test Prediction)", true_column='test_true', pred_column='test_pred')
    print(f"No-TTA inference time : {t1_no_tta - t0_no_tta:.2f} s  ({len(test_df)} structures)")
    print(f"No-TTA predictions saved to {test_csv_name_no_tta}")
    print(f"No-TTA parity plot saved to {test_parity_plot_path_no_tta}")

    ### Prediction with TTA
    print(f"\nEvaluating on test set with {N_TTA_ROTATIONS}-rotation TTA...")
    t0_tta = time.perf_counter()
    test_loss, test_preds, test_targets, test_row_indicies = evaluate_tta(model, test_loader, dummy_loss_fn, device, n_rotations=N_TTA_ROTATIONS)
    t1_tta = time.perf_counter()

    test_mae = compute_mae(test_targets, test_preds)
    test_r2 = compute_r2(test_targets, test_preds)

    valid_indicies = test_df.index.values
    valid_test_mask = np.isin(test_row_indicies, valid_indicies)
    test_safe_indices = test_row_indicies[valid_test_mask]
    test_metadata_df = test_df.loc[test_safe_indices].reset_index(drop=True)

    test_target_df = pd.DataFrame({
        'test_true': test_targets,
        'test_pred': test_preds
    })

    combined_test_df = pd.concat([test_metadata_df, test_target_df], axis=1)
    combined_test_df.to_csv(test_csv_name, index=False)
    parity_plot(combined_test_df, test_parity_plot_path, f"Pointnet++ Regression — {N_TTA_ROTATIONS}-Rotation TTA (Test Prediction)", true_column='test_true', pred_column='test_pred')
    print(f"TTA inference time    : {t1_tta - t0_tta:.2f} s  ({len(test_df)} structures, {N_TTA_ROTATIONS} rotations)")
    print(f"TTA predictions saved to {test_csv_name}")
    print(f"TTA parity plot saved to {test_parity_plot_path}")