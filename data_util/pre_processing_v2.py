import h5py
import torch
import numpy as np
from tqdm import tqdm

def fit_feature_scalers(
    train_ids,              # list of structure IDs in the training set
    h5_path,                # path to the HDF5 file
    feature_scalers,        # list of sklearn scalers (or None for identity), one per atomic feature column
):
    """
    Fit one sklearn scaler per atomic feature column on the training set.

    feature_scalers: list of length num_atomic_features, e.g.
        [StandardScaler(), StandardScaler(), MaxAbsScaler(), None]
    None entries are left as-is (identity transform).

    Returns the same list with each scaler fitted in-place.
    """
    num_atomic_features = len(feature_scalers)
    # Collect all training data per column
    columns = [[] for _ in range(num_atomic_features)]

    with h5py.File(h5_path, "r") as f:
        for sid in tqdm(train_ids, desc="Fitting feature scalers"):
            if str(sid) not in f:
                print(f"WARNING: Structure {sid} not found in HDF5 file.")
                continue
            pc_array = f[str(sid)]["point_cloud"][:]
            feature_array = pc_array[:, 3:3 + num_atomic_features]  # [N, num_atomic_features]
            for i in range(num_atomic_features):
                columns[i].append(feature_array[:, i])

    for i, scaler in enumerate(feature_scalers):
        if scaler is not None:
            col_data = np.concatenate(columns[i]).reshape(-1, 1)
            scaler.fit(col_data)

    return feature_scalers

# FiLM Addition
def fit_film_scalers(
    train_df,               # training DataFrame (already loaded)
    film_feature_cols,      # list of column names to use as FiLM inputs
    film_scalers,           # list of sklearn scalers (or None for identity), one per column
):
    """
    Fit one sklearn scaler per FiLM feature column on the training DataFrame.

    film_feature_cols: list of column names present in the DataFrame, e.g. ["density", "volume"]
    film_scalers: list of length len(film_feature_cols), e.g. [StandardScaler(), None]
    None entries are left as-is (identity transform).

    Returns the same list with each scaler fitted in-place.
    """
    for col, scaler in zip(film_feature_cols, film_scalers):
        if scaler is not None:
            col_data = train_df[col].values.reshape(-1, 1).astype(np.float32)
            scaler.fit(col_data)
    return film_scalers

# FiLM Addition
# def prepare_data(
#     data_df,
#     hdf5_path,
#     feature_scalers,        # fitted list of sklearn scalers (or None) from fit_feature_scalers
#     target_col="target",
#     task_mode=0,            # 0 = crystal (loads lattice, applies PBC), 1 = molecule (no lattice, no PBC)
#     film_feature_cols=None, # list of DataFrame column names to use as FiLM inputs (or None)
#     film_scalers=None       # fitted list of sklearn scalers (or None) from fit_film_scalers
# ):
#     if task_mode not in (0, 1):
#         raise ValueError("Invalid task_mode. Use 0 for crystal (PBC) or 1 for molecule (no PBC).")

#     num_atomic_features = len(feature_scalers)
#     data_samples = []
    
#     with h5py.File(hdf5_path, "r") as f:
#         for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Preparing samples"):

#             structure_id = str(row["ID"])
#             if structure_id not in f:
#                 print(f"WARNING: Structure {structure_id} not found in HDF5.")
#                 continue

#             target = row[target_col]
            
#             grp      = f[structure_id]
#             pc_array = grp["point_cloud"][:][:, :3 + num_atomic_features].astype(np.float32)

#             # Apply per-column scalers to atomic feature channels
#             for i, scaler in enumerate(feature_scalers):
#                 if scaler is not None:
#                     pc_array[:, 3 + i] = scaler.transform(pc_array[:, 3 + i].reshape(-1, 1)).ravel()
            
#             # Convert to tensors
#             pc_tensor = torch.tensor(pc_array.T, dtype=torch.float32)    # [C, N]

#             # Crystal mode: load lattice from H5 for PBC-aware geometry
#             if task_mode == 0:
#                 lattice_np     = grp["lattice"][:]                           # [3, 3] rows = a, b, c
#                 lattice_tensor = torch.tensor(lattice_np, dtype=torch.float32)  # [3, 3]
#             else:  # molecule mode: no periodic boundary conditions
#                 lattice_tensor = None

#             if film_feature_cols:
#                 film_vals = []
#                 for col, scaler in zip(film_feature_cols, film_scalers):
#                     val = np.float32(row[col])
#                     if scaler is not None:
#                         val = np.float32(scaler.transform([[val]])[0, 0])
#                     film_vals.append(val)
#                 extra_features_tensor = torch.tensor(film_vals, dtype=torch.float32)
#             else:
#                 extra_features_tensor = None

#             target_tensor = torch.tensor(target, dtype=torch.float32)
            
#             # Tuple order: (pc, lattice, extra_features, target, row_idx)
#             data_samples.append((pc_tensor, lattice_tensor, extra_features_tensor, target_tensor, idx))
    
#     print(f"Total samples prepared: {len(data_samples)}")
#     return data_samples

# Version for FiLM (reuse duplicate PC's by storing in memory)
def prepare_data(
    data_df,
    hdf5_path,
    feature_scalers,        # fitted list of sklearn scalers (or None) from fit_feature_scalers
    target_col="target",
    structure_id_col="ID",
    task_mode=0,            # 0 = crystal (loads lattice, applies PBC), 1 = molecule (no lattice, no PBC)
    film_feature_cols=None, # list of DataFrame column names to use as FiLM inputs (or None)
    film_scalers=None       # fitted list of sklearn scalers (or None) from fit_film_scalers
):
    if task_mode not in (0, 1):
        raise ValueError("Invalid task_mode. Use 0 for crystal (PBC) or 1 for molecule (no PBC).")

    num_atomic_features = len(feature_scalers)
    data_samples = []

    # Cache (pc_tensor, lattice_tensor) by structure ID so that structures
    # appearing multiple times in the DataFrame (e.g. isotherm data with many
    # pressure/temperature rows per ID) are only read and scaled once from HDF5.
    pc_cache = {}  # str(sid) -> (pc_tensor, lattice_tensor)
    
    with h5py.File(hdf5_path, "r") as f:
        for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Preparing samples"):

            structure_id = str(row[structure_id_col])
            if structure_id not in f:
                print(f"WARNING: Structure {structure_id} not found in HDF5.")
                continue

            target = row[target_col]

            # ── Point cloud: load once per unique ID, reuse for duplicates ────
            if structure_id not in pc_cache:
                grp      = f[structure_id]
                pc_array = grp["point_cloud"][:][:, :3 + num_atomic_features].astype(np.float32)

                # Apply per-column scalers to atomic feature channels
                for i, scaler in enumerate(feature_scalers):
                    if scaler is not None:
                        pc_array[:, 3 + i] = scaler.transform(pc_array[:, 3 + i].reshape(-1, 1)).ravel()

                pc_tensor = torch.tensor(pc_array.T, dtype=torch.float32)    # [C, N]

                # Crystal mode: load lattice from H5 for PBC-aware geometry
                if task_mode == 0:
                    lattice_np     = grp["lattice"][:]
                    lattice_tensor = torch.tensor(lattice_np, dtype=torch.float32)  # [3, 3]
                else:
                    lattice_tensor = None

                pc_cache[structure_id] = (pc_tensor, lattice_tensor)
            else:
                pc_tensor, lattice_tensor = pc_cache[structure_id]

            if film_feature_cols:
                film_vals = []
                for col, scaler in zip(film_feature_cols, film_scalers):
                    val = np.float32(row[col])
                    if scaler is not None:
                        val = np.float32(scaler.transform([[val]])[0, 0])
                    film_vals.append(val)
                extra_features_tensor = torch.tensor(film_vals, dtype=torch.float32)
            else:
                extra_features_tensor = None

            target_tensor = torch.tensor(target, dtype=torch.float32)
            
            # Tuple order: (pc, lattice, extra_features, target, row_idx)
            data_samples.append((pc_tensor, lattice_tensor, extra_features_tensor, target_tensor, idx))
    
    print(f"Total samples prepared: {len(data_samples)}")
    return data_samples

def upsample_collate(batch, enable_train_augmentations=True,
                     xyz_noise_std=0.25, feat_noise_std=0.005,
                     min_points=1024, drop_fraction=0.1,
                     color_drop_fraction=0.0,
                     color_drop_mode="vector"):
    """
    color_drop_mode:
      "vector"  — zeros ALL feature channels for a random fraction of atoms
                  (color_drop_fraction = fraction of atoms zeroed).
      "channel" — zeros individual (atom, channel) values chosen uniformly at
                  random across the full feature matrix; colour_drop_fraction
                  is the fraction of total feature values zeroed, regardless
                  of which atoms or channels they fall in.
    """

    pcs, lattices, extras, targets, indices = zip(*batch)

    # Find max number of points across batch (before dropping)
    max_points_batch = max(pc.shape[1] for pc in pcs)
    max_points = max(max_points_batch, min_points)

    upsampled_pcs = []

    for pc in pcs:
        C, N = pc.shape

        # Step 1: Drop random fraction of points (if enabled)
        if enable_train_augmentations:
            num_points_to_drop = int(round(drop_fraction * N))
            if num_points_to_drop > 0:
                keep_idx = torch.randperm(N, device=pc.device)[num_points_to_drop:]
                pc = pc[:, keep_idx]
                C, N = pc.shape  # update shape after dropping

        # Step 1b: Color drop — zero atomic features (PointNext style).
        if enable_train_augmentations and C > 3 and color_drop_fraction > 0.0:
            pc = pc.clone()
            if color_drop_mode == "vector":
                # Zero the entire feature vector for a random fraction of atoms.
                num_to_color_drop = int(round(color_drop_fraction * N))
                if num_to_color_drop > 0:
                    color_drop_idx = torch.randperm(N, device=pc.device)[:num_to_color_drop]
                    pc[3:, color_drop_idx] = 0.0
            elif color_drop_mode == "channel":
                # Zero a random fraction of individual (channel, atom) entries
                # across the full feature matrix — drops may cluster on any
                # combination of atoms and channels.
                F = C - 3  # number of feature channels
                num_to_drop = int(round(color_drop_fraction * F * N))
                if num_to_drop > 0:
                    flat_idx = torch.randperm(F * N, device=pc.device)[:num_to_drop]
                    row_idx  = flat_idx // N   # feature channel (0-indexed within features)
                    col_idx  = flat_idx  % N   # atom index
                    pc[3 + row_idx, col_idx] = 0.0

        # Step 2: Upsample to max_points
        num_points_to_add = max_points - N
        if num_points_to_add > 0:
            idx = torch.randint(0, N, (num_points_to_add,), device=pc.device)
            extra_points = pc[:, idx]
            pc_new = torch.cat([pc, extra_points], dim=1)
        else:
            pc_new = pc

        # Step 3: Add noise (training only — enable_train_augmentations=False for val/test)
        if enable_train_augmentations:
            xyz_noise = xyz_noise_std * torch.randn_like(pc_new[:3, :])
            pc_new[:3, :] += xyz_noise

            if C > 3:
                feat_noise = feat_noise_std * torch.randn_like(pc_new[3:, :])
                pc_new[3:, :] += feat_noise

        upsampled_pcs.append(pc_new)

    pcs_tensor = torch.stack(upsampled_pcs, dim=0)
    if lattices[0] is not None:
        lattices_tensor = torch.stack(list(lattices), dim=0)   # [B, 3, 3]
    else:
        lattices_tensor = None

    extra_feat_dim = batch[0][2].shape[0] if batch[0][2] is not None else 0
    extras = [
        e if e is not None else torch.zeros(extra_feat_dim, device=pcs_tensor.device)
        for e in extras
    ]
    extras_tensor  = torch.stack(extras, dim=0)
    targets_tensor = torch.stack(targets, dim=0).squeeze()
    indices_tensor = torch.tensor(indices, dtype=torch.long)

    return pcs_tensor, lattices_tensor, extras_tensor, targets_tensor, indices_tensor