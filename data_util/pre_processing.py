import h5py
import torch
import numpy as np
from tqdm import tqdm

def fit_feature_scalers(
    train_ids,              # list of structure IDs in the training set (duplicates allowed)
    h5_path,                # path to the HDF5 file
    feature_scalers,        # list of sklearn scalers (or None for identity), one per atomic feature column
):
    """
    Fit one sklearn scaler per atomic feature column on the training set.

    feature_scalers: list of length num_atomic_features, e.g.
        [StandardScaler(), StandardScaler(), MaxAbsScaler(), None]
    None entries are left as-is (identity transform).

    Each unique structure is used once, even if it appears in many CSV rows
    (e.g. isotherm data), so scaler statistics are not weighted by row count.

    Returns the same list with each scaler fitted in-place.
    """
    num_atomic_features = len(feature_scalers)
    unique_ids = list(dict.fromkeys(str(sid) for sid in train_ids))  # order-preserving dedupe
    # Collect all training data per column
    columns = [[] for _ in range(num_atomic_features)]

    with h5py.File(h5_path, "r") as f:
        for sid in tqdm(unique_ids, desc="Fitting feature scalers"):
            if sid not in f:
                print(f"WARNING: Structure {sid} not found in HDF5 file.")
                continue
            pc_array = f[sid]["point_cloud"][:]
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

# Point clouds are cached by structure ID, so duplicate rows reuse the same tensors.
def prepare_data(
    data_df,
    hdf5_path,
    feature_scalers,        # fitted list of sklearn scalers (or None) from fit_feature_scalers
    target_col="target",
    structure_id_col="ID",
    task_mode=0,            # 0 = crystal (loads lattice, applies PBC), 1 = molecule (no lattice, no PBC)
    film_feature_cols=None, # list of DataFrame column names to use as FiLM inputs (or None)
    film_scalers=None,      # fitted list of sklearn scalers (or None) from fit_film_scalers
    center_coordinates=True # subtract each structure's atomic centroid from its xyz
):
    """
    center_coordinates: translating a structure does not change it physically, but
        absolute xyz enters the network (SA3 groups absolute positions, and training
        rotations are about the origin). Centering removes the dependence on where
        the structure happens to sit in the source file's frame (e.g. PDB coordinates).
        MIC distances are translation-invariant, so this is also safe for crystals.
    """
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

                if center_coordinates:
                    pc_array[:, :3] -= pc_array[:, :3].mean(axis=0, keepdims=True)

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


def check_cell_widths(samples, max_radius, split_name=""):
    """
    Warn if any lattice is too small for the largest ball-query radius.

    The minimum image convention only finds the single nearest image of each atom.
    If a cell's smallest perpendicular width (volume / largest face area) is below
    2 × radius, neighbours from further periodic images are silently missed and the
    local neighbourhood is truncated. Use supercells in the H5 to avoid this.
    """
    lattices = {}
    for pc, lattice, _, _, _ in samples:
        if lattice is not None:
            lattices[id(lattice)] = lattice   # cached tensors are shared across duplicate rows
    if not lattices:
        return

    L = torch.stack(list(lattices.values())).double()                       # [M, 3, 3]
    volume = torch.det(L).abs()                                             # [M]
    a, b, c = L[:, 0], L[:, 1], L[:, 2]
    face_areas = torch.stack([torch.linalg.cross(b, c).norm(dim=-1),
                              torch.linalg.cross(c, a).norm(dim=-1),
                              torch.linalg.cross(a, b).norm(dim=-1)], dim=-1)  # [M, 3]
    min_width = volume / face_areas.max(dim=-1).values                      # [M]

    n_bad = int((min_width < 2 * max_radius).sum())
    if n_bad > 0:
        print(f"WARNING [{split_name}]: {n_bad}/{len(min_width)} cells have a perpendicular width "
              f"< 2 × SA radius ({2 * max_radius:.2f} Å; smallest = {min_width.min():.2f} Å). "
              f"MIC neighbourhoods are truncated for these structures — consider supercells.")


def upsample_collate(batch, min_points=1024):
    """
    Pad every point cloud in the batch to a common size by re-sampling random
    existing atoms (with replacement), so the batch can be stacked into one tensor.
    The target size is max(largest cloud in the batch, min_points).

    Also returns src_idx [B, max_points]: the index of the original atom each point
    was copied from (itself for originals), so the model can ignore the copies.
    """
    pcs, lattices, extras, targets, indices = zip(*batch)

    max_points = max(max(pc.shape[1] for pc in pcs), min_points)

    upsampled_pcs = []
    src_idx = []
    for pc in pcs:
        N = pc.shape[1]
        src = torch.arange(N, device=pc.device)
        num_points_to_add = max_points - N
        if num_points_to_add > 0:
            idx = torch.randint(0, N, (num_points_to_add,), device=pc.device)
            pc = torch.cat([pc, pc[:, idx]], dim=1)
            src = torch.cat([src, idx])
        upsampled_pcs.append(pc)
        src_idx.append(src)

    pcs_tensor = torch.stack(upsampled_pcs, dim=0)
    src_idx_tensor = torch.stack(src_idx, dim=0)
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
    targets_tensor = torch.stack(targets, dim=0).reshape(-1)   # [B], also for a batch of size 1
    indices_tensor = torch.tensor(indices, dtype=torch.long)

    return pcs_tensor, src_idx_tensor, lattices_tensor, extras_tensor, targets_tensor, indices_tensor
