import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from data_util.calculate_stats import compute_r2, compute_mae, compute_rmse


def _random_rotation_batch(B, device):
    """
    Haar-uniform random SO(3) rotations via QR decomposition.
    Mirrors random_rotation_batch in pointnet2_util — kept here to avoid
    cross-module imports. Used for test-time augmentation (TTA).
    The entire batch is generated on `device` in one kernel call.
    """
    rand_mat = torch.randn(B, 3, 3, device=device)                 # [B, 3, 3] on GPU
    Q, R     = torch.linalg.qr(rand_mat)
    signs    = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))     # [B, 3]
    Q        = Q * signs.unsqueeze(-2)                             # [B, 3, 3]
    return Q  # det(Q) = +1 for all entries

class StructureDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pc, lattice, extra, target, row_idx = self.samples[idx]
        return pc, lattice, extra, target, row_idx

def train_one_epoch(model, train_loader, optimizer, loss_fn, device, amp_scaler, n_augments=1):
    """
    n_augments: number of full passes through the training data per epoch.
    Each pass applies a fresh random rotation inside model.forward() (training mode).
    n_augments=3 means each structure sees 3 independently rotated views per epoch.
    amp_scaler: GradScaler instance created once outside the epoch loop.
    """
    model.train()
    total_loss = 0

    all_preds = []
    all_targets = []
    all_indicies = []

    for aug in range(n_augments):
        desc = f"Train {aug+1}/{n_augments}" if n_augments > 1 else "Train"
        progress_bar = tqdm(train_loader, desc=desc, leave=False)

        for pc, lattice, extra_features, target, idx in progress_bar:
            pc             = pc.to(device)
            lattice        = lattice.to(device) if lattice is not None else None   # [B, 3, 3] or None
            extra_features = extra_features.to(device)
            target         = target.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                pred, _ = model(pc, extra_features, lattice=lattice)
                loss = loss_fn(pred, target)

            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()

            total_loss += loss.item()

            all_preds.append(pred.detach().float().cpu().numpy().reshape(-1))
            all_targets.append(target.cpu().numpy().reshape(-1))
            all_indicies.append(idx.cpu().numpy().reshape(-1))

            avg_loss = total_loss / (progress_bar.n + 1 + aug * len(train_loader))
            progress_bar.set_postfix(loss=avg_loss)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_indicies = np.concatenate(all_indicies)

    r2 = compute_r2(all_targets, all_preds)
    mae = compute_mae(all_targets, all_preds)
    rmse = compute_rmse(all_targets, all_preds)
    print(f"  Train R2: {r2:.5f}")
    print(f"  Train MAE: {mae:.5f}")
    print(f"  Train RMSE: {rmse:.5f}")

    return total_loss / (len(train_loader) * n_augments), all_preds, all_targets, all_indicies

def evaluate(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_indicies = []

    with torch.no_grad():
        for pc, lattice, extra_features, target, idx in tqdm(loader, desc="Evaluating", leave=False):
            pc             = pc.to(device)
            lattice        = lattice.to(device) if lattice is not None else None   # [B, 3, 3] or None
            extra_features = extra_features.to(device)
            target         = target.to(device)

            with torch.amp.autocast('cuda'):
                pred, _ = model(pc, extra_features, lattice=lattice)
                loss = loss_fn(pred, target)
           
            running_loss += loss.item() * pc.shape[0]
            all_preds.append(pred.float().cpu().numpy().reshape(-1))
            all_targets.append(target.cpu().numpy().reshape(-1))
            all_indicies.append(idx.cpu().numpy().reshape(-1))

    epoch_loss = running_loss / len(loader.dataset)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_indicies = np.concatenate(all_indicies, axis=0)

    r2 = compute_r2(all_targets, all_preds)
    mae = compute_mae(all_targets, all_preds)
    rmse = compute_rmse(all_targets, all_preds)
    print(f"  Val R2: {r2:.5f}")
    print(f"  Val MAE: {mae:.5f}")
    print(f"  Val RMSE: {rmse:.5f}")

    return epoch_loss, all_preds, all_targets, all_indicies


def evaluate_tta(model, loader, loss_fn, device, n_rotations=16):
    """
    Test-time augmentation (TTA) via rotational averaging.

    For each batch, applies n_rotations independent random SO(3) rotations
    externally to the point cloud and lattice, runs a forward pass per rotation,
    then averages all predictions. This approximates a rotationally-invariant
    prediction without retraining.

    Note: rotation is applied EXTERNALLY here because model.eval() disables
    the internal rotation in forward(). The averaged prediction is used for
    both the reported loss and the saved predictions.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_indicies = []

    with torch.no_grad():
        for pc, lattice, extra_features, target, idx in tqdm(loader, desc=f"TTA ({n_rotations} rot)", leave=False):
            pc             = pc.to(device)
            lattice        = lattice.to(device).float() if lattice is not None else None  # [B, 3, 3] or None
            extra_features = extra_features.to(device)
            target         = target.to(device)

            B = pc.shape[0]
            rotation_preds = []

            for _ in range(n_rotations):
                R = _random_rotation_batch(B, device)                 # [B, 3, 3]

                # Rotate xyz (first 3 channels) — atomic features are scalars, unchanged
                xyz_rot = torch.bmm(R, pc[:, :3, :].float())          # [B, 3, N]
                lattice_rot = torch.bmm(lattice, R.transpose(-1, -2)) if lattice is not None else None  # [B, 3, 3] or None

                if pc.shape[1] > 3:
                    pc_rot = torch.cat([xyz_rot, pc[:, 3:, :]], dim=1)
                else:
                    pc_rot = xyz_rot

                with torch.amp.autocast('cuda'):
                    pred, _ = model(pc_rot, extra_features, lattice=lattice_rot)
                rotation_preds.append(pred)

            # Average predictions across all rotations
            avg_pred = torch.stack(rotation_preds, dim=0).mean(dim=0)  # [B, 1]

            loss = loss_fn(avg_pred, target)
            running_loss += loss.item() * B

            all_preds.append(avg_pred.float().cpu().numpy().reshape(-1))
            all_targets.append(target.cpu().numpy().reshape(-1))
            all_indicies.append(idx.cpu().numpy().reshape(-1))

    epoch_loss = running_loss / len(loader.dataset)

    all_preds   = np.concatenate(all_preds,    axis=0)
    all_targets = np.concatenate(all_targets,  axis=0)
    all_indicies = np.concatenate(all_indicies, axis=0)

    r2  = compute_r2(all_targets, all_preds)
    mae = compute_mae(all_targets, all_preds)
    rmse = compute_rmse(all_targets, all_preds)
    print(f"  TTA R2  ({n_rotations} rot): {r2:.5f}")
    print(f"  TTA MAE ({n_rotations} rot): {mae:.5f}")
    print(f"  TTA RMSE ({n_rotations} rot): {rmse:.5f}")

    return epoch_loss, all_preds, all_targets, all_indicies


### This new version returns individual predictions...!
def evaluate_tta_2(model, loader, loss_fn, device, n_rotations=16):
    """
    Test-time augmentation (TTA) via rotational averaging.

    Same as evaluate_tta(), but additionally returns the prediction from every
    individual rotation.

    Returns
    -------
    epoch_loss : float
    all_preds : ndarray, shape (N,)
        Mean prediction across all rotations.
    all_targets : ndarray, shape (N,)
    all_indices : ndarray, shape (N,)
    all_rotation_preds : ndarray, shape (N, n_rotations)
        Individual prediction from every TTA rotation.
    """
    model.eval()
    running_loss = 0.0

    all_preds = []
    all_targets = []
    all_indicies = []
    all_rotation_preds = []

    with torch.no_grad():
        for pc, lattice, extra_features, target, idx in tqdm(
                loader,
                desc=f"TTA ({n_rotations} rot)",
                leave=False):

            pc = pc.to(device)
            lattice = lattice.to(device).float() if lattice is not None else None
            extra_features = extra_features.to(device)
            target = target.to(device)

            B = pc.shape[0]
            rotation_preds = []

            for _ in range(n_rotations):
                R = _random_rotation_batch(B, device)

                # Rotate xyz coordinates
                xyz_rot = torch.bmm(R, pc[:, :3, :].float())

                # Rotate lattice if present
                lattice_rot = (
                    torch.bmm(lattice, R.transpose(-1, -2))
                    if lattice is not None else None
                )

                # Reassemble point cloud
                if pc.shape[1] > 3:
                    pc_rot = torch.cat([xyz_rot, pc[:, 3:, :]], dim=1)
                else:
                    pc_rot = xyz_rot

                with torch.amp.autocast('cuda'):
                    pred, _ = model(pc_rot, extra_features, lattice=lattice_rot)

                rotation_preds.append(pred)

            # rotation_preds: list of [B,1]
            rotation_preds = torch.stack(rotation_preds, dim=0)   # [R, B, 1]

            avg_pred = rotation_preds.mean(dim=0)                 # [B,1]

            loss = loss_fn(avg_pred, target)
            running_loss += loss.item() * B

            all_preds.append(avg_pred.float().cpu().numpy().reshape(-1))
            all_targets.append(target.cpu().numpy().reshape(-1))
            all_indicies.append(idx.cpu().numpy().reshape(-1))

            # Save every individual TTA prediction
            # becomes (B, R)
            all_rotation_preds.append(
                rotation_preds.squeeze(-1).transpose(0, 1).cpu().numpy()
            )

    epoch_loss = running_loss / len(loader.dataset)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_indicies = np.concatenate(all_indicies, axis=0)

    # Shape: (N_samples, n_rotations)
    all_rotation_preds = np.concatenate(all_rotation_preds, axis=0)

    r2 = compute_r2(all_targets, all_preds)
    mae = compute_mae(all_targets, all_preds)
    rmse = compute_rmse(all_targets, all_preds)

    print(f"  TTA R2  ({n_rotations} rot): {r2:.5f}")
    print(f"  TTA MAE ({n_rotations} rot): {mae:.5f}")
    print(f"  TTA RMSE ({n_rotations} rot): {rmse:.5f}")

    return (
        epoch_loss,
        all_preds,
        all_targets,
        all_indicies,
        all_rotation_preds,
    )