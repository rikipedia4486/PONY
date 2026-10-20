import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from data_util.calculate_stats import compute_r2, compute_mae, compute_rmse
from models.pointnet2_util import random_rotation_batch, rotate_batch


class StructureDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pc, lattice, extra, target, row_idx = self.samples[idx]
        return pc, lattice, extra, target, row_idx


def _to_device(pc, src_idx, lattice, extra_features, target, device):
    pc             = pc.to(device)
    src_idx        = src_idx.to(device)                                             # [B, N] source point of each point
    lattice        = lattice.to(device).float() if lattice is not None else None   # [B, 3, 3] or None
    extra_features = extra_features.to(device)
    target         = target.to(device)
    return pc, src_idx, lattice, extra_features, target


def _print_metrics(label, targets, preds):
    print(f"  {label} R2: {compute_r2(targets, preds):.5f}")
    print(f"  {label} MAE: {compute_mae(targets, preds):.5f}")
    print(f"  {label} RMSE: {compute_rmse(targets, preds):.5f}")


def train_one_epoch(model, train_loader, optimizer, loss_fn, device, amp_scaler, n_augments=1):
    """
    n_augments: number of full passes through the training data per epoch.
    Each pass applies a fresh random rotation inside model.forward() (training mode).
    n_augments=3 means each structure sees 3 independently rotated views per epoch.
    amp_scaler: GradScaler instance created once outside the epoch loop.

    Returns only the mean loss: predictions made in training mode (dropout, random
    rotations, weights changing mid-epoch) are not a valid train-set evaluation —
    use evaluate() with the best checkpoint for that. The train-mode metrics are
    still printed for monitoring.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_targets = []

    for aug in range(n_augments):
        desc = f"Train {aug+1}/{n_augments}" if n_augments > 1 else "Train"
        progress_bar = tqdm(train_loader, desc=desc, leave=False)

        for pc, src_idx, lattice, extra_features, target, idx in progress_bar:
            pc, src_idx, lattice, extra_features, target = _to_device(pc, src_idx, lattice, extra_features, target, device)

            optimizer.zero_grad()

            with torch.amp.autocast(device.type):
                pred, _ = model(pc, extra_features, lattice=lattice, src_idx=src_idx)
                loss = loss_fn(pred, target)

            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()

            total_loss += loss.item()
            n_batches += 1
            progress_bar.set_postfix(loss=total_loss / n_batches)

            all_preds.append(pred.detach().float().cpu().numpy().reshape(-1))
            all_targets.append(target.cpu().numpy().reshape(-1))

    _print_metrics("Train (train-mode)", np.concatenate(all_targets), np.concatenate(all_preds))

    return total_loss / n_batches


def evaluate(model, loader, loss_fn, device, label="Val"):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_indicies = []

    with torch.no_grad():
        for pc, src_idx, lattice, extra_features, target, idx in tqdm(loader, desc="Evaluating", leave=False):
            pc, src_idx, lattice, extra_features, target = _to_device(pc, src_idx, lattice, extra_features, target, device)

            with torch.amp.autocast(device.type):
                pred, _ = model(pc, extra_features, lattice=lattice, src_idx=src_idx)
                loss = loss_fn(pred, target)

            running_loss += loss.item() * pc.shape[0]
            all_preds.append(pred.float().cpu().numpy().reshape(-1))
            all_targets.append(target.cpu().numpy().reshape(-1))
            all_indicies.append(idx.cpu().numpy().reshape(-1))

    epoch_loss = running_loss / len(loader.dataset)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_indicies = np.concatenate(all_indicies, axis=0)

    _print_metrics(label, all_targets, all_preds)

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
    the reported loss.

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
        for pc, src_idx, lattice, extra_features, target, idx in tqdm(loader, desc=f"TTA ({n_rotations} rot)", leave=False):
            pc, src_idx, lattice, extra_features, target = _to_device(pc, src_idx, lattice, extra_features, target, device)

            B = pc.shape[0]
            rotation_preds = []

            for _ in range(n_rotations):
                R = random_rotation_batch(B, device)                       # [B, 3, 3]
                # Rotate xyz (first 3 channels) — atomic features are scalars, unchanged
                xyz_rot, lattice_rot = rotate_batch(pc[:, :3, :], R, lattice)
                pc_rot = torch.cat([xyz_rot, pc[:, 3:, :].float()], dim=1) if pc.shape[1] > 3 else xyz_rot

                with torch.amp.autocast(device.type):
                    pred, _ = model(pc_rot, extra_features, lattice=lattice_rot, src_idx=src_idx)
                rotation_preds.append(pred.float())

            rotation_preds = torch.stack(rotation_preds, dim=0)   # [R, B]
            avg_pred = rotation_preds.mean(dim=0)                 # [B]

            loss = loss_fn(avg_pred, target)
            running_loss += loss.item() * B

            all_preds.append(avg_pred.cpu().numpy().reshape(-1))
            all_targets.append(target.cpu().numpy().reshape(-1))
            all_indicies.append(idx.cpu().numpy().reshape(-1))
            all_rotation_preds.append(rotation_preds.reshape(n_rotations, B).transpose(0, 1).cpu().numpy())  # [B, R]

    epoch_loss = running_loss / len(loader.dataset)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_indicies = np.concatenate(all_indicies, axis=0)
    all_rotation_preds = np.concatenate(all_rotation_preds, axis=0)  # (N_samples, n_rotations)

    _print_metrics(f"TTA ({n_rotations} rot)", all_targets, all_preds)

    return epoch_loss, all_preds, all_targets, all_indicies, all_rotation_preds
