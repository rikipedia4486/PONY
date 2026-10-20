"""
Check that predictions do not depend on which atoms upsample_collate copies.

The same structure is padded several times with different random copies, and the
eval-mode predictions are compared with the FPS start point fixed. With src_idx
the predictions must agree; without it (copies treated as real atoms) they differ.

Run on a GPU node (requires torch_cluster), from the repository root:
    python tests/check_padding_invariance.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import models.pointnet2_util as pointnet2_util
from models.pointnet2 import get_model
from data_util.pre_processing import upsample_collate

N_FEATURES = 8
N_PADDINGS = 4
MIN_POINTS = 1024


def make_structure(n_atoms, crystal, generator):
    """Random structure as a dataset sample: (pc [C, N], lattice, extra, target, row_idx)."""
    if crystal:
        lattice = torch.diag(torch.tensor([14.0, 15.0, 16.0]))
        xyz = torch.rand(n_atoms, 3, generator=generator) @ lattice
    else:
        lattice = None
        xyz = 8.0 * torch.randn(n_atoms, 3, generator=generator)
    features = torch.randn(n_atoms, N_FEATURES, generator=generator)
    pc = torch.cat([xyz, features], dim=1).T.contiguous()
    return pc, lattice, None, torch.tensor(0.0), 0


def predictions(model, sample, use_src_idx, device):
    preds = []
    for seed in range(N_PADDINGS):
        torch.manual_seed(seed)   # a different set of random copies for each padding
        pc, src_idx, lattice, extras, _, _ = upsample_collate([sample], min_points=MIN_POINTS)
        with torch.no_grad():
            pred, _ = model(pc.to(device), extras.to(device),
                            lattice=lattice.to(device) if lattice is not None else None,
                            src_idx=src_idx.to(device) if use_src_idx else None)
        preds.append(pred.item())
    return torch.tensor(preds)


def main():
    device = torch.device("cuda")
    pointnet2_util.FPS_RANDOM_START = False

    torch.manual_seed(0)
    model = get_model(atomic_feature_channel_num=N_FEATURES, model_size="medium").to(device).eval()

    cases = [
        ("molecule, 300 atoms (fewer than 512 centroids)", 300, False),
        ("molecule, 700 atoms", 700, False),
        ("crystal, 300 atoms", 300, True),
        ("molecule, 1500 atoms (no copies needed)", 1500, False),
    ]
    all_passed = True
    for name, n_atoms, crystal in cases:
        sample = make_structure(n_atoms, crystal, torch.Generator().manual_seed(n_atoms))
        with_src = predictions(model, sample, use_src_idx=True, device=device)
        without_src = predictions(model, sample, use_src_idx=False, device=device)

        spread_with = (with_src.max() - with_src.min()).item()
        spread_without = (without_src.max() - without_src.min()).item()
        passed = spread_with <= 1e-4 * max(1.0, with_src.abs().max().item())
        all_passed &= passed
        print(f"{name:48s} spread with src_idx: {spread_with:.2e} | "
              f"copies treated as atoms: {spread_without:.2e} | {'PASS' if passed else 'FAIL'}")

    print("\nAll checks passed." if all_passed else "\nSome checks FAILED.")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
