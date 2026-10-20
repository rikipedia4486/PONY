# Implementation of PointNet++ Set Abstraction (SA) module with dual-stat pooling
# and per-channel depthwise Conv1d fusion.
#
# Pool fusion strategy: Conv1d(C, C, kernel_size=2, groups=C)
#   groups=C means each output channel connects only to its own input channel —
#   i.e. C independent [2→1] linear maps, one per feature channel. Each feature
#   channel learns its own way to combine (max, mean), which is appropriate
#   because different channels encode semantically different quantities
#   (e.g. bond-length spread vs. atomic-radius distribution).
#
# Parameter count per SA layer:
#   Conv1d(C, C, 2, groups=C): C × 2 weights + C biases
#   SA1 (C=128):  256 + 128 = 384  parameters
#   SA2 (C=256):  512 + 256 = 768  parameters
#   SA3 (C=1024): 2048 + 1024 = 3072 parameters
#   Total added: ~4224 parameters across all SA layers — completely negligible.
#
# Downstream architecture (in_channel values) is unchanged: output shape
# [B, C, npoint] is identical to vanilla max pooling.

import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).reshape(B, N, 1)
    dist += torch.sum(dst ** 2, -1).reshape(B, 1, M)
    return dist


# =============================================================================
# PBC ADDITIONS — Minimum Image Convention helpers
# Convention: L is [B, 3, 3] with lattice vectors as ROWS (a, b, c), matching
# the ASE Atoms.cell layout.  Cartesian <-> fractional: r = s @ L, s = r @ L_inv.
# =============================================================================

# ORIGINAL VERSION
# def mic_displacement(dr, L, L_inv):
#     """
#     Apply the Minimum Image Convention to Cartesian displacement vectors.

#     Args:
#         dr      : [..., 3]   arbitrary-batch displacement vectors (Cartesian)
#         L       : [B, 3, 3]  lattice matrix — rows are lattice vectors a, b, c
#         L_inv   : [B, 3, 3]  pre-computed inverse of L  (avoids repeated inversion)
#     Returns:
#         [..., 3] MIC-corrected Cartesian displacement vectors
#     """
#     orig_shape = dr.shape
#     B          = L.shape[0]
#     dr_flat    = dr.reshape(B, -1, 3)            # [B, M, 3]
#     ds         = torch.bmm(dr_flat, L_inv)       # fractional displacements [B, M, 3]
#     ds         = ds - torch.round(ds)            # MIC: map to [-0.5, 0.5)
#     dr_mic     = torch.bmm(ds, L)                # back to Cartesian [B, M, 3]
#     return dr_mic.reshape(orig_shape)

# V3 - Vectorized on GPU
def mic_displacement(dr, L, L_inv):
    """
    Apply the Minimum Image Convention to Cartesian displacement vectors.

    Decomposed 27-image algorithm:
      1. Convert dr → fractional, apply round() to map to [-0.5, 0.5).
      2. Back to Cartesian: dr_base = ds @ L            — bmm([B, M, 3])
      3. Precompute 27 Cartesian shift vectors:          — einsum → [B, 27, 3]
             shift_cart[b, i] = shifts[i] @ L[b]
      4. All 27 candidates via broadcast add:
             [B, 1, M, 3] + [B, 27, 1, 3] → [B, 27, M, 3]
      5. argmin + gather selects the shortest image.

    Previously a single bmm([B, 27·M, 3], [B, 3, 3]) was used. That shape
    triggers torch.inductor's pad_mm benchmarker, which allocates a padded
    copy of the tensor at compile time and causes an OOM on the GPU.
    The decomposed form only ever calls bmm on [B, M, 3] tensors, so
    pad_mm sees a small K-dimension (3) on a modestly-sized matrix and
    does not attempt a benchmark allocation.

    Args:
        dr      : [..., 3]   arbitrary-batch displacement vectors (Cartesian)
        L       : [B, 3, 3]  lattice matrix — rows are lattice vectors a, b, c
        L_inv   : [B, 3, 3]  pre-computed inverse of L  (avoids repeated inversion)
    Returns:
        [..., 3] MIC-corrected Cartesian displacement vectors
    """
    orig_shape = dr.shape
    B          = L.shape[0]
    dr_flat    = dr.reshape(B, -1, 3)                         # [B, M, 3]
    M          = dr_flat.shape[1]

    ds = torch.bmm(dr_flat, L_inv)                            # [B, M, 3] fractional
    ds = ds - torch.round(ds)                                 # map to [-0.5, 0.5)

    # All 27 shift vectors {-1,0,1}³ including (0,0,0) — built once, on device
    shifts = ds.new_tensor(
        [[ni, nj, nk] for ni in (-1, 0, 1)
                      for nj in (-1, 0, 1)
                      for nk in (-1, 0, 1)]
    )                                                         # [27, 3]

    # Base Cartesian vector (round-corrected, zero shift): bmm([B, M, 3])
    dr_base    = torch.bmm(ds, L)                             # [B, M, 3]

    # Cartesian shift vectors per crystal: shifts[i] @ L[b]
    # einsum: [27, 3] × [B, 3, 3] → [B, 27, 3]  (tiny tensor)
    shift_cart = torch.einsum('ij,bjk->bik', shifts, L)       # [B, 27, 3]

    # All 27 candidates via broadcast add — no bmm on a 27·M batch dim
    # [B, 1, M, 3] + [B, 27, 1, 3] → [B, 27, M, 3]
    dr_cands   = dr_base.unsqueeze(1) + shift_cart.unsqueeze(2)

    # Select the image with minimum squared distance
    best_idx   = dr_cands.pow(2).sum(-1).argmin(dim=1)        # [B, M]

    # Gather winning vectors: expand index to [B, 1, M, 3] then squeeze
    idx_exp    = best_idx.unsqueeze(1).unsqueeze(-1).expand(B, 1, M, 3)
    dr_mic     = dr_cands.gather(1, idx_exp).squeeze(1)       # [B, M, 3]

    return dr_mic.reshape(orig_shape)


# LATTICE VECTORS ARE ROWS BY DEFAULT? INTERESTING THAT IT'S ASE CONVENTION...
def random_rotation_batch(B, device):
    """
    Generate B independent Haar-uniform random rotation matrices (SO(3)).

    Method: QR decomposition of a standard-normal matrix — the standard
    Haar-measure sampler for SO(3). The entire batch is generated directly
    on `device` in a single kernel call, avoiding the Python loop and
    CPU→GPU copy of the previous implementation.

    Args:
        B      : int     batch size
        device : torch.device
    Returns:
        R : [B, 3, 3]  rotation matrices, det(R) = +1 for all entries
    """
    rand_mat = torch.randn(B, 3, 3, device=device)                 # [B, 3, 3] on GPU
    Q, R     = torch.linalg.qr(rand_mat)                           # Q orthogonal
    # Enforce det(Q) = +1: multiply each column i of Q by sign(R[i,i])
    signs = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))        # [B, 3]
    Q     = Q * signs.unsqueeze(-2)                                # [B, 3, 3]
    return Q  # proper rotation matrices in SO(3)


def pbc_square_distance(src, dst, L, L_inv):
    """
    Pairwise squared distances under the Minimum Image Convention.
    Drop-in replacement for square_distance when a lattice is available.

    NOTE: The ||a-b||² = ||a||² + ||b||² - 2a·b identity CANNOT be used here
    because round() in MIC is non-linear — explicit pairwise vectors are required.

    Args:
        src     : [B, S, 3]  query (centroid) positions
        dst     : [B, N, 3]  database positions
        L       : [B, 3, 3]  lattice matrix
        L_inv   : [B, 3, 3]  inverse lattice matrix
    Returns:
        [B, S, N] squared MIC distances
    """
    B, S, _ = src.shape
    _, N, _ = dst.shape
    # Explicit pairwise displacements: [B, S, N, 3]
    dr     = dst.unsqueeze(1).expand(B, S, N, 3) \
           - src.unsqueeze(2).expand(B, S, N, 3)
    dr_mic = mic_displacement(dr, L, L_inv)      # [B, S, N, 3]
    return torch.sum(dr_mic ** 2, dim=-1)         # [B, S, N]
# =============================================================================


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    reshape_shape = list(idx.shape)
    reshape_shape[1:] = [1] * (len(reshape_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).reshape(reshape_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


# PBC: L and L_inv added so FPS selects centroids using MIC distances.
#      Falls back to naive Euclidean when L is None (backward-compatible).
def farthest_point_sample(xyz, npoint, L=None, L_inv=None):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].reshape(B, 1, 3)
        # PBC: use MIC distance if lattice provided, else naive Euclidean
        if L is not None and L_inv is not None:
            dr   = xyz - centroid                    # [B, N, 3]
            dr   = mic_displacement(dr, L, L_inv)    # [B, N, 3] MIC-corrected
            dist = torch.sum(dr ** 2, -1)            # [B, N]
        else:
            dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

# ---------------------------------------------------------------------------
# CUDA-accelerated FPS via torch_cluster (requires separate installation):
#   pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
# Uses Euclidean distances for both molecule and crystal mode. PBC correctness
# is fully preserved: mic_displacement and pbc_square_distance in sample_and_group
# and query_ball_point apply exact MIC corrections to all neighbour vectors.
# To activate: in sample_and_group, comment out `farthest_point_sample(...)`
#              and uncomment `farthest_point_sample_cuda(...)`.
# ---------------------------------------------------------------------------
@torch.compiler.disable()
def farthest_point_sample_cuda(xyz, npoint, L=None, L_inv=None):
    """
    CUDA-accelerated FPS using torch_cluster.fps (Euclidean distances).
    L and L_inv are accepted for API compatibility but not used.
    Requires: pip install torch-cluster
    """
    from torch_cluster import fps as tc_fps
    device = xyz.device
    B, N, _ = xyz.shape
    xyz_flat = xyz.reshape(B * N, 3).float()  # float32 required by torch_cluster; safe under AMP
    batch    = torch.arange(B, device=device).repeat_interleave(N)
    ratio    = npoint / N
    sampled  = tc_fps(xyz_flat, batch, ratio=ratio, random_start=True)  # global indices [B*npoint]
    offsets  = (torch.arange(B, device=device) * N).unsqueeze(1)        # [B, 1]
    fps_idx  = (sampled.reshape(B, npoint) - offsets).long()            # [B, npoint]
    return fps_idx

# FIXED FUNCTION (Returns Query Point Itself):
# PBC: L and L_inv added for MIC-based radius search.
#      Falls back to naive distances when L is None (backward-compatible).
def query_ball_point(radius, nsample, xyz, new_xyz, L=None, L_inv=None):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    # PBC: use MIC distances if lattice provided, else naive distances
    if L is not None and L_inv is not None:
        sqrdists = pbc_square_distance(new_xyz, xyz, L, L_inv)  # [B, S, N]
    else:
        sqrdists = square_distance(new_xyz, xyz)                # [B, S, N]

    # Partial topk selection — O(N log nsample) vs full sort O(N log N)
    sorted_dists, group_idx = sqrdists.topk(nsample, dim=-1, largest=False, sorted=True)  # [B, S, nsample]

    # Mask out points beyond radius — out-of-place for torch.compile compatibility
    beyond_radius = sorted_dists > radius ** 2
    group_idx = group_idx.masked_fill(beyond_radius, N)

    # Determine first neighbor in each group
    group_first = group_idx[:, :, 0]        # [B, S]
    empty_mask = group_first == N           # True if no valid neighbors for that group

    # For empty groups, use the query point itself as the neighbor
    self_idx = torch.arange(S, device=device).reshape(1, S).expand(B, S)  # [B, S]
    group_first = torch.where(empty_mask, self_idx, group_first)

    # Replace any remaining N sentinel indices with the first valid index
    group_first_expanded = group_first.reshape(B, S, 1).expand(B, S, nsample)  # [B, S, nsample]
    group_idx = torch.where(group_idx == N, group_first_expanded, group_idx)

    return group_idx

# PBC: L and L_inv threaded into FPS, ball query, and local displacement.
#      Falls back to non-PBC behaviour when L is None (backward-compatible).
def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False, L=None, L_inv=None):
    B, N, C = xyz.shape
    S = npoint
    # fps_idx = farthest_point_sample(xyz, npoint, L=L, L_inv=L_inv)       # Python FPS — active (MIC distances, exact)
    fps_idx = farthest_point_sample_cuda(xyz, npoint)                   # CUDA FPS  — faster (Euclidean; requires torch_cluster)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz, L=L, L_inv=L_inv)  # PBC
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    # PBC: local displacements (what the MLP actually learns from) must be MIC-corrected
    #      so that atoms across a cell boundary get the correct short-range vector.
    if L is not None and L_inv is not None:
        raw_disp         = grouped_xyz - new_xyz.reshape(B, S, 1, C)
        grouped_xyz_norm = mic_displacement(raw_disp, L, L_inv)  # [B, S, nsample, 3]
    else:
        grouped_xyz_norm = grouped_xyz - new_xyz.reshape(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    # return FPS indices when requested (for Grad-CAM)
    if returnfps:
        return new_xyz, new_points, fps_idx
    else:
        return new_xyz, new_points

def sample_and_group_all(xyz, points):
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.reshape(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.reshape(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all
        # Per-channel depthwise fusion: Conv1d(C, C, kernel_size=3, groups=C).
        # groups=C gives each feature channel its own independent [3→1] kernel —
        # unlike the shared-weight MLP/linear approaches, channel 0 and channel 127
        # can learn completely different strategies for fusing (max, mean, std).
        # Applied to [B*npoint, C, 3] → output [B*npoint, C, 1], then reshaped
        # back to [B, C, npoint]. Output shape is identical to vanilla max pooling.
        c_out = mlp[-1]
        self.pool_fusion = nn.Conv1d(c_out, c_out, kernel_size=2, groups=c_out, bias=True)

    # PBC: lattice and lattice_inv are optional kwargs threaded into sampling/grouping.
    #      SA layers with group_all=True (SA3) ignore them (global pooling, no local search).
    def forward(self, xyz, points, lattice=None, lattice_inv=None):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
            fps_idx = None
        else:
            new_xyz, new_points, fps_idx = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points,
                returnfps=True, L=lattice, L_inv=lattice_inv              # PBC
            )
        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.silu(bn(conv(new_points)))

        # Triple-stat pooling fused by per-channel depthwise Conv1d.
        # torch.std with unbiased=False (population std) avoids NaN when nsample=1.
        max_pts  = torch.max(new_points, 2)[0]                        # [B, C, npoint]
        mean_pts = torch.mean(new_points, 2)                          # [B, C, npoint]
        stats    = torch.stack([max_pts, mean_pts], dim=-1)           # [B, C, npoint, 2]

        # Note: LayerNorm over 2 elements always produces ±1/√2 regardless of
        # the magnitude of (max - mean), erasing the most informative signal.
        # No normalization applied — max and mean share the same scale after
        # ReLU-activated BatchNorm Conv2d layers.

        # Reshape for Conv1d: [B, C, npoint, 2] → [B*npoint, C, 2]
        # Conv1d(groups=C) sweeps the kernel_size=2 window over the stat dim,
        # producing one output value per channel → [B*npoint, C, 1].
        # Reshape back to [B, C, npoint].
        B_sz, C_sz, np_sz, _ = stats.shape
        stats      = stats.permute(0, 2, 1, 3).reshape(B_sz * np_sz, C_sz, 2)  # [B*npoint, C, 2]
        new_points = self.pool_fusion(stats)                                     # [B*npoint, C, 1]
        new_points = new_points.reshape(B_sz, np_sz, C_sz).permute(0, 2, 1)    # [B, C, npoint]

        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points, fps_idx

    
""" # This code has been adapted from:
@article{Pytorch_Pointnet_Pointnet2,
      Author = {Xu Yan},
      Title = {Pointnet/Pointnet++ Pytorch},
      Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
      Year = {2019}
}
"""