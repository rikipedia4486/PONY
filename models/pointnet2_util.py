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
#
# NUMERICAL PRECISION:
#   All geometry (rotations, distances, MIC, ball query) runs in exact float32:
#     - autocast is disabled around sampling/grouping, so nothing is downcast to fp16;
#     - no matmul/bmm/einsum is used on coordinates, because
#       torch.set_float32_matmul_precision('high') makes float32 matmuls use TF32
#       (~3 significant digits). 3×3 transforms use matmul3() (broadcast-sum) and
#       squared distances use explicit differences instead of ||a||² + ||b||² − 2a·b,
#       which cancels catastrophically for coordinates far from the origin.
#   The learned MLPs (Conv2d/BN/SiLU) still run under AMP.
#
# UPSAMPLED DUPLICATES:
#   upsample_collate pads clouds with copies of random atoms. src_idx [B, N] gives,
#   for every point, the index of the original point it was copied from (itself for
#   originals). Copies are excluded from ball queries, and mean pooling averages only
#   over distinct real neighbours (padding slots and repeated centroids are masked).
#   Max pooling is unaffected by copies.

import torch
import torch.nn as nn
import torch.nn.functional as F


def matmul3(v, M):
    """
    Exact float32 v @ M for a trailing 3×3 matrix, without matmul kernels (no TF32).

    Args:
        v : [B, ..., 3]  row vectors
        M : [B, 3, 3]
    Returns:
        [B, ..., 3]
    """
    B = M.shape[0]
    M = M.reshape(B, *([1] * (v.dim() - 2)), 3, 3)     # broadcast over middle dims
    return v[..., 0:1] * M[..., 0, :] + v[..., 1:2] * M[..., 1, :] + v[..., 2:3] * M[..., 2, :]


def square_distance(src, dst):
    """
    Pairwise squared Euclidean distances via explicit differences (exact in float32).

    Args:
        src : [B, S, 3]
        dst : [B, N, 3]
    Returns:
        [B, S, N]
    """
    return ((src.unsqueeze(2) - dst.unsqueeze(1)) ** 2).sum(-1)


# =============================================================================
# PBC ADDITIONS — Minimum Image Convention helpers
# Convention: L is [B, 3, 3] with lattice vectors as ROWS (a, b, c), matching
# the ASE Atoms.cell layout.  Cartesian <-> fractional: r = s @ L, s = r @ L_inv.
# =============================================================================

def mic_displacement(dr, L, L_inv):
    """
    Apply the Minimum Image Convention to Cartesian displacement vectors.

    Decomposed 27-image algorithm:
      1. Convert dr → fractional, apply round() to map to [-0.5, 0.5).
      2. Back to Cartesian: dr_base = ds @ L
      3. Precompute 27 Cartesian shift vectors:  shift_cart[b, i] = shifts[i] @ L[b]
      4. All 27 candidates via broadcast add:
             [B, 1, M, 3] + [B, 27, 1, 3] → [B, 27, M, 3]
      5. argmin + gather selects the shortest image.

    No bmm is used on the [B, M, 3] tensors (matmul3 instead): this keeps the
    transform exact in float32 and avoids torch.inductor's pad_mm benchmarker,
    which previously allocated a padded copy at compile time and caused an OOM.

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

    ds = matmul3(dr_flat, L_inv)                              # [B, M, 3] fractional
    ds = ds - torch.round(ds)                                 # map to [-0.5, 0.5)

    # All 27 shift vectors {-1,0,1}³ including (0,0,0) — built once, on device
    shifts = ds.new_tensor(
        [[ni, nj, nk] for ni in (-1, 0, 1)
                      for nj in (-1, 0, 1)
                      for nk in (-1, 0, 1)]
    )                                                         # [27, 3]

    dr_base    = matmul3(ds, L)                               # [B, M, 3]
    shift_cart = matmul3(shifts.unsqueeze(0).expand(B, 27, 3), L)  # [B, 27, 3]

    # [B, 1, M, 3] + [B, 27, 1, 3] → [B, 27, M, 3]
    dr_cands   = dr_base.unsqueeze(1) + shift_cart.unsqueeze(2)

    # Select the image with minimum squared distance
    best_idx   = dr_cands.pow(2).sum(-1).argmin(dim=1)        # [B, M]

    # Gather winning vectors: expand index to [B, 1, M, 3] then squeeze
    idx_exp    = best_idx.unsqueeze(1).unsqueeze(-1).expand(B, 1, M, 3)
    dr_mic     = dr_cands.gather(1, idx_exp).squeeze(1)       # [B, M, 3]

    return dr_mic.reshape(orig_shape)


def random_rotation_batch(B, device):
    """
    Generate B independent Haar-uniform random rotation matrices (SO(3)).

    Method: QR decomposition of a standard-normal matrix — the standard
    Haar-measure sampler for SO(3). The entire batch is generated directly
    on `device` in a single kernel call.

    Args:
        B      : int     batch size
        device : torch.device
    Returns:
        R : [B, 3, 3]  rotation matrices, det(R) = +1 for all entries
    """
    rand_mat = torch.randn(B, 3, 3, device=device)                 # [B, 3, 3]
    Q, R     = torch.linalg.qr(rand_mat)                           # Q orthogonal
    # Enforce det(Q) = +1: multiply each column i of Q by sign(R[i,i])
    signs = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))        # [B, 3]
    Q     = Q * signs.unsqueeze(-2)                                # [B, 3, 3]
    return Q  # proper rotation matrices in SO(3)


def rotate_batch(xyz, R, lattice=None):
    """
    Rotate column-vector coordinates and row-vector lattices by the same R (exact float32).
        xyz'     = R @ xyz           (xyz: [B, 3, N] column vectors)
        lattice' = lattice @ R^T     (lattice: [B, 3, 3], rows = a, b, c)
    This keeps the fractional-coordinate transform consistent, so MIC is unaffected.
    """
    xyz = xyz.float()
    R_T = R.float().transpose(-1, -2)
    xyz_rot = matmul3(xyz.transpose(1, 2), R_T).transpose(1, 2)            # [B, 3, N]
    lattice_rot = matmul3(lattice.float(), R_T) if lattice is not None else None
    return xyz_rot, lattice_rot


def pbc_square_distance(src, dst, L, L_inv):
    """
    Pairwise squared distances under the Minimum Image Convention.
    Drop-in replacement for square_distance when a lattice is available.

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


# Pure-PyTorch FPS (MIC distances when a lattice is given). Slower than the CUDA
# version below; use it if torch_cluster is unavailable (swap in sample_and_group).
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

# Random FPS start point; set False for deterministic centroids (e.g. reproducibility checks).
FPS_RANDOM_START = True

# ---------------------------------------------------------------------------
# CUDA-accelerated FPS via torch_cluster (requires separate installation):
#   pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
# Uses Euclidean distances for both molecule and crystal mode. PBC correctness
# of the neighbourhoods is preserved: mic_displacement and pbc_square_distance in
# sample_and_group and query_ball_point apply exact MIC corrections.
# ---------------------------------------------------------------------------
@torch.compiler.disable()
def farthest_point_sample_cuda(xyz, npoint, L=None, L_inv=None):
    """
    CUDA-accelerated FPS using torch_cluster.fps (Euclidean distances).
    L and L_inv are accepted for API compatibility but not used.
    """
    from torch_cluster import fps as tc_fps
    device = xyz.device
    B, N, _ = xyz.shape
    xyz_flat = xyz.reshape(B * N, 3).float()  # float32 required by torch_cluster
    batch    = torch.arange(B, device=device).repeat_interleave(N)
    ratio    = npoint / N
    sampled  = tc_fps(xyz_flat, batch, ratio=ratio, random_start=FPS_RANDOM_START)  # global indices [B*npoint]
    offsets  = (torch.arange(B, device=device) * N).unsqueeze(1)        # [B, 1]
    fps_idx  = (sampled.reshape(B, npoint) - offsets).long()            # [B, npoint]
    return fps_idx


def map_centroids_to_sources(fps_idx, src_idx):
    """
    Replace each FPS centroid by the original point it was copied from, and build
    src_idx for the next SA level: a centroid whose source point was already picked
    by an earlier centroid points to that earlier centroid. This happens whenever a
    cloud has fewer distinct points than npoint; those repeated centroids are kept
    (fixed tensor shapes) but treated as copies at the next level.

    Args:
        fps_idx : [B, S]  FPS centroid indices into the current points
        src_idx : [B, N]  source point of every current point (itself for originals)
    Returns:
        centroid_src : [B, S]  source point of every centroid (used as the centroid index)
        next_src_idx : [B, S]  source centroid of every centroid, for the next level
    """
    centroid_src = src_idx.gather(1, fps_idx)
    B, S = centroid_src.shape
    order = torch.arange(S, device=fps_idx.device).expand(B, S)
    # first_pick[b, n] = position of the first centroid whose source is point n (S if none)
    first_pick = torch.full_like(src_idx, S).scatter_reduce(1, centroid_src, order, reduce="amin")
    next_src_idx = first_pick.gather(1, centroid_src)
    return centroid_src, next_src_idx


# PBC: L and L_inv added for MIC-based radius search (naive distances when L is None).
def query_ball_point(radius, nsample, xyz, new_xyz, query_idx, L=None, L_inv=None, is_source=None):
    """
    query_idx : [B, S] indices of the query points within xyz, used to fill groups
                with no neighbour inside the radius.
    is_source : [B, N] bool, False for upsampled copies, which are never selected.
    Returns:
        group_idx : [B, S, nsample] neighbour indices; unused slots repeat the first neighbour
        slot_mask : [B, S, nsample] bool, True for real neighbours inside the radius
    """
    B, N, C = xyz.shape

    if L is not None and L_inv is not None:
        sqrdists = pbc_square_distance(new_xyz, xyz, L, L_inv)  # [B, S, N]
    else:
        sqrdists = square_distance(new_xyz, xyz)                # [B, S, N]
    if is_source is not None:
        sqrdists = sqrdists.masked_fill(~is_source.unsqueeze(1), float("inf"))

    # Partial topk selection — O(N log nsample) vs full sort O(N log N)
    sorted_dists, group_idx = sqrdists.topk(nsample, dim=-1, largest=False, sorted=True)  # [B, S, nsample]

    # Mask out points beyond radius — out-of-place for torch.compile compatibility
    beyond_radius = sorted_dists > radius ** 2
    group_idx = group_idx.masked_fill(beyond_radius, N)

    # First neighbour in each group; empty groups fall back to the query point itself
    group_first = group_idx[:, :, 0]                        # [B, S]
    group_first = torch.where(group_first == N, query_idx, group_first)

    # Replace any remaining N sentinel indices with the first valid index
    group_first_expanded = group_first.unsqueeze(-1).expand_as(group_idx)  # [B, S, nsample]
    group_idx = torch.where(group_idx == N, group_first_expanded, group_idx)

    return group_idx, ~beyond_radius

# PBC: L and L_inv threaded into ball query and local displacement.
def sample_and_group(npoint, radius, nsample, xyz, points, src_idx=None, L=None, L_inv=None):
    """
    Geometry (FPS, ball query, local displacements) is computed in exact float32
    with autocast disabled. Grouped point features keep their incoming dtype.

    src_idx : [B, N] source point of every point (None = all points are originals).
    Returns new_xyz, new_points, fps_idx, slot_mask, next_src_idx.
    """
    B, N, C = xyz.shape
    S = npoint
    point_order = torch.arange(N, device=xyz.device).expand(B, N)
    if src_idx is None:
        src_idx = point_order
    with torch.autocast(device_type=xyz.device.type, enabled=False):
        xyz = xyz.float()
        if L is not None:
            L, L_inv = L.float(), L_inv.float()
        # fps_idx = farthest_point_sample(xyz, npoint, L=L, L_inv=L_inv)  # Python FPS (MIC distances, exact, slow)
        fps_idx = farthest_point_sample_cuda(xyz, npoint)                 # CUDA FPS (Euclidean; requires torch_cluster)
        fps_idx, next_src_idx = map_centroids_to_sources(fps_idx, src_idx)
        new_xyz = index_points(xyz, fps_idx)
        idx, slot_mask = query_ball_point(radius, nsample, xyz, new_xyz, fps_idx, L=L, L_inv=L_inv,
                                          is_source=src_idx == point_order)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
        # Local displacements (what the MLP learns from) are MIC-corrected so that
        # atoms across a cell boundary get the correct short-range vector.
        raw_disp = grouped_xyz - new_xyz.reshape(B, S, 1, C)
        if L is not None and L_inv is not None:
            grouped_xyz_norm = mic_displacement(raw_disp, L, L_inv)  # [B, S, nsample, 3]
        else:
            grouped_xyz_norm = raw_disp

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points.to(grouped_xyz_norm.dtype)], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points, fps_idx, slot_mask, next_src_idx

def sample_and_group_all(xyz, points, src_idx=None):
    """Returns new_xyz, new_points, slot_mask ([B, 1, N], False for repeated points)."""
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.reshape(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.reshape(B, 1, N, -1).to(grouped_xyz.dtype)], dim=-1)
    else:
        new_points = grouped_xyz
    if src_idx is None:
        slot_mask = torch.ones(B, 1, N, dtype=torch.bool, device=device)
    else:
        slot_mask = (src_idx == torch.arange(N, device=device)).unsqueeze(1)
    return new_xyz, new_points, slot_mask

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
        # Per-channel depthwise fusion of (max, mean): Conv1d(C, C, kernel_size=2, groups=C).
        # groups=C gives each feature channel its own independent [2→1] kernel.
        # Applied to [B*npoint, C, 2] → output [B*npoint, C, 1], then reshaped
        # back to [B, C, npoint]. Output shape is identical to vanilla max pooling.
        c_out = mlp[-1]
        self.pool_fusion = nn.Conv1d(c_out, c_out, kernel_size=2, groups=c_out, bias=True)

    # src_idx: [B, N] source point of every input point (None = all originals).
    # PBC: lattice and lattice_inv are optional kwargs threaded into sampling/grouping.
    #      SA layers with group_all=True (SA3) ignore them (global pooling, no local search).
    # Returns new_xyz, new_points, fps_idx, next_src_idx (fps_idx/next_src_idx are None for group_all).
    def forward(self, xyz, points, src_idx=None, lattice=None, lattice_inv=None):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points, slot_mask = sample_and_group_all(xyz, points, src_idx)
            fps_idx = next_src_idx = None
        else:
            new_xyz, new_points, fps_idx, slot_mask, next_src_idx = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points,
                src_idx=src_idx, L=lattice, L_inv=lattice_inv              # PBC
            )
        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.silu(bn(conv(new_points)))

        # Dual-stat (max, mean) pooling fused by per-channel depthwise Conv1d.
        # The mean covers only real neighbours (slot_mask); the max is unaffected by repeats.
        slot_mask = slot_mask.transpose(1, 2).unsqueeze(1).to(new_points.dtype)   # [B, 1, nsample, npoint]
        max_pts  = torch.max(new_points, 2)[0]                                    # [B, C, npoint]
        mean_pts = (new_points * slot_mask).sum(2) / slot_mask.sum(2)             # [B, C, npoint]
        stats    = torch.stack([max_pts, mean_pts], dim=-1)                       # [B, C, npoint, 2]

        # No normalization applied — max and mean share the same scale after
        # the BatchNorm/SiLU Conv2d layers (LayerNorm over 2 elements would
        # always produce ±1/√2 and erase the (max − mean) signal).

        # [B, C, npoint, 2] → [B*npoint, C, 2] → Conv1d → [B*npoint, C, 1] → [B, C, npoint]
        B_sz, C_sz, np_sz, _ = stats.shape
        stats      = stats.permute(0, 2, 1, 3).reshape(B_sz * np_sz, C_sz, 2)
        new_points = self.pool_fusion(stats)
        new_points = new_points.reshape(B_sz, np_sz, C_sz).permute(0, 2, 1)

        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points, fps_idx, next_src_idx


""" # This code has been adapted from:
@article{Pytorch_Pointnet_Pointnet2,
      Author = {Xu Yan},
      Title = {Pointnet/Pointnet++ Pytorch},
      Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
      Year = {2019}
}
"""
