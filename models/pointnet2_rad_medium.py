import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_util_double_conv import PointNetSetAbstraction, random_rotation_batch

class get_model(nn.Module):
    def __init__(
        self,
        extra_feat_dim=0,
        atomic_feature_channel_num=0,  # number of atomic feature channels (after xyz)
        film_hidden=64,
        dropout=[0.3, 0.3, 0.4],
        enable_rotation_aug=True
    ):
        super(get_model, self).__init__()

        self.extra_feat_dim = extra_feat_dim
        self.atomic_feature_channel_num = atomic_feature_channel_num
        self.enable_rotation_aug = enable_rotation_aug

        # Input dimension: 3 xyz + n atomic feature channels
        in_channel = 3 + atomic_feature_channel_num

        # ---- PointNet++ SA Layers ----
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=2.5, nsample=32,
            in_channel=in_channel, mlp=[64, 64, 128], group_all=False
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=5.0, nsample=64,
            in_channel=128 + 3, mlp=[128, 128, 256], group_all=False
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True
        )

        # FiLM (Flexible for any number of atomic feature channels)
        if extra_feat_dim > 0 and atomic_feature_channel_num > 0:
            self.film_fc = nn.Sequential(
                nn.Linear(extra_feat_dim, film_hidden),
                nn.SiLU(),
                nn.Linear(film_hidden, atomic_feature_channel_num * 2)  # α and β per channel
            )
        else:
            self.film_fc = None

        # ---- Fully Connected Layers ----
        total_feat_dim = 1024
        self.fc1 = nn.Linear(total_feat_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(dropout[0])

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(dropout[1])

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.drop3 = nn.Dropout(dropout[2])

        self.fc4 = nn.Linear(256, 1)

        # For Grad-CAM
        self.saved_features = {}
        self.fps_indices = {}

    # PBC: lattice is [B, 3, 3] with rows = lattice vectors a, b, c (ASE convention).
    # Rotational augmentation: pass (R @ xyz_cart) and (R @ L) for the same rotation R
    # applied at the data level — the model then handles PBC correctly in the rotated frame.
    # If lattice=None the model is fully backward-compatible (no PBC).
    def forward(self, xyz_features, extra_features=None, lattice=None):
        B, C, N = xyz_features.shape
        xyz = xyz_features[:, :3, :]

        # PBC: cast to float32 before any linalg — autocast (AMP) may downcast to
        #      float16, which torch.linalg.inv does not support.
        if lattice is not None:
            lattice = lattice.float()
        lattice_inv = torch.linalg.inv(lattice) if lattice is not None else None

        # ROTATIONAL AUGMENTATION (training only)
        # Each sample in the batch receives its own unique rotation R, seeded from
        # nanosecond wall time so no two forward passes produce the same set of rotations.
        # Both xyz (column vectors) and the lattice (row vectors) are rotated by the same R:
        #   xyz'     = R @ xyz            (R applied left to column vectors)
        #   lattice' = lattice @ R^T      (equivalent transform for row-vector layout)
        # This keeps the fractional-coordinate transform consistent, so MIC is unaffected.
        # Atomic features (channels 3+) are scalars and are not rotated.
        if self.training and self.enable_rotation_aug:
            R   = random_rotation_batch(B, xyz.device)               # [B, 3, 3]
            xyz = torch.bmm(R, xyz)                                  # [B, 3, N]
            if lattice is not None:
                # .float() needed: torch.bmm under autocast may return float16
                lattice     = torch.bmm(lattice, R.transpose(-1, -2)).float()  # [B, 3, 3]
                lattice_inv = torch.linalg.inv(lattice)                        # recompute for rotated L

        # Split xyz and atomic features
        atomic_feats = None
        if self.atomic_feature_channel_num > 0:
            atomic_feats = xyz_features[:, 3:, :]  # shape: [B, n, N]

        # Apply FiLM to atomic features
        if self.film_fc is not None and extra_features is not None:
            film_params = self.film_fc(extra_features)  # [B, n*2]
            alpha, beta = film_params[:, :self.atomic_feature_channel_num], film_params[:, self.atomic_feature_channel_num:]
            
            # Constraining FiLM with activations (for Grad-CAM):
            alpha = torch.sigmoid(alpha) * 4.0
            beta = torch.sigmoid(beta) * 1.0
            
            alpha = alpha.unsqueeze(-1)  # [B, n, 1]
            beta = beta.unsqueeze(-1)
            atomic_feats = atomic_feats * alpha + beta

        # Pass through SA layers
        # PBC: lattice/lattice_inv passed to SA1 and SA2 (local neighbourhood layers).
        #      SA3 uses group_all=True (global pooling over abstract features) — no PBC needed.
        l1_xyz, l1_points, fps1 = self.sa1(xyz, atomic_feats, lattice=lattice, lattice_inv=lattice_inv)
        self.saved_features["sa1"] = l1_points
        self.fps_indices["sa1"] = fps1
        self.saved_features["sa1_xyz"] = l1_xyz

        l2_xyz, l2_points, fps2 = self.sa2(l1_xyz, l1_points, lattice=lattice, lattice_inv=lattice_inv)
        self.saved_features["sa2"] = l2_points
        self.fps_indices["sa2"] = fps2
        self.saved_features["sa2_xyz"] = l2_xyz

        l3_xyz, l3_points, _ = self.sa3(l2_xyz, l2_points)  # PBC: not needed for global pooling
        x = l3_points.view(B, 1024)

        # Fully Connected Layers\
        x = self.drop1(F.silu(self.bn1(self.fc1(x))))
        x = self.drop2(F.silu(self.bn2(self.fc2(x))))
        x = self.drop3(F.silu(self.bn3(self.fc3(x))))
        x = self.fc4(x)
        return x.squeeze(-1), l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat=None):
        return F.smooth_l1_loss(pred, target)
    

""" # This code has been adapted from:
@article{Pytorch_Pointnet_Pointnet2,
      Author = {Xu Yan},
      Title = {Pointnet/Pointnet++ Pytorch},
      Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
      Year = {2019}
}
"""