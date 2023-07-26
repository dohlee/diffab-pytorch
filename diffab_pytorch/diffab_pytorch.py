import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch import einsum


class SingleAminoAcidEmbeddingMLP(nn.Module):
    def __init__(self, d_type):
        self.embed_type = nn.Embedding(20, d_type)

        d_feat = d_type + 20
        pass

    def forward(self, x):
        pass


class PairwiseEmbeddingMLP(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class InvariantPointAttention(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class AminoAcidDenoising(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class CaCoordinateDenoising(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class OrientationDenoising(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class OrientationLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_rotmat: torch.Tensor, target_rotmat: torch.Tensor) -> torch.Tensor:
        device = pred_rotmat.device

        rot_discrepancy = einsum("b l i j, b l i k -> b l j k", pred_rotmat, target_rotmat)
        eye = torch.eye(3).to(device).expand_as(rot_discrepancy)

        return F.mse_loss(rot_discrepancy - eye, reduction=self.reduction)


class DiffAb(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.aa_loss = nn.KLDivLoss()
        self.coordinate_loss = nn.MSELoss()
        self.oreintation_loss = OrientationLoss()

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass
