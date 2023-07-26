import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch import einsum
from diffab_pytorch.diffusion import (
    SequenceDiffuser,
    CoordinateDiffuser,
    OrientationDiffuser,
)


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

        return F.mse_loss(rot_discrepancy, eye, reduction=self.reduction)


class DiffAb(pl.LightningModule):
    def __init__(self, T=100, s=0.01, beta_max=0.999):
        super().__init__()

        self.seq_diffuser = SequenceDiffuser(T=T, s=s, beta_max=beta_max)
        self.coordinate_diffuser = CoordinateDiffuser(T=T, s=s, beta_max=beta_max)
        self.orientation_diffuser = OrientationDiffuser(T=T, s=s, beta_max=beta_max)

        self.aa_loss = nn.KLDivLoss()
        self.coordinate_loss = nn.MSELoss()
        self.orientation_loss = OrientationLoss()

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        device = batch.device
        bsz = batch.size(0)

        # sample t from [1, T]. Shape: bsz,
        t = torch.randint(low=1, high=self.T + 1, size=(bsz,)).to(device)

        # sample noisy sequence at timestep t.
        seq_t0 = batch["seq"]
        seq_t, seq_posterior = self.seq_diffuser.diffuse_from_t0(
            seq_t0, t, return_posterior=True
        )

        # sample noisy coordinates at timestep t.
        xyz_t0 = batch["xyz"]
        xyz_t, xyz_eps = self.coordinate_diffuser.diffuse_from_t0(xyz_t0, t, return_eps=True)

        # sample noisy orientations at timestep t.
        rotmat_t0 = batch["rotmat"]
        rotmat_t = self.orientation_diffuser.diffuse_from_t0(rotmat_t0, t)

        # predict sequence posterior probs at timestep t-1 (`seq_posterior`),
        # noise added to xyz (`xyz_eps`), and orientation at timestep t0 (`rotmat`)
        out = self.forward(seq_t, xyz_t, rotmat_t)
        # out['seq_posterior'], out['xyz_eps'], out['rotmat_t0']

        # compute loss
        seq_loss = self.aa_loss(out["seq_posterior"].log(), seq_posterior)
        xyz_loss = self.coordinate_loss(out["xyz_eps"], xyz_eps)
        rotmat_loss = self.orientation_loss(out["rotmat_t0"], rotmat_t0)

        loss = seq_loss + xyz_loss + rotmat_loss

        # log loss to wandb and progressbar
        self.log_dict(
            {
                "train/seq_loss": seq_loss,
                "train/xyz_loss": xyz_loss,
                "train/rotmat_loss": rotmat_loss,
                "train/loss": loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass
