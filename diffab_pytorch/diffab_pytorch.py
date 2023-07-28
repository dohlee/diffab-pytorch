import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from einops import rearrange, repeat

from torch import einsum
from diffab_pytorch.diffusion import (
    SequenceDiffuser,
    CoordinateDiffuser,
    OrientationDiffuser,
)


class AngularEncoding(nn.Module):
    def __init__(self, num_funcs=3):
        super().__init__()
        self.num_funcs = num_funcs
        self.freq_bands = torch.tensor(
            [i + 1.0 for i in range(num_funcs)] + [1.0 / (i + 1.0) for i in range(num_funcs)]
        ).float()

    def get_output_dimension(self, d_in):
        return d_in * (self.num_funcs * 2 * 2 + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute angular encoding of a vector x.

        Args:
            x (torch.Tensor): Shape: ..., d_in

        Returns:
            torch.Tensor: ..., d_in * (num_funcs * 2 * 2 + 1)
        """
        x = x.unsqueeze(-1)  # ..., d_in, 1

        encoded = torch.cat(
            [
                x,  # ..., d_in, 1
                torch.sin(self.freq_bands * x),  # ..., d_in, num_funcs * 2
                torch.cos(self.freq_bands * x),  # ..., d_in, num_funcs * 2
            ],
            dim=-1,
        )  # ..., d_in, num_funcs * 2 * 2 + 1

        encoded = rearrange(encoded, "... d1 d2 -> ... (d1 d2)")

        return encoded


class ResidueEmbedding(nn.Module):
    def __init__(self, max_n_atoms_per_residue, d_feat):
        super().__init__()
        self.max_n_amino_acid_types = 20  # TODO: why 22?
        self.max_n_atoms_per_residue = max_n_atoms_per_residue

        self.amino_acid_type_embedding = nn.Embedding(self.max_n_amino_acid_types, d_feat)
        self.dihedral_embedding = AngularEncoding(num_funcs=3)
        self.chain_embedding = nn.Embedding(10, d_feat, padding_idx=0)

        d_coord = self.max_n_amino_acid_types * max_n_atoms_per_residue * 3
        d_dihedral = self.dihedral_embedding.get_output_dimension(3)
        d_flag = 1

        self.mlp = nn.Sequential(
            nn.Linear(d_feat + d_coord + d_dihedral + d_flag, d_feat * 2),
            nn.ReLU(),
            nn.Linear(d_feat * 2, d_feat),
            nn.ReLU(),
            nn.Linear(d_feat, d_feat),
            nn.ReLU(),
            nn.Linear(d_feat, d_feat),
        )

    def forward(
        self,
        seq: torch.Tensor,
        xyz: torch.Tensor,
        dihedrals: torch.Tensor,
        chain_idx: torch.Tensor,
        orientation: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute residue-wise embedding using sequence, coordinate and orientation.

        Args:
            seq (torch.Tensor): Shape: bsz, L
            xyz (torch.Tensor): Shape: bsz, L, A, 3
            dihedrals (torch.Tensor): Shape: bsz, L, 3
            chain_idx (torch.Tensor): Shape: bsz, L
            orientation (torch.Tensor): Shape: bsz, L, 3, 3
            atom_mask (torch.Tensor): Mask denoting whether it is a valid Shape: bsz, L, A

        Returns:
            torch.Tensor: Shape: bsz, L, d_feat
        """
        bsz, L = seq.shape
        N_IDX, CA_IDX, C_IDX, CB_IDX = 0, 1, 2, 3
        residue_mask = atom_mask[:, :, CA_IDX]

        # amino acid type embedding
        aa_type_feat = self.amino_acid_type_embedding(seq)  # bsz, L, d_feat

        # coordinate embedding
        # first compute local coordinate O^{T} * (X - X_{CA})
        xyz_rel = xyz - xyz[:, :, CA_IDX, :].unsqueeze(-2)  # bsz, L, A, 3
        xyz_local = orientation.transpose(-1, -2) @ xyz_rel

        seq_expanded = repeat(
            seq,
            "b l -> b l t a d",
            f=self.max_n_amino_acid_types,
            a=self.max_n_atoms_per_residue,
            d=3,
        )
        xyz_expanded = repeat(
            xyz_local,
            "b l a d -> b l t a d",
            f=self.max_n_amino_acid_types,
        )
        idx_expanded = repeat(
            torch.arange(self.max_n_amino_acid_types).to(xyz.device),
            "t -> b l t a d",
            b=bsz,
            l=L,
            a=self.max_n_atoms_per_residue,
            d=3,
        )
        coord_feat = torch.where(
            seq_expanded == idx_expanded,
            xyz_expanded,
            torch.zeros_like(xyz_expanded),
        )

        coord_feat = rearrange(coord_feat, "b l t a d -> b l (t a d)")  # bsz, L, D
        # where D = self.max_n_amino_acid_types * max_n_atoms_per_residue * 3

        # dihedral embedding by applying angular encoding
        dihedral_feat = self.dihedral_embedding(dihedrals)

        # chain embedding
        chain_feat = self.chain_embedding(chain_idx)  # bsz, L, d_feat

        x = torch.cat(
            [
                aa_type_feat,
                coord_feat,
                dihedral_feat,
                chain_feat,
            ]
        )
        return self.mlp(x)


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
