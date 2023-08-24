import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from einops import rearrange, repeat
from torch import einsum

from protstruc.general import ATOM, AA

from diffab_pytorch.diffusion import (
    cosine_variance_schedule,
    CoordinateDiffuser,
    OrientationDiffuser,
    SequenceDiffuser,
)
from diffab_pytorch.so3 import vector_to_rotation_matrix


class AngularEncoding(nn.Module):
    def __init__(self, num_funcs=3):
        super().__init__()
        self.num_funcs = num_funcs
        self.freq_bands = torch.tensor(
            [i + 1.0 for i in range(num_funcs)]
            + [1.0 / (i + 1.0) for i in range(num_funcs)]
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
        self.max_n_aa_types = 21  # TODO: why 22?
        self.max_n_atoms_per_residue = max_n_atoms_per_residue

        self.amino_acid_type_embedding = nn.Embedding(self.max_n_aa_types, d_feat)
        self.dihedral_embedding = AngularEncoding(num_funcs=3)
        self.chain_embedding = nn.Embedding(10, d_feat, padding_idx=0)

        d_coord = self.max_n_aa_types * max_n_atoms_per_residue * 3
        d_dihedral = self.dihedral_embedding.get_output_dimension(3)
        # d_flag = 1

        self.mlp = nn.Sequential(
            nn.Linear(d_feat + d_coord + d_dihedral + d_feat, d_feat * 2),
            nn.ReLU(),
            nn.Linear(d_feat * 2, d_feat),
            nn.ReLU(),
            nn.Linear(d_feat, d_feat),
            nn.ReLU(),
            nn.Linear(d_feat, d_feat),
        )

    def forward(
        self,
        seq_idx: torch.Tensor,
        xyz: torch.Tensor,
        orientation: torch.Tensor,
        dihedrals: torch.Tensor,
        chain_idx: torch.Tensor,
        atom_mask: torch.Tensor,
        structure_context_mask: torch.Tensor = None,
        sequence_context_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute residue-wise embedding using sequence, coordinate and orientation.

        Args:
            seq_idx (torch.Tensor): Shape: bsz, L
            xyz (torch.Tensor): Shape: bsz, L, A, 3
            orientation (torch.Tensor): Shape: bsz, L, 3, 3
            dihedrals (torch.Tensor): Shape: bsz, L, 3
            chain_idx (torch.Tensor): Shape: bsz, L
            atom_mask (torch.Tensor): Mask denoting whether it is a valid Shape: bsz, L, A
            structure_context_mask: Mask denoting whether a residue is in the structural context.
                Shape: bsz, L
            sequence_context_mask: Mask denoting whether a residue is in the sequence context.
                Shape: bsz, L

        Returns:
            torch.Tensor: Shape: bsz, L, d_feat
        """
        bsz, L = seq_idx.shape
        CA_IDX = 1

        # amino acid type embedding
        if sequence_context_mask is not None:  # fill non-context residues with UNK
            _mask = sequence_context_mask.bool()
            seq_idx = torch.where(_mask, seq_idx, torch.full_like(seq_idx, AA.UNK))
        aa_type_feat = self.amino_acid_type_embedding(seq_idx)  # bsz, L, d_feat

        # coordinate embedding
        # first compute local coordinate O^{T} * (X - X_{CA})
        xyz_rel = xyz - xyz[:, :, CA_IDX, :].unsqueeze(
            -2
        )  # bsz, L, A, 3 (global coords)
        orientation_transposed = rearrange(orientation, "b l r1 r2 -> b l r2 r1")
        xyz_local = einsum(
            "b l i j, b l a j -> b l a i", orientation_transposed, xyz_rel
        )
        xyz_local *= atom_mask[:, :, :, None]  # bsz, L, A, 3 (local coords)

        seq_expanded = repeat(
            seq_idx,
            "b l -> b l t a d",
            t=self.max_n_aa_types,
            a=self.max_n_atoms_per_residue,
            d=3,
        )
        xyz_expanded = repeat(
            xyz_local,
            "b l a d -> b l t a d",
            t=self.max_n_aa_types,
        )
        idx_expanded = repeat(
            torch.arange(self.max_n_aa_types).to(xyz.device),
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
        if structure_context_mask is not None:
            coord_feat *= structure_context_mask[:, :, None]

        # dihedral embedding by applying angular encoding
        dihedral_feat = self.dihedral_embedding(dihedrals)
        if structure_context_mask is not None:
            dihedral_mask = torch.stack(
                [
                    torch.roll(structure_context_mask, shifts=s, dims=1)
                    for s in range(-1, 1)
                ]
            ).all(axis=0)
            dihedral_feat *= dihedral_mask[:, :, None]

        # chain embedding
        chain_feat = self.chain_embedding(chain_idx)  # bsz, L, d_feat

        x = torch.cat(
            [
                aa_type_feat,
                coord_feat,
                dihedral_feat,
                chain_feat,
            ],
            dim=-1,
        )
        return self.mlp(x)


class PairEmbedding(nn.Module):
    def __init__(self, max_n_atoms_per_residue, d_feat, max_dist_to_consider=32):
        super().__init__()

        self.d_feat = d_feat
        self.max_dist_to_consider = max_dist_to_consider

        self.max_n_aa_types = 21  # TODO: why 22?
        self.aa_pair_type_embedding = nn.Embedding(self.max_n_aa_types**2, d_feat)

        self.relpos_embedding = nn.Embedding(2 * max_dist_to_consider + 1, d_feat)

        self.pair2distcoef = nn.Embedding(
            self.max_n_aa_types**2, max_n_atoms_per_residue**2
        )
        nn.init.zeros_(self.pair2distcoef.weight)
        self.distance_embedding = nn.Sequential(
            nn.Linear(max_n_atoms_per_residue**2, d_feat),
            nn.ReLU(),
            nn.Linear(d_feat, d_feat),
            nn.ReLU(),
        )

        self.dihedral_embedding = AngularEncoding(2)  # phi and psi
        d_dihedral = self.dihedral_embedding.get_output_dimension(2)

        self.mlp = nn.Sequential(
            nn.Linear(d_feat + d_feat + d_feat + d_dihedral, d_feat),
            nn.ReLU(),
            nn.Linear(d_feat, d_feat),
            nn.ReLU(),
            nn.Linear(d_feat, d_feat),
        )

    def forward(
        self,
        seq_idx: torch.Tensor,
        distmat: torch.Tensor,
        dihedrals: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_idx: torch.Tensor,
        atom_mask: torch.Tensor,
        structure_context_mask: torch.Tensor,
        sequence_context_mask: torch.Tensor,
    ) -> torch.Tensor:
        """_summary_

        Args:
            seq_idx: Amino acid sequence. Shape: bsz, L
            residue_idx: Residue index. Shape: bsz, L
            chain_idx: Chain index. Shape: bsz, L
            distmat: Inter-residue atom distances. Shape: bsz, L, L, A, A
            dihedrals: Inter-residue phi and psi angles. Shape: bsz, L, L, 2
            atom_mask: Whether an atom is valid for each residue. Shape: bsz, L, A
            structure_context_mask: Whether a residue is in the structural context.
                Shape: bsz, L
            sequence_context_mask: Whether a residue is in the sequence context.
                Shape: bsz, L

        Returns:
            Shape: bsz, L, L, d_feat
        """
        bsz, L = seq_idx.shape
        CA_IDX = 1

        if structure_context_mask is not None:
            pair_structure_context_mask = (
                structure_context_mask[:, :, None] * structure_context_mask[:, None, :]
            )
        if sequence_context_mask is not None:
            pair_sequence_context_mask = (
                sequence_context_mask[:, :, None] * sequence_context_mask[:, None, :]
            )

        atom_mask_pair = atom_mask[:, :, None, :, None] * atom_mask[:, None, :, None, :]
        atom_mask_pair = rearrange(
            atom_mask_pair, "b l1 l2 a1 a2 -> b l1 l2 (a1 a2)"
        )  # bsz, L, L, A * A

        residue_mask = atom_mask[:, :, CA_IDX]  # bsz, L
        residue_mask_pair = (
            residue_mask[:, :, None] * residue_mask[:, None, :]
        )  # bsz, L, L

        # amino acid type pair embedding
        if sequence_context_mask is not None:  # fill non-context residues with UNK
            _mask = sequence_context_mask.bool()
            seq_idx = torch.where(_mask, seq_idx, torch.full_like(seq_idx, AA.UNK))

        seq_pair = seq_idx[:, :, None] * self.max_n_aa_types + seq_idx[:, None, :]
        seq_pair_feat = self.aa_pair_type_embedding(seq_pair)  # bsz, L, L, d_feat

        # relative 1D position embedding
        same_chain_mask = chain_idx[:, :, None] * chain_idx[:, None, :]
        relpos = residue_idx[:, :, None] - residue_idx[:, None, :]
        relpos = relpos.clamp(
            -self.max_dist_to_consider, self.max_dist_to_consider
        )  # bsz, L, L
        relpos_feat = self.relpos_embedding(relpos + self.max_dist_to_consider)
        relpos_feat = relpos_feat * same_chain_mask[:, :, :, None]

        # distance embedding
        coef = F.softplus(self.pair2distcoef(seq_pair))  # bsz, L, L, A * A
        distmat = rearrange(distmat, "b l1 l2 a1 a2 -> b l1 l2 (a1 a2)")

        distmat = torch.exp(-1 * coef * distmat**2)  # bsz, L, L, A * A
        dist_feat = self.distance_embedding(
            distmat * atom_mask_pair
        )  # bsz, L, L, d_feat
        if structure_context_mask is not None:
            distmat *= pair_structure_context_mask[:, :, :, None]

        # dihedral embedding
        dihedral_feat = self.dihedral_embedding(dihedrals)  # bsz, L, L, d_dihedral
        if structure_context_mask is not None:
            distmat *= pair_structure_context_mask[:, :, :, None]

        x = torch.cat(
            [
                seq_pair_feat,
                relpos_feat,
                dist_feat,
                dihedral_feat,
            ],
            dim=-1,
        )
        return self.mlp(x) * residue_mask_pair[:, :, :, None]


def euclidean_transform(x, r, t):
    """r: rotation matrix of size (b, l, 3, 3)
    t: translation vector of size (b, l, 3)
    """
    # infer number of heads
    n = x.size(1)
    r = repeat(r, "b l x y -> b n l x y", n=n)
    t = repeat(t, "b l x -> b n l () x", n=n)

    return einsum("b n l d k, b n l k c -> b n l d c", x, r) + t


def inverse_euclidean_transform(x, r, t):
    """r: rotation matrix of size (b, l, 3, 3)
    t: translation vector of size (b, l, 3)
    """
    # infer number of heads
    n = x.size(1)
    r_inv = repeat(r, "b l x y  -> b n l y x", n=n)  # note that R^-1 = R^T
    t = repeat(t, "b l x -> b n l () x", n=n)

    return einsum("b n l d k, b n l k c -> b n l d c", x - t, r_inv)


class InvariantPointAttentionLayer(nn.Module):
    def __init__(
        self,
        d_residue_emb,
        d_pair_emb,
        d_scalar_per_head=16,
        n_query_point_per_head=4,
        n_value_point_per_head=4,
        n_head=8,
        use_pair_bias=True,
    ):
        super().__init__()
        self.n_head = n_head
        self.use_pair_bias = use_pair_bias

        # standard self-attention (scalar attention)
        d_scalar = d_scalar_per_head * n_head
        self.to_q_scalar = nn.Linear(d_residue_emb, d_scalar, bias=False)
        self.to_k_scalar = nn.Linear(d_residue_emb, d_scalar, bias=False)
        self.to_v_scalar = nn.Linear(d_residue_emb, d_scalar, bias=False)
        self.scale_scalar = d_scalar_per_head**-0.5

        # modulation by pair representation
        if self.use_pair_bias:
            self.to_pair_bias = nn.Linear(d_pair_emb, n_head, bias=False)

        # point attention
        d_query_point = (n_query_point_per_head * 3) * n_head
        d_value_point = (n_value_point_per_head * 3) * n_head
        n_value_point = n_value_point_per_head * n_head
        self.to_q_point = nn.Linear(d_residue_emb, d_query_point, bias=False)
        self.to_k_point = nn.Linear(d_residue_emb, d_query_point, bias=False)
        self.to_v_point = nn.Linear(d_residue_emb, d_value_point, bias=False)
        self.scale_point = (4.5 * n_query_point_per_head) ** -0.5
        self.gamma = nn.Parameter(torch.log(torch.exp(torch.ones(n_head)) - 1.0))

        if use_pair_bias:
            d_pair = d_pair_emb * n_head
            self.to_out = nn.Linear(
                d_scalar + d_pair + d_value_point + n_value_point, d_residue_emb
            )
        else:
            self.to_out = nn.Linear(
                d_scalar + d_value_point + n_value_point, d_residue_emb
            )

        self.num_independent_logits = 3 if use_pair_bias else 2

        self.scale_total = self.num_independent_logits**-0.5

    def forward(self, x, e, r, t):
        # query, key and values for scalar
        q_scalar = self.to_q_scalar(x)
        k_scalar = self.to_k_scalar(x)
        v_scalar = self.to_v_scalar(x)

        q_scalar, k_scalar, v_scalar = map(
            lambda t: rearrange(t, "b l (n d) -> b n l d", n=self.n_head),
            (q_scalar, k_scalar, v_scalar),
        )

        # query, key and values for points
        q_point = self.to_q_point(x)
        k_point = self.to_k_point(x)
        v_point = self.to_v_point(x)

        q_point, k_point, v_point = map(
            lambda t: rearrange(t, "b l (n p c) -> b n l p c", n=self.n_head, c=3),
            (q_point, k_point, v_point),
        )

        q_point, k_point, v_point = map(
            lambda v: euclidean_transform(v, r, t),
            (q_point, k_point, v_point),
        )

        # standard self-attention (scalar attention)
        logit_scalar = (
            einsum("b n i d, b n j d -> b n i j", q_scalar, k_scalar)
            * self.scale_scalar
        )

        # modulation by pair representation
        if self.use_pair_bias:
            bias_pair = rearrange(self.to_pair_bias(e), "b i j n -> b n i j")

        # point attention
        all_pairwise_diff = rearrange(q_point, "b n i p c -> b n i () p c") - rearrange(
            k_point, "b n j p c -> b n () j p c"
        )
        gamma = rearrange(self.gamma, "n -> () n () ()")

        logit_point = (
            -0.5
            * self.scale_point
            * gamma
            * (all_pairwise_diff**2).sum(dim=-1).sum(dim=-1)
        )

        if self.use_pair_bias:
            logit = self.scale_total * (logit_scalar + bias_pair + logit_point)
        else:
            logit = self.scale_total * (logit_scalar + logit_point)

        attn = logit.softmax(dim=-1)

        out_scalar = einsum("b n i j, b n j d -> b n i d", attn, v_scalar)
        out_scalar = rearrange(out_scalar, "b n i d -> b i (n d)")

        if self.use_pair_bias:
            out_pair = einsum("b n i j, b i j d -> b n i d", attn, e)
            out_pair = rearrange(out_pair, "b n i d -> b i (n d)")

        out_point = einsum("b n i j, b n j p c -> b n i p c", attn, v_point)
        out_point = inverse_euclidean_transform(out_point, r, t)
        out_point_norm = out_point.norm(dim=-1, keepdim=True)

        out_point = rearrange(out_point, "b n i p c -> b i (n p c)")
        out_point_norm = rearrange(out_point_norm, "b n i p c -> b i (n p c)")

        if self.use_pair_bias:
            out = torch.cat([out_scalar, out_pair, out_point, out_point_norm], dim=-1)
        else:
            out = torch.cat([out_scalar, out_point, out_point_norm], dim=-1)

        x = self.to_out(out)
        return x


class InvariantPointAttentionModule(nn.Module):
    def __init__(
        self,
        n_layers,
        d_residue_emb,
        d_pair_emb,
        d_scalar_per_head,
        n_query_point_per_head,
        n_value_point_per_head,
        n_head,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                InvariantPointAttentionLayer(
                    d_residue_emb,
                    d_pair_emb,
                    d_scalar_per_head,
                    n_query_point_per_head,
                    n_value_point_per_head,
                    n_head,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, res_emb, pair_emb, orientations, translations):
        for layer in self.layers:
            res_emb = layer(res_emb, pair_emb, orientations, translations)

        return res_emb


class Denoiser(nn.Module):
    def __init__(
        self,
        d_residue_emb,
        d_pair_emb,
        n_ipa_layers,
        d_scalar_per_head,
        n_query_point_per_head,
        n_value_point_per_head,
        n_head,
        aa_vocab_size,
    ):
        super().__init__()
        self.sequence_embedding = nn.Embedding(25, d_residue_emb)
        self.to_res_emb = nn.Sequential(
            nn.Linear(d_residue_emb * 2, d_residue_emb),
            nn.ReLU(),
            nn.Linear(d_residue_emb, d_residue_emb),
        )

        self.ipa = InvariantPointAttentionModule(
            n_ipa_layers,
            d_residue_emb,
            d_pair_emb,
            d_scalar_per_head,
            n_query_point_per_head,
            n_value_point_per_head,
            n_head,
        )

        d_beta_emb = 3  # dimension for beta encoding

        self.coordinate_denoising = nn.Sequential(
            nn.Linear(d_residue_emb + d_beta_emb, d_residue_emb),
            nn.ReLU(),
            nn.Linear(d_residue_emb, d_residue_emb),
            nn.ReLU(),
            nn.Linear(d_residue_emb, 3),
        )

        self.orientation_denoising = nn.Sequential(
            nn.Linear(d_residue_emb + d_beta_emb, d_residue_emb),
            nn.ReLU(),
            nn.Linear(d_residue_emb, d_residue_emb),
            nn.ReLU(),
            nn.Linear(d_residue_emb, 3),
        )

        self.sequence_denoising = nn.Sequential(
            nn.Linear(d_residue_emb + d_beta_emb, d_residue_emb),
            nn.ReLU(),
            nn.Linear(d_residue_emb, d_residue_emb),
            nn.ReLU(),
            nn.Linear(d_residue_emb, aa_vocab_size),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        seq_idx_t: torch.Tensor,
        translations_t: torch.Tensor,
        orientations_t: torch.Tensor,
        res_context_emb: torch.Tensor,
        pair_context_emb: torch.Tensor,
        beta: torch.Tensor,
        generation_mask: torch.Tensor,
        residue_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, n_residues = seq_idx_t.shape[:2]

        # embed current sequence information (s_t) to res_emb
        s_emb = self.sequence_embedding(seq_idx_t)  # b n d_emb
        res_context_emb = torch.cat([res_context_emb, s_emb], dim=-1)  # b n (d_emb * 2)
        res_context_emb = self.to_res_emb(res_context_emb)  # b n d_emb

        # embed pair_emb to res_emb using Invariant Point Attention
        # NOTE: current orientation and translations are used to calculate attention between
        # residue pairs
        res_context_emb = self.ipa(
            res_context_emb, pair_context_emb, orientations_t, translations_t
        )

        # variance (or timepoint, equivalently) embedding
        t_emb = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # b 3
        t_emb = repeat(t_emb, "b d -> b n d", n=n_residues)  # b n 3

        # finalize residue embedding
        res_context_emb = torch.cat([res_context_emb, t_emb], dim=-1)  # b n (d_emb + 3)

        # denoise coordinates and predict epsilon
        translations_eps = self.coordinate_denoising(res_context_emb)  # b n 3

        # denoise orientations
        v_eps = self.orientation_denoising(res_context_emb)  # b n 3, rotation vector
        o_eps = vector_to_rotation_matrix(v_eps)  # b n 3 3, rotation matrix
        o_denoised = orientations_t @ o_eps  # b n 3 3

        # denoise sequence probabilities
        s_denoised_prob = self.sequence_denoising(res_context_emb)  # b n 21

        out = {
            "translations_eps": translations_eps,
            "orientations_t0": o_denoised,
            "seq_posterior": s_denoised_prob,
        }

        return out


class OrientationLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self, pred_rotmat: torch.Tensor, target_rotmat: torch.Tensor
    ) -> torch.Tensor:
        device = pred_rotmat.device

        rot_discrepancy = einsum(
            "b l i j, b l i k -> b l j k", pred_rotmat, target_rotmat
        )
        eye = torch.eye(3).to(device).expand_as(rot_discrepancy)

        return F.mse_loss(rot_discrepancy, eye, reduction=self.reduction)


class DiffAb(pl.LightningModule):
    def __init__(
        self,
        d_residue_emb,
        d_pair_emb,
        n_ipa_layers,
        d_scalar_per_head,
        n_query_point_per_head,
        n_value_point_per_head,
        n_head,
        T=100,
        s=0.01,
        beta_max=0.999,
        n_atoms=15,
        aa_vocab_size=21,
        max_dist_to_consider=32,
        lr=1e-4,
        weight_decay=0.0,
        betas=(0.9, 0.999),
    ):
        super().__init__()

        self.sched = cosine_variance_schedule(T=T, s=s, beta_max=beta_max)
        self.residue_context_embedding = ResidueEmbedding(n_atoms, d_residue_emb)
        self.pair_context_embedding = PairEmbedding(
            n_atoms, d_pair_emb, max_dist_to_consider
        )

        self.denoiser = Denoiser(
            d_residue_emb,
            d_pair_emb,
            n_ipa_layers,
            d_scalar_per_head,
            n_query_point_per_head,
            n_value_point_per_head,
            n_head,
            aa_vocab_size,
        )

        self.seq_diffuser = SequenceDiffuser(T, s, beta_max, aa_vocab_size)
        self.coordinate_diffuser = CoordinateDiffuser(T, s, beta_max)
        self.orientation_diffuser = OrientationDiffuser(T, s, beta_max)

        self.aa_loss = nn.KLDivLoss(reduction="none")
        self.coordinate_loss = nn.MSELoss(reduction="none")
        self.orientation_loss = OrientationLoss(reduction="none")

        self.T = T
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas

    def encode_context(
        self,
        seq_idx_t0,  # b n
        xyz_t0,  # b n a 3
        orientations_t0,  # b n 3 3
        backbone_dihedrals,  # b n 3
        distmat,  # b n n a a
        pairwise_dihedrals,  # b n n 2
        atom_mask: torch.BoolTensor,  # b n a
        chain_idx: torch.LongTensor,  # b n
        residue_idx: torch.LongTensor,  # b n
        generation_mask: torch.BoolTensor,  # b n
        residue_mask: torch.BoolTensor,  # b n
        generate_structure: bool = True,
        generate_sequence: bool = True,
    ):
        # marks context residues which will be only used for
        # computing context embedding (i.e., not generated)
        context_mask = residue_mask & (~generation_mask)
        structure_context_mask = context_mask if generate_structure else None
        sequence_context_mask = context_mask if generate_sequence else None

        res_context_emb = self.residue_context_embedding(
            seq_idx_t0,
            xyz_t0,
            orientations_t0,
            backbone_dihedrals,
            chain_idx,
            atom_mask,
            structure_context_mask,
            sequence_context_mask,
        )

        pair_context_emb = self.pair_context_embedding(
            seq_idx_t0,
            distmat,
            pairwise_dihedrals,
            residue_idx,
            chain_idx,
            atom_mask,
            structure_context_mask,
            sequence_context_mask,
        )

        return res_context_emb, pair_context_emb

    def denoise(
        self,
        seq_idx_t,
        translations_t,
        orientations_t,
        res_context_emb,
        pair_context_emb,
        beta,
        generation_mask,
        residue_mask,
    ) -> Dict[str, torch.Tensor]:
        """Given a noisy sequence, translations and orientations at timestep t,
        denoise them to obtain the sequence posterior, translations at timestep t0,
        and orientations at timestep t0.

        Args:
            seq_idx_t: Noisy amino acid sequence. Shape: bsz, L
            translations_t: Noisy Ca translations. Shape: bsz, L, 3
            orientations_t: Noisy residue orientations. Shape: bsz, L, 3, 3
            res_context_emb: Residue context embedding. Shape: bsz, L, d_emb
            pair_context_emb: Residue-pair context embedding. Shape: bsz, L, L, d_emb
            beta: Scheduled variances for diffusion. Shape: bsz
            generation_mask: Whether a residue is generated. Shape: bsz, L
            residue_mask: Whether a residue is valid. Shape: bsz, L

        Returns:
            A dictionary containing the following keys:
                - seq_posterior:
                - translations_eps: translations_eps,
                - orientations_t0: o_denoised,
        """
        denoised_out = self.denoiser(
            seq_idx_t,
            translations_t,
            orientations_t,
            res_context_emb,
            pair_context_emb,
            beta,
            generation_mask,
            residue_mask,
        )  # seq_posterior, translations_eps, orientations_t0

        return denoised_out

    def sample(
        self,
        seq_idx: torch.LongTensor,
        xyz: torch.FloatTensor,
        orientations: torch.FloatTensor,
    ) -> Dict[str, torch.Tensor]:
        pass

    def _add_noise(
        self,
        seq_idx_t0: torch.LongTensor,
        translations_t0: torch.FloatTensor,
        orientations_t0: torch.FloatTensor,
        generation_mask: torch.BoolTensor,
        t: torch.LongTensor,
    ) -> Dict[str, torch.Tensor]:
        # sample noisy sequence at timestep t.
        seq_idx_t, seq_posterior = self.seq_diffuser.diffuse_from_t0(
            seq_idx_t0, t, generation_mask, return_posterior=True
        )
        # sample noisy coordinates at timestep t.
        translations_t, translations_eps = self.coordinate_diffuser.diffuse_from_t0(
            translations_t0, t, generation_mask, return_eps=True
        )
        # sample noisy orientations at timestep t.
        orientations_t = self.orientation_diffuser.diffuse_from_t0(
            orientations_t0, generation_mask, t
        )

        ret = {}
        ret["seq_idx_t"] = seq_idx_t
        ret["seq_posterior"] = seq_posterior
        ret["translations_t"] = translations_t
        ret["translations_eps"] = translations_eps
        ret["orientations_t"] = orientations_t

        return ret

    def _shared_step(self, batch, batch_idx):
        device = batch["generation_mask"].device
        bsz = batch["generation_mask"].size(0)

        # sample t from [1, T]. Shape: bsz,
        t = torch.randint(low=1, high=self.T + 1, size=(bsz,)).to(device)
        # get corresponding beta from the variance schedule. Shape: bsz,
        beta = self.sched["beta"][t]

        # original protein structures at timestep 0.
        seq_idx_t0 = batch["seq_idx"]
        xyz_t0 = batch["xyz"]
        translations_t0 = xyz_t0[:, :, ATOM.CA]
        orientations_t0 = batch["orientations"]
        generation_mask = batch["generation_mask"]

        noised = self._add_noise(
            seq_idx_t0, translations_t0, orientations_t0, generation_mask, t
        )

        res_context_emb, pair_context_emb = self.encode_context(
            seq_idx_t0,
            xyz_t0,
            orientations_t0,
            batch["backbone_dihedrals"],
            batch["distmat"],
            batch["pairwise_dihedrals"],
            batch["atom_mask"],
            batch["chain_idx"],
            batch["residue_idx"],
            batch["generation_mask"],
            batch["residue_mask"],
        )

        # predict sequence posterior probs at timestep t-1 (`seq_posterior`),
        # noise added to translations (`translations_eps`),
        # and orientation at timestep t0 (`orientations_t0`)
        denoised = self.denoise(
            noised["seq_idx_t"],
            noised["translations_t"],
            noised["orientations_t"],
            res_context_emb,
            pair_context_emb,
            beta,
            batch["generation_mask"],
            batch["residue_mask"],
        )

        # compute loss
        seq_loss = self.aa_loss(
            denoised["seq_posterior"].log(), noised["seq_posterior"]
        )
        translations_loss = self.coordinate_loss(
            denoised["translations_eps"], noised["translations_eps"]
        )
        orientations_loss = self.orientation_loss(
            denoised["orientations_t0"], batch["orientations"]
        )

        # apply masking to losses and average
        loss_mask = batch["generation_mask"] & batch["residue_mask"]
        loss_denom = loss_mask.sum()

        seq_mask = rearrange(loss_mask, "b l -> b l ()")
        seq_loss = (seq_loss * seq_mask).sum() / loss_denom

        translations_mask = rearrange(loss_mask, "b l -> b l ()")
        translations_loss = (translations_loss * translations_mask).sum() / loss_denom

        orientations_mask = rearrange(loss_mask, "b l -> b l () ()")
        orientations_loss = (orientations_loss * orientations_mask).sum() / loss_denom

        return seq_loss, translations_loss, orientations_loss

    def training_step(self, batch, batch_idx):
        seq_loss, translations_loss, orientations_loss = self._shared_step(
            batch, batch_idx
        )
        loss = seq_loss + translations_loss + orientations_loss

        # log loss to wandb and progressbar
        self.log_dict(
            {
                "train/seq_loss": seq_loss,
                "train/translations_loss": translations_loss,
                "train/orientations_loss": orientations_loss,
                "train/loss": loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        seq_loss, translations_loss, orientations_loss = self._shared_step(
            batch, batch_idx
        )
        loss = seq_loss + translations_loss + orientations_loss

        self.log_dict(
            {
                "val/seq_loss": seq_loss,
                "val/translations_loss": translations_loss,
                "val/orientations_loss": orientations_loss,
                "val/loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )
