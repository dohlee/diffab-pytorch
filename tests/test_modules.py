import torch

from diffab_pytorch.diffab_pytorch import AngularEncoding, ResidueEmbedding, PairEmbedding

from scipy.spatial.transform import Rotation as R
from protstruc import StructureBatch
import protstruc.geometry as geom


def test_AngularEncoding():
    bsz, n_res, d_in = 32, 16, 3

    x = torch.rand(bsz, n_res, d_in)
    ang_enc = AngularEncoding(num_funcs=3)

    assert ang_enc.get_output_dimension(d_in) == d_in * (3 * 2 * 2 + 1)

    d_out = d_in * (3 * 2 * 2 + 1)
    assert ang_enc(x).shape == (bsz, n_res, d_out)


def test_ResidueEmbedding():
    bsz, n_res, n_max_atoms_per_residue = 32, 16, 25
    xyz = torch.rand(bsz, n_res, n_max_atoms_per_residue, 3).numpy()
    chain_idx = torch.zeros(bsz, n_res, dtype=torch.long).numpy()
    chain_idx[:, 10:20] = 1.0
    chain_idx[:, 20:] = 2.0

    sb = StructureBatch.from_xyz(xyz)
    assert sb.get_max_n_atoms_per_residue() == n_max_atoms_per_residue

    d_feat = 32
    res_emb = ResidueEmbedding(
        max_n_atoms_per_residue=n_max_atoms_per_residue,
        d_feat=d_feat,
    )

    seq = torch.randint(0, 20, (bsz, n_res))
    xyz = torch.tensor(xyz).float()

    dihedrals, dihedral_mask = sb.get_backbone_dihedrals()
    dihedrals = torch.tensor(dihedrals).float()

    chain_idx = torch.from_numpy(chain_idx).long()
    orientation = (
        torch.tensor(R.random(bsz * n_res).as_matrix()).reshape(bsz, n_res, 3, 3).float()
    )
    atom_mask = torch.ones(bsz, n_res, n_max_atoms_per_residue)

    out = res_emb(seq, xyz, dihedrals, chain_idx, orientation, atom_mask)
    assert out.shape == (bsz, n_res, d_feat)


def test_PairEmbedding():
    bsz, n_res, n_max_atoms_per_residue = 32, 16, 25
    xyz = torch.rand(bsz, n_res, n_max_atoms_per_residue, 3).numpy()
    chain_idx = torch.zeros(bsz, n_res, dtype=torch.long).numpy()
    chain_idx[:, 10:20] = 1.0
    chain_idx[:, 20:] = 2.0

    sb = StructureBatch.from_xyz(xyz)
    assert sb.get_max_n_atoms_per_residue() == n_max_atoms_per_residue

    d_feat = 32
    pair_emb = PairEmbedding(
        max_n_atoms_per_residue=n_max_atoms_per_residue,
        d_feat=d_feat,
        max_dist_to_consider=32,
    )

    seq = torch.randint(0, 20, (bsz, n_res))
    residue_idx = torch.arange(n_res).repeat(bsz, 1)
    chain_idx = torch.from_numpy(chain_idx).long()

    xyz = torch.tensor(xyz).float()
    distmat = torch.tensor(sb.pairwise_distance_matrix())

    N_IDX, CA_IDX, C_IDX = 0, 1, 2
    n_coords, ca_coords, c_coords = xyz[:, :, N_IDX], xyz[:, :, CA_IDX], xyz[:, :, C_IDX]
    phi = geom.dihedral(
        c_coords[:, :, None].expand(-1, n_res, n_res, 3).numpy(),
        n_coords[:, None, :].expand(-1, n_res, n_res, 3).numpy(),
        ca_coords[:, None, :].expand(-1, n_res, n_res, 3).numpy(),
        c_coords[:, None, :].expand(-1, n_res, n_res, 3).numpy(),
    )
    psi = geom.dihedral(
        n_coords[:, :, None].expand(-1, n_res, n_res, 3).numpy(),
        ca_coords[:, :, None].expand(-1, n_res, n_res, 3).numpy(),
        c_coords[:, :, None].expand(-1, n_res, n_res, 3).numpy(),
        n_coords[:, None, :].expand(-1, n_res, n_res, 3).numpy(),
    )

    phi, psi = torch.tensor(phi).float(), torch.tensor(psi).float()
    dihedrals = torch.stack([phi, psi], dim=-1)

    atom_mask = torch.ones(bsz, n_res, n_max_atoms_per_residue)

    out = pair_emb(
        seq,
        residue_idx,
        chain_idx,
        distmat,
        dihedrals,
        atom_mask,
    )

    assert out.shape == (bsz, n_res, n_res, d_feat)
