import torch

from diffab_pytorch.diffab_pytorch import (
    AngularEncoding,
    ResidueEmbedding,
    PairEmbedding,
    InvariantPointAttentionLayer,
    InvariantPointAttentionModule,
    Denoiser,
    DiffAb,
)

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
    bsz, num_residues, num_atoms = 32, 16, 25
    xyz = torch.rand(bsz, num_residues, num_atoms, 3).numpy()
    atom_mask = torch.ones(bsz, num_residues, num_atoms)

    chain_idx = torch.zeros(bsz, num_residues, dtype=torch.long).numpy()
    chain_idx[:, 10:20] = 1.0
    chain_idx[:, 20:] = 2.0

    sb = StructureBatch.from_xyz(xyz=xyz, atom_mask=atom_mask)
    assert sb.get_max_n_atoms_per_residue() == num_atoms

    d_feat = 32
    res_emb = ResidueEmbedding(
        max_n_atoms_per_residue=num_atoms,
        d_feat=d_feat,
    )

    seq = torch.randint(0, 20, (bsz, num_residues))
    xyz = torch.tensor(xyz).float()

    dihedrals, dihedral_mask = sb.backbone_dihedrals()
    dihedrals = torch.tensor(dihedrals).float()

    chain_idx = torch.from_numpy(chain_idx).long()
    # generate random orientations using scipy.spatial.transform.Rotation
    orientation = sb.backbone_orientations()

    # orientation = (
    # torch.tensor(R.random(bsz * num_residues).as_matrix())
    # .reshape(bsz, num_residues, 3, 3)
    # .float()
    # )

    out = res_emb(seq, xyz, dihedrals, chain_idx, orientation, atom_mask)
    assert out.shape == (bsz, num_residues, d_feat)


def test_PairEmbedding():
    bsz, n_res, n_max_atoms_per_residue = 32, 16, 25
    xyz = torch.rand(bsz, n_res, n_max_atoms_per_residue, 3).numpy()
    chain_idx = torch.zeros(bsz, n_res, dtype=torch.long).numpy()
    chain_idx[:, 10:20] = 1.0
    chain_idx[:, 20:] = 2.0

    atom_mask = torch.ones(bsz, n_res, n_max_atoms_per_residue)

    sb = StructureBatch.from_xyz(xyz=xyz, atom_mask=atom_mask)
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
    distmat, distmat_mask = sb.pairwise_distance_matrix()
    assert distmat.shape == (
        bsz,
        n_res,
        n_res,
        n_max_atoms_per_residue,
        n_max_atoms_per_residue,
    )
    assert distmat_mask.shape == (
        bsz,
        n_res,
        n_res,
        n_max_atoms_per_residue,
        n_max_atoms_per_residue,
    )

    phi = sb.pairwise_dihedrals(atoms_i=["C"], atoms_j=["N", "CA", "C"])
    psi = sb.pairwise_dihedrals(atoms_i=["N", "CA", "C"], atoms_j=["N"])
    assert phi.shape == (bsz, n_res, n_res)
    assert psi.shape == (bsz, n_res, n_res)

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


def test_InvariantPointAttentionLayer():
    d_orig = 32
    d_scalar_per_head = 16
    n_query_point_per_head = 4
    n_value_point_per_head = 4
    n_head = 8

    ipa = InvariantPointAttentionLayer(
        d_orig,
        d_scalar_per_head,
        n_query_point_per_head,
        n_value_point_per_head,
        n_head,
    )

    bsz, n_res = 32, 16
    x = torch.rand(bsz, n_res, d_orig)
    e = torch.rand(bsz, n_res, n_res, d_orig)

    r = torch.rand(bsz, n_res, 3, 3)
    t = torch.rand(bsz, n_res, 3)

    out = ipa(x, e, r, t)
    assert out.shape == (bsz, n_res, d_orig)


def test_InvariantPointAttentionModule():
    d_emb = 32
    d_scalar_per_head = 16

    n_query_point_per_head = 4
    n_value_point_per_head = 4
    n_head = 8

    n_layers = 4

    ipa = InvariantPointAttentionModule(
        n_layers,
        d_emb,
        d_scalar_per_head,
        n_query_point_per_head,
        n_value_point_per_head,
        n_head,
    )

    bsz, n_residues = 32, 16
    res_emb = torch.rand(bsz, n_residues, d_emb)
    pair_emb = torch.randn(bsz, n_residues, n_residues, d_emb)

    orientations = torch.rand(bsz, n_residues, 3, 3)
    translations = torch.rand(bsz, n_residues, 3)

    out = ipa(res_emb, pair_emb, orientations, translations)
    assert out.shape == (bsz, n_residues, d_emb)


def test_Denoiser():
    d_emb = 32
    n_ipa_layers = 4
    d_scalar_per_head = 12
    n_query_point_per_head = 4
    n_value_point_per_head = 4
    n_head = 8

    denoiser = Denoiser(
        d_emb,
        n_ipa_layers,
        d_scalar_per_head,
        n_query_point_per_head,
        n_value_point_per_head,
        n_head,
    )

    bsz, n_residues = 32, 16
    res_emb = torch.rand(bsz, n_residues, d_emb)
    pair_emb = torch.randn(bsz, n_residues, n_residues, d_emb)
    beta = torch.rand(bsz)

    generation_mask = torch.randint(0, 2, (bsz, n_residues))
    residue_mask = torch.randint(0, 2, (bsz, n_residues))

    s_t = torch.randint(0, 20, (bsz, n_residues))  # sequence
    x_t = torch.rand(bsz, n_residues, 3)  # translations
    o_t = torch.rand(bsz, n_residues, 3, 3)  # orientations

    out = denoiser(s_t, x_t, o_t, res_emb, pair_emb, beta, generation_mask, residue_mask)

    assert out["xyz_eps"].shape == (bsz, n_residues, 3)
    assert out["rotmat_t0"].shape == (bsz, n_residues, 3, 3)
    assert out["seq_posterior"].shape == (bsz, n_residues, 20)


def test_DiffAb():
    d_emb = 32
    n_ipa_layers = 4
    d_scalar_per_head = 12
    n_query_point_per_head = 4
    n_value_point_per_head = 4
    n_head = 8

    diffab = DiffAb(
        d_emb,
        n_ipa_layers,
        d_scalar_per_head,
        n_query_point_per_head,
        n_value_point_per_head,
        n_head,
    )

    sb = StructureBatch.from_pdb_id("1REX")

    batch = {
        "xyz": sb.get_xyz(),
        "atom_mask": sb.get_atom_mask(),
        "chain_idx": sb.get_chain_idx(),
        "chain_ids": sb.get_chain_ids(),
        "residue_idx": sb.get_residue_idx(),
    }
