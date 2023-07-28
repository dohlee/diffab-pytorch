import torch

from diffab_pytorch.diffab_pytorch import AngularEncoding, ResidueEmbedding

from protstruc import StructureBatch


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

    res_emb = ResidueEmbedding()

    seq = torch.randint(0, 20, (bsz, n_res))
    dihedrals = sb.get_backbone_dihedrals()
