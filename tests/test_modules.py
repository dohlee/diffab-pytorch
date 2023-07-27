import torch

from diffab_pytorch.diffab_pytorch import AngularEncoding


def test_AngularEncoding():
    bsz, n_res, d_in = 32, 16, 3

    x = torch.rand(bsz, n_res, d_in)
    ang_enc = AngularEncoding(num_funcs=3)

    assert ang_enc.get_output_dimension(d_in) == d_in * (3 * 2 * 2 + 1)

    d_out = d_in * (3 * 2 * 2 + 1)
    assert ang_enc(x).shape == (bsz, n_res, d_out)
