import pytest
import torch

from diffab_pytorch.so3 import (
    tensor_trace,
    log_rotmat,
    skew_symmetric_mat_to_vector,
    uniform,
)


def test_tensor_trace():
    bsz, L = 32, 100
    R = uniform(bsz, L, 3, 3)

    assert tensor_trace(R).shape == (bsz, L)


def test_log_rotmat():
    bsz, L = 32, 100
    R = uniform(bsz, L, 3, 3)

    S = log_rotmat(R)

    assert S.shape == (bsz, L, 3, 3)
    assert torch.allclose(S, -S.transpose(2, 3))


def test_skew_symmetric_mat_to_vector():
    bsz, L = 32, 100

    R = uniform(bsz, L, 3, 3)
    S = log_rotmat(R)

    v = skew_symmetric_mat_to_vector(S)
    assert v.shape == (bsz, L, 3)


def test_uniform():
    bsz, L = 32, 100

    R = uniform(bsz, L, 3, 3)
    assert R.shape == (bsz, L, 3, 3)
