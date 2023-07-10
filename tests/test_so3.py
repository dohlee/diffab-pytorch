import pytest
import torch

import numpy as np

from torch import einsum
from diffab_pytorch.so3 import (
    uniform,
    tensor_trace,
    log_rotmat,
    skew_symmetric_mat_to_vector,
    exp_skew_symmetric_mat,
    scale_rot,
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


def test_exp_skew_symmetric_mat():
    bsz, L = 32, 100

    R = uniform(bsz, L, 3, 3)
    S = log_rotmat(R)
    R_recon = exp_skew_symmetric_mat(S)

    for i in range(bsz):
        for j in range(L):
            tr = tensor_trace(R[i, j])
            cos_theta = (tr - 1) / 2

            # skip if theta is close to 0 or pi
            # since the computation becomes unstable
            if (cos_theta - 1.0).abs() < 1e-2 or (cos_theta - (-1.0)).abs() < 1e-2:
                continue

            diff = (R[i, j] - R_recon[i, j]).abs().sum()
            assert diff < 1e-4


def test_uniform():
    bsz, L = 32, 100

    R = uniform(bsz, L, 3, 3)
    assert R.shape == (bsz, L, 3, 3)

    # Check that R is a rotation matrix
    # i.e. R^T R = I
    product = einsum("b l i j, b l j k -> b l i k", R.transpose(2, 3), R)
    assert torch.allclose(product, torch.eye(3).expand_as(product), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("k", np.linspace(0.1, 1, 10))
def test_scale_rot(k):
    bsz, L = 32, 100

    R = uniform(bsz, L, 3, 3)
    R_scaled = scale_rot(R, k)

    assert R_scaled.shape == (bsz, L, 3, 3)

    # Check that R_scaled is a rotation matrix
    # i.e. R^T R = I
    product = einsum("b l i j, b l j k -> b l i k", R_scaled.transpose(2, 3), R_scaled)
    assert torch.allclose(product, torch.eye(3).expand_as(product), rtol=1e-5, atol=1e-5)
