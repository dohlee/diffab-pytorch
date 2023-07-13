import torch
import torch.nn.functional as F

from einops import rearrange
from scipy.spatial.transform import Rotation


def uniform(*size):
    assert len(size) >= 2, "size must be at least 2-dimensional"
    assert size[-2] == size[-1] == 3, "last two dimensions must be 3"

    # sample from uniform distribution
    n_sample = 1
    for dim in size[:-2]:
        n_sample *= dim

    R = Rotation.random(n_sample).as_matrix().reshape(*size)
    return torch.tensor(R).float()

def isotropic_gaussian_on_so3(sigma, size):
    assert len(size) >= 2, "size must be at least 2-dimensional"
    assert size[-2] == size[-1] == 3, "last two dimensions must be 3"

    # sample rotational axis from uniform distribution on S^2 (unit sphere)
    # this can be effectively done by sampling 3D standard Gaussian and
    # normalizing the sampled vector to unit length
    u = F.normalize(torch.randn(*size[:-1]), dim=-1) # *, L, 3

    # sample rotation angle from Gaussian distribution
    pass


def tensor_trace(T):
    return T.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1)


def log_rotmat(R):
    """
    Compute skew-symmetric matrix of R,
    which resides in the Lie algebra of SO(3).

    log R = theta / (2 * sin(theta)) * (R - R^T)

    R (torch.tensor): bsz, L, 3, 3
    """
    tr = tensor_trace(R)

    cos_theta = (tr - 1) / 2
    theta = torch.acos(cos_theta)
    theta = rearrange(theta, "b l -> b l () ()")

    # TODO: should handle when theta == 0
    return theta / (2 * torch.sin(theta)) * (R - R.transpose(-1, -2))


def skew_symmetric_mat_to_vector(S):
    v_x = S[..., 2, 1]
    v_y = S[..., 0, 2]
    v_z = S[..., 1, 0]

    return torch.stack([v_x, v_y, v_z], dim=-1)


def exp_skew_symmetric_mat(S):
    """
    Compute the exponential of a skew-symmetric matrix S
    to produce a rotation matrix R in SO(3).

    exp S = I + S * sin(theta) + S^2 * (1 - cos(theta)).

    S (torch.tensor): bsz, L, 3, 3
    """
    v = skew_symmetric_mat_to_vector(S)

    norm = v.norm(dim=-1)
    norm = rearrange(norm, "b l -> b l () ()")

    iden = torch.eye(3).to(S.device).expand_as(S)

    R = iden + S * torch.sin(norm) / norm + S @ S * (1 - torch.cos(norm)) / norm**2

    return R


def scale_rot(R, k):
    """
    Scale a rotation matrix R by a scalar k.

    k (int): scalar
    """
    return exp_skew_symmetric_mat(k * log_rotmat(R))
