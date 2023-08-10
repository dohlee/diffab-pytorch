import os
import torch
import torch.nn.functional as F

from einops import rearrange
from scipy.spatial.transform import Rotation


class SO3:
    def __init__(
        self,
        sigmas_to_consider,
        cache_prefix=".cache/so3_histograms",
        sigma_threshold=0.1,
        n_bins=8192,
        num_iters=1024,
    ):
        h = hash((n_bins, num_iters, str(sigmas_to_consider)))
        self.cache_path = f"{cache_prefix}/{h}/cache.pt"

        self.n_bins = n_bins
        self.num_iters = num_iters

        # candidate list of sigmas to consider.
        # this may be a whole list of variance along the variance schedule.
        self.sigmas_to_consider = sigmas_to_consider

        # threshold for sigma to determine whether to use histogram-based
        # sampling or not.
        # if sigma < sigma_threshold, use histogram-based sampling,
        # otherwise sample from Gaussian(2*sigma, sigma) truncated to [0, pi)
        self.sigma_threshold = sigma_threshold

        self._initialize()
        self.histograms = torch.load(self.cache_path)  # num_sigmas, bins

    def _initialize(self):
        if os.path.exists(self.cache_path):
            return

        cache_dir = os.path.dirname(self.cache_path)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        histograms = []
        for sigma in self.sigmas_to_consider:
            histograms.append(self._precompute_histogram(sigma))
        histograms = torch.stack(histograms)  # num_sigmas, bins

        torch.save(histograms, self.cache_path)

    def _precompute_histogram(self, sigma):
        # evenly partition [0, pi] into 8192 bins,
        # and use probability density p(theta|sigma^2)
        # at the center of each bin as the bin weight
        binsize = torch.pi / self.n_bins
        bin_centers = torch.arange(0, torch.pi, binsize) + binsize / 2.0

        # compute probability density p(theta|sigma^2) at each bin center
        probs = self._angular_pdf(bin_centers, sigma, self.num_iters)
        probs = torch.nan_to_num(probs).clamp_min(0.0)

        return probs

    def _angular_pdf(self, theta, sigma, num_iters):
        l = torch.arange(self.num_iters).view(-1, 1)  # noqa: E741

        a = (1 - torch.cos(theta)) / torch.pi
        b = (2 * l + 1) * torch.exp(-l * (l + 1) * sigma**2)
        c = torch.sin((l + 0.5) * theta) / torch.sin(theta / 2.0)

        return (a * b * c).sum(axis=0)

    def sample_from_histogram(self, sigma_idx):
        probs = self.histograms[sigma_idx]  # len(sigma_idx), n_bins

        # first sample bin according to the probability
        bin_idx = torch.multinomial(probs, num_samples=1).flatten()
        binsize = torch.pi / self.n_bins
        bin_starts = torch.arange(0, torch.pi, binsize)

        # uniform sampling within the bin
        sampled = bin_starts[bin_idx] + binsize * torch.rand(bin_idx.shape)
        return sampled

    def sample_from_gaussian(self, sigma_idx):
        mu = self.sigmas_to_consider[sigma_idx] * 2.0
        std = self.sigmas_to_consider[sigma_idx]

        sampled = mu + std * torch.randn_like(mu)
        sampled = sampled % torch.pi  # truncate to [0, pi)

        return sampled

    def sample_isotropic_gaussian(self, sigma_idx):
        n = len(sigma_idx)
        #
        # Sample rotational axis from uniform distribution on S^2 (unit sphere).
        #
        # This can be effectively done by sampling 3D standard Gaussian and
        # normalizing the sampled vector to unit length
        u = F.normalize(torch.randn(n, 3), dim=-1)  # n, 3

        #
        # Sample rotation angle
        #
        # from histogram
        theta_hist = self.sample_from_histogram(sigma_idx)

        # from Gaussian distribution
        theta_gaussian = self.sample_from_gaussian(sigma_idx)

        use_hist = self.sigmas_to_consider[sigma_idx] < self.sigma_threshold
        theta = torch.where(use_hist, theta_hist, theta_gaussian)

        return u * theta[:, None]


def uniform(*size):
    assert len(size) >= 2, "size must be at least 2-dimensional"
    assert size[-2] == size[-1] == 3, "last two dimensions must be 3"

    # sample from uniform distribution
    n_sample = 1
    for dim in size[:-2]:
        n_sample *= dim

    R = Rotation.random(n_sample).as_matrix().reshape(*size)
    return torch.tensor(R).float()


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


def rotation_matrix_to_vector(R: torch.Tensor) -> torch.Tensor:
    """Convert a rotation matrix to a so3 vector.

    Args:
        R: A rotation matrix. Shape: *, 3, 3

    Returns:
        v: An so3 vector. Shape: *, 3
    """
    return skew_symmetric_mat_to_vector(log_rotmat(R))


def vector_to_skew_symmetric_mat(v: torch.Tensor) -> torch.Tensor:
    """Convert a vector v to a skew-symmetric matrix S.

    Args:
        v (torch.Tensor): Shape: bsz, L, 3

    Returns:
        torch.Tensor: Shape: bsz, L, 3, 3
    """
    v_x, v_y, v_z = v[..., 0], v[..., 1], v[..., 2]

    S = torch.zeros(*v.shape[:-1], 3, 3).to(v.device)
    S[..., 0, 1] = -v_z
    S[..., 0, 2] = v_y
    S[..., 1, 0] = v_z
    S[..., 1, 2] = -v_x
    S[..., 2, 0] = -v_y
    S[..., 2, 1] = v_x

    return S


def vector_to_rotation_matrix(v: torch.Tensor) -> torch.Tensor:
    """Convert a vector v to a rotation matrix R.

    Args:
        v (torch.Tensor): Shape: bsz, L, 3

    Returns:
        torch.Tensor: Shape: bsz, L, 3, 3
    """
    return exp_skew_symmetric_mat(vector_to_skew_symmetric_mat(v))


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
