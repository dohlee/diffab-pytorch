import torch
import math

import torch.nn as nn
import torch.nn.functional as F

import diffab_pytorch.so3 as so3


def cosine_variance_schedule(T, s=8e-3, beta_max=0.999):
    # cosine variance schedule
    # T: total timesteps
    # s: small offset to prevent beta from being too small
    # beta_max: to prevent singularities at the end of the diffusion process
    t = torch.arange(T + 1)  # 0, 1, ..., T

    f_t = torch.cos((t / T + s) / (1 + s) * math.pi / 2.0).square()
    alpha_bar = f_t / f_t[0]
    beta = torch.cat(
        [
            torch.tensor([0.0]),
            torch.clip(1 - alpha_bar[1:] / alpha_bar[:-1], min=1e-5, max=beta_max),
        ]
    )
    alpha = 1 - beta

    sched = {
        "alpha": alpha,
        "alpha_bar": alpha_bar,
        "alpha_bar_sqrt": alpha_bar.sqrt(),
        "one_minus_alpha_bar_sqrt": (1 - alpha_bar).sqrt(),
        "beta": beta,
    }
    return sched


def weighted_multinomial(p1, p2, w1, w2):
    return w1 * p1 + w2 * p2


class SequenceDiffuser(object):
    def __init__(self, T, s=0.01, beta_max=0.999):
        self.sched = cosine_variance_schedule(T, s=s, beta_max=beta_max)

    def forward_prob_single_step(self, seq, t):
        """
        Compute the probability of each amino acid at timestep t,
        given the sequence at timestep t-1.

        seq (torch.LongTensor): bsz, L
        t (int): timestep
        """
        seq = F.one_hot(seq)
        w_seq = 1 - self.sched["beta"][t]

        unif_noise = torch.ones_like(seq) / 20.0
        w_unif_noise = self.sched["beta"][t]

        return weighted_multinomial(seq, unif_noise, w_seq, w_unif_noise)

    def diffuse_single_step(self, seq, t):
        """ """
        p = self.forward_prob_single_step(seq, t)
        return torch.multinomial(p.view(-1, 20), num_samples=1).view(p.shape[:-1])

    def forward_prob_from_t0(self, seq_t0, t):
        """
        Compute the probability of each amino acid at timestep t,
        given the sequence at timestep 0.

        seq0 (torch.LongTensor): bsz, L
        t (int): timestep
        """

        seq = F.one_hot(seq_t0)
        w_seq = self.sched["alpha_bar"][t]

        unif_noise = torch.ones_like(seq) / 20.0
        w_unif_noise = 1 - self.sched["alpha_bar"][t]

        return weighted_multinomial(seq, unif_noise, w_seq, w_unif_noise)

    def diffuse_from_t0(self, seq_t0: torch.Tensor, t: int) -> torch.Tensor:
        """Sample a sequence at timestep t, given the sequence at timestep 0.

        Args:
            seq_t0 (torch.Tensor): Sequence at timestep 0. Shape: bsz, L
            t (int): Timestep

        Returns:
            torch.Tensor: Sequence at timestep t. Shape: bsz, L
        """
        p = self.forward_prob_from_t0(seq_t0, t)
        return torch.multinomial(p.view(-1, 20), num_samples=1).view(p.shape[:-1])

    def posterior_single_step(self, seq_t, seq_t0, t):
        """
        Compute the posterior probability of each amino acid at timestep t-1,
        given the sequence at timestep t.

        seq (torch.LongTensor): bsz, L
        t (int): timestep

        TODO: See if normalizing with self.diffuse_from_t0(seq_t0, t) gives better performance.
        """
        p = self.forward_prob_single_step(seq_t, t) * self.forward_prob_from_t0(seq_t0, t - 1)

        return p / p.sum(dim=-1, keepdim=True)


class CoordinateDiffuser(object):
    def __init__(self, T, s=0.01, beta_max=0.999):
        self.sched = cosine_variance_schedule(T, s=s, beta_max=beta_max)

    def diffuse_from_t0(
        self, xyz_t0: torch.Tensor, t: int, return_eps: bool = True
    ) -> torch.Tensor:
        """Sample a coordinate at timestep t, given the coordinate at timestep 0.

        Args:
            xyz_t0 (torch.Tensor): Coordinate at timestep 0. Shape: bsz, L, 3
            t (int): Timestep
            return_eps (bool, optional):
                Whether to return noise added to the coordinate. Defaults to True.

        Returns:
            torch.Tensor: Sampled coordinate at timestep t. Shape: bsz, L, 3
        """
        alpha_bar_sqrt = self.sched["alpha_bar_sqrt"][t]
        one_minus_alpha_bar_sqrt = self.sched["one_minus_alpha_bar_sqrt"][t]

        eps = torch.randn_like(xyz_t0)
        xyz_t = alpha_bar_sqrt * xyz_t0 + one_minus_alpha_bar_sqrt * eps

        if return_eps:
            return xyz_t, eps
        else:
            return xyz_t


class OrientationDiffuser(object):
    def __init__(self, T: int, s: float = 0.01, beta_max: float = 0.999):
        """Diffuser for SO(3) rotations.

        Args:
            T (int): Maximum number of timesteps.
            s (float, optional):
                Small offset added to cosine variance schedule to prevent
                beta being too small. Defaults to 0.01.
            beta_max (float, optional):
                To prevent singularities at the end of the diffusion process.
                Defaults to 0.999.
        """
        self.sched = cosine_variance_schedule(T, s=s, beta_max=beta_max)

        self.so3 = so3.SO3(
            sigmas_to_consider=self.sched["one_minus_alpha_bar_sqrt"],
            cache_prefix=".cache/so3_histograms",
            sigma_threshold=0.1,
            n_bins=8192,
            num_iters=1024,
        )

    def diffuse_from_t0(self, o: torch.Tensor, t: int) -> torch.Tensor:
        """Sample an orientation at timestep t, given the orientation at timestep 0.

        Args:
            o (torch.Tensor):
                Orientation (i.e., rotation matrix) at timestep 0. Shape: bsz, L, 3, 3
            t (int): Timestep

        Returns:
            torch.Tensor: Sampled orientation at timestep t. Shape: bsz, L, 3, 3
        """
        mean_orientation = so3.scale_rot(o, self.sched["alpha_bar_sqrt"][t])

        # sample from isotropic Gaussian IGSO3(R, sqrt(1-a))
        noise = so3.vector_to_rotation_matrix(
            self.so3.sample_isotropic_gaussian(t)
        )  # bsz, 3, 3

        return mean_orientation @ noise
