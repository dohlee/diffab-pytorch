import torch
import math

import torch.nn as nn
import torch.nn.functional as F

import diffab_pytorch.so3 as so3
from einops import rearrange


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
    w1 = rearrange(w1, "b -> b () ()")
    w2 = rearrange(w2, "b -> b () ()")
    return w1 * p1 + w2 * p2


class SequenceDiffuser(object):
    def __init__(self, T, s=0.01, beta_max=0.999, aa_vocab_size=21):
        self.sched = cosine_variance_schedule(T, s=s, beta_max=beta_max)
        self.aa_vocab_size = 21

    def forward_prob_single_step(
        self,
        seq_idx: torch.LongTensor,
        t: torch.LongTensor,
        generation_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        """Compute the probability of each amino acid at timestep t,
        given the sequence at timestep t-1.

        Args:
            seq_idx: Sequence index. Shape: (bsz, L)
            t: Timestep. Shape: (bsz,)
            generation_mask: Mask for residues for which noise to be added.
                Shape: (bsz, L)

        Returns:
            seq_idx_noised: A tensor representing noised amino acid probabilities.
                Shape: (bsz, L)
        """
        seq_onehot = F.one_hot(seq_idx, num_classes=self.aa_vocab_size)
        w_orig = 1 - self.sched["beta"][t]

        unif_noise = torch.ones_like(seq_onehot) / self.aa_vocab_size
        w_noise = self.sched["beta"][t]

        seq_onehot_noised = weighted_multinomial(
            seq_onehot, unif_noise, w_orig, w_noise
        )

        _mask = generation_mask.unsqueeze(-1).expand_as(seq_onehot)
        return torch.where(_mask, seq_onehot_noised, seq_onehot)

    def diffuse_single_step(
        self,
        seq_idx: torch.LongTensor,
        t: torch.LongTensor,
        generation_mask: torch.BoolTensor,
    ) -> torch.LongTensor:
        """Take a single diffusion step and return a batch of noised sequence.

        Args:
            seq_idx: Sequence index. Shape: (bsz, L)
            t: Timestep. Shape: (bsz,)
            generation_mask: Mask for residues for which noise to be added.
                Shape: (bsz, L)

        Returns:
            seq_noised: A sequence index tensor sampled from noised amino acid
                probability distributions. Shape: (bsz, L)
        """
        p = self.forward_prob_single_step(seq_idx, t, generation_mask)
        print(p.shape)
        return torch.multinomial(p.view(-1, self.aa_vocab_size), num_samples=1).view(
            p.shape[:-1]
        )

    def forward_prob_from_t0(
        self,
        seq_idx_t0: torch.LongTensor,
        t: torch.LongTensor,
        generation_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        """Compute the probability of each amino acid at timestep t,
        given the sequence at timestep 0.

        Args:
            seq_idx_t0: Original sequence index. Shape: (bsz, L)
            t: Timestep. Shape: (bsz,)
            generation_mask: Mask for residues for which noise to be added.
                Shape: (bsz, L)

        Returns:
            seq_idx_noised: A tensor representing noised amino acid probabilities.
                Shape: (bsz, L)
        """
        seq_onehot_t0 = F.one_hot(seq_idx_t0, num_classes=self.aa_vocab_size)
        w_seq = self.sched["alpha_bar"][t]

        unif_noise = torch.ones_like(seq_onehot_t0) / self.aa_vocab_size
        w_noise = 1 - self.sched["alpha_bar"][t]

        seq_onehot_noised = weighted_multinomial(
            seq_onehot_t0, unif_noise, w_seq, w_noise
        )

        _mask = generation_mask.unsqueeze(-1).expand_as(seq_onehot_t0)
        return torch.where(_mask, seq_onehot_noised, seq_onehot_t0)

    def diffuse_from_t0(
        self,
        seq_idx_t0: torch.LongTensor,
        t: torch.LongTensor,
        generation_mask: torch.BoolTensor,
        return_posterior: bool = True,
    ) -> torch.Tensor:
        """Sample a sequence at timestep t, given the sequence at timestep 0.

        Args:
            seq_idx_t0: Sequence at timestep 0. Shape: bsz, L
            t: Timestep
            return_posterior (bool, optional): Whether to return the
                posterior probability. Defaults to True.

        Returns:
            torch.Tensor: Sequence at timestep t. Shape: bsz, L
        """
        p = self.forward_prob_from_t0(seq_idx_t0, t, generation_mask)
        seq_idx_t = torch.multinomial(
            p.view(-1, self.aa_vocab_size), num_samples=1
        ).view(p.shape[:-1])

        if return_posterior:
            posterior = self.posterior_single_step(
                seq_idx_t, seq_idx_t0, t, generation_mask
            )
            return seq_idx_t, posterior
        else:
            return seq_idx_t

    def posterior_single_step(
        self,
        seq_idx_t: torch.LongTensor,
        seq_idx_t0: torch.LongTensor,
        t: torch.LongTensor,
        generation_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        """Compute the posterior probability of each amino acid at timestep t-1,
        given the sequence at timestep t.

        Args:
            seq_idx_t: Shape: (bsz, L)
            seq_idx_t0: Shape: (bsz, L)
            t: Shape: (bsz,)
            generation_mask: Shape: (bsz, L)

        Returns:
            seq_posterior: Shape: (bsz, L)
        TODO: See if normalizing with self.diffuse_from_t0(seq_t0, t) gives better performance.
        """
        p_single = self.forward_prob_single_step(seq_idx_t, t, generation_mask)
        p_from_t0 = self.forward_prob_from_t0(seq_idx_t0, t - 1, generation_mask)

        p = p_single * p_from_t0
        return p / p.sum(dim=-1, keepdim=True)


class CoordinateDiffuser(object):
    def __init__(self, T, s=0.01, beta_max=0.999):
        self.sched = cosine_variance_schedule(T, s=s, beta_max=beta_max)

    def diffuse_from_t0(
        self,
        translations_t0: torch.FloatTensor,
        t: torch.LongTensor,
        generation_mask: torch.BoolTensor,
        return_eps: bool = True,
    ) -> torch.Tensor:
        """Sample a coordinate at timestep t, given the coordinate at timestep 0.

        Args:
            translations_t0: Translation vectors of residue frames at timestep 0.
                Shape: bsz, L, 3
            t: Timestep. Shape: bsz,
            generation_mask: Shape: bsz, L
            return_eps: Whether to return noise added to the coordinate.
                Defaults to True.

        Returns:
            Sampled coordinate at timestep t. Shape: bsz, L, 3
        """
        alpha_bar_sqrt = self.sched["alpha_bar_sqrt"][t]
        one_minus_alpha_bar_sqrt = self.sched["one_minus_alpha_bar_sqrt"][t]

        alpha_bar_sqrt = rearrange(alpha_bar_sqrt, "b -> b () ()")
        one_minus_alpha_bar_sqrt = rearrange(one_minus_alpha_bar_sqrt, "b -> b () ()")

        eps = torch.randn_like(translations_t0)
        translations_t = (
            alpha_bar_sqrt * translations_t0 + one_minus_alpha_bar_sqrt * eps
        )

        _mask = generation_mask.unsqueeze(-1)
        translations_t = torch.where(_mask, translations_t, translations_t0)

        if return_eps:
            return translations_t, eps
        else:
            return translations_t


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

    def diffuse_from_t0(
        self,
        orientations_t0: torch.FloatTensor,
        generation_mask: torch.LongTensor,
        t: torch.LongTensor,
    ) -> torch.Tensor:
        """Sample an orientation at timestep t, given the orientation at timestep 0.

        Args:
            orientations_t0: Orientation (i.e., rotation matrix) at timestep 0.
                Shape: bsz, L, 3, 3
            generation_mask: True if a noise would be added to that residue.
                0 otherwise. Shape: bsz, L
            t (int): Timestep. Shape: bsz

        Returns:
            Sampled orientation at timestep t. Shape: bsz, L, 3, 3
        """
        mean_orientation = so3.scale_rot(
            orientations_t0, self.sched["alpha_bar_sqrt"][t]
        )

        # sample from isotropic Gaussian IGSO3(R, sqrt(1-a))
        n_residues = orientations_t0.shape[1]
        rotvec_sampled = self.so3.sample_isotropic_gaussian(t, num_samples=n_residues)
        noise = so3.vector_to_rotation_matrix(rotvec_sampled)

        orientations_t = torch.einsum("bnij,bnjk->bnik", mean_orientation, noise)

        _mask = rearrange(generation_mask, "b l -> b l () ()")
        orientations_t = torch.where(_mask, orientations_t, orientations_t0)

        return orientations_t
