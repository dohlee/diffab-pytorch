import torch
import pytest

from diffab_pytorch.diffusion import (
    cosine_variance_schedule,
    SequenceDiffuser,
    CoordinateDiffuser,
    OrientationDiffuser,
)


def test_cosine_variance_schedule():
    pass


def test_SequenceDiffuser():
    seq_diffuser = SequenceDiffuser(T=100, s=0.01, beta_max=0.999)

    bsz, L = 32, 100
    seq = torch.randint(0, 20, (bsz, L))

    generation_mask = torch.randint(0, 2, (bsz, L)).bool()
    generate_all = torch.ones(bsz, L).bool()

    p_1 = seq_diffuser.forward_prob_single_step(
        seq, t=torch.ones(bsz).long(), generation_mask=generate_all
    )
    p_90 = seq_diffuser.forward_prob_single_step(
        seq, t=torch.full((bsz,), 90).long(), generation_mask=generate_all
    )

    assert p_1.shape == p_90.shape == (bsz, L, 21)

    for seq_idx in range(bsz):
        for pos_idx in range(L):
            # variance is larger at t=90 than at t=1
            # so probability of being original amino acid is smaller
            # at t=90 than at t=1
            aa = seq[seq_idx][pos_idx]
            assert p_1[seq_idx][pos_idx][aa] > p_90[seq_idx][pos_idx][aa]

    p_1 = seq_diffuser.forward_prob_from_t0(
        seq, t=torch.ones(bsz).long(), generation_mask=generate_all
    )
    p_90 = seq_diffuser.forward_prob_from_t0(
        seq, t=torch.full((bsz,), fill_value=90).long(), generation_mask=generate_all
    )

    assert p_1.shape == p_90.shape == (bsz, L, 21)

    for seq_idx in range(bsz):
        for pos_idx in range(L):
            aa = seq[seq_idx][pos_idx]
            assert p_1[seq_idx][pos_idx][aa] > p_90[seq_idx][pos_idx][aa]

    p_10 = seq_diffuser.forward_prob_from_t0(
        seq, t=torch.full((bsz,), fill_value=10).long(), generation_mask=generation_mask
    )
    seq_sampled = torch.multinomial(p_10.view(-1, 21), num_samples=1).view(bsz, L)

    posterior = seq_diffuser.posterior_single_step(
        seq_sampled,
        seq,
        t=torch.full((bsz,), fill_value=10).long(),
        generation_mask=generation_mask,
    )
    assert p_10.shape == posterior.shape == (bsz, L, 21)

    for seq_idx in range(bsz):
        for pos_idx in range(L):
            # posterior probability of original amino acid
            # may be greater than the others
            aa = seq[seq_idx][pos_idx]
            assert posterior[seq_idx][pos_idx][aa] > 1 / 20.0


def test_SequenceDiffuser_diffuse():
    seq_diffuser = SequenceDiffuser(T=100, s=0.01, beta_max=0.999)

    bsz, L = 32, 100
    seq = torch.randint(0, 20, (bsz, L))

    generate_all = torch.ones(bsz, L).bool()

    seq_t2, post_t2 = seq_diffuser.diffuse_from_t0(
        seq,
        t=torch.full((bsz,), fill_value=2).long(),
        generation_mask=generate_all,
        return_posterior=True,
    )
    seq_t99, post_t99 = seq_diffuser.diffuse_from_t0(
        seq,
        t=torch.full((bsz,), fill_value=99).long(),
        generation_mask=generate_all,
        return_posterior=True,
    )

    assert seq_t2.shape == seq_t99.shape == (bsz, L)
    assert post_t2.shape == (bsz, L, 21)
    assert post_t99.shape == (bsz, L, 21)

    # sequence at t99 should deviate more from original sequence
    assert (seq_t2 != seq).sum() < (seq_t99 != seq).sum()


def test_CoordinateDiffuser():
    coord_diffuser = CoordinateDiffuser(T=100, s=0.01, beta_max=0.999)

    bsz, L = 32, 100
    xyz = torch.randn(bsz, L, 3)
    t = torch.randint(0, 100, (bsz,)).long()

    generation_mask = torch.randint(0, 2, (bsz, L)).bool()

    xyz_t, eps = coord_diffuser.diffuse_from_t0(
        xyz, t=t, generation_mask=generation_mask, return_eps=True
    )
    assert xyz_t.shape == (bsz, L, 3)
    assert eps.shape == (bsz, L, 3)


def test_OrientationDiffuser():
    orient_diffuser = OrientationDiffuser(T=100, s=0.01, beta_max=0.999)

    bsz, L = 32, 100
    orientations_t0 = torch.randn(bsz, L, 3, 3)
    generation_mask = torch.randint(0, 2, (bsz, L)).bool()

    t = torch.full((bsz,), 50).long()
    orientations_t = orient_diffuser.diffuse_from_t0(
        orientations_t0,
        generation_mask,
        t=t,
    )
    assert orientations_t.shape == (bsz, L, 3, 3)
