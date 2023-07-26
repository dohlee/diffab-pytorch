import torch

from diffab_pytorch.diffab_pytorch import OrientationLoss
from scipy.spatial.transform import Rotation as R

import pytest


def test_OrientationLoss():
    bsz, n_res = 16, 20

    rotmat = R.random(bsz * n_res).as_matrix()
    rotmat = rotmat.reshape(bsz, n_res, 3, 3)
    rotmat = torch.tensor(rotmat)
    assert rotmat.shape == (bsz, n_res, 3, 3)

    criterion = OrientationLoss(reduction="mean")
    loss = criterion(rotmat, rotmat)

    assert loss.shape == ()
    assert loss.item() == pytest.approx(0.0)


def test_KLDivLoss():
    bsz, n_res = 16, 20

    pred = torch.rand(bsz, n_res, 20).softmax(dim=-1)

    criterion = torch.nn.KLDivLoss(reduction="mean")
    # KLDivLoss takes log-probability as input
    loss = criterion(pred.log(), pred)

    assert loss.shape == ()
    assert loss.item() == pytest.approx(0.0)

    target = torch.randint(low=0, high=20, size=(bsz, n_res))
    target = torch.nn.functional.one_hot(target, num_classes=20).float()

    correct_pred = target + torch.randn_like(target) * 1e-5
    correct_pred = correct_pred.softmax(dim=-1)

    random_pred = torch.rand(bsz, n_res, 20).softmax(dim=-1)

    correct_loss = criterion(correct_pred.log(), target)
    random_loss = criterion(random_pred.log(), target)

    assert correct_loss < random_loss
