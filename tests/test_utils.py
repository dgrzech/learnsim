from model.loss import EntropyMultivariateNormal
from utils.transformation import SVF
from utils.util import init_identity_grid_3d, init_identity_grid_2d

import math
import numpy as np
import pytest
import torch


def test_entropy():
    entropy = EntropyMultivariateNormal()

    n = 2  # no. of voxels in each dimension

    # initialise sigma_v
    log_var_v = torch.log(torch.abs(torch.randn(n, n, n)))
    sigma_v = torch.exp(0.5 * log_var_v) + 1e-5
    var_v = sigma_v ** 2

    # initialise the first mode of variation
    u_v = torch.zeros((n, n, n))

    # calculate the entropy
    val = entropy.forward(log_var_v, u_v).item()
    val_true = 0.5 * math.log(np.linalg.det(np.diag(var_v.data.numpy().flatten())))

    assert pytest.approx(val, 0.01) == val_true


def test_scaling_and_squaring_2d_translation():
    print('--- TRANSLATION 2D ---')

    n = 4
    dim = (1, n, n)

    identity_grid_2d = init_identity_grid_2d(dim)
    transformation = SVF()

    print(identity_grid_2d)

    v = torch.zeros((1, 2, n, n))
    warp_field = transformation.forward_2d_add(identity_grid_2d, v)
    print('zero velocity field\n', warp_field)

    # warp_field = transformation.forward_2d_comp(identity_grid_2d, v)
    # print('zero velocity field\n', warp_field)

    v = 0.2 * torch.ones((1, 2, n, n))
    warp_field = transformation.forward_2d_add(identity_grid_2d, v)
    print('uniform velocity field\n', warp_field)

    # warp_field = transformation.forward_2d_comp(identity_grid_2d, v)
    # print('uniform velocity field \n', warp_field)


def test_scaling_and_squaring_3d_translation():
    print('--- TRANSLATION 3D ---')

    n = 4
    dim = (1, n, n, n)

    identity_grid_3d = init_identity_grid_3d(dim)
    transformation = SVF()

    print(identity_grid_3d)

    v = torch.zeros((1, 3, n, n, n))
    warp_field = transformation.forward_3d_add(identity_grid_3d, v)
    print('zero velocity field\n', warp_field)

    # warp_field = transformation.forward_3d_comp(identity_grid_3d, v)
    # print('zero velocity field\n', warp_field)

    v = 0.2 * torch.ones((1, 3, n, n, n))
    warp_field = transformation.forward_3d_add(identity_grid_3d, v)
    print('uniform velocity field\n', warp_field)

    # warp_field = transformation.forward_3d_comp(identity_grid_3d, v)
    # print('uniform velocity field \n', warp_field)


def test_scaling_and_squaring_2d_rotation():
    print('--- ROTATION 2D ---')

    n = 4
    dim = (1, n, n)

    identity_grid_2d = init_identity_grid_2d(dim)
    transformation = SVF()

    v = torch.zeros((1, 2, n, n))

    for idx_x in range(v.shape[3]):
        for idx_y in range(v.shape[2]):
            v[0, 0, idx_x, idx_y] = 1.0 - 2.0 * idx_x / float(v.shape[3])
            v[0, 1, idx_x, idx_y] = -1.0 + 2.0 * idx_y / float(v.shape[2])

    print(v)
    warp_field = transformation.forward_2d_add(identity_grid_2d, v)
    print(warp_field)

    # warp_field = transformation.forward_2d_comp(identity_grid_2d, v)
    # print(warp_field)


def test_scaling_and_squaring_3d_rotation():
    print('--- ROTATION 3D---')

    n = 4
    dim = (1, n, n, n)

    identity_grid_3d = init_identity_grid_3d(dim)
    transformation = SVF()

    v = torch.zeros((1, 3, n, n, n))

    for idx_x in range(v.shape[4]):
        for idx_y in range(v.shape[3]):
            v[0, 1, :, idx_y, idx_x] = 1.0 - 2.0 * idx_x / float(v.shape[3])
            v[0, 2, :, idx_y, idx_x] = -1.0 + 2.0 * idx_y / float(v.shape[2])

    warp_field = transformation.forward_3d_add(identity_grid_3d, v)
    print(warp_field)

    # warp_field = transformation.forward_3d_comp(identity_grid_3d, v)
    # print(warp_field)


def test_utils():
    test_entropy()

    test_scaling_and_squaring_2d_translation()
    test_scaling_and_squaring_3d_translation()

    test_scaling_and_squaring_2d_rotation()
    test_scaling_and_squaring_3d_rotation()


if __name__ == '__main__':
    test_utils()
