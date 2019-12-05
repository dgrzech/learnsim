from model.loss import EntropyMultivariateNormal

from utils.plots import plot_2d, plot_3d
from utils.transformation import SVF
from utils.util import init_identity_grid_3d, init_identity_grid_2d

import math
import numpy as np
import pytest
import torch
import unittest


def pixel_to_normalised_2d(px_idx, dims):
    dim_x = dims[2]
    dim_y = dims[3]

    x = -1.0 + 2.0 * px_idx[0] / (dim_x - 1.0)
    y = -1.0 + 2.0 * px_idx[1] / (dim_y - 1.0)

    return x, y


def pixel_to_normalised_3d(px_idx, dims):
    dim_x = dims[2]
    dim_y = dims[3]
    dim_z = dims[4]

    x = -1.0 + 2.0 * px_idx[0] / (dim_x - 1.0)
    y = -1.0 + 2.0 * px_idx[1] / (dim_y - 1.0)
    z = -1.0 + 2.0 * px_idx[2] / (dim_z - 1.0)

    return x, y, z


class LossTestMethods(unittest.TestCase):
    def setUp(self):
        n = 2
        self.dims = (n, n, n)

        print(self._testMethodName)

    def test_entropy(self):
        n = 2  # no. of voxels in each dimension
        entropy = EntropyMultivariateNormal()

        # initialise sigma_v
        log_var_v = torch.log(torch.abs(torch.randn(self.dims)))
        sigma_v = torch.exp(0.5 * log_var_v) + 1e-5
        var_v = sigma_v ** 2

        # initialise the first mode of variation
        u_v = torch.zeros((n, n, n))

        # calculate the entropy
        val = entropy.forward(log_var_v, u_v).item()
        val_true = 0.5 * math.log(np.linalg.det(np.diag(var_v.data.numpy().flatten())))

        assert pytest.approx(val, 0.01) == val_true


class UtilsTestMethods(unittest.TestCase):
    def setUp(self):
        n = 8

        dim_2d = (1, n, n)
        self.identity_grid_2d = init_identity_grid_2d(dim_2d)
        self.dim_2d = (1, 2, n, n)

        dim_3d = (1, n, n, n)
        self.identity_grid_3d = init_identity_grid_3d(dim_3d)
        self.dim_3d = (1, 3, n, n, n)

        print(self._testMethodName)

    def test_scaling_and_squaring_2d_translation(self):
        transformation = SVF()

        # v = torch.zeros(self.dim_2d)
        # warp_field = transformation.forward_2d_add(self.identity_grid_2d, v)
        # warp_field = transformation.forward_2d_comp(self.identity_grid_2d, v)
        # print('zero velocity field\n', warp_field)

        v = 0.2 * torch.ones(self.dim_2d)
        warp_field = transformation.forward_2d_add(self.identity_grid_2d, v)
        # warp_field = transformation.forward_2d_comp(self.identity_grid_2d, v)
        # print('uniform velocity field\n', warp_field)
        plot_2d(v, warp_field)

    def test_scaling_and_squaring_3d_translation(self):
        transformation = SVF()

        # v = torch.zeros(self.dim_3d)
        # warp_field = transformation.forward_3d_add(self.identity_grid_3d, v)
        # warp_field = transformation.forward_3d_comp(self.identity_grid_3d, v)
        # print('zero velocity field\n', warp_field)

        v = 0.2 * torch.ones(self.dim_3d)
        warp_field = transformation.forward_3d_add(self.identity_grid_3d, v)
        # warp_field = transformation.forward_3d_comp(self.identity_grid_3d, v)
        # print('uniform velocity field\n', warp_field)
        plot_3d(v, warp_field)

    def test_scaling_and_squaring_2d_rotation(self):
        transformation = SVF()

        v = torch.zeros(self.dim_2d)
        for idx_x in range(v.shape[3]):
            for idx_y in range(v.shape[2]):
                x, y = pixel_to_normalised_2d((idx_x, idx_y), self.dim_2d)

                v[0, 0, idx_x, idx_y] = y
                v[0, 1, idx_x, idx_y] = -1.0 * x

        warp_field = transformation.forward_2d_add(self.identity_grid_2d, v)
        # warp_field = transformation.forward_2d_comp(self.identity_grid_2d, v)
        # print(warp_field)
        plot_2d(v, warp_field)

    def test_scaling_and_squaring_3d_rotation(self):
        transformation = SVF()

        v = torch.zeros(self.dim_3d)
        for idx_z in range(v.shape[2]):
            for idx_y in range(v.shape[3]):
                for idx_x in range(v.shape[4]):
                    x, y, z = pixel_to_normalised_3d((idx_x, idx_y, idx_z), self.dim_3d)

                    v[0, 0, idx_x, idx_y, idx_z] = y
                    v[0, 1, idx_x, idx_y, idx_z] = -1.0 * x
                    v[0, 2, idx_x, idx_y, idx_z] = 0.0

        warp_field = transformation.forward_3d_add(self.identity_grid_3d, v)
        # warp_field = transformation.forward_3d_comp(self.identity_grid_3d, v)
        # print(warp_field)
        plot_3d(v, warp_field)


if __name__ == '__main__':
    unittest.main()
