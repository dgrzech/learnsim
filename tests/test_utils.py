from utils import compute_norm, pixel_to_normalised_3d, pixel_to_normalised_2d, plot_2d, plot_3d,\
    sobolev_kernel_1D, separable_conv_3d, SVF_2D, SVF_3D

import math
import numpy as np
import torch
import unittest

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.autograd.set_detect_anomaly(True)


class UtilsTestMethods(unittest.TestCase):
    def setUp(self):
        print(self._testMethodName + '\n')

        n = 8
        self.dim_x, self.dim_y, self.dim_z = n, n, n

    def test_norm(self):
        v = torch.ones((3, self.dim_x, self.dim_y, self.dim_z))

        v_norm = compute_norm(v)
        val_true = math.sqrt(3) * torch.ones((3, self.dim_x, self.dim_y, self.dim_z))

        assert torch.all(torch.eq(v_norm, val_true))

    def test_scaling_and_squaring_2d_translation(self):
        transformation_module = SVF_2D(self.dim_x, self.dim_y)

        # v = torch.zeros(1, 2, self.dim_x, self.dim_y)
        # transformation, displacement = transformation_module(v)
        # print('zero velocity field\n', transformation)

        v = 0.2 * torch.ones(1, 2, self.dim_x, self.dim_y)
        transformation, displacement = transformation_module(v)
        # print('uniform velocity field\n', transformation)
        plot_2d(v, transformation)

    def test_scaling_and_squaring_3d_translation(self):
        transformation_module = SVF_3D(self.dim_x, self.dim_y, self.dim_z)

        # v = torch.zeros(1, 3, self.dim_x, self.dim_y, self.dim_z)
        # transformation, displacement = transformation_module(v)
        # print('zero velocity field\n', transformation)

        v = 0.2 * torch.ones(1, 3, self.dim_x, self.dim_y, self.dim_z)
        transformation, displacement = transformation_module(v)
        # print('uniform velocity field\n', transformation)
        plot_3d(v, transformation)

    def test_scaling_and_squaring_2d_rotation(self):
        transformation_module = SVF_2D(self.dim_x, self.dim_y)

        v = torch.zeros(1, 2, self.dim_x, self.dim_y)
        for idx_x in range(v.shape[3]):
            for idx_y in range(v.shape[2]):
                x, y = pixel_to_normalised_2d(idx_x, idx_y, self.dim_x, self.dim_y)

                v[0, 0, idx_x, idx_y] = y
                v[0, 1, idx_x, idx_y] = -1.0 * x

        transformation, displacement = transformation_module(v)
        # print(transformation)
        plot_2d(v, transformation)

    def test_scaling_and_squaring_3d_rotation(self):
        transformation_module = SVF_3D(self.dim_x, self.dim_y, self.dim_z)

        v = torch.zeros(1, 3, self.dim_x, self.dim_y, self.dim_z)
        for idx_z in range(v.shape[2]):
            for idx_y in range(v.shape[3]):
                for idx_x in range(v.shape[4]):
                    x, y, z = pixel_to_normalised_3d(idx_x, idx_y, idx_z, self.dim_x, self.dim_y, self.dim_z)

                    v[0, 0, idx_x, idx_y, idx_z] = y
                    v[0, 1, idx_x, idx_y, idx_z] = -1.0 * x
                    v[0, 2, idx_x, idx_y, idx_z] = 0.0

        transformation, displacement = transformation_module(v)
        # print(transformation)
        plot_3d(v, transformation)

    def test_separable_conv_3d(self):
        N = 2  # batch size
        D = H = W = 16  # no. of voxels in each dimension

        _s = 3  # Sobolev kernel size
        S_numpy = np.ones(_s)
        S = torch.from_numpy(S_numpy).float()
        S.unsqueeze_(0)
        S = torch.stack((S, S, S), 0)

        S_x = S.unsqueeze(2).unsqueeze(2)
        S_y = S.unsqueeze(2).unsqueeze(4)
        S_z = S.unsqueeze(3).unsqueeze(4)

        padding_sz = _s // 2

        # velocity fields
        v = torch.zeros([N, 3, D, H, W]).float()  # velocity fields

        v[0, 1] = 1.0
        v[1, 2] = 1.0

        # separable convolution
        # v_out = separable_conv_3d(v, S, padding_sz)
        v_out = separable_conv_3d(v, S_x, S_y, S_z, padding_sz)
        v_out_size = v_out.size()

        assert v_out_size[0] == 2
        assert v_out_size[1] == 3
        assert v_out_size[2] == D
        assert v_out_size[3] == H
        assert v_out_size[4] == W

        # assert torch.all(torch.eq(v[0, 0], 0.0))
        # assert torch.all(torch.eq(v[0, 1], 27.0))
        # assert torch.all(torch.eq(v[0, 2], 0.0))

        # assert torch.all(torch.eq(v[1, 0], 0.0))
        # assert torch.all(torch.eq(v[1, 1], 0.0))
        # assert torch.all(torch.eq(v[1, 2], 27.0))

