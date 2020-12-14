import math
import unittest

import numpy as np
import pytest
import torch

from utils import calc_norm, pixel_to_normalised_3D, pixel_to_normalised_2D, plot_2D, plot_3D, \
    separable_conv_3D, SVF_2D, SVF_3D

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
        self.dim_x = self.dim_y = self.dim_z = n

    def test_norm(self):
        v = torch.ones((1, 3, self.dim_x, self.dim_y, self.dim_z))

        v_norm = calc_norm(v)
        val_true = math.sqrt(3) * torch.ones(size=(1, 1, self.dim_x, self.dim_y, self.dim_z))

        assert torch.all(torch.eq(v_norm, val_true))

    def test_norm_batch(self):
        v1 = torch.ones((3, self.dim_x, self.dim_y, self.dim_z))
        v2 = 2.0 * torch.ones_like(v1)

        v = torch.stack([v1, v2], dim=0)
        v_norm = calc_norm(v)

        v1_norm_true = math.sqrt(3) * torch.ones(1, 1, self.dim_x, self.dim_y, self.dim_z)
        v2_norm_true = math.sqrt(12) * torch.ones_like(v1_norm_true)

        assert torch.all(torch.eq(v_norm[0], v1_norm_true))
        assert torch.all(torch.eq(v_norm[1], v2_norm_true))

    def test_scaling_and_squaring_2D_translation(self):
        transformation_module = SVF_2D(self.dim_x, self.dim_y)

        v = 0.2 * torch.ones(1, 2, self.dim_x, self.dim_y)
        transformation, displacement = transformation_module(v)
        plot_2D(v, transformation)

    def test_scaling_and_squaring_3D_translation(self):
        transformation_module = SVF_3D(self.dim_x, self.dim_y, self.dim_z)

        v = 0.2 * torch.ones(1, 3, self.dim_x, self.dim_y, self.dim_z)
        transformation, displacement = transformation_module(v)
        plot_3D(v, transformation)

    def test_scaling_and_squaring_2D_rotation(self):
        transformation_module = SVF_2D(self.dim_x, self.dim_y)

        v = torch.zeros(1, 2, self.dim_x, self.dim_y)
        for idx_x in range(v.shape[3]):
            for idx_y in range(v.shape[2]):
                x, y = pixel_to_normalised_2D(idx_x, idx_y, self.dim_x, self.dim_y)

                v[0, 0, idx_x, idx_y] = y
                v[0, 1, idx_x, idx_y] = -1.0 * x

        transformation, displacement = transformation_module(v)
        plot_2D(v, transformation)

    def test_scaling_and_squaring_3D_rotation(self):
        transformation_module = SVF_3D(self.dim_x, self.dim_y, self.dim_z)

        v = torch.zeros(1, 3, self.dim_x, self.dim_y, self.dim_z)
        for idx_z in range(v.shape[2]):
            for idx_y in range(v.shape[3]):
                for idx_x in range(v.shape[4]):
                    x, y, z = pixel_to_normalised_3D(idx_x, idx_y, idx_z, self.dim_x, self.dim_y, self.dim_z)

                    v[0, 0, idx_x, idx_y, idx_z] = y
                    v[0, 1, idx_x, idx_y, idx_z] = -1.0 * x
                    v[0, 2, idx_x, idx_y, idx_z] = 0.0

        transformation, displacement = transformation_module(v)
        plot_3D(v, transformation)

    def test_separable_conv_3D(self):
        N = 2  # batch size
        D = H = W = 16  # no. of voxels in each dimension

        _s = 3  # Sobolev kernel size
        S_numpy = np.ones(_s)
        S = torch.from_numpy(S_numpy).float().unsqueeze(0)
        S = torch.stack((S, S, S), 0)

        S_x = S.unsqueeze(2).unsqueeze(2)
        S_y = S.unsqueeze(2).unsqueeze(4)
        S_z = S.unsqueeze(3).unsqueeze(4)

        padding_sz = _s // 2

        # velocity fields
        v = torch.zeros([N, 3, D, H, W]).float()  # velocity fields

        v[0, 1] = 1.0
        v[1, 2] = 1.0

        # separable convolution implemented as three 1D convolutions
        v_out = separable_conv_3D(v, S, padding_sz)
        v_out_size = v_out.shape

        assert v_out_size[0] == 2
        assert v_out_size[1] == 3
        assert v_out_size[2] == D
        assert v_out_size[3] == H
        assert v_out_size[4] == W

        for idx_z in range(self.dim_z - 1):
            for idx_y in range(self.dim_y - 1):
                for idx_x in range(self.dim_x - 1):
                    v_out_0_x = v_out[0, 0, idx_z, idx_y, idx_x].item()
                    v_out_0_y = v_out[0, 1, idx_z, idx_y, idx_x].item()
                    v_out_0_z = v_out[0, 2, idx_z, idx_y, idx_x].item()

                    assert pytest.approx(v_out_0_x, 1e-5) == 0.0
                    assert pytest.approx(v_out_0_y, 1e-5) == 27.0
                    assert pytest.approx(v_out_0_z, 1e-5) == 0.0

                    v_out_1_x = v_out[1, 0, idx_z, idx_y, idx_x].item()
                    v_out_1_y = v_out[1, 1, idx_z, idx_y, idx_x].item()
                    v_out_1_z = v_out[1, 2, idx_z, idx_y, idx_x].item()

                    assert pytest.approx(v_out_1_x, 1e-5) == 0.0
                    assert pytest.approx(v_out_1_y, 1e-5) == 0.0
                    assert pytest.approx(v_out_1_z, 1e-5) == 27.0

        # separable convolution implemented as 3D convolutions
        v_out = separable_conv_3D(v, S_x, S_y, S_z, padding_sz)
        v_out_size = v_out.shape

        assert v_out_size[0] == 2
        assert v_out_size[1] == 3
        assert v_out_size[2] == D
        assert v_out_size[3] == H
        assert v_out_size[4] == W

        for idx_z in range(self.dim_z - 1):
            for idx_y in range(self.dim_y - 1):
                for idx_x in range(self.dim_x - 1):
                    v_out_0_x = v_out[0, 0, idx_z, idx_y, idx_x].item()
                    v_out_0_y = v_out[0, 1, idx_z, idx_y, idx_x].item()
                    v_out_0_z = v_out[0, 2, idx_z, idx_y, idx_x].item()

                    assert pytest.approx(v_out_0_x, 1e-5) == 0.0
                    assert pytest.approx(v_out_0_y, 1e-5) == 27.0
                    assert pytest.approx(v_out_0_z, 1e-5) == 0.0

                    v_out_1_x = v_out[1, 0, idx_z, idx_y, idx_x].item()
                    v_out_1_y = v_out[1, 1, idx_z, idx_y, idx_x].item()
                    v_out_1_z = v_out[1, 2, idx_z, idx_y, idx_x].item()

                    assert pytest.approx(v_out_1_x, 1e-5) == 0.0
                    assert pytest.approx(v_out_1_y, 1e-5) == 0.0
                    assert pytest.approx(v_out_1_z, 1e-5) == 27.0
