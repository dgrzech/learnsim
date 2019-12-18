from utils.plots import plot_2d, plot_3d
from utils.transformation import SVF
from utils.util import init_identity_grid_3d, init_identity_grid_2d, pixel_to_normalised_3d, pixel_to_normalised_2d

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
        print(self._testMethodName)

        n = 8

        self.dim_x = n
        self.dim_y = n
        self.dim_z = n

        self.identity_grid_2d = init_identity_grid_2d(self.dim_x, self.dim_y)
        self.identity_grid_3d = init_identity_grid_3d(self.dim_x, self.dim_y, self.dim_z)

    def test_scaling_and_squaring_2d_translation(self):
        transformation = SVF()

        # v = torch.zeros(1, 2, self.dim_x, self.dim_y)
        # warp_field = transformation.forward_2d(self.identity_grid_2d, v)
        # print('zero velocity field\n', warp_field)

        v = 0.2 * torch.ones(1, 2, self.dim_x, self.dim_y)
        warp_field = transformation.forward_2d(self.identity_grid_2d, v)
        # print('uniform velocity field\n', warp_field)
        plot_2d(v, warp_field)

    def test_scaling_and_squaring_3d_translation(self):
        transformation = SVF()

        # v = torch.zeros(1, 3, self.dim_x, self.dim_y, self.dim_z)
        # warp_field = transformation.forward_3d(self.identity_grid_3d, v)
        # print('zero velocity field\n', warp_field)

        v = 0.2 * torch.ones(1, 3, self.dim_x, self.dim_y, self.dim_z)
        warp_field = transformation.forward_3d(self.identity_grid_3d, v)
        # print('uniform velocity field\n', warp_field)
        plot_3d(v, warp_field)

    def test_scaling_and_squaring_2d_rotation(self):
        transformation = SVF()

        v = torch.zeros(1, 2, self.dim_x, self.dim_y)
        for idx_x in range(v.shape[3]):
            for idx_y in range(v.shape[2]):
                x, y = pixel_to_normalised_2d(idx_x, idx_y, self.dim_x, self.dim_y)

                v[0, 0, idx_x, idx_y] = y
                v[0, 1, idx_x, idx_y] = -1.0 * x

        warp_field = transformation.forward_2d(self.identity_grid_2d, v)
        # print(warp_field)
        plot_2d(v, warp_field)

    def test_scaling_and_squaring_3d_rotation(self):
        transformation = SVF()

        v = torch.zeros(1, 3, self.dim_x, self.dim_y, self.dim_z)
        for idx_z in range(v.shape[2]):
            for idx_y in range(v.shape[3]):
                for idx_x in range(v.shape[4]):
                    x, y, z = pixel_to_normalised_3d(idx_x, idx_y, idx_z, self.dim_x, self.dim_y, self.dim_z)

                    v[0, 0, idx_x, idx_y, idx_z] = y
                    v[0, 1, idx_x, idx_y, idx_z] = -1.0 * x
                    v[0, 2, idx_x, idx_y, idx_z] = 0.0

        warp_field = transformation.forward_3d(self.identity_grid_3d, v)
        # print(warp_field)
        plot_3d(v, warp_field)