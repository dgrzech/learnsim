from utils.registration import RegistrationModule
from utils.util import init_identity_grid_3d, pixel_to_normalised_3d, save_im_to_disk

import numpy as np
import torch
import unittest

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

torch.autograd.set_detect_anomaly(True)


class WarpingTestMethods(unittest.TestCase):
    def setUp(self):
        print(self._testMethodName)

        """
        utils
        """

        n = 64
        self.dim_x = n
        self.dim_y = n
        self.dim_z = n

        self.dims_im = (1, 1, self.dim_x, self.dim_y, self.dim_z)
        self.dims_v = (1, 3, self.dim_x, self.dim_y, self.dim_z)

        self.identity_grid = init_identity_grid_3d(self.dim_x, self.dim_y, self.dim_z).to('cuda:0')

        """
        modules
        """

        self.registration_module = RegistrationModule().to('cuda:0')

    def test_sphere_translation(self):
        """
        initialise 3D image of a sphere
        """

        im_moving = -1.0 + torch.zeros(self.dims_im).to('cuda:0')
        r = 0.02

        for idx_z in range(im_moving.shape[2]):
            for idx_y in range(im_moving.shape[3]):
                for idx_x in range(im_moving.shape[4]):
                    x, y, z = pixel_to_normalised_3d(idx_x, idx_y, idx_z, self.dim_x, self.dim_y, self.dim_z)

                    if x ** 2 + y ** 2 + z ** 2 <= r:
                        im_moving[0, 0, idx_x, idx_y, idx_z] = 1.0

        """
        initialise a warp field
        """

        offset = 5.0

        warp_field = self.identity_grid.permute([0, 4, 1, 2, 3]) + offset / self.dim_x * torch.ones(self.dims_v).to('cuda:0')
        im_moving_warped = self.registration_module(im_moving, warp_field)

        """"
        save the images to disk
        """

        save_im_to_disk(im_moving, './temp/moving.nii.gz')
        save_im_to_disk(im_moving_warped, './temp/moving_warped.nii.gz')
