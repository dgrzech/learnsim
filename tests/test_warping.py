import math
import unittest

import SimpleITK as sitk
import numpy as np
import pytest
import torch
from skimage import transform

from logger import save_im_to_disk
from model.loss import SSD
from utils import RegistrationModule, init_identity_grid_3D, pixel_to_normalised_3D, rescale_im

# fix random seeds for reproducibility
SEED = 123

np.random.seed(SEED)
torch.manual_seed(SEED)

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class WarpingTestMethods(unittest.TestCase):
    def setUp(self):
        print(self._testMethodName + '\n')
        self.atol = 1e-5
        self.device = 'cuda:0'

        n = 32
        self.dim_x = self.dim_y = self.dim_z = n

        self.dims = (n, ) * 3
        self.dims_im = (1, 1, *self.dims)
        self.dims_v = (1, 3, *self.dims)

        self.identity_grid = init_identity_grid_3D(self.dims).to(self.device)
        self.loss = SSD().to(self.device)
        self.registration_module = RegistrationModule().to(self.device)

    def tearDown(self):
        del self.registration_module

    def test_loss_value_zero_deformation(self):
        im_fixed = torch.randn(self.dims_im, device=self.device)
        im_moving = torch.randn(self.dims_im, device=self.device)
        mask = torch.ones_like(im_fixed).bool()

        transformation = self.identity_grid.permute([0, 4, 1, 2, 3])
        im_moving_warped = self.registration_module(im_moving, transformation)

        z_unwarped = (im_fixed - im_moving) ** 2
        z_unwarped_masked = z_unwarped[mask]

        z_warped = (im_fixed - im_moving_warped) ** 2
        z_warped_masked = z_warped[mask]

        unwarped_loss_value = self.loss(z_unwarped_masked).item()
        warped_loss_value = self.loss(z_warped_masked).item()

        assert pytest.approx(unwarped_loss_value, self.atol) == warped_loss_value

    def test_sphere_translation(self):
        # initialise 3D image of a sphere
        im_moving = -1.0 + torch.zeros(self.dims_im, device=self.device)
        r = 0.02

        for idx_z in range(im_moving.shape[2]):
            for idx_y in range(im_moving.shape[3]):
                for idx_x in range(im_moving.shape[4]):
                    x, y, z = pixel_to_normalised_3D(idx_x, idx_y, idx_z, self.dim_x, self.dim_y, self.dim_z)

                    if x ** 2 + y ** 2 + z ** 2 <= r:
                        im_moving[0, 0, idx_x, idx_y, idx_z] = 1.0

        # initialise a warp field
        offset = 5.0

        displacement = offset / self.dim_x * torch.ones(self.dims_v, device=self.device)
        transformation = self.identity_grid.permute([0, 4, 1, 2, 3]) + displacement
        im_moving_warped = self.registration_module(im_moving, transformation)

        # save the images to disk
        save_im_to_disk(im_moving[0, 0].cpu().numpy(), './temp/test_output/moving.nii.gz')
        save_im_to_disk(im_moving_warped[0, 0].cpu().numpy(), './temp/test_output/moving_warped.nii.gz')

    def test_sphere_translation_large(self):
        # initialise 3D image of a sphere
        im_moving = -1.0 + torch.zeros(self.dims_im, device=self.device)
        r = 0.02

        for idx_z in range(im_moving.shape[2]):
            for idx_y in range(im_moving.shape[3]):
                for idx_x in range(im_moving.shape[4]):
                    x, y, z = pixel_to_normalised_3D(idx_x, idx_y, idx_z, self.dim_x, self.dim_y, self.dim_z)

                    if x ** 2 + y ** 2 + z ** 2 <= r:
                        im_moving[0, 0, idx_x, idx_y, idx_z] = 1.0

        # initialise a warp field
        offset = 20

        displacement = offset / self.dim_x * torch.ones(self.dims_v, device=self.device)
        transformation = self.identity_grid.permute([0, 4, 1, 2, 3]) + displacement
        im_moving_warped = self.registration_module(im_moving, transformation)

        # save the images to disk
        save_im_to_disk(im_moving[0, 0].cpu().numpy(), './temp/test_output/moving.nii.gz')
        save_im_to_disk(im_moving_warped[0, 0].cpu().numpy(), './temp/test_output/moving_warped_large.nii.gz')

    def test_brain_rotation(self):
        # initialise a rotation matrix
        theta = math.pi / 2.0  # 90 degrees
        R_arr = [[math.cos(theta), -1.0 * math.sin(theta), 0.0], [math.sin(theta), math.cos(theta), 0.0], [0.0, 0.0, 1.0]]
        R = torch.tensor(R_arr, device=self.device)

        # initialise a warp field
        dim_x, dim_y, dim_z = 128, 128, 128
        dims = (dim_x, dim_y, dim_z)

        identity_grid = init_identity_grid_3D(dims).to(self.device)
        transformation = identity_grid.permute([0, 4, 1, 2, 3])

        for idx_z in range(transformation.shape[2]):
            for idx_y in range(transformation.shape[3]):
                for idx_x in range(transformation.shape[4]):
                    p = transformation[0, :, idx_z, idx_y, idx_x]
                    transformation[0, :, idx_z, idx_y, idx_x] = torch.mv(R, p)

        # load an image an warp it
        im_path = '/vol/bitbucket/dig15/datasets/mine/biobank/biobank_08/1034854_T2_FLAIR_unbiased_brain_affine_to_mni.nii.gz'
        im_moving = sitk.ReadImage(im_path, sitk.sitkFloat32)

        im_moving = torch.from_numpy(transform.resize(np.transpose(sitk.GetArrayFromImage(im_moving), (2, 1, 0)), (dim_x, dim_y, dim_z)))
        im_moving = rescale_im(im_moving).unsqueeze(0).unsqueeze(0)
        im_moving = im_moving.to(self.device)

        im_moving_warped = self.registration_module(im_moving, transformation)

        # save the images to disk
        save_im_to_disk(im_moving[0, 0].cpu().numpy(), './temp/test_output/brain_moving.nii.gz')
        save_im_to_disk(im_moving_warped[0, 0].cpu().numpy(), './temp/test_output/brain_moving_warped.nii.gz')
