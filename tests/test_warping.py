from utils import init_identity_grid_3d, pixel_to_normalised_3d, rescale_im, standardise_im, save_im_to_disk, RegistrationModule

from skimage import transform

import math
import numpy as np
import SimpleITK as sitk
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

        warp_field = offset / self.dim_x * torch.ones(self.dims_v).to('cuda:0')
        transformation = self.identity_grid.permute([0, 4, 1, 2, 3]) + warp_field

        im_moving_warped = self.registration_module(im_moving, transformation)

        """"
        save the images to disk
        """

        save_im_to_disk(im_moving[0, :, :, :, :], './temp/moving.nii.gz')
        save_im_to_disk(im_moving_warped[0, :, :, :, :], './temp/moving_warped.nii.gz')

    def test_sphere_translation_large(self):
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

        offset = 20
    
        warp_field = offset / self.dim_x * torch.ones(self.dims_v).to('cuda:0')
        transformation = self.identity_grid.permute([0, 4, 1, 2, 3]) + warp_field

        im_moving_warped = self.registration_module(im_moving, transformation)

        """"
        save the images to disk
        """

        save_im_to_disk(im_moving[0, :, :, :, :], './temp/moving.nii.gz')
        save_im_to_disk(im_moving_warped[0, :, :, :, :], './temp/moving_warped_large.nii.gz')

    def test_brain_rotation(self):
        """
        initialise a rotation matrix
        """

        theta = math.pi / 2.0  # 90 degrees

        R_arr = [[math.cos(theta), -1.0 * math.sin(theta), 0.0],
                 [math.sin(theta), math.cos(theta), 0.0],
                 [0.0, 0.0, 1.0]]
        R = torch.Tensor(R_arr).to('cuda:0')

        """
        initialise a warp field
        """

        dim_x = 128
        dim_y = 128
        dim_z = 128

        identity_grid = init_identity_grid_3d(dim_x, dim_y, dim_z).to('cuda:0')
        transformation = identity_grid.permute([0, 4, 1, 2, 3])

        for idx_z in range(transformation.shape[2]):
            for idx_y in range(transformation.shape[3]):
                for idx_x in range(transformation.shape[4]):
                    p = transformation[0, :, idx_z, idx_y, idx_x]

                    p_new = torch.mv(R, p)
                    transformation[0, :, idx_z, idx_y, idx_x] = p_new

        """
        load image and warp it
        """

        im_path = '/vol/bitbucket/dig15/biobank_test/1007582_T2_FLAIR_unbiased_brain_affine_to_mni.nii.gz'
        im_moving = sitk.ReadImage(im_path, sitk.sitkFloat32)

        im_moving = torch.from_numpy(transform.resize(
            np.transpose(sitk.GetArrayFromImage(im_moving), (2, 1, 0)), (dim_x, dim_y, dim_z)))

        im_moving = standardise_im(im_moving)
        im_moving = rescale_im(im_moving)

        im_moving.unsqueeze_(0).unsqueeze_(0)
        im_moving = im_moving.to('cuda:0')

        im_moving_warped = self.registration_module(im_moving, transformation)

        """"
        save the images to disk
        """

        save_im_to_disk(im_moving[0, :, :, :, :], './temp/brain_moving.nii.gz')
        save_im_to_disk(im_moving_warped[0, :, :, :, :], './temp/brain_moving_warped.nii.gz')
