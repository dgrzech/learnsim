import pytest
import torch.nn.functional as F
from skimage.data import shepp_logan_phantom

from logger import save_im_to_disk
from utils import pixel_to_normalised_3D, rescale_im
from .test_setup import *


class WarpingTestMethods(unittest.TestCase):
    def setUp(self):
        print(self._testMethodName + '\n')

    def test_identity_transformation(self):
        im_fixed = torch.rand(1, 1, *dims, device=device)
        im_moving = torch.rand_like(im_fixed)
        mask = torch.ones_like(im_fixed).bool()

        transformation = identity_grid.permute([0, 4, 1, 2, 3])
        im_moving_warped = registration_module(im_moving, transformation)

        z_unwarped = (im_fixed - im_moving) ** 2
        z_unwarped_masked = z_unwarped[mask]

        z_warped = (im_fixed - im_moving_warped) ** 2
        z_warped_masked = z_warped[mask]

        unwarped_loss_value = loss_SSD(z_unwarped_masked).item()
        warped_loss_value = loss_SSD(z_warped_masked).item()

        assert pytest.approx(unwarped_loss_value, rel=rtol) == warped_loss_value

    def test_sphere_translation(self):
        # initialise 3D image of a sphere
        im_moving = -1.0 + torch.zeros(1, 1, *dims, device=device)
        r = 0.02

        for idx_z in range(im_moving.shape[2]):
            for idx_y in range(im_moving.shape[3]):
                for idx_x in range(im_moving.shape[4]):
                    x, y, z = pixel_to_normalised_3D(idx_x, idx_y, idx_z, dim_x, dim_y, dim_z)

                    if x ** 2 + y ** 2 + z ** 2 <= r:
                        im_moving[0, 0, idx_x, idx_y, idx_z] = 1.0

        # initialise a warp field
        offset = 5.0

        displacement = offset / dim_x * torch.ones(dims_v, device=device)
        transformation = identity_grid.permute([0, 4, 1, 2, 3]) + displacement
        im_moving_warped = registration_module(im_moving, transformation)

        # save the images to disk
        save_im_to_disk(im_moving[0, 0].cpu().numpy(), './temp/test_output/moving.nii.gz')
        save_im_to_disk(im_moving_warped[0, 0].cpu().numpy(), './temp/test_output/moving_warped.nii.gz')

    def test_sphere_translation_large(self):
        # initialise 3D image of a sphere
        im_moving = -1.0 + torch.zeros((1, 1, *dims), device=device)
        r = 0.02

        for idx_z in range(im_moving.shape[2]):
            for idx_y in range(im_moving.shape[3]):
                for idx_x in range(im_moving.shape[4]):
                    x, y, z = pixel_to_normalised_3D(idx_x, idx_y, idx_z, dim_x, dim_y, dim_z)

                    if x ** 2 + y ** 2 + z ** 2 <= r:
                        im_moving[0, 0, idx_x, idx_y, idx_z] = 1.0

        # initialise a warp field
        offset = 20

        displacement = offset / dim_x * torch.ones(dims_v, device=device)
        transformation = identity_grid.permute([0, 4, 1, 2, 3]) + displacement
        im_moving_warped = registration_module(im_moving, transformation)

        # save the images to disk
        save_im_to_disk(im_moving[0, 0].cpu().numpy(), './temp/test_output/moving.nii.gz')
        save_im_to_disk(im_moving_warped[0, 0].cpu().numpy(), './temp/test_output/moving_warped_large.nii.gz')

    def test_brain_rotation(self):
        # initialise a rotation matrix
        theta = math.pi / 2.0  # 90 degrees

        R1 = [math.cos(theta), -1.0 * math.sin(theta), 0.0]
        R2 = [math.sin(theta), math.cos(theta), 0.0]
        R3 = [0.0, 0.0, 1.0]
        R = torch.tensor([R1, R2, R3], device=device)

        # initialise a warp field
        transformation = identity_grid.permute([0, 4, 1, 2, 3])

        for idx_z in range(transformation.shape[2]):
            for idx_y in range(transformation.shape[3]):
                for idx_x in range(transformation.shape[4]):
                    p = transformation[0, :, idx_z, idx_y, idx_x]
                    transformation[0, :, idx_z, idx_y, idx_x] = torch.mv(R, p)

        # load an image an warp it
        im_moving = np.expand_dims(shepp_logan_phantom(), axis=0)
        im_moving_arr = np.transpose(im_moving, (2, 1, 0))

        padding = (max(im_moving_arr.shape) - np.asarray(im_moving_arr.shape)) // 2
        padding = ((padding[0], padding[0]), (padding[1], padding[1]), (padding[2], padding[2]))

        im_moving_arr_padded = np.pad(im_moving_arr, padding, mode='minimum')
        im_moving_tensor = torch.from_numpy(im_moving_arr_padded).unsqueeze(0).unsqueeze(0).float()
        im_moving = F.interpolate(im_moving_tensor, size=dims, mode='trilinear', align_corners=True)
        im_moving = rescale_im(im_moving).to(device)

        im_moving_warped = registration_module(im_moving, transformation)

        # save the images to disk
        save_im_to_disk(im_moving[0, 0].cpu().numpy(), './temp/test_output/brain_moving.nii.gz')
        save_im_to_disk(im_moving_warped[0, 0].cpu().numpy(), './temp/test_output/brain_moving_warped.nii.gz')
