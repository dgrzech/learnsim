from torch import nn
import torch.nn.functional as F


class WarpingModule(nn.Module):
    def __init__(self):
        super(WarpingModule, self).__init__()

    def forward(self, moving, identity_grid, warp_field):
        """
        warp an image

        :param moving: moving image
        :param identity_grid:
        :param warp_field: dense deformation field to apply to the moving image

        :return: image obtained by applying warp_field to moving
        """

        warp_field = identity_grid + warp_field.permute([0, 2, 3, 4, 1])
        return F.grid_sample(moving, warp_field, padding_mode='border')


class RegistrationModule(nn.Module):
    def __init__(self):
        super(RegistrationModule, self).__init__()
        self.warping_module = WarpingModule()

    def forward(self, moving, identity_grid, warp_field):
        return self.warping_module.forward(moving, identity_grid, warp_field)
