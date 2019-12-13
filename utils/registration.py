from torch import nn

import torch.nn.functional as F


class RegistrationModule(nn.Module):
    def __init__(self):
        super(RegistrationModule, self).__init__()

    def forward(self, im_moving, warp_field):
        grid = warp_field.permute([0, 2, 3, 4, 1])
        return F.grid_sample(im_moving, grid, padding_mode='border')
