from torch import nn

import torch.nn.functional as F


class RegistrationModule(nn.Module):
    def __init__(self):
        super(RegistrationModule, self).__init__()

    def forward(self, im_or_seg_moving, transformation, mode='bilinear'):
        grid = transformation.permute([0, 2, 3, 4, 1])
        return F.grid_sample(im_or_seg_moving, grid, mode=mode, padding_mode='border')
