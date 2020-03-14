from torch import nn

import torch.nn.functional as F


class RegistrationModule(nn.Module):
    """
    module for warping images and masks
    """

    def __init__(self):
        super(RegistrationModule, self).__init__()

    def forward(self, im_or_seg_moving, transformation):
        grid = transformation.permute([0, 2, 3, 4, 1])

        if im_or_seg_moving.type() == 'torch.BoolTensor' or im_or_seg_moving.type() == 'torch.cuda.BoolTensor':
            im_or_seg_moving_float = im_or_seg_moving.float()
            return F.grid_sample(im_or_seg_moving_float, grid,
                                 mode='nearest', padding_mode='border', align_corners=True).bool()
        
        if im_or_seg_moving.type() == 'torch.ShortTensor' or im_or_seg_moving.type() == 'torch.cuda.ShortTensor':
            im_or_seg_moving_float = im_or_seg_moving.float()
            return F.grid_sample(im_or_seg_moving_float, grid,
                                 mode='nearest', padding_mode='border', align_corners=True).short()

        return F.grid_sample(im_or_seg_moving, grid, mode='bilinear', padding_mode='border', align_corners=True)
