import torch.nn.functional as F
from torch import nn


class RegistrationModule(nn.Module):
    """
    module for warping images and masks
    """

    def __init__(self):
        super(RegistrationModule, self).__init__()

    def forward(self, im_or_seg_moving, transformation):
        grid = transformation.permute([0, 2, 3, 4, 1])

        if im_or_seg_moving.type() in {'torch.BoolTensor', 'torch.cuda.BoolTensor',
                                       'torch.ShortTensor', 'torch.cuda.ShortTensor'}:
            im_or_seg_moving_float = im_or_seg_moving.float()
            im_or_seg_moving_warped = F.grid_sample(im_or_seg_moving_float, grid, mode='nearest', padding_mode='border',
                                                    align_corners=True)

            if im_or_seg_moving.type() in {'torch.BoolTensor', 'torch.cuda.BoolTensor'}:
                return im_or_seg_moving_warped.bool()
            elif im_or_seg_moving.type() in {'torch.ShortTensor', 'torch.cuda.ShortTensor'}:
                return im_or_seg_moving_warped.short()

        return F.grid_sample(im_or_seg_moving, grid, mode='bilinear', padding_mode='border', align_corners=True)
