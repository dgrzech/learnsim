import torch.nn.functional as F
from torch import nn


class RegistrationModule(nn.Module):
    """
    module for warping images and masks
    """

    def __init__(self):
        super(RegistrationModule, self).__init__()

        self.im_types = ['torch.FloatTensor', 'torch.cuda.FloatTensor']
        self.mask_types = ['torch.BoolTensor', 'torch.cuda.BoolTensor']
        self.seg_types = ['torch.LongTensor', 'torch.cuda.LongTensor']

    def forward(self, im_or_seg_moving, transformation):
        grid = transformation.permute([0, 2, 3, 4, 1])

        if self.is_mask(im_or_seg_moving) or self.is_seg(im_or_seg_moving):
            im_or_seg_moving_float = im_or_seg_moving.float()
            im_or_seg_moving_warped = F.grid_sample(im_or_seg_moving_float, grid, mode='nearest', padding_mode='border', align_corners=True)

            if self.is_mask(im_or_seg_moving):
                return im_or_seg_moving_warped.bool()
            elif self.is_seg(im_or_seg_moving):
                return im_or_seg_moving_warped.long()

        elif self.is_im(im_or_seg_moving):
            return F.grid_sample(im_or_seg_moving, grid, mode='bilinear', padding_mode='border', align_corners=True)

        raise NotImplementedError

    def is_im(self, input):
        return input.type() in self.im_types

    def is_mask(self, input):
        return input.type() in self.mask_types

    def is_seg(self, input):
        return input.type() in self.seg_types
