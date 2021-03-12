import math

import torch
from torch import nn

from base import BaseModel


class CNN_LCC(BaseModel):
    def __init__(self, learnable=False, no_features=None, s=1, activation=nn.Identity()):
        super(CNN_LCC, self).__init__(learnable, no_features, activation)

        self.sz = float((2 * s + 1) ** 3)
        self.kernel = nn.Conv3d(1, 1, kernel_size=2*s+1, stride=1, padding=s, bias=False, padding_mode='replicate')
        nn.init.ones_(self.kernel.weight)
        self.kernel.weight.requires_grad_(learnable)

    def map(self, im_fixed, im_moving, mask):
        u_F = self.kernel(im_fixed) / self.sz
        u_M = self.kernel(im_moving) / self.sz

        cross = self.kernel((im_fixed - u_F) * (im_moving - u_M))
        var_F = self.kernel((im_fixed - u_F) ** 2)
        var_M = self.kernel((im_moving - u_M) ** 2)

        z = cross * cross / (var_F * var_M + 1e-5)
        return z[mask]


class CNN_MI(BaseModel):
    def __init__(self, learnable=False, no_features=None, activation=nn.Identity(),
                 min=0.0, max=1.0, no_bins=64, sample_ratio=0.1, normalised=True):
        super(CNN_MI, self).__init__(learnable, no_features, activation)

        self.normalised = normalised
        self.sample_ratio = sample_ratio

        bins = torch.linspace(min, max, no_bins).unsqueeze(1)
        self.register_buffer('bins', bins, persistent=False)

        # set the std. dev. of the Gaussian kernel so that FWHM is one bin width
        bin_width = (max - min) / no_bins
        self.sigma = bin_width * 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))

    def __joint_prob(self, im_fixed, im_moving):
        # compute the Parzen window function response
        win_F = torch.exp(-0.5 * (im_fixed - self.bins) ** 2 / (self.sigma ** 2)) / (math.sqrt(2 * math.pi) * self.sigma)
        win_M = torch.exp(-0.5 * (im_moving - self.bins) ** 2 / (self.sigma ** 2)) / (math.sqrt(2 * math.pi) * self.sigma)

        # compute the histogram
        hist = win_F.bmm(win_M.transpose(1, 2))

        # normalise the histogram to get the joint distr.
        hist_normalised = hist.flatten(start_dim=1, end_dim=-1).sum(dim=1) + 1e-5
        return hist / hist_normalised.view(-1, 1, 1)

    def map(self, im_fixed, im_moving, mask):
        with torch.no_grad():
            batch_size = im_fixed.shape[0]
            no_voxels = mask[0].sum()
            no_voxels_sampled = int(self.sample_ratio * no_voxels)
            idxs = torch.randperm(no_voxels)[:no_voxels_sampled].view(batch_size, -1)
        
        im_fixed_masked = im_fixed[mask].view(batch_size, -1)
        im_moving_masked = im_moving[mask].view(batch_size, -1)

        im_fixed_sampled = im_fixed_masked[idxs].view(batch_size, 1, no_voxels_sampled_per_im)
        im_moving_sampled = im_moving_masked[idxs].view(batch_size, 1, no_voxels_sampled_per_im)

        p_FM = self.__joint_prob(im_fixed_sampled, im_moving_sampled)
        p_F, p_M = torch.sum(p_FM, dim=2), torch.sum(p_FM, dim=1)

        # calculate entropy of the distr.
        H_FM = -1.0 * torch.sum(p_FM * torch.log(p_FM + 1e-5), dim=(1, 2))
        H_F, H_M = -1.0 * torch.sum(p_F * torch.log(p_F + 1e-5), dim=1), -1.0 * torch.sum(p_M * torch.log(p_M + 1e-5), dim=1)

        z = (H_F + H_M) / H_FM if self.normalised else H_F + H_M - H_FM
        return z


class CNN_SSD(BaseModel):
    def __init__(self, learnable=False, no_features=None, activation=nn.Identity()):
        super(CNN_SSD, self).__init__(learnable, no_features, activation)

    def map(self, im_fixed, im_moving, mask):
        z = (im_fixed - im_moving) ** 2
        return z[mask]
