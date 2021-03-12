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

    def encode(self, im_fixed, im_moving):
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
        self.bins = nn.Parameter(bins, requires_grad=False)

        # set the std. dev. of the Gaussian kernel so that FWHM is one bin width
        bin_width = (max - min) / no_bins
        self.sigma = bin_width * 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))

    def __joint_prob(self, im_fixed, im_moving):
        # compute the Parzen window function response
        win_F = torch.exp(-1.0 * (im_fixed - self.bins) ** 2 / (2.0 * self.sigma ** 2)) \
                / (math.sqrt(2 * math.pi) * self.sigma)
        win_M = torch.exp(-1.0 * (im_moving - self.bins) ** 2 / (2.0 * self.sigma ** 2)) \
                / (math.sqrt(2 * math.pi) * self.sigma)

        # compute the histogram
        hist = win_F.bmm(win_M.transpose(1, 2))

        # normalise the histogram to get the joint distr.
        hist_normalised = hist.flatten(start_dim=1, end_dim=-1).sum(dim=1) + 1e-5
        return hist / hist_normalised.view(-1, 1, 1)

    def map(self, im_fixed, im_moving, mask):
        no_voxels = int(mask[0].sum())
        sampled_no_voxels = int(self.sample_ratio * no_voxels)
        idxs = torch.randperm(no_voxels)[:sampled_no_voxels]

        batch_size = mask.shape[0]
        im_fixed_sampled = torch.cat(tuple(im_fixed[idx, ...].flatten()[idxs].unsqueeze(0).unsqueeze(0) for idx in range(batch_size)), dim=0)
        im_moving_sampled = torch.cat(tuple(im_moving[idx, ...].flatten()[idxs].unsqueeze(0).unsqueeze(0) for idx in range(batch_size)), dim=0)

        p_FM = self.__joint_prob(im_fixed_sampled, im_moving_sampled)
        p_F, p_M = torch.sum(p_FM, dim=2), torch.sum(p_FM, dim=1)

        # calculate the entropies
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
