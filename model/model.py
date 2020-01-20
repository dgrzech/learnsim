from base import BaseModel

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_LCC(BaseModel):
    """
    initialised to LCC
    """

    def __init__(self, s, no_feature_maps):
        super(CNN_LCC, self).__init__()

        self.no_feature_maps = no_feature_maps

        self.kernel_size = s * 2 + 1
        self.sz = float(self.kernel_size ** 3)
        self.padding = (s, s, s, s, s, s)

        # initialise convolutional layers
        self.conv1 = nn.Conv3d(1, self.no_feature_maps, kernel_size=self.kernel_size, stride=1, bias=False)
        self.conv2 = nn.Conv3d(1, 1, kernel_size=self.kernel_size, stride=1, bias=False)
        
        # initialise kernels
        nn.init.ones_(self.conv2.weight)  # sum

        if self.no_feature_maps == 1:  # FIXME: when not learning the similarity metric, use a regular loss function
            nn.init.zeros_(self.conv1.weight)

            w1 = self.conv1.weight
            w1[0, 0, s, s, s] = 1.0
        else:
            w1 = self.conv1.weight * 1e-5
            w1[0, 0, s, s, s] = 1.0

        with torch.no_grad():
            self.conv1.weight = nn.Parameter(w1, requires_grad=True)

    def encode(self, im_fixed, im_moving_warped):
        im_fixed, im_moving_warped = F.pad(im_fixed, self.padding, mode='replicate'), \
                                     F.pad(im_moving_warped, self.padding, mode='replicate')
        im_fixed, im_moving_warped = self.conv1(im_fixed), \
                                     self.conv1(im_moving_warped)

        if self.no_feature_maps > 1:
            N, C, D, H, W = im_fixed.size()

            im_fixed = im_fixed.reshape(N, 1, self.no_feature_maps, D, H, W).sum(2)
            im_moving_warped = im_moving_warped.reshape(N, 1, self.no_feature_maps, D, H, W).sum(2)

        im_fixed, im_moving_warped = F.pad(im_fixed, self.padding, mode='replicate'), \
                                     F.pad(im_moving_warped, self.padding, mode='replicate')
        u_F, u_M = F.pad(self.conv2(im_fixed), self.padding, mode='replicate') / self.sz, \
                   F.pad(self.conv2(im_moving_warped), self.padding, mode='replicate') / self.sz

        cross = self.conv2((im_fixed - u_F) * (im_moving_warped - u_M))
        F_var = self.conv2((im_fixed - u_F) * (im_fixed - u_F))
        M_var = self.conv2((im_moving_warped - u_M) * (im_moving_warped - u_M))

        return cross, F_var, M_var

    def forward(self, im_fixed, im_moving_warped):
        cross, F_var, M_var = self.encode(im_fixed, im_moving_warped)
        return cross * cross / (F_var * M_var + 1e-5)


class CNN_SSD(BaseModel):
    """
    initialised to SSD
    """

    def __init__(self, s, no_feature_maps):
        super(CNN_SSD, self).__init__()

        self.kernel_size = 2 * s + 1
        n = self.kernel_size ** 3

        # convolutional layers
        self.conv1 = nn.Conv3d(1, 1, kernel_size=self.kernel_size, stride=1, padding=s, bias=False)
        self.conv2 = nn.Conv3d(1, 1, kernel_size=self.kernel_size, stride=1, padding=s, bias=False)

        # initialise kernels
        nn.init.zeros_(self.conv1.weight)  # identity
        nn.init.zeros_(self.conv2.weight)  # identity

        w1 = self.conv1.weight.view(n)
        w1[math.floor(n / 2)] = 1.0
        w1 = w1.view(1, 1, self.kernel_size, self.kernel_size, self.kernel_size)

        w2 = self.conv2.weight.view(n)
        w2[math.floor(n / 2)] = 1.0
        w2 = w2.view(1, 1, self.kernel_size, self.kernel_size, self.kernel_size)

        with torch.no_grad():
            self.conv1.weight = nn.Parameter(w1, requires_grad=True)
            self.conv2.weight = nn.Parameter(w2, requires_grad=True)

    def encode(self, im_fixed, im_moving_warped):
        diff = im_fixed - im_moving_warped
        h1 = self.conv1(diff)
        return self.conv2(h1)

    def forward(self, im_fixed, im_moving_warped):
        return self.encode(im_fixed, im_moving_warped)


class SimEnc(BaseModel):
    """
    encoding function
    """

    def __init__(self, init_type, s=5, no_feature_maps=4):
        super(SimEnc, self).__init__()

        if init_type == 'SSD':  # initialisation to SSD
            self.CNN = CNN_SSD(s, no_feature_maps)
        elif init_type == 'LCC':  # initialisation to LCC
            self.CNN = CNN_LCC(s, no_feature_maps)

    def set_grad_enabled(self, mode):
        for p in self.parameters():
            p.requires_grad_(mode)

    def forward(self, im_fixed, im_moving_warped):
        return self.CNN.forward(im_fixed, im_moving_warped)
