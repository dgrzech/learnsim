from base import BaseModel

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_LCC(BaseModel):
    def __init__(self, s):
        super(CNN_LCC, self).__init__()

        self.kernel_size = s * 2 + 1
        self.sz = float(self.kernel_size ** 3)

        self.conv1 = nn.Conv3d(1, 1, kernel_size=self.kernel_size, stride=1, padding=s, bias=False)
        self.conv2 = nn.Conv3d(1, 1, kernel_size=self.kernel_size, stride=1, bias=False)

        self.padding = (s, s, s, s, s, s)

        # initialise kernels
        nn.init.zeros_(self.conv1.weight)
        nn.init.ones_(self.conv2.weight)

        w1 = self.conv1.weight.view(self.kernel_size ** 3)
        w1[math.floor((self.kernel_size ** 3) / 2)] = 1.0
        w1 = w1.view(1, 1, self.kernel_size, self.kernel_size, self.kernel_size)

        with torch.no_grad():
            self.conv1.weight = nn.Parameter(w1, requires_grad=True)

    def encode(self, im_fixed, im_moving_warped):
        im_fixed, im_moving_warped = F.pad(im_fixed, self.padding, mode='replicate'), F.pad(im_moving_warped, self.padding, mode='replicate')
        im_fixed, im_moving_warped = self.conv1(im_fixed), self.conv1(im_moving_warped)

        F2 = im_fixed * im_fixed
        M2 = im_moving_warped * im_moving_warped
        FM = im_fixed * im_moving_warped

        u_F, u_M = self.conv2(im_fixed) / self.sz, self.conv2(im_moving_warped) / self.sz

        cross = self.conv2(FM) - u_F * u_M * self.sz
        F_var = self.conv2(F2) - u_F * u_F * self.sz
        M_var = self.conv2(M2) - u_M * u_M * self.sz

        return cross, F_var, M_var

    def forward(self, im_fixed, im_moving_warped):
        cross, F_var, M_var = self.encode(im_fixed, im_moving_warped)
        return cross * cross / (F_var * M_var + 1e-1)


class CNN_SSD(BaseModel):
    def __init__(self, s):
        super(CNN_SSD, self).__init__()

        self.kernel_size = 2 * s + 1
        n = self.kernel_size ** 3

        self.conv1 = nn.Conv3d(1, 1, kernel_size=self.kernel_size, stride=1, padding=s, bias=False)
        self.conv2 = nn.Conv3d(1, 1, kernel_size=self.kernel_size, stride=1, padding=s, bias=False)

        # initialise kernels to identity
        nn.init.zeros_(self.conv1.weight)
        nn.init.zeros_(self.conv2.weight)

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

    def __init__(self, init_type, s=5):
        super(SimEnc, self).__init__()

        if init_type == 'SSD':
            self.CNN = CNN_SSD(s)
        elif init_type == 'LCC':
            self.CNN = CNN_LCC(s)

    def set_grad_enabled(self, mode):
        for p in self.parameters():
            p.requires_grad_(mode)

    def forward(self, im_fixed, im_moving_warped):
        return self.CNN.forward(im_fixed, im_moving_warped)
