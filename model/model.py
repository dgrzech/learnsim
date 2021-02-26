import torch
import torch.nn as nn

from base import BaseModel
from utils import get_noise_uniform


class CNN_LCC(BaseModel):
    def __init__(self, learnable=False, s=1, no_feature_maps=1):
        super(CNN_LCC, self).__init__(learnable)

        kernel_size = s * 2 + 1
        self.sz = float(kernel_size ** 3)
        self.kernel = nn.Conv3d(1, 1, kernel_size=kernel_size, stride=1, padding=s, bias=False, padding_mode='replicate')

        with torch.no_grad():
            nn.init.ones_(self.kernel.weight)
            self.kernel.weight.requires_grad_(False)

        if self.learnable:
            with torch.no_grad():
                self.conv1 = nn.Conv3d(1, no_feature_maps, kernel_size=kernel_size, stride=1, padding=s, bias=False, padding_mode='replicate')
                self.agg = nn.Conv1d(no_feature_maps, 1, kernel_size=1, stride=1, bias=False)
                
                nn.init.dirac_(self.conv1.weight, groups=no_feature_maps)
                nn.init.ones_(self.agg.weight)

                self.conv1.weight /= no_feature_maps

                # add noise
                alpha = 1e-5
                epsilon = get_noise_uniform(self.conv1.weight.shape, alpha)
                self.conv1.weight = nn.Parameter(self.conv1.weight + epsilon)

                alpha = 1e-3
                epsilon = get_noise_uniform(self.kernel.weight.shape, alpha)
                self.kernel.weight = nn.Parameter(self.kernel.weight + epsilon)

    def print_weights(self):
        print(self.conv1.weight)
        print(self.agg.weight)
        print(self.kernel.weight)

    def encode(self, im_fixed, im_moving):
        if self.learnable:
            im_fixed_out1 = self.conv1(im_fixed)
            im_moving_out1 = self.conv1(im_moving)

            N_f, C_f, D_f, H_f, W_f = im_fixed_out1.shape
            N_m, C_m, D_m, H_m, W_m = im_moving_out1.shape

            im_fixed_out = self.agg(im_fixed_out1.view(N_f, C_f, D_f * H_f * W_f)).view(N_f, 1, D_f, H_f, W_f)
            im_moving_out = self.agg(im_moving_out1.view(N_m, C_m, D_m * H_m * W_m)).view(N_m, 1, D_m, H_m, W_m)
        else:
            im_fixed_out = im_fixed
            im_moving_out = im_moving

        u_F = self.kernel(im_fixed_out) / self.sz
        u_M = self.kernel(im_moving_out) / self.sz

        cross = self.kernel((im_fixed_out - u_F) * (im_moving_out - u_M))
        var_F = self.kernel((im_fixed_out - u_F) ** 2)
        var_M = self.kernel((im_moving_out - u_M) ** 2)

        z = cross * cross / (var_F * var_M + 1e-10)
        return z


class CNN_SSD(BaseModel):
    def __init__(self, learnable=False, s=1, no_feature_maps=1):
        super(CNN_SSD, self).__init__(learnable)

        if self.learnable:
            with torch.no_grad():
                kernel_size = s * 2 + 1

                self.conv1 = nn.Conv3d(1, no_feature_maps, kernel_size=kernel_size, stride=1, padding=s, bias=False, padding_mode='replicate')
                self.agg = nn.Conv1d(no_feature_maps, 1, kernel_size=1, stride=1, bias=False)

                nn.init.dirac_(self.conv1.weight, groups=no_feature_maps)
                nn.init.ones_(self.agg.weight)
                
                self.conv1.weight /= no_feature_maps

                # add noise
                alpha = 1e-5
                epsilon = get_noise_uniform(self.conv1.weight.shape, alpha)
                self.conv1.weight = nn.Parameter(self.conv1.weight + epsilon)

    def print_weights(self):
        print(self.conv1.weight)
        print(self.agg.weight)

    def encode(self, im_fixed, im_moving):
        if self.learnable:
            im_fixed_out1 = self.conv1(im_fixed)
            im_moving_out1 = self.conv1(im_moving)

            N_f, C_f, D_f, H_f, W_f = im_fixed_out1.shape
            N_m, C_m, D_m, H_m, W_m = im_moving_out1.shape

            im_fixed_out = self.agg(im_fixed_out1.view(N_f, C_f, D_f * H_f * W_f)).view(N_f, 1, D_f, H_f, W_f)
            im_moving_out = self.agg(im_moving_out1.view(N_m, C_m, D_m * H_m * W_m)).view(N_m, 1, D_m, H_m, W_m)
        else:
            im_fixed_out = im_fixed
            im_moving_out = im_moving

        z = (im_fixed_out - im_moving_out) ** 2
        return z
