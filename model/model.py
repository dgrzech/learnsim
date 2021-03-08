import torch
import torch.nn as nn

from base import BaseModel


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, s=1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernel_size=2*s+1, stride=1, padding=s, padding_mode='replicate')
        self.activation = nn.LeakyReLU(0.2)

        with torch.no_grad():
            epsilon = self.main.weight * 1e-5
            nn.init.dirac_(self.main.weight)
            nn.init.zeros_(self.main.bias)

            self.main.weight = nn.Parameter(self.main.weight + epsilon)

    def forward(self, x):
        out = self.main(x)
        return self.activation(out)


class UNetEncoder(nn.Module):
    def __init__(self, no_features, s=1):
        super(UNetEncoder, self).__init__()
        self.enc = nn.ModuleList()
        self.no_features = no_features
        prev_no_features = 1

        for no_features in self.no_features:
            self.enc.append(ConvBlock(prev_no_features, no_features, s=s))
            prev_no_features = no_features

    def forward(self, im):
        for layer in self.enc:
            im = layer(im)

        return im


class CNN_SSD(BaseModel):
    def __init__(self, learnable=False, no_features=None):
        super(CNN_SSD, self).__init__(learnable)

        if self.learnable:
            self.enc = UNetEncoder(no_features)
            self.agg = nn.Conv1d(no_features[-1], 1, kernel_size=1, stride=1, bias=False)

            with torch.no_grad():
                w = self.agg.weight * 1e-5
                w[0, 0] = 1.0
                self.agg.weight = nn.Parameter(w)
        
        self.disable_grads()

    def encode(self, im_fixed, im_moving):
        if self.learnable:
            z_fixed = self.enc(im_fixed)
            N, C, D, H, W = z_fixed.shape
            im_fixed = self.agg(z_fixed.view(N, C, -1)).view(N, 1, D, H, W)

            z_moving = self.enc(im_moving)
            N, C, D, H, W = z_moving.shape
            im_moving = self.agg(z_moving.view(N, C, -1)).view(N, 1, D, H, W)

        z = (im_fixed - im_moving) ** 2
        return z


class CNN_LCC(BaseModel):
    def __init__(self, learnable=False, no_features=None, s=1):
        super(CNN_LCC, self).__init__(learnable)

        kernel_size = 2 * s + 1
        self.sz = float(kernel_size ** 3)
        self.kernel = nn.Conv3d(1, 1, kernel_size=kernel_size, stride=1, padding=s, bias=False, padding_mode='replicate')

        with torch.no_grad():
            nn.init.ones_(self.kernel.weight)
            self.kernel.requires_grad_(False)

        if self.learnable:
            self.kernel.requires_grad_(True)

            self.no_features = no_features
            self.enc = UNetEncoder(no_features)
            self.agg = nn.Conv1d(no_features[-1], 1, kernel_size=1, stride=1, bias=False)

            with torch.no_grad():
                w = self.agg.weight * 1e-5
                w[0, 0] = 1.0
                self.agg.weight = nn.Parameter(w)
        
        self.disable_grads()

    def encode(self, im_fixed, im_moving):
        if self.learnable:
            z_fixed = self.enc(im_fixed)
            N, C, D, H, W = z_fixed.shape
            im_fixed = self.agg(z_fixed.view(N, C, -1)).view(N, 1, D, H, W)

            z_moving = self.enc(im_moving)
            N, C, D, H, W = z_moving.shape
            im_moving = self.agg(z_moving.view(N, C, -1)).view(N, 1, D, H, W)

        u_F = self.kernel(im_fixed) / self.sz
        u_M = self.kernel(im_moving) / self.sz

        cross = self.kernel((im_fixed - u_F) * (im_moving - u_M))
        var_F = self.kernel((im_fixed - u_F) ** 2)
        var_M = self.kernel((im_moving - u_M) ** 2)

        z = cross * cross / (var_F * var_M + 1e-10)
        return z
