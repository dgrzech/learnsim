import torch
import torch.nn as nn

from base import BaseModel


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, s=1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, 2 * s + 1, stride=1, padding=s, padding_mode='replicate')
        self.activation = nn.LeakyReLU(0.2)

        with torch.no_grad():
            epsilon = self.main.weight * 1e-5
            nn.init.dirac_(self.main.weight, groups=2)
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
        prev_no_features = 2

        for no_features in self.no_features:
            self.enc.append(ConvBlock(prev_no_features, no_features, s=s))
            prev_no_features = no_features

        self.mid_feature_idx = self.no_features[-1] // 2 + 1

    def forward(self, im_fixed, im_moving):
        x = torch.cat([im_fixed, im_moving], dim=1)
        x_enc = x

        for layer in self.enc:
            x_enc = layer(x_enc)
        
        z_fixed = torch.cat([x_enc[:, 0].unsqueeze(1), x_enc[:, self.mid_feature_idx:]], dim=1)
        z_moving = x_enc[:, 1:self.mid_feature_idx]

        return z_fixed, z_moving


class CNN_SSD(BaseModel):
    def __init__(self, learnable=False, no_features=None):
        super(CNN_SSD, self).__init__(learnable)

        if self.learnable:
            self.enc = UNetEncoder(no_features)
            self.agg = nn.Conv1d(no_features[-1], 1, kernel_size=1, stride=1, bias=False)

            with torch.no_grad():
                nn.init.zeros_(self.agg.weight)
                self.agg.weight[0, 0] = 1.0
                self.agg.weight[0, 1] = -1.0

    def encode(self, im_fixed, im_moving):
        if self.learnable:
            z_fixed, z_moving = self.enc(im_fixed, im_moving)
            z = torch.cat([z_fixed, z_moving], dim=1)
            N, C, D, H, W = z.shape
            z = self.agg(z.view(N, C, -1)).view(N, 1, D, H, W)
        else:
            z = im_fixed - im_moving

        return z ** 2


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
            self.agg = nn.Conv1d(no_features[-1] // 2, 1, kernel_size=1, stride=1, bias=False)

            with torch.no_grad():
                nn.init.zeros_(self.agg.weight)
                self.agg.weight[0, 0] = 1.0

    def encode(self, im_fixed, im_moving):
        if self.learnable:
            z_fixed, z_moving = self.enc(im_fixed, im_moving)
            N, C, D, H, W = z_fixed.shape

            im_fixed = self.agg(z_fixed.view(N, C, -1)).view(N, 1, D, H, W)
            im_moving = self.agg(z_moving.view(N, C, -1)).view(N, 1, D, H, W)

        u_F = self.kernel(im_fixed) / self.sz
        u_M = self.kernel(im_moving) / self.sz

        cross = self.kernel((im_fixed - u_F) * (im_moving - u_M))
        var_F = self.kernel((im_fixed - u_F) ** 2)
        var_M = self.kernel((im_moving - u_M) ** 2)

        z = cross * cross / (var_F * var_M + 1e-10)
        return z
