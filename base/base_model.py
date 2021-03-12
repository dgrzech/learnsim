from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, s=1, activation=nn.Identity()):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernel_size=2*s+1, stride=1, padding=s, padding_mode='replicate')
        self.activation = activation

        with torch.no_grad():
            self.main.weight.multiply_(1e-7)
            self.main.weight[..., s, s, s] = F.normalize(torch.rand_like(self.main.weight[..., s, s, s]), p=1, dim=0)

            nn.init.zeros_(self.main.bias)
            self.main.bias.add_(torch.randn_like(self.main.bias).multiply(1e-7))

    def forward(self, x):
        out = self.main(x)
        return self.activation(out)


class UNetEncoder(nn.Module):
    def __init__(self, no_features, s=1, activation=nn.Identity()):
        super(UNetEncoder, self).__init__()
        self.enc = nn.ModuleList()
        self.no_features = no_features
        prev_no_features = 1

        for no_features in self.no_features:
            self.enc.append(ConvBlock(prev_no_features, no_features, s=s, activation=activation))
            prev_no_features = no_features

    def forward(self, im):
        im_enc = [im]

        for layer in self.enc:
            im_enc.append(layer(im_enc[-1]))

        return im_enc[-1]


class BaseModel(nn.Module):
    """
    base class for all models
    """

    def __init__(self, learnable, no_features=None, activation=nn.Identity()):
        super(BaseModel, self).__init__()
        self.learnable = learnable

        if self.learnable:
            self.enc = UNetEncoder(no_features, activation=activation)
            self.agg = nn.Conv1d(no_features[-1], 1, kernel_size=1, stride=1, bias=False)

            with torch.no_grad():
                nn.init.ones_(self.agg.weight)
                self.agg.weight.add_(torch.randn_like(self.agg.weight).multiply(1e-7))

        self.disable_grads()

    def enable_grads(self):
        for param in self.parameters():
            param.requires_grad_(True)

        self.train()

    def disable_grads(self):
        for param in self.parameters():
            param.requires_grad_(False)

        self.eval()

    @abstractmethod
    def map(self, im_fixed, im_moving, mask):
        pass

    def feature_extraction(self, im_fixed, im_moving):
        if self.learnable:
            z_fixed = self.enc(im_fixed)
            N, C, D, H, W = z_fixed.shape
            z_fixed = self.agg(z_fixed.view(N, C, -1)).view(N, 1, D, H, W)

            z_moving = self.enc(im_moving)
            z_moving = self.agg(z_moving.view(N, C, -1)).view(N, 1, D, H, W)
        else:
            z_fixed = im_fixed
            z_moving = im_moving

        return z_fixed, z_moving

    def forward(self, im_fixed, im_moving, mask):
        """
        forward pass logic

        :return: Model output
        """
        
        z_fixed, z_moving = self.feature_extraction(im_fixed, im_moving)
        z = self.map(z_fixed, z_moving, mask)

        return z

    def __str__(self):
        """
        model prints with number of trainable parameters
        """
        
        model_parameters = list(self.parameters())
        no_trainable_params = sum([np.prod(p.size()) for p in model_parameters]) if self.learnable else 0
        return super().__str__() + '\ntrainable parameters: {}\n'.format(no_trainable_params)

