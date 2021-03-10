from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, s=1, activation=nn.Identity()):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernel_size=2*s+1, stride=1, padding=s, padding_mode='zeros')
        self.activation = activation

        with torch.no_grad():
            self.main.weight.multiply_(1e-5)
            self.main.weight[..., s, s, s] = F.normalize(torch.rand_like(self.main.weight[..., s, s, s]), p=1, dim=0)

            nn.init.zeros_(self.main.bias)
            self.main.bias.add_(torch.randn_like(self.main.bias).multiply(1e-5))

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
        for layer in self.enc:
            im = layer(im)

        return im


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
                self.agg.weight.add_(torch.randn_like(self.agg.weight).multiply(1e-5))

        self.disable_grads()
        self.register_buffer_enabled = False  # NOTE (DG): workaround to enable logging the graph to tensorboard

    def enable_grads(self):
        for param in self.parameters():
            param.requires_grad_(True)

        self.train()

    def disable_grads(self):
        for param in self.parameters():
            param.requires_grad_(False)

        self.eval()

    @abstractmethod
    def encode(self, im_fixed, im_moving):
        pass

    def feature_extraction(self, im_fixed, im_moving):
        if self.learnable:
            if self.training:
                z_fixed = self.enc(im_fixed)
                N, C, D, H, W = z_fixed.shape
                im_fixed = self.agg(z_fixed.view(N, C, -1)).view(N, 1, D, H, W)
            else:
                if hasattr(self, 'im_fixed'):
                    im_fixed = self.im_fixed
                else:
                    z_fixed = self.enc(im_fixed)
                    N, C, D, H, W = z_fixed.shape
                    im_fixed = self.agg(z_fixed.view(N, C, -1)).view(N, 1, D, H, W)

                    if self.register_buffer_enabled:
                        self.register_buffer('im_fixed', im_fixed, persistent=False)

            z_moving = self.enc(im_moving)
            N, C, D, H, W = z_moving.shape
            im_moving = self.agg(z_moving.view(N, C, -1)).view(N, 1, D, H, W)

        return im_fixed, im_moving

    def forward(self, im_fixed, im_moving, mask):
        """
        forward pass logic

        :return: Model output
        """
        
        z_fixed, z_moving = self.feature_extraction(im_fixed, im_moving)
        z = self.encode(z_fixed, z_moving)

        return z[mask]

    def __str__(self):
        """
        model prints with number of trainable parameters
        """

        model_parameters = list(self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\ntrainable parameters: {}\n'.format(params)
