import torch.nn as nn

from base import BaseModel


class CNN_SSD(BaseModel):
    def __init__(self, learnable=False, no_features=None, activation=nn.Identity()):
        super(CNN_SSD, self).__init__(learnable, no_features, activation)

    def encode(self, im_fixed, im_moving):
        z = (im_fixed - im_moving) ** 2
        return z


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

        z = cross * cross / (var_F * var_M + 1e-10)
        return z
