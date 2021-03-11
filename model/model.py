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
        if hasattr(self, 'u_F'):
            u_F = self.u_F
        else:
            u_F = self.kernel(im_fixed) / self.sz

            if self.register_buffer_enabled:
                self.register_buffer('u_F', u_F.detach().clone(), persistent=False)

        if hasattr(self, 'var_F'):
            var_F = self.var_F
        else:
            var_F = self.kernel((im_fixed - u_F) ** 2)

            if self.register_buffer_enabled:
                self.register_buffer('var_F', var_F.detach().clone(), persistent=False)

        u_M = self.kernel(im_moving) / self.sz
        var_M = self.kernel((im_moving - u_M) ** 2)

        cross = self.kernel((im_fixed - u_F) * (im_moving - u_M))
        z = cross * cross / (var_F * var_M + 1e-10)

        return z
