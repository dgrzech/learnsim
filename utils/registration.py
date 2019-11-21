from torch import nn

import torch.nn.functional as F


class RegistrationModule(nn.Module):
    def __init__(self):
        super(RegistrationModule, self).__init__()

    def forward(self, m, identity_grid, warp_field):
        warp_field = identity_grid + warp_field.permute([0, 2, 3, 4, 1])
        return F.grid_sample(m, warp_field, padding_mode='border')
