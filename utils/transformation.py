from abc import ABC, abstractmethod
from torch import nn

import torch.nn.functional as F


class TransformationModel(nn.Module, ABC):
    """
    abstract class for a transformation model, e.g. B-splines or a stationary velocity field
    """

    def __init__(self):
        super(TransformationModel, self).__init__()

    @abstractmethod
    def forward_2d_add(self):
        pass

    @abstractmethod
    def forward_2d_comp(self):
        pass

    @abstractmethod
    def forward_3d_add(self):
        pass

    @abstractmethod
    def forward_3d_comp(self):
        pass


class SVF(TransformationModel):
    """
    stationary velocity field transformation model
    """

    def __init__(self):
        super(SVF, self).__init__()
        self.no_steps = 12

    def forward_2d_add(self, identity_grid, v):
        warp_field = v / float(2 ** self.no_steps)

        for _ in range(self.no_steps):
            grid_sample_input = warp_field
            grid = identity_grid + grid_sample_input.permute([0, 2, 3, 1])

            warp_field = grid_sample_input + F.grid_sample(grid_sample_input, grid, padding_mode='border')

        return identity_grid.permute([0, 3, 1, 2]) + warp_field

    def forward_2d_comp(self, identity_grid, v):
        warp_field = identity_grid.permute([0, 3, 1, 2]) + v / float(2 ** self.no_steps)

        for _ in range(self.no_steps):
            grid_sample_input = warp_field
            grid = grid_sample_input.permute([0, 2, 3, 1])

            warp_field = F.grid_sample(grid_sample_input, grid, padding_mode='border')

        return warp_field

    def forward_3d_add(self, identity_grid, v):
        warp_field = v / float(2 ** self.no_steps)

        for _ in range(self.no_steps):
            grid_sample_input = warp_field
            grid = identity_grid + grid_sample_input.permute([0, 2, 3, 4, 1])

            warp_field = grid_sample_input + F.grid_sample(grid_sample_input, grid, padding_mode='border')

        return identity_grid.permute([0, 4, 1, 2, 3]) + warp_field

    def forward_3d_comp(self, identity_grid, v):
        warp_field = identity_grid.permute([0, 4, 1, 2, 3]) + v / float(2 ** self.no_steps)

        for _ in range(self.no_steps):
            grid_sample_input = warp_field
            grid = grid_sample_input.permute([0, 2, 3, 4, 1])

            warp_field = F.grid_sample(grid_sample_input, grid, padding_mode='border')

        return warp_field
