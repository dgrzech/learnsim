from utils import init_identity_grid_2d, init_identity_grid_3d

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
    def forward(self, v):
        pass


class SVF_2D(TransformationModel):
    """
    stationary velocity field transformation model
    """

    def __init__(self, dim_x, dim_y):
        super(SVF_2D, self).__init__()
        self.no_steps = 16

        identity_grid = init_identity_grid_2d(dim_x, dim_y)
        self.identity_grid = nn.Parameter(identity_grid, requires_grad=False)

    def forward(self, v):
        """
        integrate a 2D stationary velocity field through scaling and squaring
        """

        warp_field = v / float(2 ** self.no_steps)

        for _ in range(self.no_steps):
            grid_sample_input = warp_field
            grid = self.identity_grid + grid_sample_input.permute([0, 2, 3, 1])

            warp_field = grid_sample_input + F.grid_sample(grid_sample_input, grid, padding_mode='border')

        transformation = warp_field + self.identity_grid.permute([0, 3, 1, 2])
        return transformation, warp_field


class SVF_3D(TransformationModel):
    """
    stationary velocity field transformation model
    """

    def __init__(self, dim_x, dim_y, dim_z):
        super(SVF_3D, self).__init__()
        self.no_steps = 16

        identity_grid = init_identity_grid_3d(dim_x, dim_y, dim_z)
        self.identity_grid = nn.Parameter(identity_grid, requires_grad=False)

    def forward(self, v):
        """
        integrate a 3D stationary velocity field through scaling and squaring
        """

        warp_field = v / float(2 ** self.no_steps)

        for _ in range(self.no_steps):
            grid_sample_input = warp_field
            grid = self.identity_grid + grid_sample_input.permute([0, 2, 3, 4, 1])

            warp_field = grid_sample_input + F.grid_sample(grid_sample_input, grid, padding_mode='border')

        transformation = warp_field + self.identity_grid.permute([0, 4, 1, 2, 3])
        return transformation, warp_field
