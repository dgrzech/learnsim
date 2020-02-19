from abc import ABC, abstractmethod
from torch import nn

import torch
import torch.nn.functional as F


class DifferentialOperator(nn.Module, ABC):
    """
    base class for defining a differential operator on a vector field
    """

    def __init__(self):
        super(DifferentialOperator, self).__init__()

    @abstractmethod
    def forward(self, v):
        return


class GradientOperator(DifferentialOperator):
    def __init__(self):
        super(GradientOperator, self).__init__()

        # paddings
        self.px = (0, 1, 0, 0, 0, 0)
        self.py = (0, 0, 0, 1, 0, 0)
        self.pz = (0, 0, 0, 0, 0, 1)

        # F.grid_sample(..) takes values in range (-1, 1), so needed for det(J) = 1 when the transformation is identity
        self.pixel_spacing = None

    def forward(self, v):
        if self.pixel_spacing is None:
            dim_x = v.size()[4]
            dim_y = v.size()[3]
            dim_z = v.size()[2]

            self.pixel_spacing = (2.0 / float(dim_x - 1), 2.0 / float(dim_y - 1), 2.0 / float(dim_z - 1))

        # forward differences
        dv_dx = F.pad(v[:, :, :, :, 1:] - v[:, :, :, :, :-1], self.px, mode='replicate') / self.pixel_spacing[0]
        dv_dy = F.pad(v[:, :, :, 1:] - v[:, :, :, :-1], self.py, mode='replicate') / self.pixel_spacing[1]
        dv_dz = F.pad(v[:, :, 1:] - v[:, :, :-1], self.pz, mode='replicate') / self.pixel_spacing[2]

        nabla_vx = torch.stack((dv_dx[:, 0], dv_dy[:, 0], dv_dz[:, 0]), 1)
        nabla_vy = torch.stack((dv_dx[:, 1], dv_dy[:, 1], dv_dz[:, 1]), 1)
        nabla_vz = torch.stack((dv_dx[:, 2], dv_dy[:, 2], dv_dz[:, 2]), 1)

        return nabla_vx, nabla_vy, nabla_vz
