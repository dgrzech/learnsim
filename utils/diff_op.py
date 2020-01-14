from abc import ABC, abstractmethod
from torch import nn

import torch
import torch.nn.functional as F


class DifferentialOperator(nn.Module, ABC):
    """
    abstract class for defining a differential operator on a vector field
    """

    def __init__(self):
        super(DifferentialOperator, self).__init__()

    @abstractmethod
    def forward(self, v):
        return


class GradientOperator(DifferentialOperator):
    """
    Jacobian differential operator
    """

    def __init__(self):
        super(GradientOperator, self).__init__()

        self.px = (1, 1, 0, 0, 0, 0)  # paddings
        self.py = (0, 0, 1, 1, 0, 0)
        self.pz = (0, 0, 0, 0, 1, 1)

    def forward(self, v):
        # central differences
        dv_dx = 0.5 * F.pad(v[:, :, :, :, 2:] - v[:, :, :, :, :-2], self.px, 'replicate')
        dv_dy = 0.5 * F.pad(v[:, :, :, 2:, :] - v[:, :, :, :-2, :], self.py, 'replicate')
        dv_dz = 0.5 * F.pad(v[:, :, 2:, :, :] - v[:, :, :-2, :, :], self.pz, 'replicate')

        nabla_vx = torch.cat((dv_dx[:, 0], dv_dy[:, 0], dv_dz[:, 0]), 0)
        nabla_vy = torch.cat((dv_dx[:, 1], dv_dy[:, 1], dv_dz[:, 1]), 0)
        nabla_vz = torch.cat((dv_dx[:, 2], dv_dy[:, 2], dv_dz[:, 2]), 0)

        nabla_vx.unsqueeze_(0)
        nabla_vy.unsqueeze_(0)
        nabla_vz.unsqueeze_(0)

        return nabla_vx, nabla_vy, nabla_vz
