from abc import ABC, abstractmethod

import torch.nn.functional as F
from torch import nn
import torch

from utils import init_identity_grid_2D, init_identity_grid_3D, transform_coordinates, transform_coordinates_inv


class TransformationModule(nn.Module, ABC):
    """
    abstract class for a transformation model, e.g. B-splines or a stationary velocity field
    """

    def __init__(self):
        super(TransformationModule, self).__init__()

    @abstractmethod
    def forward(self, v):
        pass


def BSpline_1D_value(x, order):
    t = abs(x)

    if t >= 2:
        return 0

    if order == 0:
        if t < 1:
            return 2 / 3 + (0.5 * t - 1) * t ** 2

        return -1.0 * ((t - 2) ** 3) / 6

    if order == 1:
        if t < 1:
            return (1.5 * t - 2.0) * x
        if x < 0:
            return 0.5 * (t - 2) ** 2

        return -0.5 * (t - 2) ** 2

    if order == 2:
        if t < 1:
            return 3 * t - 2
        return -1.0 * t + 2


def BSpline_1D(stride, order):
    kernel = torch.ones(4 * stride - 1)
    radius = kernel.shape[0] // 2

    for i in range(kernel.shape[0]):
        kernel[i] = BSpline_1D_value((i - radius) / stride, order)

    return kernel


class BSpline_3D(TransformationModule):
    def __init__(self):
        super(BSpline_3D, self).__init__()
        self.kernels = list()
        self.stride = 1

        for s in self.stride:
            self.kernels.append(BSpline_1D(s))

        self.padding = [(len(kernel) - 1) // 2 for kernel in self.kernels]

    def forward(self, v):
        flow = v
        for i, (k, s, p) in enumerate(zip(self.kernels, self.stride, self.padding)):
            v = F.conv1d(v, dim=i+2, kernel=k, stride=s, padding=p, transpose=True)

        return flow


class SVF_2D(TransformationModule):
    """
    stationary velocity field transformation model
    """

    def __init__(self, dims, no_steps=12):
        super(SVF_2D, self).__init__()
        identity_grid = init_identity_grid_2D(dims)

        self.identity_grid = nn.Parameter(identity_grid, requires_grad=False)
        self.no_steps = no_steps

    def forward(self, v):
        """
        integrate a 2D stationary velocity field through scaling and squaring
        """

        displacement = transform_coordinates(v) / float(2 ** self.no_steps)

        for _ in range(self.no_steps):
            transformation = self.identity_grid + displacement.permute([0, 2, 3, 1])
            displacement = displacement + F.grid_sample(displacement, transformation,
                                                        padding_mode='border', align_corners=True)

        transformation = self.identity_grid.permute([0, 3, 1, 2]) + displacement
        return transformation, transform_coordinates_inv(displacement)


class SVF_3D(TransformationModule):
    """
    stationary velocity field transformation model
    """

    def __init__(self, dims, no_steps=12):
        super(SVF_3D, self).__init__()
        identity_grid = init_identity_grid_3D(dims)

        self.identity_grid = nn.Parameter(identity_grid, requires_grad=False)
        self.no_steps = no_steps

    def forward(self, v):
        """
        integrate a 3D stationary velocity field through scaling and squaring
        """

        displacement = transform_coordinates(v) / float(2 ** self.no_steps)

        for _ in range(self.no_steps):
            transformation = self.identity_grid + displacement.permute([0, 2, 3, 4, 1])
            displacement = displacement + F.grid_sample(displacement, transformation,
                                                        padding_mode='border', align_corners=True)

        transformation = self.identity_grid.permute([0, 4, 1, 2, 3]) + displacement
        return transformation, transform_coordinates_inv(displacement)
