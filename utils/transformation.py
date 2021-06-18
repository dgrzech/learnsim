from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn

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


def cubic_B_spline_1D_value(x):
    """
    evaluate a 1D cubic B-spline
    """

    t = abs(x)

    if t >= 2:  # outside the local support region
        return 0

    if t < 1:
        return 2.0 / 3.0 + (0.5 * t - 1.0) * t ** 2

    return -1.0 * ((t - 2.0) ** 3) / 6.0


def B_spline_1D_kernel(stride):
    kernel = torch.ones(4 * stride - 1)
    radius = kernel.shape[0] // 2

    for i in range(kernel.shape[0]):
        kernel[i] = cubic_B_spline_1D_value((i - radius) / stride)

    return kernel


def conv1D(x, kernel, dim=-1, stride=1, dilation=1, padding=0, transpose=False):
    """
    convolve data with 1-dimensional kernel along specified dimension
    """

    x = x.type(kernel.dtype)  # (N, ndim, *sizes)
    x = x.transpose(dim, -1)  # (N, ndim, *other_sizes, sizes[dim])
    shape_ = x.size()

    # reshape into channel (N, ndim * other_sizes, sizes[dim])
    groups = int(torch.prod(torch.tensor(shape_[1:-1])))
    weight = kernel.expand(groups, 1, kernel.shape[-1])  # (ndim*other_sizes, 1, kernel_size)
    x = x.reshape(shape_[0], groups, shape_[-1])  # (N, ndim*other_sizes, sizes[dim])
    conv_fn = F.conv_transpose1d if transpose else F.conv1d

    x = conv_fn(x, weight, stride=stride, dilation=dilation, padding=padding, groups=groups)
    x = x.reshape(shape_[0:-1] + x.shape[-1:])  # (N, ndim, *other_sizes, size[dim])

    return x.transpose(-1, dim)  # (N, ndim, *sizes)


class Cubic_B_spline_FFD_3D(TransformationModule):
    def __init__(self, dims, cps):
        """
        compute dense velocity field of the cubic B-spline FFD transformation model from input control point parameters
        :param cps: control point spacing
        """

        super(Cubic_B_spline_FFD_3D, self).__init__()

        self.dims = dims
        self.stride = cps
        self.kernels, self.padding = nn.ParameterList(), list()

        for s in self.stride:
            kernel = B_spline_1D_kernel(s)

            self.kernels.append(nn.Parameter(kernel, requires_grad=False))
            self.padding.append((len(kernel) - 1) // 2)

    def forward(self, v):
        # compute B-spline tensor product via separable 1D convolutions
        for i, (k, s, p) in enumerate(zip(self.kernels, self.stride, self.padding)):
            v = conv1D(v, dim=i + 2, kernel=k, stride=s, padding=p, transpose=True)

        #  crop the output to image size
        slicer = (slice(0, v.shape[0]), slice(0, v.shape[1])) + tuple(slice(s, s + self.dims[i]) for i, s in enumerate(self.stride))
        return v[slicer]


class SVFFD_3D(TransformationModule):
    def __init__(self, dims, cps):
        super(SVFFD_3D, self).__init__()

        self.cubic_B_spline_FFD = Cubic_B_spline_FFD_3D(dims, cps)
        self.SVF_3D = SVF_3D(dims)

    def forward(self, v):
        return self.SVF_3D(self.cubic_B_spline_FFD(v))
