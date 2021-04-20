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


def cubic_bspline_1d_value(x):
    r"""Evaluate 1-dimensional cubic B-spline."""
    t = abs(x)

    # outside local support region
    if t >= 2:
        return 0

    if t < 1:
        return 2 / 3 + (0.5 * t - 1) * t ** 2
    return -((t - 2) ** 3) / 6


def bspline_1d_kernel(stride):
    kernel = torch.ones(4 * stride - 1)
    radius = kernel.shape[0] // 2

    for i in range(kernel.shape[0]):
        kernel[i] = cubic_bspline_1d_value((i - radius) / stride)
    return kernel


def conv1d(x, kernel, dim=-1, stride=1, dilation=1, padding=0, transpose=False):
    r"""Convolve data with 1-dimensional kernel along specified dimension."""
    x = x.type(kernel.dtype)  # (N, ndim, *sizes)
    x = x.transpose(dim, -1)  # (N, ndim, *other_sizes, sizes[dim])
    shape_ = x.size()

    # reshape into channel (N, ndim * other_sizes, sizes[dim])
    groups = int(torch.prod(torch.tensor(shape_[1:-1])))
    weight = kernel.expand(groups, 1, kernel.shape[-1])  # (ndim*other_sizes, 1, kernel_size)
    x = x.reshape(shape_[0], groups, shape_[-1])  # (N, ndim*other_sizes, sizes[dim])
    conv_fn = F.conv_transpose1d if transpose else F.conv1d
    x = conv_fn(
        x,
        weight,
        stride=stride,
        dilation=dilation,
        padding=padding,
        groups=groups,
    )
    x = x.reshape(shape_[0:-1] + x.shape[-1:])  # (N, ndim, *other_sizes, size[dim])
    x = x.transpose(-1, dim)  # (N, ndim, *sizes)
    return x


class CubicBSplineFFD(TransformationModule):
    def __init__(self, ndim, img_size=192, cps=5):
        """
        Compute dense displacement field of Cubic B-spline FFD transformation model
        from input control point parameters.
        Args:
            ndim: (int) image dimension
            img_size: (int or tuple) size of the image
            cps: (int or tuple) control point spacing (in number of intervals between pixel/voxel centres)
        """
        super(CubicBSplineFFD, self).__init__()
        self.kernels = list()

        if isinstance(cps, (int, float)):
            # assume isotropic
            self.stride = (cps, ) * ndim
        else:
            self.stride = cps

        if isinstance(img_size, (int, float)):
            # assume same size each dim
            self.img_size = (img_size, ) * ndim
        else:
            self.img_size = img_size

        for s in self.stride:
            self.kernels.append(bspline_1d_kernel(s))

        self.padding = [(len(kernel) - 1) // 2 for kernel in self.kernels]

    def forward(self, v):
        # compute B-spline tensor product via separated 1d convolutions
        for i, (k, s, p) in enumerate(zip(self.kernels, self.stride, self.padding)):
            v = conv1d(v, dim=i+2, kernel=k, stride=s, padding=p, transpose=True)

        #  crop the output to image size
        slicer = (slice(0, v.shape[0]), slice(0, v.shape[1])) \
                 + tuple(slice(s, s + self.img_size[i]) for i, s in enumerate(self.stride))
        v = v[slicer]
        return v


if __name__ == '__main__':
    import math
    ndim = 3
    img_size = (192, ) * ndim
    cps = (4, ) * ndim

    # input to B-spline module needs to be configured to this size
    control_grid_sizes = tuple([int(math.ceil((imsz - 1) / c) + 1 + 2)
                                for imsz, c in zip(img_size, cps)])
    v = torch.randn(1, ndim, *control_grid_sizes)

    # b-spline model
    cubic_bspline_transform = CubicBSplineFFD(ndim, img_size=img_size, cps=cps)
    v = cubic_bspline_transform(v)
    print(f'Input sizes: {control_grid_sizes}, output sizes: {v.shape[2:]}')

    # svf model compatibility
    svf_transform = SVF_3D(img_size)
    disp_normed, disp = svf_transform(v)
    print(disp.shape)
