from abc import ABC
from torch import nn

import torch
import torch.nn.functional as F


"""
differential operator
"""    


def _get_all_subclasses(cls):
    for scls in cls.__subclasses__():
        yield scls
        for scls in _get_all_subclasses(scls):
            yield scls


class DifferentialOperator(nn.Module, ABC):
    """
    from_string is just a convenience method to create an instance of subclass s from string s.
    e.g. DifferentialOperator.from_string('GradientOperator')
    """
    def __init__(self):
        super(DifferentialOperator, self).__init__()
        
    @staticmethod
    def from_string(s, *args, **kwargs):
        for cls in _get_all_subclasses(DifferentialOperator):
            # pick the class based on the string
            if cls.__name__ in s:
                return cls(*args, **kwargs)
        raise ValueError('Unknown differential operator: {}'.format(s))
        
    def forward(self, input):
        """
        Override in children classes, default behaviour is identity map.
        Our GradientOperator should stack the 3 x, y, z maps on a new dimension.
        """
        return input

    
class Fourier1stDerivativeOperator(DifferentialOperator):
    """
    Square root of the Laplacian operator computed in frequency domain:
        alpha -> alpha * |omega|
    (technically it's not the nabla operator)
    
    image_size (int): number of voxels along x, y, or z (assumed to be the same)
    """
    
    def __init__(self, image_size):
        super(DifferentialOperator, self).__init__()
        omega_sq = get_omega_norm_sq((image_size, image_size, image_size)).transpose(1, 4)
        self.omega_abs = nn.Parameter(omega_sq.sqrt_().unsqueeze(-1)).requires_grad_(False)

    def forward(self, input):
        """
        input: a field in frequency domain ('z_hat'), with its real and imaginary parts along dim=-1
        """
        return input * self.omega_abs
    
    
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
            dim_x = v.shape[4]
            dim_y = v.shape[3]
            dim_z = v.shape[2]

            self.pixel_spacing = (2.0 / float(dim_x - 1), 2.0 / float(dim_y - 1), 2.0 / float(dim_z - 1))

        # forward differences
        dv_dx = F.pad(v[:, :, :, :, 1:] - v[:, :, :, :, :-1], self.px, mode='replicate') / self.pixel_spacing[0]
        dv_dy = F.pad(v[:, :, :, 1:] - v[:, :, :, :-1], self.py, mode='replicate') / self.pixel_spacing[1]
        dv_dz = F.pad(v[:, :, 1:] - v[:, :, :-1], self.pz, mode='replicate') / self.pixel_spacing[2]

        nabla_vx = torch.stack((dv_dx[:, 0], dv_dy[:, 0], dv_dz[:, 0]), 1)
        nabla_vy = torch.stack((dv_dx[:, 1], dv_dy[:, 1], dv_dz[:, 1]), 1)
        nabla_vz = torch.stack((dv_dx[:, 2], dv_dy[:, 2], dv_dz[:, 2]), 1)

        return torch.stack([nabla_vx, nabla_vy, nabla_vz], dim=-1)
