from abc import abstractmethod, ABC
from torch import nn

from utils import GradientOperator

import torch
import torch.nn.functional as F

"""
data loss
"""


class DataLoss(nn.Module, ABC):
    def __init__(self):
        super(DataLoss, self).__init__()

    @abstractmethod
    def forward(self, im_fixed, im_moving, z, mask):
        pass
    
    @abstractmethod
    def map(self, im_fixed, im_moving):
        pass

    @abstractmethod
    def reduce(self):
        pass


class LCC(DataLoss):
    """
    local cross-correlation
    """

    def __init__(self, s=None):
        super(LCC, self).__init__()

        if s is not None:
            self.s = s
            self.kernel_size = self.s * 2 + 1
            self.sz = float(self.kernel_size ** 3)

            self.padding = (self.s, self.s, self.s, self.s, self.s, self.s)

            self.kernel = nn.Conv3d(1, 1, kernel_size=self.kernel_size, stride=1, bias=False)
            nn.init.ones_(self.kernel.weight)
            self.kernel.weight.requires_grad_(False)

    def forward(self, im_fixed=None, im_moving=None, z=None, mask=None):
        if z is not None:
            if mask is not None:
                return -1.0 * torch.sum(z * mask)
            else:
                return -1.0 * torch.sum(z)

        cross, var_F, var_M = self.map(im_fixed, im_moving)
        return self.reduce(cross, var_F, var_M, mask)

    def map(self, im_fixed, im_moving):
        im_fixed = F.pad(im_fixed, self.padding, mode='replicate')
        im_moving = F.pad(im_moving, self.padding, mode='replicate')

        u_F = F.pad(self.kernel(im_fixed), self.padding, mode='replicate') / self.sz
        u_M = F.pad(self.kernel(im_moving), self.padding, mode='replicate') / self.sz

        cross = self.kernel((im_fixed - u_F) * (im_moving - u_M))
        var_F = self.kernel((im_fixed - u_F) * (im_fixed - u_F))
        var_M = self.kernel((im_moving - u_M) * (im_moving - u_M))

        return cross, var_F, var_M

    def reduce(self, cross, var_F, var_M, mask=None):
        lcc = cross * cross / (var_F * var_M + 1e-5)

        if mask is not None:
            return -1.0 * torch.sum(lcc * mask)

        return -1.0 * torch.sum(lcc)


class SSD(DataLoss):
    """
    sum of squared differences
    """

    def __init__(self):
        super(SSD, self).__init__()

    def forward(self, im_fixed=None, im_moving=None, z=None, mask=None):
        if z is not None:
            if mask is not None:
                return self.reduce(z * mask)
            else:
                return self.reduce(z)

        z = self.map(im_fixed, im_moving)
        return self.reduce(z, mask)

    def map(self, im_fixed, im_moving):
        return im_fixed - im_moving

    def reduce(self, z, mask=None):
        if mask is not None:
            return 0.5 * torch.sum(torch.pow(z * mask, 2))

        return 0.5 * torch.sum(torch.pow(z, 2))


"""
regularisation loss
"""


class RegLoss(nn.Module, ABC):
    def __init__(self, w_reg=1.0):
        super(RegLoss, self).__init__()
        self.w_reg = float(w_reg)

    @abstractmethod
    def forward(self, v):
        pass


class RegLossL2(RegLoss):
    def __init__(self, diff_op, w_reg):
        super(RegLossL2, self).__init__(w_reg)

        if diff_op == 'GradientOperator':
            self.diff_op = GradientOperator()
        else:
            raise Exception('Unknown differential operator')

    def forward(self, v):
        nabla_vx, nabla_vy, nabla_vz = self.diff_op(v)
        return self.w_reg * (torch.sum(torch.pow(nabla_vx, 2)) +
                             torch.sum(torch.pow(nabla_vy, 2)) +
                             torch.sum(torch.pow(nabla_vz, 2)))


class RegLossL2_Student(RegLoss):
    def __init__(self, diff_op, w_reg, nu0=2e-6, lambda0=1e-6, a0=1e-6, b0=1e-6):
        """
        t_nu0(x | 0, lambda0) with  nu0 = 2 * a0 deg. of freedom and scale lambda0 = a0 / b0,
        following a canonical parameterisation of the multivariate student distribution t_nu(x | mu, Lambda)

        if lambda0 is specified, b0 is ignored
        if nu0 is specified, a0 is ignored

        recall that t_nu0(x | 0, a0 / b0) = int_s N(x | 0, lambda) Gamma(lambda | a0, b0) ds

        lambda0 would be the expected "precision" E_{lambda ~ Gamma(lambda | a0, b0)}[lambda] = a0 / b0,
        hence the design choice

        lambda0 gives a more direct access to the strength of the prior
        """

        super(RegLossL2_Student, self).__init__(w_reg)

        if nu0 != 2e-6:
            self.a0 = nu0 / 2.0
        else:
            self.a0 = a0

        if lambda0 != 1e-6:
            b0 = self.a0 / lambda0

        self.b0_twice = b0 * 2.0

        if diff_op == 'GradientOperator':
            self.diff_op = GradientOperator()
        else:
            raise Exception('Unknown differential operator')

    def forward(self, v):
        nabla_vx, nabla_vy, nabla_vz = self.diff_op(v)
        return self.w_reg * torch.log(self.b0_twice
                                      + torch.sum(torch.pow(nabla_vx, 2))
                                      + torch.sum(torch.pow(nabla_vy, 2))
                                      + torch.sum(torch.pow(nabla_vz, 2))) * (self.a0 + 0.5)


"""
entropy
"""


class Entropy(nn.Module, ABC):
    def __init__(self):
        super(Entropy, self).__init__()

    @abstractmethod
    def forward(self, log_var_v, u_v):
        pass


class EntropyMultivariateNormal(Entropy):
    """
    multivariate normal distribution
    """

    def __init__(self):
        super(EntropyMultivariateNormal, self).__init__()

    def forward(self, log_var_v, u_v):
        sigma_v = torch.exp(0.5 * log_var_v) + 1e-5
        return -0.5 * (torch.log(1.0 + torch.sum(u_v * torch.pow(sigma_v, -2) * u_v)) + torch.sum(log_var_v))


"""
KL divergence
"""


class KL(nn.Module):
    def __init__(self, diff_op):
        super(KL, self).__init__()

        if diff_op == 'GradientOperator':
            self.diff_op = GradientOperator()
        else:
            raise Exception('Unknown differential operator')

    def forward(self, mu_v, log_var_v, u_v):
        nabla_ux, nabla_uy, nabla_uz = self.diff_op(u_v)
        nabla_vx, nabla_vy, nabla_vz = self.diff_op(mu_v)

        sigma_v = torch.exp(0.5 * log_var_v) + 1e-5

        t1 = 36.0 * torch.sum(sigma_v ** 2) \
             + torch.sum(torch.pow(nabla_ux, 2)) + torch.sum(torch.pow(nabla_uy, 2)) + torch.sum(torch.pow(nabla_uz, 2))
        t2 = torch.sum(nabla_vx ** 2) + torch.sum(nabla_vy ** 2) + torch.sum(nabla_vz ** 2)
        t3 = -1.0 * (torch.log(1.0 + torch.sum(u_v * torch.pow(sigma_v, -2) * u_v)) + torch.sum(log_var_v))

        return -0.5 * (t1 + t2 + t3)
