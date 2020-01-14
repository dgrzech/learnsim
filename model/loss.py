from abc import abstractmethod, ABC
from torch import nn

from utils import GradientOperator

import torch

"""
data loss
"""


class DataLoss(nn.Module, ABC):
    def __init__(self):
        super(DataLoss, self).__init__()
    
    @abstractmethod
    def forward(self):
        pass
    
    @abstractmethod
    def map(self, im_fixed, im_moving):
        pass

    @abstractmethod
    def reduce(self, z):
        pass


class LCC(DataLoss):
    """
    local cross-correlation
    """

    def __init__(self):
        super(LCC, self).__init__()

    def forward(self, z):
        return self.reduce(z)

    def map(self, im_fixed, im_moving):
        pass

    def reduce(self, z):
        return -1.0 * torch.sum(z)


class SSD(DataLoss):
    """
    sum of squared differences
    """

    def __init__(self):
        super(SSD, self).__init__()

    def forward(self, z):
        return self.reduce(z)

    def map(self, im_fixed, im_moving):
        return im_fixed - im_moving

    def reduce(self, z):
        return 0.5 * torch.sum(torch.pow(z, 2))


"""
regularisation loss
"""


class RegLoss(nn.Module, ABC):
    def __init__(self):
        super(RegLoss, self).__init__()

    @abstractmethod
    def forward(self, v):
        pass


class RegLossL2(RegLoss):
    def __init__(self, diff_op):
        super(RegLossL2, self).__init__()

        if diff_op == 'GradientOperator':
            self.diff_op = GradientOperator()
        else:
            raise Exception('Unknown differential operator')

    def forward(self, v):
        nabla_vx, nabla_vy, nabla_vz = self.diff_op(v)
        return 0.5 * (torch.sum(torch.pow(nabla_vx, 2)) + torch.sum(torch.pow(nabla_vy, 2)) + torch.sum(torch.pow(nabla_vz, 2)))


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
