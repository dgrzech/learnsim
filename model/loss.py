from abc import abstractmethod, ABC
from torch import nn
from utils.diff_op import GradientOperator

import torch

"""
data loss
"""


class DataLoss(nn.Module, ABC):
    def __init__(self):
        super(DataLoss, self).__init__()

    def forward(self, z):
        return self.reduce(z)

    @abstractmethod
    def reduce(self, z):
        pass


class SSD(DataLoss):
    def __init__(self):
        super(SSD, self).__init__()

    def reduce(self, z):
        return 0.5 * torch.sum(z ** 2)


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
        dv_dx, dv_dy, dv_dz = self.diff_op.apply(v)
        return torch.sum(dv_dx ** 2) + torch.sum(dv_dy ** 2) + torch.sum(dv_dz ** 2)


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
    def __init__(self):
        super(EntropyMultivariateNormal, self).__init__()

    def forward(self, log_var_v, u_v):
        sigma_v = torch.exp(0.5 * log_var_v) + 1e-5
        return 0.5 * (torch.log(1.0 + torch.sum(u_v * 1.0 / (sigma_v ** 2) * u_v)) + torch.sum(log_var_v))


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

    def forward(self, v, log_var_v, u_v):
        du_v_dx, du_v_dy, du_v_dz = self.diff_op.apply(u_v)
        dv_dx, dv_dy, dv_dz = self.diff_op.apply(v)

        sigma_v = torch.exp(0.5 * log_var_v) + 1e-5

        t1 = 36.0 * torch.sum(sigma_v ** 2) + torch.sum(du_v_dx ** 2) + torch.sum(du_v_dy ** 2) + torch.sum(du_v_dz ** 2)
        t2 = torch.sum(dv_dx ** 2) + torch.sum(dv_dy ** 2) + torch.sum(dv_dz ** 2)
        t3 = -1.0 * (torch.log(1.0 + torch.sum(u_v * 1.0 / (sigma_v ** 2) * u_v)) + torch.sum(log_var_v))

        return -0.5 * (t1 + t2 + t3)
