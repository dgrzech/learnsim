from abc import abstractmethod, ABC
from torch import nn

from utils.diff_op import GradientOperator

import torch
import torch.nn.functional as F

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


class SSD(DataLoss):
    def __init__(self):
        super(SSD, self).__init__()

    def forward(self, z):
        return self.reduce(z)

    def map(self, im_fixed, im_moving):
        return im_fixed - im_moving

    def reduce(self, z):
        return 0.5 * torch.sum(torch.pow(z, 2))


class LCC(DataLoss):
    def __init__(self, s):
        super(LCC, self).__init__()

        kernel_size = s * 2 + 1
        kernel = torch.ones([kernel_size, kernel_size, kernel_size])
        kernel.unsqueeze_(0).unsqueeze_(0)

        self.padding = (s, s, s, s, s, s)
        self.sz = float(kernel_size ** 3)
        self.kernel = kernel.to('cuda:0')

    def compute_local_means(self, im_fixed, im_moving_warped):
        return F.conv3d(im_fixed, self.kernel) / self.sz, F.conv3d(im_moving_warped, self.kernel) / self.sz

    def compute_local_sums(self, im_fixed, im_moving_warped):
        F2 = im_fixed * im_fixed
        M2 = im_moving_warped * im_moving_warped
        FM = im_fixed * im_moving_warped

        u_F, u_M = self.compute_local_means(im_fixed, im_moving_warped)

        cross = F.conv3d(FM, self.kernel) - u_F * u_M * self.sz
        F_var = F.conv3d(F2, self.kernel) - u_F * u_F * self.sz
        M_var = F.conv3d(M2, self.kernel) - u_M * u_M * self.sz

        return cross, F_var, M_var
    
    def forward(self, im_fixed, im_moving_warped):
        return self.reduce(im_fixed, im_moving_warped)

    def map(self, im_fixed, im_moving):
        pass

    def reduce(self, im_fixed, im_moving_warped):
        im_fixed = F.pad(im_fixed, self.padding, mode='replicate')
        im_moving_warped = F.pad(im_moving_warped, self.padding, mode='replicate')

        cross, F_var, M_var = self.compute_local_sums(im_fixed, im_moving_warped)
        lcc = cross * cross / (F_var * M_var + 1e-1)

        return -1.0 * torch.sum(lcc)


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
        return 0.5 * (torch.sum(torch.pow(dv_dx, 2)) + torch.sum(torch.pow(dv_dy, 2)) + torch.sum(torch.pow(dv_dz, 2)))


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
        du_v_dx, du_v_dy, du_v_dz = self.diff_op.apply(u_v)
        dv_dx, dv_dy, dv_dz = self.diff_op.apply(mu_v)

        sigma_v = torch.exp(0.5 * log_var_v) + 1e-5

        t1 = 36.0 * torch.sum(sigma_v ** 2) \
             + torch.sum(torch.pow(du_v_dx, 2)) + torch.sum(torch.pow(du_v_dy, 2)) + torch.sum(torch.pow(du_v_dz, 2))
        t2 = torch.sum(dv_dx ** 2) + torch.sum(dv_dy ** 2) + torch.sum(dv_dz ** 2)
        t3 = -1.0 * (torch.log(1.0 + torch.sum(u_v * torch.pow(sigma_v, -2) * u_v)) + torch.sum(log_var_v))

        return -0.5 * (t1 + t2 + t3)
