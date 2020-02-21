from abc import abstractmethod, ABC
from torch import nn
from torch.nn.functional import log_softmax

from utils import gaussian_kernel_3d, GaussianGrad, GradientOperator

import math
import torch
import torch.nn.functional as F


"""
data loss
"""


class DataLoss(nn.Module, ABC):
    """
    base class for all data losses
    """

    def __init__(self):
        super(DataLoss, self).__init__()

    @abstractmethod
    def forward(self, im_fixed, im_moving, mask):
        pass
    
    @abstractmethod
    def map(self, im_fixed, im_moving):
        pass

    @abstractmethod
    def reduce(self, z, mask):
        pass


class LCC(DataLoss):
    """
    local cross-correlation
    """

    def __init__(self, s):
        super(LCC, self).__init__()

        self.s = s
        self.padding = (s, s, s, s, s, s)

        self.kernel_size = self.s * 2 + 1
        self.sz = float(self.kernel_size ** 3)

        self.kernel = nn.Conv3d(1, 1, kernel_size=self.kernel_size, stride=1, bias=False)
        nn.init.ones_(self.kernel.weight)
        self.kernel.weight.requires_grad_(False)

    def forward(self, im_fixed, im_moving, mask):
        cross = self.map(im_fixed, im_moving)
        return self.reduce(cross, mask)

    def map(self, im_fixed, im_moving):
        im_fixed_padded = F.pad(im_fixed, self.padding, mode='replicate')
        im_moving_padded = F.pad(im_moving, self.padding, mode='replicate')

        u_F = self.kernel(im_fixed_padded) / self.sz
        var_F = self.kernel(torch.pow(F.pad(im_fixed - u_F, self.padding, mode='replicate'), 2)) / self.sz
        sigma_F = torch.sqrt(var_F)

        u_M = self.kernel(im_moving_padded) / self.sz
        var_M = self.kernel(torch.pow(F.pad(im_moving - u_M, self.padding, mode='replicate'), 2)) / self.sz
        sigma_M = torch.sqrt(var_M)

        n_F = (im_fixed - u_F) / (sigma_F + 1e-10)
        n_M = (im_moving - u_M) / (sigma_M + 1e-10)

        return self.kernel(F.pad(n_F * n_M, self.padding, mode='replicate'))

    def reduce(self, z, mask):
        return -1.0 * torch.sum(torch.pow(z, 2) * mask)


class SSD(DataLoss):
    """
    sum of squared differences
    """

    def __init__(self):
        super(SSD, self).__init__()

    def forward(self, im_fixed, im_moving, mask):
        z = self.map(im_fixed, im_moving)
        return self.reduce(z, mask)

    def map(self, im_fixed, im_moving):
        return im_fixed - im_moving

    def reduce(self, z, mask):
        return 0.5 * torch.sum(torch.pow(z, 2) * mask)


# © Loic Le Folgoc, l.le-folgoc@imperial.ac.uk
class GaussianMixtureLoss(DataLoss):
    """
    Gaussian mixture loss
    """

    def __init__(self, num_components, s):
        super(GaussianMixtureLoss, self).__init__()

        # parameters of the Gaussian mixture
        self.num_components = num_components

        self.log_std = nn.Parameter(torch.Tensor(num_components))
        self.logits = nn.Parameter(torch.zeros(num_components))
        self.register_buffer('_log_sqrt_2pi', torch.log(torch.Tensor([math.pi * 2.0])) / 2.0)

        # parameters of LCC
        self.s = s
        self.padding = (s, s, s, s, s, s)

        self.kernel_size = self.s * 2 + 1
        self.sz = float(self.kernel_size ** 3)

        self.kernel = nn.Conv3d(1, 1, kernel_size=self.kernel_size, stride=1, bias=False)
        self.kernel.weight.requires_grad_(False)

        nn.init.ones_(self.kernel.weight)

    def init_parameters(self, sigma):
        nn.init.zeros_(self.logits)

        sigma_min, sigma_max = sigma / 100.0, sigma * 5.0
        log_std_init = torch.linspace(math.log(sigma_min), math.log(sigma_max), steps=self.num_components)

        self.log_std.data.copy_(log_std_init.data)

    def log_pdf(self, z):
        # could equally apply the retraction trick to the mean (Riemannian metric for the tangent space of mean is
        # (v|w)_{mu,Sigma} = v^t Sigma^{-1} w), but it is less important with adaptive optimizers
        # and I also like the behaviour of the standard gradient intuitively

        z_flattened = z.view(1, -1, 1)

        E = (z_flattened * torch.exp(-self.log_std)) ** 2 / 2.0
        log_proportions = self.log_proportions()
        log_Z = self.log_std + self._log_sqrt_2pi

        return torch.logsumexp((log_proportions - log_Z) - E, dim=-1)

    def log_pdf_vd(self, z_scaled):
        E = (z_scaled ** 2) / 2.0
        log_proportions = self.log_proportions()
        log_Z = self.log_std + self._log_sqrt_2pi

        return torch.logsumexp((log_proportions - log_Z) - E, dim=-1)

    def log_proportions(self):
        return log_softmax(self.logits + 1e-2, dim=0, _stacklevel=5)
    
    def log_scales(self):
        return self.log_std
    
    def precision(self):
        return torch.exp(-2 * self.log_std)

    def forward(self, z):
        return self.reduce(z)
    
    def map(self, im_fixed, im_moving):
        im_fixed_padded = F.pad(im_fixed, self.padding, mode='replicate')
        im_moving_padded = F.pad(im_moving, self.padding, mode='replicate')

        u_F = self.kernel(im_fixed_padded) / self.sz
        var_F = self.kernel(torch.pow(F.pad(im_fixed - u_F, self.padding, mode='replicate'), 2)) / self.sz
        sigma_F = torch.sqrt(var_F + 1e-10)

        u_M = self.kernel(im_moving_padded) / self.sz
        var_M = self.kernel(torch.pow(F.pad(im_moving - u_M, self.padding, mode='replicate'), 2)) / self.sz
        sigma_M = torch.sqrt(var_M + 1e-10)

        return (im_fixed - u_F) / sigma_F, (im_moving - u_M) / sigma_M

    def reduce(self, z):
        return -1.0 * torch.sum(self.log_pdf(z))
    

class DirichletPrior(nn.Module):
    """
    alpha: None, scalar or (num_classes,) tensor expected. The concentration parameters for the prior.
    
    Wrapping torch distribution is even worse than copy paste -_-
    PS remove constants if you want
    """

    def __init__(self, num_classes, alpha=None):
        super(DirichletPrior, self).__init__() 
        
        if alpha is None:
            alpha = .5
            
        is_float = True

        try:
            val = float(alpha)
        except:
            is_float = False

        if is_float:
            self.concentration = nn.Parameter(torch.full(size=[num_classes], fill_value=alpha), requires_grad=False)
        else:
            if len(alpha) != num_classes:
                raise ValueError("Invalid tensor size. Expected {}".format(num_classes) +
                                 ", got: {}".format(len(alpha)))
            self.concentration = nn.Parameter(alpha.clone().squeeze().detach(), requires_grad=False)
        
    def forward(self, log_proportions):
        return (log_proportions * (self.concentration - 1.0)).sum(-1) + \
               torch.lgamma(self.concentration.sum(-1)) - torch.lgamma(self.concentration).sum(-1)


class ScaleLogNormalPrior(nn.Module):
    """
    Lots of wrapping for not much -_-
    """

    def __init__(self, loc=None, scale=None):
        super(ScaleLogNormalPrior, self).__init__()

        if loc is None:
            loc = 0.
        if scale is None:
            scale = math.log(10)

        loc_is_float = True
        try:
            val = float(loc)
        except:
            loc_is_float = False
        if loc_is_float:
            loc = torch.Tensor([loc])
        else:
            if len(loc) != 1:
                raise ValueError("Invalid tensor size. Expected 1, got: {}".format(len(loc)))
            loc = loc.clone()

        scale_is_float = True
        try:
            val = float(scale)
        except:
            scale_is_float = False
        if scale_is_float:
            scale = torch.Tensor([scale])
        else:
            if len(scale) != 1:
                raise ValueError("Invalid tensor size. Expected 1, got: {}".format(len(scale)))
            scale = scale.clone()

        self.loc = nn.Parameter(loc.detach(), requires_grad=False)
        self.log_scale = nn.Parameter(torch.log(scale).detach(), requires_grad=False)
        self.register_buffer('_log_sqrt_2pi', (torch.log(torch.Tensor([math.pi * 2.0])) / 2.0).detach())

    def forward(self, log_scales):
        E = ((log_scales - self.loc) * torch.exp(-self.log_scale)) ** 2 / 2.0
        return -E - self.log_scale - self._log_sqrt_2pi


class ScaleGammaPrior(nn.Module):
    """
    Wrapping torch distribution is even worse than copy paste -_-

    PS remove constants if you want
    """

    def __init__(self, shape=1e-3, rate=1e-3):
        super(ScaleGammaPrior, self).__init__() 
            
        shape_is_float = True
        try:
            val = float(shape)
        except:
            shape_is_float = False

        if shape_is_float:
            self.shape = nn.Parameter(torch.Tensor([shape]), requires_grad=False)
        else:
            if len(shape) != 1:
                raise ValueError("Invalid tensor size. Expected 1, got: {}".format(len(shape)))
            self.shape = shape.clone().squeeze().detach()
            
        rate_is_float = True

        try:
            val = float(rate)
        except:
            rate_is_float = False

        if rate_is_float:
            self.rate = nn.Parameter(torch.Tensor([rate]), requires_grad=False)
        else:
            if len(rate) != 1:
                raise ValueError("Invalid tensor size. Expected 1, got: {}".format(len(rate)))
            self.rate = rate.clone().squeeze().detach()
    
    def forward(self, log_precision):
        return self.shape * torch.log(self.rate) + (self.shape - 1) * log_precision - \
               self.rate * torch.exp(log_precision) - torch.lgamma(self.shape)


"""
regularisation loss
"""


class RegLoss(nn.Module, ABC):
    """
    base class for all regularisation losses
    """

    def __init__(self):
        super(RegLoss, self).__init__()

    @abstractmethod
    def forward(self, v):
        pass


class RegLossL2(RegLoss):
    def __init__(self, diff_op, w_reg):
        super(RegLossL2, self).__init__()
        self.w_reg = w_reg

        if diff_op == 'GradientOperator':
            self.diff_op = GradientOperator()
        else:
            raise Exception('Unknown differential operator')

    def forward(self, v):
        nabla_vx, nabla_vy, nabla_vz = self.diff_op(v)
        return self.w_reg * (torch.sum(torch.pow(nabla_vx, 2) + torch.pow(nabla_vy, 2) + torch.pow(nabla_vz, 2)))


class RegLossL2_Learnable(RegLoss):
    def __init__(self, diff_op, sigma_init, w_reg_init=1.0):
        super(RegLossL2_Learnable, self).__init__()

        if diff_op == 'GradientOperator':
            self.diff_op = GradientOperator()
        else:
            raise Exception('Unknown differential operator')

        gaussian_kernel = torch.from_numpy(gaussian_kernel_3d(96, sigma=sigma_init)).float()
        self.__gaussian_kernel_hat = nn.Parameter(torch.rfft(gaussian_kernel, 3, onesided=False))
        self.__gaussian_kernel_hat.requires_grad_(False)

        # voxel-specific regularisation weight
        self.log_lambda = nn.Parameter(math.log(w_reg_init) + torch.zeros((96, 96, 96)))

    def log_scales(self):
        return self.log_lambda

    def forward(self, v):
        log_lambda_smoothed = GaussianGrad.apply(self.log_lambda, self.__gaussian_kernel_hat)
        w_reg = torch.exp(log_lambda_smoothed)

        nabla_vx, nabla_vy, nabla_vz = self.diff_op(v)
        reg_term = torch.pow(nabla_vx, 2) + torch.pow(nabla_vy, 2) + torch.pow(nabla_vz, 2)

        return torch.sum(w_reg * reg_term) + log_lambda_smoothed.sum() / 2.0


class RegLossL2_Student(RegLoss):
    def __init__(self, diff_op, nu0=2e-6, lambda0=1e-6, a0=1e-6, b0=1e-6):
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

        super(RegLossL2_Student, self).__init__()

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

        self.N = None  # no. of voxels

    def forward(self, v):
        nabla_vx, nabla_vy, nabla_vz = self.diff_op(v)

        if self.N is None:
            self.N = v.numel() / 3.0

        return torch.log(self.b0_twice +
                         torch.sum(torch.pow(nabla_vx, 2)) +
                         torch.sum(torch.pow(nabla_vy, 2)) +
                         torch.sum(torch.pow(nabla_vz, 2))) * (self.a0 + self.N * 0.5)


"""
entropy
"""


class Entropy(nn.Module, ABC):
    """
    base class for the entropy of a probability distribution
    """

    def __init__(self):
        super(Entropy, self).__init__()

    @abstractmethod
    def forward(self, **kwargs):
        pass


class EntropyMultivariateNormal(Entropy):
    """
    multivariate normal distribution
    """

    def __init__(self):
        super(EntropyMultivariateNormal, self).__init__()

    def forward(self, **kwargs):
        if len(kwargs) == 2:
            log_var_v = kwargs['log_var_v']
            u_v = kwargs['u_v']

            sigma_v = torch.exp(0.5 * log_var_v)
            return 0.5 * (torch.log1p(torch.sum(torch.pow(u_v / sigma_v, 2))) + torch.sum(log_var_v))
        elif len(kwargs) == 4:
            v_sample = kwargs['v_sample']

            mu_v = kwargs['mu_v']
            log_var_v = kwargs['log_var_v']
            u_v = kwargs['u_v']

            sigma_v = torch.exp(0.5 * log_var_v)

            v = (v_sample - mu_v) / sigma_v
            u_n = u_v / sigma_v

            t1 = torch.sum(torch.pow(v, 2))
            t2 = torch.pow(torch.sum(v * u_n), 2) / (1.0 + torch.sum(torch.pow(u_n, 2)))

            return 0.5 * (t1 - t2)
