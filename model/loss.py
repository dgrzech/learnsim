from abc import abstractmethod, ABC
from torch import nn
from torch.nn.functional import log_softmax

from utils import gaussian_kernel_3d, transform_coordinates_inv, vd_reg, GaussianGrad, DifferentialOperator

import model.distributions as model_distr

import math
import numpy as np
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


"""
regularisation loss
"""


class RegLoss(nn.Module, ABC):
    """
    base class for all regularisation losses.
    
    diff_op: takes an input tensor and outputs a tensor, the result of the action of diff_op on the input.
    Or strings that can be parsed to a diff_op
    """

    def __init__(self, diff_op=None):
        super(RegLoss, self).__init__()

        if diff_op is not None:
            if isinstance(diff_op, str):  
                # create from string, e.g. diff_op='GradientOperator'
                self.diff_op = DifferentialOperator.from_string(diff_op)
            else:
                if isinstance(diff_op, DifferentialOperator):
                    # point to passed instance, e.g. diff_op=GradientOperator()
                    self.diff_op = diff_op
                else:
                    # create instance from class name, e.g. diff_op=GradientOperator
                    self.diff_op = diff_op()
        else:
            # identity operator
            self.diff_op = DifferentialOperator()
            
    def forward(self, input, *args, **kwargs):
        """
        All RegLosses operate as functions of the energy defined by the differential operator.
        This is not meant to be overriden, just define the proper diff_op and _loss.
        """

        D_input = self.diff_op(input)
        y = torch.sum(D_input ** 2)  # "chi-square" variable / energy
        
        return self._loss(y, *args, **kwargs)
    
    @abstractmethod
    def _loss(y, *args, **kwargs):
        """
        Override this in children classes. For Bayesian RegLosses, this _loss is of the form 
            - log_pdf(...) and potentially includes hyperpriors for tunable parameters.
        """
        pass


class RegLoss_L2(RegLoss):
    """
    This implements all spatial and Fourier logGaussian losses, by passing
    1) a suitable differential operator
    2) a suitable input to forward (implemented in parent class)
    
    Example 1. Penalty on the squared Froebenius norm of the Jacobian of the velocity, implemented in spatial domain:
        diff_op = GradientOperator
        input = v.
    Example 2. Smooth Sobolev space equivalent of the above.
        diff_op = GradientOperator
        v = K^{1/2} alpha is the (Sobolev-smooth) velocity field obtained from its L2-preimage alpha
        input = alpha
    Example 3. Equivalent of example 2, but implemented in frequency domain.
        diff_op = Fourier1stDerivativeOperator
        alpha is an L2 preimage of the velocity, it lives in the frequency domain.
        v = fft(half_kernel_hat * alpha * 1 / sqrt(|omega|**2 + 1)) is the (Sobolev-smooth) velocity field
    """
    
    def __init__(self, dims, w_reg, diff_op=None, learnable=False):
        super(RegLoss_L2, self).__init__(diff_op=diff_op)

        log_w_reg = math.log(w_reg)
        self.log_w_reg = nn.Parameter(torch.tensor([log_w_reg])).requires_grad_(learnable)

    def _loss(self, y, dof=0):
        """
        num degrees of freedom dof only matters for learnable w_reg, the argument doesn't need to be passed otherwise.
        """

        return self.log_w_reg.exp() * 0.5 * y - .5 * dof * self.log_w_reg, y.log()

    
class RegLoss_Student(RegLoss):
    """
    See RegLossL2 for tips.
    """
    
    def __init__(self, dims, diff_op=None, nu0=2e-6, lambda0=1e-6, a0=1e-6, b0=1e-6):
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

        super(RegLoss_Student, self).__init__(diff_op=diff_op)
        
        if nu0 != 2e-6:
            self.a0 = nu0 / 2.0
        else:
            self.a0 = a0

        if lambda0 != 1e-6:
            b0 = self.a0 / lambda0

        self.b0_twice = b0 * 2.0

    def _loss(self, y, dof):
        """
        num degrees of freedom dof is the number of free variables in the field to which this is a prior
        WARNING. dof = 3*number of voxels for a prior on a velocity field
            I changed your multiplicative factor from 1.5 to .5, but dof = 3*N
        """

        return torch.log(self.b0_twice + y) * (self.a0 + .5 * dof), y.log()
    

class RegLoss_EnergyBased(RegLoss):
    """
    All these EnergyBasedRegLosses derive from expressing a prior on the scalar energy (returned by forward()),
    and converting it to a prior on the underlying field with dof number of degrees of freedom.
    
    RegLossL2 actually belongs to this family, 
    with _mlog_energy_prior corresponding to -log Gamma(dof/2,wreg/2),
    up to the quality of numerical implementation in here.
    """
    
    def __init__(self, diff_op=None):
        super(RegLoss_EnergyBased, self).__init__(diff_op=diff_op)
    
    @abstractmethod
    def _mlog_energy_prior(self, y, *args, **kwargs):
        """
        Override in children classes, this is -log pdf of the prior on the energy y.
        """
        pass
        
    def _loss(self, y, dof, *args, **kwargs):
        """
        All EnergyBasedRegLosses should use this _loss, not meant to be overriden.
        To adjust the behaviour, adjust the _mlog_energy_prior (and the class attributes)
        """
        
        return self._mlog_energy_prior(y, *args, **kwargs) + (dof * 0.5 - 1.0) * torch.log(y), torch.log(y)
        

class RegLoss_LogNormal(RegLoss_EnergyBased):
    """
    log-Normal prior on the energy y, as returned by self.forward.
    """
    
    def __init__(self, dims, w_reg=1.0, diff_op=None, learnable=False, loc_learnable=False, scale_learnable=False):
        """
        The default values have no reason to be good values. If no prior knowledge, use learnable=True along with:
        - A Normal hyperprior on loc, or another choice (see below)
        - A suitable scale hyperprior on scale (Gamma or Log-Normal). Careful since you will probably implement it
          as a prior on log_scale, which induces a change of variable p(log_scale)=p(scale)*scale. Convenient when it
          turns a LogNormal prior on scale into a Normal prior on LogScale; more tricky when using a Gamma prior on scale.
          
        A potential choice of hyperprior on loc is actually an exponential-Gamma(dof/2, wreg/2). 
        Recall that loc is the logarithm of an energy. What this prior means is that exp(loc) is Gamma(dof/2,wreg/2).
        This allows to fall back onto the "familiar" regularisation strength wreg to calibrate and initialize loc.
        expGamma(dof/2,wreg/2) has a mean at digamma(dof/2) - ln(wreg/2), and it will be sharply peaked if dof is big.
        This would yield a very informative prior on loc. A variant if necessary would be Gamma(nu/2,wreg/2) with
        learnable degrees of freedom nu.
        
        Choosing the hyperprior on scale is more intuitive and based on how vague or informative we want it to be, it 
        regulates the amount of deviations of ln(y), the log of the actual energy from the mean loc.
        """

        super(RegLoss_EnergyBased, self).__init__(diff_op=diff_op)

        if not learnable:
            loc_learnable = False
            scale_learnable = False

        dof = np.prod(dims) * 3.0
        log_energy_exp_gamma_prior = model_distr.LogEnergyExpGammaPrior(w_reg, dof)

        loc_init = log_energy_exp_gamma_prior.expectation()
        self.loc = nn.Parameter(torch.Tensor([loc_init]), requires_grad=loc_learnable)

        log_scale = math.log(4.0) + math.log(loc_init)
        self.log_scale = nn.Parameter(torch.Tensor([log_scale]), requires_grad=scale_learnable)
        
    def _mlog_energy_prior(self, y, *args, **kwargs):
        scale = torch.exp(self.log_scale)
        log_y = torch.log(y)
        
        return log_y + self.log_scale + 0.5 * ((log_y - self.loc) / scale) ** 2


class RegLoss_LogNormal_L2(RegLoss_EnergyBased):
    def __init__(self, diff_op=None):
        super(RegLoss_EnergyBased, self).__init__(diff_op=diff_op)
        self.gamma_distr = model_distr._GammaDistribution(96.0 ** 3 * 1.5, 0.1, learnable=False)
        
    def _mlog_energy_prior(self, y, *args, **kwargs):
        log_y = torch.log(y)
        return -1.0 * self.gamma_distr(log_y)


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

            v = transform_coordinates_inv(v_sample - mu_v) / sigma_v  # FIXME: aiyoo..
            u_n = u_v / sigma_v

            t1 = torch.sum(torch.pow(v, 2))
            t2 = torch.pow(torch.sum(v * u_n), 2) / (1.0 + torch.sum(torch.pow(u_n, 2)))

            return 0.5 * (t1 - t2)
