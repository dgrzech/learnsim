import math

import torch
from torch import nn


class LowRankMultivariateNormalDistribution(nn.Module):
    def __init__(self, model, mu, loc_learnable=False, cov_learnable=False):
        super(LowRankMultivariateNormalDistribution, self).__init__()

        if model not in ['LCC', 'SSD']:
            raise NotImplementedError

        self.model = model
        self.loc_learnable, self.cov_learnable = loc_learnable, cov_learnable

        with torch.no_grad():
            log_var = torch.log((0.1 ** 2) * torch.ones_like(mu))
            u = torch.zeros_like(mu)

            self.mu = nn.Parameter(mu, requires_grad=loc_learnable)
            self.log_var = nn.Parameter(log_var, requires_grad=cov_learnable)
            self.u = nn.Parameter(u, requires_grad=cov_learnable)

    def forward(self, no_samples=1):
        if self.model == 'SSD':
            with torch.no_grad():
                self.log_var.clamp_(max=0.0)
                self.u.clamp_(min=-1.0, max=0.0)

        sigma = torch.exp(0.5 * self.log_var)

        eps = torch.randn_like(sigma)
        x = torch.randn(1, device=self.u.device)

        if no_samples == 1:
            return self.mu + eps * sigma + x * self.u
        elif no_samples == 2:
            return self.mu + (eps * sigma + x * self.u), self.mu - (eps * sigma + x * self.u)

        raise NotImplementedError

"""
Useful distributions
"""


class NormalDistribution(nn.Module):
    """
    Univariate Normal distribution x~N(loc, scale).
    """

    def __init__(self, loc=None, scale=None, learnable=False):
        super(NormalDistribution, self).__init__()

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

        self.loc = nn.Parameter(loc.detach(), requires_grad=learnable)
        self.log_scale = nn.Parameter(torch.log(scale).detach(), requires_grad=learnable)
        self.register_buffer('_log_sqrt_2pi', (torch.log(torch.Tensor([math.pi * 2.0])) / 2.0).detach())

    def forward(self, x):
        E = ((x - self.loc) * torch.exp(-self.log_scale)) ** 2 / 2.0
        return -E - self.log_scale - self._log_sqrt_2pi


class _GammaDistribution(nn.Module):
    """
    x~Gamma(shape, rate). The support is the positive real line. The forward is the log pdf.
    The forward is such that log p(x) = forward(log_x). 
    
    Don't use this distribution, it's just a building block for distributions defined on log_x. You are most likely
    looking for ExpGammaDistribution.
    """

    def __init__(self, shape=1e-3, rate=1e-3, shape_learnable=False, rate_learnable=False, learnable=False):
        super(_GammaDistribution, self).__init__()

        if not learnable:
            shape_learnable = False
            rate_learnable = False

        shape_is_float = True
        try:
            val = float(shape)
        except:
            shape_is_float = False

        if shape_is_float:
            self.shape = nn.Parameter(torch.Tensor([shape]), requires_grad=shape_learnable)
        else:
            if len(shape) != 1:
                raise ValueError("Invalid tensor size. Expected 1, got: {}".format(len(shape)))
            self.shape = nn.Parameter(shape.clone().squeeze().detach(), requires_grad=shape_learnable)

        rate_is_float = True

        try:
            val = float(rate)
        except:
            rate_is_float = False

        if rate_is_float:
            self.rate = nn.Parameter(torch.Tensor([rate]), requires_grad=rate_learnable)
        else:
            if len(rate) != 1:
                raise ValueError("Invalid tensor size. Expected 1, got: {}".format(len(rate)))
            self.rate = nn.Parameter(rate.clone().squeeze().detach(), requires_grad=rate_learnable)

    def expectation(self):
        return self.shape / self.rate

    def forward(self, log_x):
        return gamma_log_pdf(log_x, self.shape, self.rate)


def gamma_log_pdf(log_x, shape, rate):
    return shape * torch.log(rate) + (shape - 1) * log_x - rate * torch.exp(log_x) - torch.lgamma(shape)


class _InverseGammaDistribution(nn.Module):
    """
    Distribution of x = 1/z if z~Gamma(shape, rate).
    The forward is such that log p(x) = forward(log_x). 
    
    Don't use this distribution, it's just a building block for distributions defined on log_x. You are most likely
    looking for ExpInverseGammaDistribution.
    """

    def __init__(self, shape=1e-3, rate=1e-3, shape_learnable=False, rate_learnable=False, learnable=False):
        super(_InverseGammaDistribution, self).__init__()
        self.gamma_distribution = _GammaDistribution(shape, rate, shape_learnable, rate_learnable, learnable)

    def expectation(self):
        """
        For shape > 1.
        """
        return self.gamma_distribution.rate / (self.gamma_distribution.shape - 1)

    def forward(self, log_x):
        return self.gamma_distribution(-log_x) - 2 * log_x


class ExpInverseGammaDistribution(nn.Module):
    """
    This is the distribution of X = log(Z) if Z is InverseGamma(shape, rate) distributed.
    """

    def __init__(self, shape=1e-3, rate=1e-3, shape_learnable=False, rate_learnable=False, learnable=False):
        super(ExpInverseGammaDistribution, self).__init__()
        self.igamma_distribution = _InverseGammaDistribution(shape, rate, shape_learnable, rate_learnable, learnable)

    def forward(self, x):
        return self.igamma_distribution(x) + x


class ExpGammaDistribution(nn.Module):
    """
    This is the distribution of X = log(Z) if Z is Gamma(shape, rate) distributed.
    """

    def __init__(self, shape=1e-3, rate=1e-3, shape_learnable=False, rate_learnable=False, learnable=False):
        super(ExpGammaDistribution, self).__init__()
        self.gamma_distribution = _GammaDistribution(shape, rate, shape_learnable, rate_learnable, learnable)

    def expectation(self):
        return expgamma_expectation(self.gamma_distribution.shape, self.gamma_distribution.rate)

    def forward(self, x):
        return self.gamma_distribution(x) + x


def expgamma_log_pdf(x, shape, rate):
    return gamma_log_pdf(x, shape, rate) + x


def expgamma_expectation(shape, rate):
    return torch.digamma(shape) - torch.log(rate)


"""
Useful hyperpriors
"""


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


class LogPrecisionExpGammaPrior(nn.Module):
    """
    Hyperprior over wreg is Gamma <-> hyperprior over log_wreg is expGamma
    Parametrized by shape and rate.
    """

    def __init__(self, shape=1e-3, rate=1e-3, shape_learnable=False, rate_learnable=False, learnable=False):
        super(LogPrecisionExpGammaPrior, self).__init__()
        self.expgamma_distribution = ExpGammaDistribution(shape, rate, shape_learnable, rate_learnable, learnable)

    def forward(self, x):
        return self.expgamma_distribution(x)


class LogEnergyExpGammaPrior(nn.Module):
    """
    Suitable as a prior over the location parameter (loc) for a Log-Normal distribution on the energy y. 
    """

    def __init__(self, w_reg, dof, nu=1.0, learnable=False):
        super(LogEnergyExpGammaPrior, self).__init__()

        self.register_buffer('w_reg', torch.Tensor([w_reg]))  # never learnable
        self.register_buffer('dof', torch.Tensor([dof]))  # never learnable

        self.nu = nn.Parameter(torch.Tensor([nu]), requires_grad=learnable)

    def expectation(self):
        return expgamma_expectation(0.5 * self.nu * self.dof, 0.5 * self.nu * self.w_reg)

    def forward(self, log_energy):
        return expgamma_log_pdf(log_energy, 0.5 * self.nu * self.dof, 0.5 * self.nu * self.w_reg)


class LogScaleNormalPrior(nn.Module):
    """
    Suitable as a prior on log(scale) parameters, such as a log standard deviation.
    """

    def __init__(self, loc, scale, learnable=False):
        super(LogScaleNormalPrior, self).__init__()
        self.normal = NormalDistribution(loc, scale, learnable)

    def forward(self, log_scale):
        return self.normal(log_scale)
